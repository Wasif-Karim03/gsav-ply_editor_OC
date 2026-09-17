"""Disk-backed, source-fingerprinted research cache; no editor mutations."""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.gsplay.motion_crop.tracking import Features, MatchSettings, Tracker


VERSION = 3
FIELDS = {
    "means": (3, "float32"),
    "scales": (3, "float32"),
    "opacities": (1, "float32"),
    "presence": (1, "bool"),
    "tracks": (1, "int32"),
}


def fingerprint(source):
    with Path(source).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(source, destination, *, radius_fraction=0.03, max_bytes=1024**3, progress=print):
    """Decode once and cache geometry/tracks; complete.json is published last.

    Radius is a prototype parameter relative to the first active frame's robust
    bounds. It is not calibrated for all motion speeds or coordinate systems.
    """
    from src.models.gsav import GsavModel

    if not np.isfinite(radius_fraction) or radius_fraction <= 0:
        raise ValueError("Invalid radius fraction")
    source, destination = Path(source), Path(destination)
    source_hash = fingerprint(source)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = GsavModel(source, "cpu")
    arrays = {}
    try:
        n = model.get_total_frames()
        tracker = None
        steps = []
        for index in range(n):
            data = model.get_gaussians_at_normalized_time(model.get_frame_time(index))
            presence = model._raw(index)["presence"]
            rows = len(data.means)
            if not arrays:
                needed = n * rows * 33
                if needed > max_bytes:
                    raise ValueError(f"Cache requires {needed} bytes; budget is {max_bytes}")
                for key, (width, dtype) in FIELDS.items():
                    shape = (n, rows, width) if width > 1 else (n, rows)
                    arrays[key] = np.lib.format.open_memmap(
                        destination / f"{key}.npy", mode="w+", dtype=dtype, shape=shape
                    )
            if rows != arrays["presence"].shape[1]:
                raise ValueError("GSAV frame storage dimensions changed")
            for key in ("means", "scales", "opacities"):
                value = getattr(data, key)
                arrays[key][index] = value.reshape(arrays[key][index].shape)
            arrays["presence"][index] = presence
            arrays["tracks"][index] = -1
            active = np.flatnonzero(presence)
            if tracker is None and len(active):
                low, high = np.quantile(data.means[active], [0.01, 0.99], axis=0)
                radius = max(float(np.linalg.norm(high - low)) * radius_fraction, 1e-6)
                tracker = Tracker(MatchSettings(radius))
            if tracker is not None:
                colors = data.sh0[active]
                if not data.is_sh0_rgb:
                    colors = colors * 0.28209479177387814 + 0.5
                features = Features(
                    data.means[active],
                    colors,
                    np.sort(np.log(np.maximum(data.scales[active], 1e-12)), axis=1),
                )
                ids, links = tracker.update(features)
                arrays["tracks"][index, active] = ids
                steps.append(
                    {"frame": index, "active": len(active), "linked": int((links >= 0).sum())}
                )
            progress(f"Motion cache: {index + 1}/{n}")
        if fingerprint(source) != source_hash:
            raise ValueError("Source changed during cache construction")
        for value in arrays.values():
            value.flush()
        metadata = {
            "version": VERSION,
            "source_sha256": source_hash,
            "frames": n,
            "rows": arrays["presence"].shape[1],
            "tracks": tracker.next_id if tracker else 0,
            "settings": asdict(tracker.settings) if tracker else None,
            "seconds": time.perf_counter() - started,
            "steps": steps,
            "identity": "estimated; no authoritative IDs supplied",
            "complete": True,
        }
        (destination / "complete.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    finally:
        model.on_shutdown()
        for value in arrays.values():
            value._mmap.close()


class MotionCache:
    def __init__(self, directory, source):
        directory = Path(directory)
        self.arrays = {}
        self.meta = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
        if self.meta.get("version") != VERSION or self.meta.get("complete") is not True:
            raise ValueError("Incomplete or incompatible motion cache")
        if self.meta["source_sha256"] != fingerprint(source):
            raise ValueError("Motion cache belongs to different source content")
        try:
            n, rows = self.meta["frames"], self.meta["rows"]
            tracks = self.meta["tracks"]
            if (
                any(type(value) is not int for value in (n, rows, tracks))
                or n <= 0
                or rows <= 0
                or n * rows * 33 > 1024**3
                or not 0 <= tracks <= n * rows
            ):
                raise ValueError("Invalid motion cache metadata or memory budget")
            for key, (width, dtype) in FIELDS.items():
                value = np.load(directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                self.arrays[key] = value
                shape = (n, rows, width) if width > 1 else (n, rows)
                if value.shape != shape or value.dtype != np.dtype(dtype):
                    raise ValueError("Invalid motion cache dimensions or type")
        except Exception:
            self.close()
            raise

    def close(self):
        for value in self.arrays.values():
            value._mmap.close()
        self.arrays.clear()

    def evaluate(self, values, policy, min_observations=3):
        """Return packed masks; short fragments fall back to per-frame cropping.

        Policies apply to observed fragments only, not verified whole objects.
        No correspondence is inferred across a missed/ambiguous association.
        """
        from dataclasses import replace

        from gsmod import FilterValues
        from gsmod.filter.apply import compute_filter_mask

        if policy not in ("keep", "remove") or min_observations < 2:
            raise ValueError("Invalid crop policy or minimum observations")
        if values.invert:
            raise ValueError("Inverted filters are not supported in this prototype")
        spatial = replace(values, min_opacity=0, max_opacity=1, min_scale=0, max_scale=float("inf"))
        attributes = FilterValues(
            min_opacity=values.min_opacity,
            max_opacity=values.max_opacity,
            min_scale=values.min_scale,
            max_scale=values.max_scale,
        )
        count = self.meta["tracks"]
        inside_seen = np.zeros(count, bool)
        outside_seen = np.zeros(count, bool)
        lengths = np.zeros(count, np.int32)
        packed_inside, packed_attributes = [], []
        started = time.perf_counter()
        for index in range(self.meta["frames"]):
            data = SimpleNamespace(
                **{key: self.arrays[key][index] for key in ("means", "scales", "opacities")},
                is_scales_ply=False,
                is_opacities_ply=False,
            )
            present = self.arrays["presence"][index]
            inside = compute_filter_mask(data, spatial) & present
            ids = self.arrays["tracks"][index, present]
            if len(ids) and (
                ids.min() < 0 or ids.max() >= count or len(np.unique(ids)) != len(ids)
            ):
                raise ValueError("Invalid or duplicated track IDs in cache")
            inside_seen[ids] |= inside[present]
            outside_seen[ids] |= ~inside[present]
            lengths[ids] += 1
            packed_inside.append(np.packbits(inside))
            packed_attributes.append(np.packbits(compute_filter_mask(data, attributes) & present))
        eligible = lengths >= min_observations
        selection = inside_seen if policy == "keep" else inside_seen & ~outside_seen
        masks, eligible_samples, total = [], 0, 0
        for index in range(self.meta["frames"]):
            present = self.arrays["presence"][index]
            ids = self.arrays["tracks"][index, present]
            mask = np.unpackbits(packed_inside[index], count=self.meta["rows"]).astype(bool)
            active_mask = mask[present]
            reliable = eligible[ids]
            active_mask[reliable] = selection[ids[reliable]]
            mask[present] = active_mask
            mask &= np.unpackbits(packed_attributes[index], count=self.meta["rows"]).astype(bool)
            masks.append(np.packbits(mask))
            eligible_samples += int(reliable.sum())
            total += len(ids)
        report = {
            "seconds": time.perf_counter() - started,
            "estimated_track_sample_fraction": eligible_samples / max(1, total),
            "median_track_frames": float(np.median(lengths)) if count else 0,
            "full_timeline_tracks": int((lengths == self.meta["frames"]).sum()),
            "track_count": count,
            "samples": total,
            "quality_validated": False,
        }
        return np.asarray(masks), report
