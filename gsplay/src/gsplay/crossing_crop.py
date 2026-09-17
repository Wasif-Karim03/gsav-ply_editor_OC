"""Explicit chunk-local crop decisions; source rows are not object identities."""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


POLICIES = ("Show only inside", "Keep crossing rows", "Remove crossing rows")


def filter_key(values):
    return json.dumps(asdict(values), sort_keys=True)


def choose_rows(inside, presence, policy):
    """Classify only frames where each row exists, within one encoded chunk."""
    if policy not in POLICIES:
        raise ValueError("Unknown crossing policy")
    inside, presence = np.asarray(inside), np.asarray(presence)
    if inside.dtype != np.bool_ or presence.dtype != np.bool_ or inside.shape != presence.shape:
        raise ValueError("Expected matching boolean frame/row masks")
    if inside.ndim != 2 or not len(inside):
        raise ValueError("Expected at least one frame")
    any_inside = (inside & presence).any(axis=0)
    any_outside = (~inside & presence).any(axis=0)
    if policy == POLICIES[0]:
        return inside & presence
    selected = any_inside if policy == POLICIES[1] else any_inside & ~any_outside
    return presence & selected[None, :]


@dataclass(frozen=True)
class CrossingPlan:
    source: str
    signature: str
    policy: str
    rows: int
    masks: tuple
    method: str = "Chunk-based"
    summary: str = "preview and export use the analyzed chunk masks."

    def mask(self, frame):
        return np.unpackbits(self.masks[frame], count=self.rows).astype(bool)

    def matches(self, model, config):
        return (
            str(Path(getattr(model, "path", "")).resolve()) == self.source
            and filter_key(config.filter_values) == self.signature
            and config.crossing_policy == self.policy
            and getattr(config, "crossing_method", "Chunk-based") == self.method
            and model.get_total_frames() == len(self.masks)
        )


def active_plan(model, config, *, required=False):
    if config is None or getattr(config, "crossing_policy", POLICIES[0]) == POLICIES[0]:
        return None
    plan = getattr(config, "crossing_plan", None)
    if plan is not None and plan.matches(model, config):
        return plan
    if required:
        raise ValueError("Crossing crop is not applied. Click Analyze & Apply in Filter first.")
    return None


def analyze(model, values, policy, device, progress=lambda message: None):
    """Decode independently of playback; retain packed masks, never full scenes."""
    from gsmod import FilterValues

    from src.domain.entities import GSTensor
    from src.gsplay.gsav_visibility import ExportVisibility
    from src.models.gsav import GsavModel

    if not isinstance(model, GsavModel):
        raise ValueError("Crossing choices currently require a GSAV source")
    if policy not in POLICIES[1:]:
        raise ValueError("Choose Keep or Remove crossing rows first")
    if values.invert:
        raise ValueError("Disable inverted filtering before analyzing crossing rows")
    spatial = replace(
        values, min_opacity=0.0, max_opacity=1.0, min_scale=0.0, max_scale=float("inf")
    )
    if spatial.is_neutral():
        raise ValueError("Choose and position a spatial boundary first")
    attributes = FilterValues(
        min_opacity=values.min_opacity,
        max_opacity=values.max_opacity,
        min_scale=values.min_scale,
        max_scale=values.max_scale,
    )
    n = model.get_total_frames()
    chunks = model.gsav_metadata.get("chunks", [])
    if (
        not chunks
        or chunks[0][0] != 0
        or chunks[-1][1] != n
        or any(
            start >= end or (i and start != chunks[i - 1][1])
            for i, (start, end) in enumerate(chunks)
        )
    ):
        raise ValueError("Source chunk boundaries are missing or invalid")
    decoder = GsavModel(model.path, device="cpu")
    masks = []
    count = None
    try:
        for start, end in chunks:
            spatial_masks, presence_masks, attribute_masks = [], [], []
            for index in range(start, end):
                data = decoder.get_gaussians_at_normalized_time(decoder.get_frame_time(index))
                present = decoder._raw(index)["presence"]
                rows = len(data.means)
                if count is None:
                    count = rows
                    if (n * rows + 7) // 8 > 256 * 1024**2:
                        raise ValueError(
                            "Timeline is too large for crossing analysis (256 MiB mask limit)"
                        )
                if (end - start) * rows > 128 * 1024**2:
                    raise ValueError("Encoded chunk is too large for crossing analysis")
                if rows != count:
                    raise ValueError("Source row count changes within the timeline")
                if str(device).startswith("cuda"):
                    data = GSTensor.from_gsdata(data, device=device)
                selected = []
                for criterion in (spatial, attributes):
                    capture = ExportVisibility()
                    cfg = SimpleNamespace(filter_values=criterion)
                    if str(device).startswith("cuda"):
                        capture.filter_gpu(data, cfg, None)
                    else:
                        capture.filter_cpu(data, cfg, None)
                    selected.append(capture.mask)
                spatial_masks.append(selected[0])
                attribute_masks.append(selected[1])
                presence_masks.append(present)
                progress(f"Analyzing boundary: {index + 1}/{n} frames")
            chosen = choose_rows(spatial_masks, presence_masks, policy)
            # Opacity/scale thresholds still apply per frame, even to kept crossings.
            chosen &= np.asarray(attribute_masks)
            masks.extend(np.packbits(frame) for frame in chosen)
    finally:
        decoder.on_shutdown()
    return CrossingPlan(str(model.path.resolve()), filter_key(values), policy, count, tuple(masks))


def analyze_motion(model, values, policy, directory, progress):
    """Opt-in estimated tracks; ambiguous fragments fall back to ordinary crop."""
    from src.gsplay.motion_crop.cache import VERSION, MotionCache, build, fingerprint
    from src.models.gsav import GsavModel

    if not isinstance(model, GsavModel):
        raise ValueError("Motion-aware crop requires a GSAV source")
    if policy not in POLICIES[1:] or values.invert:
        raise ValueError("Choose Keep/Remove crossing rows with non-inverted filtering")
    spatial = replace(values, min_opacity=0, max_opacity=1, min_scale=0, max_scale=float("inf"))
    if spatial.is_neutral():
        raise ValueError("Choose and position a spatial boundary first")
    progress("Checking motion cache…")
    key = fingerprint(model.path)
    root = Path(directory)
    destination = root / f"{key}-v{VERSION}"
    if not destination.exists():
        # Failed builds stay separate and are never reused as complete caches.
        import tempfile

        staging = Path(tempfile.mkdtemp(prefix="building-", dir=root)) / "cache"
        build(model.path, staging, progress=progress)
        staging.rename(destination)
    progress("Evaluating crop using cached motion…")
    cache = MotionCache(destination, model.path)
    try:
        masks, report = cache.evaluate(values, "keep" if policy == POLICIES[1] else "remove")
        fraction = report["estimated_track_sample_fraction"]
        return CrossingPlan(
            str(model.path.resolve()),
            filter_key(values),
            policy,
            cache.meta["rows"],
            tuple(masks),
            "Motion-aware (preview)",
            f"Motion masks applied. {fraction:.0%} of samples use estimated tracks; "
            f"{1 - fraction:.0%} use ordinary cropping. Coverage is not tracking accuracy. "
            "Preview and export use the same masks.",
        )
    finally:
        cache.close()
