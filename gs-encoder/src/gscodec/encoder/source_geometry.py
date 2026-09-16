"""Reuse source geometry for verified color/visibility, full-timeline exports."""

from pathlib import Path

import numpy as np

from gscodec.common.types import HAS_MASK_FLAG, QuantRanges
from gscodec.decoder.providers import GSAVFileProvider


class SourceGeometry:
    """Source bytes and atlas channels that must not be requantized for color edits."""

    def __init__(self, path: str | Path):
        self.data = Path(path).read_bytes()
        self.provider = GSAVFileProvider(self.data)
        self.header = self.provider.header
        h = self.header
        if (
            h["version"] not in (1, 3)
            or h["means_hi_payload_offset"]
            or h.get("static_asset_size", 0)
            or (h["n_atlas_rows"], h["n_atlas_cols"]) != (3, 5)
        ):
            raise ValueError("Source geometry reuse does not support this GSAV layout")
        lo_end = h["sh_payload_offset"] or h["video_payload_offset"]
        self.means_lo = self.data[h["means_lo_payload_offset"] : lo_end]
        start = h["audio_payload_offset"]
        self.audio = self.data[start : start + h["audio_payload_size"]] if start else b""
        self.mask_flag = h["flags"] & HAS_MASK_FLAG

    def merge(
        self,
        atlases: list[np.ndarray],
        ranges: QuantRanges,
        *,
        rows: int,
        fps: int,
        chunk_size: int,
        visibility: list[np.ndarray] | None = None,
    ) -> None:
        """Restore source geometry; update color channels and optional visibility."""
        h = self.header
        if (len(atlases), rows, fps, chunk_size) != (
            h["n_frames"],
            h["n_gaussians"],
            h["fps"],
            h["chunk_size"],
        ):
            raise ValueError("Source geometry reuse requires the complete unchanged timeline")
        expected = [
            [i, min(i + chunk_size, len(atlases))] for i in range(0, len(atlases), chunk_size)
        ]
        actual = [[c["start_frame"], c["end_frame"] + 1] for c in self.provider.chunk_index]
        if actual != expected:
            raise ValueError("Source geometry reuse requires aligned chunk boundaries")
        side = h["atlas_side"]
        if visibility is not None:
            if len(visibility) != len(atlases) or any(
                mask.dtype != np.bool_ or mask.shape != (rows,) for mask in visibility
            ):
                raise ValueError("Invalid source visibility dimensions or type")
            from gscodec.encoder.video_writer import _precompute_atlas_scatter

            mask_indices = _precompute_atlas_scatter(side, rows)[14]
            self.mask_flag = HAS_MASK_FLAG
        source_atlases = self.provider.decode_all_frames()
        if len(source_atlases) != len(atlases):
            raise ValueError("Source atlas frame count mismatch")
        for index, (edited, source) in enumerate(zip(atlases, source_atlases, strict=True)):
            if edited.shape != source.shape or edited.shape != (side * 3, side * 5):
                raise ValueError("Source atlas dimensions mismatch")
            # Row 2 holds opacity, three color tiles, then presence/padding.
            source[side * 2 : side * 3, side : side * 4] = edited[
                side * 2 : side * 3, side : side * 4
            ]
            edited[:] = source
            if visibility is not None:
                if h["flags"] & HAS_MASK_FLAG and np.any(
                    visibility[index] & (source.flat[mask_indices] != 16)
                ):
                    raise ValueError("Export visibility cannot revive absent source rows")
                # Keep original geometry and SH rows, replacing only visibility.
                # 16 is present and 235 absent in the version-3 mask tile.
                edited[side * 2 : side * 3, side * 4 : side * 5] = 235
                edited.flat[mask_indices] = np.where(visibility[index], 16, 235)
        original = self.provider.ranges
        for name in (
            "means_min",
            "means_max",
            "scales_min",
            "scales_max",
            "quats_min",
            "quats_max",
            "opacity_min",
            "opacity_max",
        ):
            setattr(ranges, name, getattr(original, name))
