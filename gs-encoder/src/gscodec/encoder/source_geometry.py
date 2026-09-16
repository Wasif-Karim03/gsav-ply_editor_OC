"""Reuse source geometry for verified color-only, full-timeline exports."""

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
    ) -> None:
        """Keep edited color channels 11–13; restore all other source atlas bytes."""
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
        source_atlases = self.provider.decode_all_frames()
        if len(source_atlases) != len(atlases):
            raise ValueError("Source atlas frame count mismatch")
        for edited, source in zip(atlases, source_atlases, strict=True):
            if edited.shape != source.shape or edited.shape != (side * 3, side * 5):
                raise ValueError("Source atlas dimensions mismatch")
            # Row 2 holds opacity, three color tiles, then presence/padding.
            source[side * 2 : side * 3, side : side * 4] = edited[
                side * 2 : side * 3, side : side * 4
            ]
            edited[:] = source
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
