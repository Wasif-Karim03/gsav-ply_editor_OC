"""Configuration dataclasses for the GSAV encoder."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class VideoConfig:
    """Configuration for VP9 lossless video encoding."""

    fps: int = 30
    """Frames per second for the output video."""


@dataclass
class ChunkConfig:
    """Configuration for chunk processing.

    A chunk is a group of frames processed together with shared sorting.
    """

    size: int = 30
    """Number of frames per chunk."""

    identity_mode: Literal["auto", "stable", "unstructured", "preserve"] = "auto"
    """Preserve retains supplied rows; stable requires identities; unstructured matches."""

    lossy_pruning: bool = False
    """Optionally prune low-opacity/small stable slots, in addition to never-active slots."""

    reuse_inactive: bool = True
    """Reuse quantized attributes and SH labels while slots are disabled."""

    keyframe_snap: bool = True
    """Snap small attribute changes only across continuously active samples."""

    gsflow_metadata: Path | None = None
    """Path to GSFlow metadata.json for chunk boundaries and per-chunk ranges."""

    matching_enabled: bool = False
    """Enable temporal matching for unstructured point clouds with varying counts."""

    matcher_k_passes: int = 10
    """Number of iterative NN matching passes for conflict resolution."""

    lo_snap_k: int | tuple[int, int, int] = 0
    """Round lo bytes to nearest K for compression (1=off, 0=auto).
    Auto computes optimal K from scene Gaussian scale distribution.
    Use odd K coprime with 256 for best PSNR (e.g. 27, 35, 51, 105)."""

    sh_bands: int = -1
    """SH band level for higher-order SH compression.
    -1=auto-detect from source data (default), 0=DC only, 1=SH1 (3 coeffs),
    2=SH2 (8), 3=SH3 (15)."""

    sh_max_centroids: int = 65535
    """Maximum number of SH palette entries (up to 65,535 = uint16 max)."""

    parallel_chunks: int = 1
    """Number of worker processes for parallel chunk encoding (1 = serial).
    Only used when temporal matching is enabled (the expensive path). Each worker
    holds its own CUDA context; the single GPU saturates around 4 workers.
    0 = auto (min(4, cpu_count // 8))."""
