"""
Common type definitions for the GSAV codec.

This module defines the core data structures used throughout the encoder
and decoder pipelines for the GSAV format.

GSAV Layout (frame-by-frame):
    Header (128B) -> Ranges (112B) -> ChunkIndex (16B x nChunks) -> FrameIndex (8B x nFrames)
    -> MeansLoPayload (nGaussians*3 bytes per frame) -> VideoPayload -> AudioPayload (optional)

Video payload stores 3x5 frame atlases (all attributes except means_lo per frame).
Audio payload stores a single OGG/Opus bitstream synced to video via fps.
"""

from dataclasses import dataclass
from typing import NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray


@dataclass
class QuantRanges:
    """Global quantization ranges for frame encoding.

    These ranges are computed across all frames in the sequence and used
    to normalize values before quantization.
    """

    means_min: NDArray[np.float32]  # [3]
    means_max: NDArray[np.float32]  # [3]
    scales_min: NDArray[np.float32]  # [3]
    scales_max: NDArray[np.float32]  # [3]
    quats_min: NDArray[np.float32]  # [4]
    quats_max: NDArray[np.float32]  # [4]
    opacity_min: float
    opacity_max: float
    sh0_min: NDArray[np.float32]  # [3]
    sh0_max: NDArray[np.float32]  # [3]


class ChunkEntry(TypedDict):
    """Index entry for a chunk in the GSAV file.

    Chunks exist only for GOP/I-frame alignment. offset=0, size=0.
    """

    start_frame: int
    end_frame: int  # Inclusive
    offset: int
    size: int


class FrameEntry(TypedDict):
    """Index entry for a video frame (OBU) in the GSAV file.

    Offsets are RELATIVE to video_payload_offset for easy range calculations.
    The high bit of size indicates keyframe status for seeking.
    """

    offset: int  # Relative to video_payload_offset
    size: int  # High bit (0x80000000) = keyframe flag


class GSAVHeader(TypedDict):
    """GSAV file header structure (128 bytes).

    Designed for streaming: all indices come before payloads so client
    can download the "map" first, then fetch data via Range requests.

    Atlas layout fields (self-describing):
        atlas_side: Gaussian grid side (ceil(sqrt(n_gaussians)) rounded even)
        n_atlas_cols: Number of columns in atlas grid (e.g., 6)
        n_atlas_rows: Number of rows in atlas grid (e.g., 3)

    Derived dimensions:
        atlas_width = atlas_side * n_atlas_cols
        atlas_height = atlas_side * n_atlas_rows
    """

    magic: bytes  # b"GSAV"
    version: int  # 1
    n_gaussians: int
    n_frames: int
    n_chunks: int
    chunk_size: int  # Frames per chunk (= GOP size for I-frame alignment)
    atlas_side: int  # Gaussian grid side (u16)
    n_atlas_cols: int  # Atlas columns (u8)
    n_atlas_rows: int  # Atlas rows (u8)
    fps: int
    flags: int
    # Section offsets (absolute file positions)
    ranges_offset: int
    chunk_index_offset: int
    frame_index_offset: int
    means_lo_payload_offset: int
    video_payload_offset: int
    codec: str  # WebCodecs codec string, e.g. "av01.0.04M.08" or "vp09.00.10.08"
    # Audio (optional, 0 when absent)
    audio_payload_offset: int
    audio_payload_size: int
    # Reserved (always 0, kept for binary format compatibility)
    means_hi_payload_offset: int
    # SH compression (0 when absent)
    sh_bands: int  # 0=none, 1=SH1, 2=SH2, 3=SH3
    sh_payload_offset: int  # Absolute file offset of SH payload (0 if sh_bands=0)
    static_asset_offset: NotRequired[int]
    static_asset_size: NotRequired[int]
    static_asset_encoding: NotRequired[int]


# Flag for keyframe in FrameEntry.size
KEYFRAME_FLAG = 0x80000000
SIZE_MASK = 0x7FFFFFFF

# Header flags
HAS_AUDIO_FLAG = 0x0001
HAS_SH_FLAG = 0x0002
HAS_MASK_FLAG = 0x0010
HAS_STATIC_ASSET_FLAG = 0x0020
STATIC_ENCODING_NATIVE = 1
KNOWN_V3_FLAGS = HAS_AUDIO_FLAG | HAS_SH_FLAG | HAS_MASK_FLAG | HAS_STATIC_ASSET_FLAG


@dataclass
class SHChunkData:
    """Global SH compression data for the entire video.

    Centroids and codebook are shared across all chunks/frames.
    Labels are per-frame, stored as a flat list across all frames.

    Attributes:
        codebook: [256] float32 optimal scalar codebook (Lloyd-Max quantizer).
        centroids: [K, coeffs_per_channel * 3] uint8 codebook indices per palette entry.
        labels: List of [N] uint16 arrays, one per frame (flat across all chunks).
        n_centroids: Number of palette entries.
        sh_bands: SH band level (1-3).
    """

    codebook: NDArray[np.float32]
    centroids: NDArray[np.uint8]
    labels: list[NDArray[np.uint16]]
    n_centroids: int
    sh_bands: int


def get_atlas_dimensions(header: GSAVHeader) -> tuple[int, int]:
    """Compute (atlas_width, atlas_height) from header fields.

    Args:
        header: GSAV header with atlas_side, n_atlas_cols, n_atlas_rows.

    Returns:
        Tuple of (width, height) in pixels.
    """
    return (
        header["atlas_side"] * header["n_atlas_cols"],
        header["atlas_side"] * header["n_atlas_rows"],
    )
