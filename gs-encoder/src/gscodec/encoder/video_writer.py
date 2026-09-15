"""
Lossless VP9 video encoding for frame-by-frame GSAV codec.

Frame atlas layout (3 rows x 5 cols = 15 cells):
    Row 0: [mean_hi.x][mean_hi.y][mean_hi.z][scale.x  ][scale.y  ]
    Row 1: [scale.z  ][quat.w   ][quat.x   ][quat.y   ][quat.z   ]
    Row 2: [opacity  ][sh0.r    ][sh0.g     ][sh0.b    ][padding  ]

14 active channels + 1 padding (filled with MIN_VAL=16).
All channels are 8-bit video-safe [16, 235].
means_lo is stored as raw binary in a separate GSAV section.
Each cell is side x side pixels where side = ceil(sqrt(n_gaussians)).
Gaussians are placed along a 2D Morton (Z-order) curve within each cell.
GOP size should match chunk_size so I-frames align with chunk boundaries.
"""

import functools
import logging
import math
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict

import numpy as np

from gscodec.constants import (
    MIN_VAL,
    N_ATLAS_COLS,
    N_ATLAS_ROWS,
)

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def _morton_2d_order(side: int) -> np.ndarray:
    """Compute a 2D Morton (Z-order) curve mapping for a side x side grid.

    Returns an array of length side*side where entry i is the flat grid
    index that the i-th Gaussian (in sorted order) should be placed at.
    This improves 2D spatial locality vs row-major for video compression.
    """
    ys, xs = np.meshgrid(
        np.arange(side, dtype=np.uint32), np.arange(side, dtype=np.uint32), indexing="ij"
    )
    coords_y = ys.ravel()
    coords_x = xs.ravel()

    def part1by1(n: np.ndarray) -> np.ndarray:
        n = n & np.uint32(0x0000FFFF)
        n = (n | (n << 8)) & np.uint32(0x00FF00FF)
        n = (n | (n << 4)) & np.uint32(0x0F0F0F0F)
        n = (n | (n << 2)) & np.uint32(0x33333333)
        n = (n | (n << 1)) & np.uint32(0x55555555)
        return n

    codes = part1by1(coords_x) | (part1by1(coords_y) << np.uint32(1))
    return np.argsort(codes).astype(np.int64)


class RawFrameEntry(TypedDict):
    """Raw frame entry from IVF parsing (before keyframe marking)."""

    obu_offset: int
    obu_size: int


@functools.lru_cache(maxsize=8)
def _precompute_atlas_scatter(side: int, n_gaussians: int) -> np.ndarray:
    """Precompute scatter indices for all 15 atlas cells.

    Returns [15, N] array of flat atlas indices where each Gaussian's
    channel value should be written. Computed once per (side, N) pair.
    """
    morton_map = _morton_2d_order(side)
    atlas_width = side * N_ATLAS_COLS
    cell_flat = morton_map[:n_gaussians]  # [N]
    cell_row = cell_flat // side  # [N]
    cell_col = cell_flat % side  # [N]

    scatter = np.empty((15, n_gaussians), dtype=np.int64)
    for ch_idx in range(15):
        atlas_col = ch_idx % N_ATLAS_COLS
        atlas_row = ch_idx // N_ATLAS_COLS
        y = atlas_row * side + cell_row  # [N]
        x = atlas_col * side + cell_col  # [N]
        scatter[ch_idx] = y * atlas_width + x

    return scatter


def build_frame_atlas(
    means_hi: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacity: np.ndarray,
    sh0: np.ndarray,
    n_gaussians: int,
    presence: np.ndarray | None = None,
) -> np.ndarray:
    """Build frame atlas from fully quantized channels (means_lo excluded).

    Atlas layout (3 rows x 5 cols = 15 cells):
        Row 0: mean_hi.x | mean_hi.y | mean_hi.z | scale.x   | scale.y
        Row 1: scale.z   | quat.w    | quat.x    | quat.y    | quat.z
        Row 2: opacity   | sh0.r     | sh0.g     | sh0.b     | padding

    Args:
        means_hi: [N, 3] uint8 hi-byte of 16-bit means (video-safe).
        scales: [N, 3] uint8 quantized scales (video-safe).
        quats: [N, 4] uint8 quantized quaternions (video-safe).
        opacity: [N] uint8 quantized opacity (video-safe).
        sh0: [N, 3] uint8 quantized SH0 (video-safe).
        n_gaussians: Number of Gaussians (for padding).

    Returns:
        [atlas_height, atlas_width] uint8 grayscale atlas (3 rows x 5 cols).
    """
    side = math.ceil(math.sqrt(n_gaussians))
    if side % 2 == 1:
        side += 1

    atlas_width = side * N_ATLAS_COLS
    atlas_height = side * N_ATLAS_ROWS
    atlas = np.full(atlas_height * atlas_width, MIN_VAL, dtype=np.uint8)

    # Stack all 15 channels into [15, N] for vectorized scatter
    all_ch = np.empty((15, n_gaussians), dtype=np.uint8)
    all_ch[0] = means_hi[:, 0]
    all_ch[1] = means_hi[:, 1]
    all_ch[2] = means_hi[:, 2]
    all_ch[3] = scales[:, 0]
    all_ch[4] = scales[:, 1]
    all_ch[5] = scales[:, 2]
    all_ch[6] = quats[:, 0]
    all_ch[7] = quats[:, 1]
    all_ch[8] = quats[:, 2]
    all_ch[9] = quats[:, 3]
    all_ch[10] = opacity
    all_ch[11] = sh0[:, 0]
    all_ch[12] = sh0[:, 1]
    all_ch[13] = sh0[:, 2]
    if presence is None:
        all_ch[14] = MIN_VAL
    else:
        presence = np.asarray(presence)
        if presence.shape != (n_gaussians,):
            raise ValueError("presence must have shape [n_gaussians]")
        if presence.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
            raise ValueError("presence must have dtype bool or uint8")
        if not np.isin(presence, [0, 1]).all():
            raise ValueError("presence values must be 0 or 1")
        all_ch[14] = np.where(presence, MIN_VAL, 235)

    scatter = _precompute_atlas_scatter(side, n_gaussians)
    atlas[scatter.ravel()] = all_ch.ravel()

    return atlas.reshape(atlas_height, atlas_width)


def encode_to_ivf(
    atlases: list[np.ndarray],
    fps: int,
    gop_size: int | None = None,
) -> tuple[bytes, list[RawFrameEntry]]:
    """Encode atlas frames to lossless VP9 IVF.

    Args:
        atlases: List of grayscale atlas frames [H, W].
        fps: Frames per second.
        gop_size: GOP size (I-frame interval). Defaults to len(atlases).
                  Set to chunk_size to align I-frames with chunk boundaries.

    Returns:
        Tuple of (raw VP9 frame bytes, list of frame entries with offsets/sizes).
    """
    if not atlases:
        return b"", []

    if gop_size is None:
        gop_size = len(atlases)

    try:
        import xllvp9
    except ModuleNotFoundError as exc:
        if exc.name != "xllvp9":
            raise
        logger.info("xllvp9 unavailable; using FFmpeg lossless libvpx-vp9")
        ivf_data = _encode_ffmpeg(atlases, fps, gop_size)
    else:
        frames = np.stack(atlases)
        ivf_data = xllvp9.encode(frames, fps=fps, keyframe_interval=gop_size)
    return _parse_ivf(ivf_data)


def _encode_ffmpeg(atlases: list[np.ndarray], fps: int, gop_size: int) -> bytes:
    """Encode exact luma samples, neutral chroma and fixed keyframe intervals.

    A file input avoids subprocess pipe deadlocks and duplicate sequence buffers.
    This fallback changes encoding speed/size, not the GSAV container or SH payload.
    """
    height, width = atlases[0].shape
    if fps <= 0 or gop_size <= 0 or width % 2 or height % 2:
        raise ValueError("VP9 requires positive FPS/GOP and even atlas dimensions")
    with tempfile.TemporaryDirectory(prefix="gsav-vp9-") as directory:
        raw = Path(directory) / "frames.yuv"
        output = Path(directory) / "frames.ivf"
        chroma = bytes([128]) * (width * height // 2)
        with raw.open("wb") as stream:
            for atlas in atlases:
                if atlas.shape != (height, width) or atlas.dtype != np.uint8:
                    raise ValueError("Atlas frames must have matching uint8 dimensions")
                stream.write(atlas.tobytes())
                stream.write(chroma)
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                str(raw),
                "-c:v",
                "libvpx-vp9",
                "-lossless",
                "1",
                "-g",
                str(gop_size),
                "-keyint_min",
                str(gop_size),
                "-auto-alt-ref",
                "0",
                "-lag-in-frames",
                "0",
                "-color_range",
                "tv",
                "-threads",
                "4",
                "-row-mt",
                "1",
                "-cpu-used",
                "4",
                "-f",
                "ivf",
                str(output),
            ],
            capture_output=True,
            timeout=3600,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "FFmpeg VP9 encoding failed: " + result.stderr.decode(errors="replace")[-3000:]
            )
        return output.read_bytes()


def _parse_ivf(ivf_data: bytes) -> tuple[bytes, list[RawFrameEntry]]:
    """Parse IVF container, extract raw VP9 frames.

    Args:
        ivf_data: Raw IVF file contents.

    Returns:
        Tuple of (concatenated frame bytes, list of frame entries).
    """
    if len(ivf_data) < 32 or ivf_data[:4] != b"DKIF":
        raise ValueError("Invalid IVF data")

    obu_chunks = []
    frame_entries: list[RawFrameEntry] = []
    offset = 32  # IVF header
    obu_offset = 0

    while offset < len(ivf_data):
        if offset + 12 > len(ivf_data):
            break
        frame_size = struct.unpack("<I", ivf_data[offset : offset + 4])[0]
        offset += 12
        if offset + frame_size > len(ivf_data):
            break

        frame_data = ivf_data[offset : offset + frame_size]
        obu_chunks.append(frame_data)
        frame_entries.append(RawFrameEntry(obu_offset=obu_offset, obu_size=frame_size))
        obu_offset += frame_size
        offset += frame_size

    logger.info(f"Extracted {len(frame_entries)} frames, total {obu_offset} bytes")
    return b"".join(obu_chunks), frame_entries


def compute_atlas_dimensions(n_gaussians: int) -> tuple[int, int, int]:
    """Compute atlas dimensions for given number of Gaussians.

    Atlas layout is 3 rows x 5 cols, so:
        width = side x 5
        height = side x 3

    Args:
        n_gaussians: Number of Gaussians.

    Returns:
        Tuple of (atlas_side, width, height).
    """
    side = math.ceil(math.sqrt(n_gaussians))
    if side % 2 == 1:
        side += 1
    return side, side * N_ATLAS_COLS, side * N_ATLAS_ROWS
