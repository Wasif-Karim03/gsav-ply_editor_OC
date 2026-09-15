"""
Frame-by-frame decoder for the GSAV codec.

Each frame's video-safe attributes are in a 3x5 atlas.
means_lo is provided separately as raw binary data.

Atlas layout (3 rows x 5 cols = 15 cells):
    Row 0: [mean_hi.x][mean_hi.y][mean_hi.z][scale.x  ][scale.y  ]
    Row 1: [scale.z  ][quat.w   ][quat.x   ][quat.y   ][quat.z   ]
    Row 2: [opacity  ][sh0.r    ][sh0.g     ][sh0.b    ][padding  ]
"""

import functools

import numpy as np
from gsply import GSData

from gscodec.common.types import QuantRanges, SHChunkData
from gscodec.constants import N_ATLAS_COLS
from gscodec.decoder.dequantization.strategies import (
    dequantize_anchor_8bit,
    dequantize_means_16bit_from_atlas,
)
from gscodec.decoder.sh_decompress import expand_sh
from gscodec.decoder.utils.helpers import inverse_log_transform

# 14 active channels + 1 padding in 3x5 grid
_N_CHANNELS = 15


@functools.lru_cache(maxsize=8)
def _inverse_morton_2d(side: int) -> np.ndarray:
    """Compute the inverse 2D Morton mapping for a side x side grid.

    The encoder places Gaussian i at grid position morton_map[i].
    This returns an array such that: original[i] = block.flat[inv[i]],
    i.e. inv[i] = morton_map[i] (same mapping, used to gather).
    """
    ys, xs = np.meshgrid(
        np.arange(side, dtype=np.uint32), np.arange(side, dtype=np.uint32), indexing="ij"
    )

    def part1by1(n: np.ndarray) -> np.ndarray:
        n = n & np.uint32(0x0000FFFF)
        n = (n | (n << 8)) & np.uint32(0x00FF00FF)
        n = (n | (n << 4)) & np.uint32(0x0F0F0F0F)
        n = (n | (n << 2)) & np.uint32(0x33333333)
        n = (n | (n << 1)) & np.uint32(0x55555555)
        return n

    codes = part1by1(xs.ravel()) | (part1by1(ys.ravel()) << np.uint32(1))
    # morton_map = argsort(codes): morton_map[i] = flat index for the i-th Gaussian
    # To reverse: read block.flat[morton_map[i]] to get Gaussian i's value
    return np.argsort(codes).astype(np.int64)


class ChunkDecoder:
    """Frame-by-frame chunk decoder — each frame is self-contained."""

    def __init__(
        self,
        ranges: QuantRanges,
        atlas_width: int,
        n_gaussians: int,
        n_atlas_cols: int = N_ATLAS_COLS,
        sh_bands: int = 0,
        has_mask: bool = False,
    ):
        """Initialize chunk decoder.

        Args:
            ranges: Global quantization ranges.
            atlas_width: Width of atlas in pixels.
            n_gaussians: Number of Gaussians per frame.
            n_atlas_cols: Number of columns in atlas grid.
            sh_bands: SH band level (0=none, 1-3).
        """
        self.ranges = ranges
        self.atlas_width = atlas_width
        self.n_gaussians = n_gaussians
        self.n_atlas_cols = n_atlas_cols
        self.sh_bands = sh_bands
        self.has_mask = has_mask

    def decode_frame(
        self,
        atlas: np.ndarray,
        means_lo: np.ndarray,
        sh_labels: np.ndarray | None = None,
        sh_chunk_data: SHChunkData | None = None,
    ) -> GSData:
        """Decode a frame atlas + means_lo into GSData.

        Args:
            atlas: [atlas_height, atlas_width] uint8 frame atlas (3x5).
            means_lo: [N, 3] uint8 raw lo-bytes [0, 255].
            sh_labels: [N] uint16 SH palette labels (from binary payload), or None.
            sh_chunk_data: SH chunk data for label → SH reconstruction, or None.

        Returns:
            Decoded GSData frame.
        """
        n = self.n_gaussians
        side = self.atlas_width // self.n_atlas_cols
        inv_morton = _inverse_morton_2d(side)

        channels = []
        for ch_idx in range(_N_CHANNELS):
            atlas_col = ch_idx % self.n_atlas_cols
            atlas_row = ch_idx // self.n_atlas_cols
            y_start = atlas_row * side
            x_start = atlas_col * side
            block = atlas[y_start : y_start + side, x_start : x_start + side]
            channels.append(block.flat[inv_morton[:n]])

        # Ch 0-2: means_hi
        means_hi = np.stack([channels[0], channels[1], channels[2]], axis=-1)
        means_log = dequantize_means_16bit_from_atlas(
            means_hi,
            means_lo,
            self.ranges.means_min,
            self.ranges.means_max,
        )
        means = inverse_log_transform(means_log)

        # Ch 3-5: scales
        scales_q = np.stack([channels[3], channels[4], channels[5]], axis=-1)
        scales = dequantize_anchor_8bit(scales_q, self.ranges.scales_min, self.ranges.scales_max)

        # Ch 6-9: quats (w, x, y, z)
        quats_q = np.stack([channels[6], channels[7], channels[8], channels[9]], axis=-1)
        quats_asin = dequantize_anchor_8bit(quats_q, self.ranges.quats_min, self.ranges.quats_max)
        # Inverse arcsine: sin(x * π/2) maps [-1,1] back to [-1,1]
        quats = np.sin(quats_asin * (np.pi / 2))
        quat_norms = np.linalg.norm(quats, axis=-1, keepdims=True)
        quat_norms = np.where(quat_norms > 0, quat_norms, np.ones_like(quat_norms))
        quats = quats / quat_norms

        # Ch 10: opacity
        opacity_q = channels[10].reshape(-1, 1)
        opacity_mins = np.array([self.ranges.opacity_min], dtype=np.float32)
        opacity_maxs = np.array([self.ranges.opacity_max], dtype=np.float32)
        opacities = dequantize_anchor_8bit(opacity_q, opacity_mins, opacity_maxs).squeeze(-1)
        presence = self.decode_presence(atlas)
        opacities = np.where(presence, opacities, -np.inf)

        # Ch 11-13: sh0 (stored as YCbCr, inverse to RGB)
        sh0_q = np.stack([channels[11], channels[12], channels[13]], axis=-1)
        sh0_ycbcr = dequantize_anchor_8bit(sh0_q, self.ranges.sh0_min, self.ranges.sh0_max)
        _inv_ycbcr = np.array(
            [
                [1.0, 0.0, 1.402],
                [1.0, -0.344, -0.714],
                [1.0, 1.772, 0.0],
            ],
            dtype=np.float32,
        )
        sh0 = sh0_ycbcr @ _inv_ycbcr.T

        # Reconstruct higher-order SH from labels + centroids + codebook
        # expand_sh returns [N, coeffs, 3] — keep this shape so gsply
        # correctly converts to channel-grouped PLY layout on write.
        if sh_labels is not None and sh_chunk_data is not None:
            shN = expand_sh(sh_labels, sh_chunk_data)
        else:
            shN = np.empty((n, 0, 3), dtype=np.float32)

        return GSData(
            means=means.astype(np.float32),
            scales=scales.astype(np.float32),
            quats=quats.astype(np.float32),
            opacities=opacities.astype(np.float32),
            sh0=sh0.astype(np.float32),
            shN=shN.astype(np.float32),
            masks=presence,
        )

    def decode_presence(self, atlas: np.ndarray) -> np.ndarray:
        """Decode exact [N] binary presence; absent metadata means all active."""
        if not self.has_mask:
            return np.ones(self.n_gaussians, dtype=bool)
        side = self.atlas_width // self.n_atlas_cols
        block = atlas[2 * side:3 * side, 4 * side:5 * side]
        values = block.flat[_inverse_morton_2d(side)[:self.n_gaussians]]
        if not np.isin(values, [16, 235]).all():
            raise ValueError("Invalid binary mask samples: expected lossless 16/235")
        return values == 16
