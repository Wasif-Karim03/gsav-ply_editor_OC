"""
Dequantization functions for the GSAV codec.

Reverses the quantization process:
- 16-bit means from hi (video-safe) + lo (raw binary) uint8 back to float32
- 8-bit attributes back to float32 (global ranges)
"""

import numpy as np

from gscodec.constants import (
    LO_BASE,
    MIN_VAL,
    SCALE_8BIT,
    SCALE_16BIT,
)


def dequantize_means_16bit_from_atlas(
    hi_u8: np.ndarray,
    lo_u8: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> np.ndarray:
    """Dequantize means from hi (video-safe) + lo (raw) uint8 back to float32.

    Reconstructs: quantized = (hi - 16) * 256 + lo
    normalized = quantized / 56319
    result = normalized * (maxs - mins) + mins

    Args:
        hi_u8: [N, 3] uint8 hi-byte in video-safe range [16, 235].
        lo_u8: [N, 3] uint8 lo-byte in raw range [0, 255].
        mins: [3] per-channel minimum values.
        maxs: [3] per-channel maximum values.

    Returns:
        [N, 3] float32 array (log-space means).
    """
    hi = hi_u8.astype(np.float32) - MIN_VAL
    lo = lo_u8.astype(np.float32)
    quantized = hi * LO_BASE + lo
    normalized = quantized / SCALE_16BIT
    scale = np.where(maxs > mins, maxs - mins, np.ones_like(mins))
    result: np.ndarray = normalized * scale + mins
    return result


def dequantize_anchor_8bit(
    data: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> np.ndarray:
    """Dequantize 8-bit attribute back to float32 (global ranges).

    Args:
        data: [N, C] uint8 array in video-safe range [16, 235].
        mins: [C] per-channel minimum values.
        maxs: [C] per-channel maximum values.

    Returns:
        [N, C] float32 array.
    """
    normalized = (data.astype(np.float32) - MIN_VAL) / SCALE_8BIT
    scale = np.where(maxs > mins, maxs - mins, np.ones_like(mins))
    result: np.ndarray = normalized * scale + mins
    return result
