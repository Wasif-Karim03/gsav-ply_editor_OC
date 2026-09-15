"""
Dequantization functions for the GSAV codec.
"""

from gscodec.decoder.dequantization.strategies import (
    dequantize_anchor_8bit,
    dequantize_means_16bit_from_atlas,
)

__all__ = [
    "dequantize_means_16bit_from_atlas",
    "dequantize_anchor_8bit",
]
