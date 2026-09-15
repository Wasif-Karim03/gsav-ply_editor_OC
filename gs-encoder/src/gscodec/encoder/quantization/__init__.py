"""
Quantization functions for the GSAV codec.
"""

from gscodec.encoder.quantization.strategies import (
    quantize_anchor_8bit,
    quantize_means_16bit_split,
)

__all__ = [
    "quantize_means_16bit_split",
    "quantize_anchor_8bit",
]
