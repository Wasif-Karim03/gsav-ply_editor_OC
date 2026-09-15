"""
gscodec.encoder - Encode dynamic 3D Gaussian Splatting sequences to GSAV format.

This module provides the encoding pipeline for compressing GS sequences.
The main entry point is the `SequenceEncoder` class.
"""

from gscodec.encoder.chunk_encoder import ChunkEncoder
from gscodec.encoder.config import ChunkConfig, VideoConfig
from gscodec.encoder.sequence_encoder import SequenceEncoder

__all__ = [
    "SequenceEncoder",
    "ChunkEncoder",
    "VideoConfig",
    "ChunkConfig",
]
