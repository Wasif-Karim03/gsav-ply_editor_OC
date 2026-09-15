"""
gscodec.decoder - Decode GSAV files to dynamic 3D Gaussian Splatting sequences.

This module provides the decoding pipeline for decompressing GS sequences
from GSAV container files. The main entry point is the `SequenceDecoder` class.
"""

from gsply import GSData

from gscodec.decoder.chunk_decoder import ChunkDecoder
from gscodec.decoder.providers import GSAVFileProvider
from gscodec.decoder.sequence_decoder import SequenceDecoder

__all__ = [
    "SequenceDecoder",
    "ChunkDecoder",
    "GSAVFileProvider",
    "GSData",
]
