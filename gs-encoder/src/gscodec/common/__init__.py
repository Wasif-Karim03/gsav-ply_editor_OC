"""
gscodec.common - Shared types and binary format utilities.

This module contains common data structures and I/O utilities used by both
the encoder and decoder for the GSAV streaming format.
"""

from gscodec.common.binary_format import (
    CHUNK_ENTRY_SIZE,
    FRAME_ENTRY_SIZE,
    GSAV_MAGIC,
    GSAV_VERSION,
    HEADER_SIZE,
    RANGES_SIZE,
    get_frame_size,
    is_keyframe,
    make_frame_entry,
    read_chunk_index,
    read_frame_index,
    read_header,
    read_ranges,
    write_chunk_index,
    write_frame_index,
    write_header,
    write_ranges,
)
from gscodec.common.types import (
    KEYFRAME_FLAG,
    SIZE_MASK,
    ChunkEntry,
    FrameEntry,
    GSAVHeader,
    QuantRanges,
)

__all__ = [
    # Types
    "ChunkEntry",
    "FrameEntry",
    "GSAVHeader",
    "QuantRanges",
    # Constants
    "GSAV_MAGIC",
    "GSAV_VERSION",
    "HEADER_SIZE",
    "RANGES_SIZE",
    "CHUNK_ENTRY_SIZE",
    "FRAME_ENTRY_SIZE",
    "KEYFRAME_FLAG",
    "SIZE_MASK",
    # I/O
    "write_header",
    "read_header",
    "write_ranges",
    "read_ranges",
    "write_chunk_index",
    "read_chunk_index",
    "write_frame_index",
    "read_frame_index",
    # Helpers
    "is_keyframe",
    "get_frame_size",
    "make_frame_entry",
]
