"""
Binary format I/O utilities for the GSAV container format.

GSAV is optimized for HTTP streaming with random access:

    Header (128B) → Ranges (112B) → ChunkIndex → FrameIndex → MeansLoPayload → VideoPayload → AudioPayload (optional)

The "Index-First" design allows clients to:
1. Download Header + Ranges + Indices in one request (~100KB)
2. Calculate exact byte ranges for any chunk/frame
3. Fetch geometry and video data independently via Range requests
4. Seek to any frame by finding nearest I-frame (chunk boundary)

Header (128 bytes):
    - magic: b"GSAV" (4 bytes)
    - version: uint32 (= 1)
    - n_gaussians: uint32
    - n_frames: uint32
    - n_chunks: uint32
    - chunk_size: uint32 (frames per chunk = GOP size)
    - atlas_side: uint16 (gaussian grid side)
    - n_atlas_cols: uint8 (atlas columns, e.g., 5)
    - n_atlas_rows: uint8 (atlas rows, e.g., 3)
    - fps: uint16
    - flags: uint16 (bit 0 = HAS_AUDIO)
    - ranges_offset: uint32
    - chunk_index_offset: uint32
    - frame_index_offset: uint32
    - means_lo_payload_offset: uint32
    - video_payload_offset: uint32
    - codec: 16 bytes (null-padded UTF-8 WebCodecs codec string)
    - audio_payload_offset: uint32 (0 if no audio)
    - audio_payload_size: uint32 (0 if no audio)
    - means_hi_payload_offset: uint32 (0 if lossless)
    - reserved: 48 bytes
"""

import struct
from typing import BinaryIO

import numpy as np

from gscodec.common.types import (
    KEYFRAME_FLAG,
    SIZE_MASK,
    ChunkEntry,
    FrameEntry,
    GSAVHeader,
    QuantRanges,
    SHChunkData,
)

# Format constants
GSAV_MAGIC = b"GSAV"
GSAV_VERSION = 3
SUPPORTED_VERSIONS = (1, 3)  # The v2 software branch writes wire version 1.
HEADER_SIZE = 128
RANGES_SIZE = 112  # 3+3+3+3+4+4+1+1+3+3 = 28 floats × 4 = 112 bytes
CHUNK_ENTRY_SIZE = 16  # start_frame(4) + end_frame(4) + offset(4) + size(4)
FRAME_ENTRY_SIZE = 8  # offset(4) + size(4)


def write_header(f: BinaryIO, header: GSAVHeader) -> None:
    """Write GSAV header to file.

    Args:
        f: Binary file handle opened for writing.
        header: Header data to write.
    """
    from gscodec.common.v3 import validate_extensions

    validate_extensions(header)
    data = struct.pack(
        "<4s I I I I I H B B H H I I I I I 16s I I I B I 43x",
        header["magic"],
        header["version"],
        header["n_gaussians"],
        header["n_frames"],
        header["n_chunks"],
        header["chunk_size"],
        header["atlas_side"],
        header["n_atlas_cols"],
        header["n_atlas_rows"],
        header["fps"],
        header["flags"],
        header["ranges_offset"],
        header["chunk_index_offset"],
        header["frame_index_offset"],
        header["means_lo_payload_offset"],
        header["video_payload_offset"],
        header["codec"].encode("utf-8").ljust(16, b"\x00"),
        header["audio_payload_offset"],
        header["audio_payload_size"],
        header["means_hi_payload_offset"],
        header["sh_bands"],
        header["sh_payload_offset"],
    )
    if header["version"] == 3:
        data = bytearray(data)
        struct.pack_into("<II", data, 85, header.get("static_asset_offset", 0),
                         header.get("static_asset_size", 0))
        from gscodec.common.v3 import static_encoding
        data[93] = static_encoding(header)
    f.write(data)


def read_header(f: BinaryIO) -> GSAVHeader:
    """Read GSAV header from file.

    Args:
        f: Binary file handle opened for reading.

    Returns:
        Parsed header data.

    Raises:
        ValueError: If magic bytes don't match GSAV format.
    """
    data = f.read(HEADER_SIZE)
    if len(data) < HEADER_SIZE:
        raise ValueError(f"File too small for GSAV header (got {len(data)} bytes)")

    unpacked = struct.unpack("<4s I I I I I H B B H H I I I I I 16s I I I B I 43x", data)

    magic = unpacked[0]
    if magic != GSAV_MAGIC:
        raise ValueError(f"Invalid GSAV magic: {magic!r}, expected {GSAV_MAGIC!r}")

    codec = unpacked[16].rstrip(b"\x00").decode("utf-8")

    header = GSAVHeader(
        magic=magic,
        version=unpacked[1],
        n_gaussians=unpacked[2],
        n_frames=unpacked[3],
        n_chunks=unpacked[4],
        chunk_size=unpacked[5],
        atlas_side=unpacked[6],
        n_atlas_cols=unpacked[7],
        n_atlas_rows=unpacked[8],
        fps=unpacked[9],
        flags=unpacked[10],
        ranges_offset=unpacked[11],
        chunk_index_offset=unpacked[12],
        frame_index_offset=unpacked[13],
        means_lo_payload_offset=unpacked[14],
        video_payload_offset=unpacked[15],
        codec=codec,
        audio_payload_offset=unpacked[17],
        audio_payload_size=unpacked[18],
        means_hi_payload_offset=unpacked[19],
        sh_bands=unpacked[20],
        sh_payload_offset=unpacked[21],
    )
    if header["version"] == 3:
        header["static_asset_offset"], header["static_asset_size"] = struct.unpack_from("<II", data, 85)
        header["static_asset_encoding"] = data[93]
    from gscodec.common.v3 import validate_extensions

    validate_extensions(header)
    return header


def write_ranges(f: BinaryIO, ranges: QuantRanges) -> None:
    """Write global quantization ranges to file.

    Args:
        f: Binary file handle opened for writing.
        ranges: Quantization ranges to write.
    """
    parts = [
        ranges.means_min.astype(np.float32).tobytes(),
        ranges.means_max.astype(np.float32).tobytes(),
        ranges.scales_min.astype(np.float32).tobytes(),
        ranges.scales_max.astype(np.float32).tobytes(),
        ranges.quats_min.astype(np.float32).tobytes(),
        ranges.quats_max.astype(np.float32).tobytes(),
        struct.pack("<ff", ranges.opacity_min, ranges.opacity_max),
        ranges.sh0_min.astype(np.float32).tobytes(),
        ranges.sh0_max.astype(np.float32).tobytes(),
    ]
    f.write(b"".join(parts))


def read_ranges(f: BinaryIO) -> QuantRanges:
    """Read global quantization ranges from file.

    Args:
        f: Binary file handle opened for reading.

    Returns:
        Parsed quantization ranges.
    """
    data = f.read(RANGES_SIZE)
    if len(data) < RANGES_SIZE:
        raise ValueError(f"File too small for ranges (got {len(data)} bytes)")

    offset = 0

    def read_floats(n: int) -> np.ndarray:
        nonlocal offset
        arr: np.ndarray = np.frombuffer(data[offset : offset + n * 4], dtype=np.float32).copy()
        offset += n * 4
        return arr

    means_min = read_floats(3)
    means_max = read_floats(3)
    scales_min = read_floats(3)
    scales_max = read_floats(3)
    quats_min = read_floats(4)
    quats_max = read_floats(4)
    opacity_min, opacity_max = struct.unpack("<ff", data[offset : offset + 8])
    offset += 8
    sh0_min = read_floats(3)
    sh0_max = read_floats(3)

    return QuantRanges(
        means_min=means_min,
        means_max=means_max,
        scales_min=scales_min,
        scales_max=scales_max,
        quats_min=quats_min,
        quats_max=quats_max,
        opacity_min=opacity_min,
        opacity_max=opacity_max,
        sh0_min=sh0_min,
        sh0_max=sh0_max,
    )


def write_chunk_index(f: BinaryIO, entries: list[ChunkEntry]) -> None:
    """Write chunk index to file.

    Args:
        f: Binary file handle opened for writing.
        entries: List of chunk index entries.
    """
    for entry in entries:
        data = struct.pack(
            "<IIII",
            entry["start_frame"],
            entry["end_frame"],
            entry["offset"],
            entry["size"],
        )
        f.write(data)


def read_chunk_index(f: BinaryIO, n_chunks: int) -> list[ChunkEntry]:
    """Read chunk index from file.

    Args:
        f: Binary file handle opened for reading.
        n_chunks: Number of chunk entries to read.

    Returns:
        List of parsed chunk entries.
    """
    entries = []
    for _ in range(n_chunks):
        data = f.read(CHUNK_ENTRY_SIZE)
        start_frame, end_frame, offset, size = struct.unpack("<IIII", data)
        entries.append(
            ChunkEntry(
                start_frame=start_frame,
                end_frame=end_frame,
                offset=offset,
                size=size,
            )
        )
    return entries


def write_frame_index(f: BinaryIO, entries: list[FrameEntry]) -> None:
    """Write frame index to file.

    Keyframe status is encoded in the high bit of size.

    Args:
        f: Binary file handle opened for writing.
        entries: List of frame index entries.
    """
    for entry in entries:
        data = struct.pack("<II", entry["offset"], entry["size"])
        f.write(data)


def read_frame_index(f: BinaryIO, n_frames: int) -> list[FrameEntry]:
    """Read frame index from file.

    Args:
        f: Binary file handle opened for reading.
        n_frames: Number of frame entries to read.

    Returns:
        List of parsed frame entries.
    """
    entries = []
    for _ in range(n_frames):
        data = f.read(FRAME_ENTRY_SIZE)
        offset, size = struct.unpack("<II", data)
        entries.append(FrameEntry(offset=offset, size=size))
    return entries


def is_keyframe(entry: FrameEntry) -> bool:
    """Check if a frame entry is a keyframe.

    Args:
        entry: Frame index entry.

    Returns:
        True if this is a keyframe (I-frame).
    """
    return bool(entry["size"] & KEYFRAME_FLAG)


def get_frame_size(entry: FrameEntry) -> int:
    """Get the actual byte size of a frame (masking out keyframe flag).

    Args:
        entry: Frame index entry.

    Returns:
        Frame size in bytes.
    """
    return entry["size"] & SIZE_MASK


def make_frame_entry(offset: int, size: int, keyframe: bool = False) -> FrameEntry:
    """Create a FrameEntry with optional keyframe flag.

    Args:
        offset: Byte offset relative to video_payload_offset.
        size: Frame size in bytes.
        keyframe: Whether this is a keyframe (I-frame).

    Returns:
        FrameEntry with encoded size.
    """
    encoded_size = size | (KEYFRAME_FLAG if keyframe else 0)
    return FrameEntry(offset=offset, size=encoded_size)


def write_sh_payload(f: BinaryIO, sh_data: SHChunkData) -> None:
    """Write SH payload with global centroids (no per-frame labels).

    Layout:
        u16  n_centroids
        u8   sh_bands
        256 x f32 codebook (1024 bytes)
        u32  compressed_centroids_size
        zstd centroids [K, coeffs*3] uint8

    Args:
        f: Binary file handle opened for writing.
        sh_data: Global SHChunkData with codebook and centroids.
    """
    import zstandard as zstd

    cctx = zstd.ZstdCompressor(level=13)
    f.write(struct.pack("<HB", sh_data.n_centroids, sh_data.sh_bands))
    f.write(sh_data.codebook.astype(np.float32).tobytes())
    raw = sh_data.centroids.tobytes()
    compressed = cctx.compress(raw)
    f.write(struct.pack("<I", len(compressed)))
    f.write(compressed)


def serialize_sh_payload(sh_data: SHChunkData) -> bytes:
    """Serialize SH payload (global header only) to bytes.

    Args:
        sh_data: Global SHChunkData.

    Returns:
        Serialized bytes for the SH payload section.
    """
    import io

    buf = io.BytesIO()
    write_sh_payload(buf, sh_data)
    return buf.getvalue()


def read_sh_global_data(
    f: BinaryIO,
    offset: int,
) -> SHChunkData:
    """Read global SH codebook + centroids from the SH payload.

    Args:
        f: Binary file handle opened for reading.
        offset: Absolute file offset of the SH payload start.

    Returns:
        SHChunkData with codebook and centroids (no labels).
    """
    import zstandard as zstd

    from gscodec.constants import SH_CODEBOOK_SIZE, SH_COEFFS

    f.seek(offset)
    dctx = zstd.ZstdDecompressor()

    n_centroids, sh_bands = struct.unpack("<HB", f.read(3))
    codebook = np.frombuffer(f.read(SH_CODEBOOK_SIZE * 4), dtype=np.float32).copy()
    compressed_size = struct.unpack("<I", f.read(4))[0]
    compressed_data = f.read(compressed_size)

    coeffs = SH_COEFFS[sh_bands]
    raw = dctx.decompress(compressed_data)
    centroids = np.frombuffer(raw, dtype=np.uint8).reshape(n_centroids, coeffs * 3).copy()

    return SHChunkData(
        codebook=codebook,
        centroids=centroids,
        labels=[],
        n_centroids=n_centroids,
        sh_bands=sh_bands,
    )


def build_sh_label_offsets(
    f: BinaryIO,
    payload_offset: int,
    n_chunks: int,
) -> tuple[int, list[list[tuple[int, int]]]]:
    """Build index of per-chunk label offsets in the SH payload.

    Skips the global header (codebook + centroids), then indexes per-chunk
    label sections.

    Args:
        f: Binary file handle opened for reading.
        payload_offset: Absolute file offset of SH payload start.
        n_chunks: Number of chunks.

    Returns:
        Tuple of (labels_start_offset, per-chunk label offsets).
        Each chunk has a list of (file_offset, blob_size) per frame.
    """
    from gscodec.constants import SH_CODEBOOK_SIZE

    f.seek(payload_offset)

    # Skip global header
    _n_centroids, _sh_bands = struct.unpack("<HB", f.read(3))
    f.seek(SH_CODEBOOK_SIZE * 4, 1)  # skip codebook
    compressed_size = struct.unpack("<I", f.read(4))[0]
    f.seek(compressed_size, 1)  # skip compressed centroids

    labels_start = f.tell()

    # Index per-chunk label sections
    all_label_offsets: list[list[tuple[int, int]]] = []
    for _ in range(n_chunks):
        n_frames_chunk = struct.unpack("<I", f.read(4))[0]
        frame_offsets: list[tuple[int, int]] = []
        for _ in range(n_frames_chunk):
            blob_size = struct.unpack("<I", f.read(4))[0]
            frame_offsets.append((f.tell(), blob_size))
            f.seek(blob_size, 1)
        all_label_offsets.append(frame_offsets)

    return labels_start, all_label_offsets
