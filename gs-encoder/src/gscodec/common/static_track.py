"""Flat native static track using the existing Gaussian codec payloads.

On disk: 64-byte descriptor, 112-byte ranges, means-lo, optional SH, one VP9
keyframe. In-memory adapters reuse the ordinary sequence decoder; no nested
GSAV container is stored in the file.
"""

import io
import struct

from gscodec.common.binary_format import read_header, write_header
from gscodec.common.types import HAS_MASK_FLAG, HAS_SH_FLAG, KEYFRAME_FLAG

DESCRIPTOR = struct.Struct("<4sIIIII16sIII12x")
PAYLOAD_OFFSET = DESCRIPTOR.size + 112


def validate_static_track(data: bytes) -> tuple[int, int, int, int, bytes, int, int, int]:
    if len(data) < PAYLOAD_OFFSET:
        raise ValueError("Truncated native static track")
    magic, version, count, side, bands, flags, codec, lo, sh, video = DESCRIPTOR.unpack_from(data)
    if magic != b"GSST" or version != 1:
        raise ValueError("Unsupported native static track")
    if not 0 < count <= 0x7FFFFFFF or not 0 < side <= 65535 or count > side * side:
        raise ValueError("Invalid static Gaussian count or atlas size")
    if side * 5 > 65535 or not 0 <= bands <= 3:
        raise ValueError("Invalid static atlas dimensions or SH bands")
    if flags & ~(HAS_MASK_FLAG | HAS_SH_FLAG) or bool(flags & HAS_SH_FLAG) != bool(bands):
        raise ValueError("Invalid static flags")
    if bool(sh) != bool(bands) or lo == 0 or video == 0 or video > 0x7FFFFFFF:
        raise ValueError("Missing or invalid static payload")
    if codec.rstrip(b"\0") != b"vp09.00.51.08":
        raise ValueError("Native static track requires the VP9 codec")
    if PAYLOAD_OFFSET + lo + sh + video != len(data) or len(data) + 88 > 0xFFFFFFFF:
        raise ValueError("Static payload sizes do not cover the section")
    return count, side, bands, flags, codec, lo, sh, video


def static_track_from_gsav(data: bytes) -> bytes:
    """Flatten an ordinary single-frame encode without re-encoding attributes."""
    h = read_header(io.BytesIO(data))
    if h["n_frames"] != 1 or h["n_chunks"] != 1 or h["flags"] & ~(HAS_MASK_FLAG | HAS_SH_FLAG):
        raise ValueError("Static track requires one frame without audio or embedded assets")
    r, lo, sh, video = (
        h[k]
        for k in (
            "ranges_offset",
            "means_lo_payload_offset",
            "sh_payload_offset",
            "video_payload_offset",
        )
    )
    if not (r >= 128 and r + 112 <= lo <= (sh or video) <= video < len(data)):
        raise ValueError("Invalid source GSAV payload bounds")
    if (h["n_atlas_cols"], h["n_atlas_rows"]) != (5, 3):
        raise ValueError("Static track requires a 5 by 3 atlas")
    if not 128 <= h["frame_index_offset"] <= len(data) - 8:
        raise ValueError("Invalid source GSAV frame index")
    packet_offset, packet_size = struct.unpack_from("<II", data, h["frame_index_offset"])
    if packet_offset or packet_size != (KEYFRAME_FLAG | (len(data) - video)):
        raise ValueError("Static video must be exactly one keyframe")
    descriptor = DESCRIPTOR.pack(
        b"GSST",
        1,
        h["n_gaussians"],
        h["atlas_side"],
        h["sh_bands"],
        h["flags"],
        h["codec"].encode().ljust(16, b"\0"),
        (sh or video) - lo,
        video - sh if sh else 0,
        len(data) - video,
    )
    result = descriptor + data[r : r + 112] + data[lo:]
    validate_static_track(result)
    return result


def static_track_to_gsav(data: bytes) -> bytes:
    """Build an in-memory decoder view using the shared sequence codec."""
    count, side, bands, flags, codec, lo, sh, video = validate_static_track(data)
    header = {
        "magic": b"GSAV",
        "version": 3,
        "n_gaussians": count,
        "n_frames": 1,
        "n_chunks": 1,
        "chunk_size": 1,
        "atlas_side": side,
        "n_atlas_cols": 5,
        "n_atlas_rows": 3,
        "fps": 1,
        "flags": flags,
        "ranges_offset": 128,
        "chunk_index_offset": 240,
        "frame_index_offset": 256,
        "means_lo_payload_offset": 264,
        "sh_payload_offset": 264 + lo if sh else 0,
        "video_payload_offset": 264 + lo + sh,
        "codec": codec.rstrip(b"\0").decode(),
        "audio_payload_offset": 0,
        "audio_payload_size": 0,
        "means_hi_payload_offset": 0,
        "sh_bands": bands,
    }
    out = io.BytesIO()
    write_header(out, header)
    out.write(data[64:PAYLOAD_OFFSET])
    out.write(struct.pack("<IIIIII", 0, 0, 0, 0, 0, video | KEYFRAME_FLAG))
    out.write(data[PAYLOAD_OFFSET:])
    return out.getvalue()
