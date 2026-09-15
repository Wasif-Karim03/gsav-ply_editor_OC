"""Shared atlas invariants and versioned mask/static metadata for GSAV."""

from gscodec.common.types import (
    HAS_MASK_FLAG,
    HAS_STATIC_ASSET_FLAG,
    KNOWN_V3_FLAGS,
    STATIC_ENCODING_NATIVE,
    GSAVHeader,
)


def static_encoding(header: GSAVHeader) -> int:
    """Read the declared static track encoding."""
    return header.get("static_asset_encoding", 0)


def validate_extensions(header: GSAVHeader, file_size: int | None = None) -> None:
    """Validate version and optional section before using its byte range."""
    version = header["version"]
    if version not in (1, 3):
        raise ValueError(f"Unsupported GSAV version {version}")
    side, cols, rows = (header[k] for k in ("atlas_side", "n_atlas_cols", "n_atlas_rows"))
    if not (0 < side <= 65535 and 0 < cols <= 255 and 0 < rows <= 255):
        raise ValueError("Atlas dimensions must be positive and fit their header fields")
    if side * cols > 65535 or side * rows > 65535 or cols * rows < 14:
        raise ValueError("Atlas dimensions exceed codec limits or omit attribute mappings")
    if not 0 < header["n_gaussians"] <= min(side * side, 0x7FFFFFFF):
        raise ValueError("Gaussian count must be positive and fit the atlas capacity")
    flags = header["flags"]
    offset = header.get("static_asset_offset", 0)
    size = header.get("static_asset_size", 0)
    if version == 1:
        if flags & ~0x000F or offset or size or static_encoding(header):
            raise ValueError("v3 extensions require GSAV version 3")
        return
    if flags & ~KNOWN_V3_FLAGS:
        raise ValueError("Unsupported GSAV v3 flags")
    has_asset = bool(flags & HAS_STATIC_ASSET_FLAG)
    encoding = static_encoding(header)
    if (has_asset and encoding != STATIC_ENCODING_NATIVE) or (
        not has_asset and encoding
    ):
        raise ValueError("Static asset flags and encoding must be specified together")
    if has_asset != bool(offset) or has_asset != bool(size):
        raise ValueError("Static asset flags and offset/size disagree")
    if not 0 <= offset <= 0xFFFFFFFF or not 0 <= size <= 0xFFFFFFFF:
        raise ValueError("Static asset offset/size must fit uint32")
    if has_asset:
        index_end = header["frame_index_offset"] + 8 * header["n_frames"]
        if offset != index_end or offset + size != header["means_lo_payload_offset"]:
            raise ValueError("Static asset must fill the section between frame index and means-lo")
        if file_size is not None and offset + size > file_size:
            raise ValueError("Static asset extends beyond file")
    if flags & HAS_MASK_FLAG and (header["n_atlas_cols"], header["n_atlas_rows"]) != (5, 3):
        raise ValueError("Mask mapping requires a 5 by 3 atlas")
