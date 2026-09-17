"""Crop an encoded source without PLY transport, matching or color quantization."""

import io

from gscodec.common.binary_format import (
    CHUNK_ENTRY_SIZE,
    FRAME_ENTRY_SIZE,
    HEADER_SIZE,
    RANGES_SIZE,
    make_frame_entry,
    write_chunk_index,
    write_frame_index,
    write_header,
    write_ranges,
)
from gscodec.common.types import HAS_MASK_FLAG
from gscodec.encoder.compact import compact_source
from gscodec.encoder.source_geometry import SourceGeometry
from gscodec.encoder.video_writer import encode_to_ivf


def crop_source(path, visibility, progress=lambda message: None):
    """Return v3 GSAV bytes with exact retained geometry and color samples."""
    from gscodec.encoder.sequence_encoder import SequenceEncoder

    source = SourceGeometry(path)
    provider = source.provider
    header = dict(source.header)
    progress("Reading encoded source; skipping PLY, matching and color fitting")
    atlases = provider.decode_all_frames()
    # merge validates the timeline and source visibility, and writes exact masks.
    source.merge(
        atlases,
        provider.ranges,
        rows=provider.n_gaussians,
        fps=provider.fps,
        chunk_size=provider.chunk_size,
        visibility=visibility,
    )
    sh = provider.get_sh_chunk_data(0)
    if sh is not None:
        sh.labels = [provider.get_sh_labels(i) for i in range(provider.n_frames)]
    rows, side, lows, sh = compact_source(atlases, provider, visibility, sh, provider.chunk_size)
    progress(f"Compacted storage: {provider.n_gaussians} -> {rows} rows")
    low_data = source.means_lo if lows is None else SequenceEncoder._compress_means_lo(lows)
    counts = [chunk["end_frame"] - chunk["start_frame"] + 1 for chunk in provider.chunk_index]
    sh_data = b"" if sh is None else SequenceEncoder._compress_sh_payload(sh, counts)
    progress("Encoding compact atlas with native xllvp9")
    video, entries = encode_to_ivf(atlases, fps=provider.fps, gop_size=provider.chunk_size)
    frame_entries = [
        make_frame_entry(
            offset=entry["obu_offset"],
            size=entry["obu_size"],
            keyframe=index % provider.chunk_size == 0,
        )
        for index, entry in enumerate(entries)
    ]
    header.update(
        version=3,
        n_gaussians=rows,
        atlas_side=side,
        flags=header["flags"] | HAS_MASK_FLAG,
        ranges_offset=HEADER_SIZE,
        chunk_index_offset=HEADER_SIZE + RANGES_SIZE,
        frame_index_offset=HEADER_SIZE + RANGES_SIZE + CHUNK_ENTRY_SIZE * provider.n_chunks,
        codec="vp09.00.51.08",
        sh_bands=0 if sh is None else sh.sh_bands,
    )
    header["means_lo_payload_offset"] = (
        header["frame_index_offset"] + FRAME_ENTRY_SIZE * provider.n_frames
    )
    header["sh_payload_offset"] = (
        header["means_lo_payload_offset"] + len(low_data) if sh_data else 0
    )
    header["video_payload_offset"] = (
        header["means_lo_payload_offset"] + len(low_data) + len(sh_data)
    )
    header["audio_payload_offset"] = (
        header["video_payload_offset"] + len(video) if source.audio else 0
    )
    with io.BytesIO() as stream:
        write_header(stream, header)
        write_ranges(stream, provider.ranges)
        write_chunk_index(stream, provider.chunk_index)
        write_frame_index(stream, frame_entries)
        for data in (low_data, sh_data, video, source.audio):
            stream.write(data)
        return stream.getvalue()
