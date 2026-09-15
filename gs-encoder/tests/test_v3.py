"""v2 compatibility, exact video masks, and native static embedding roundtrips."""

import io
import struct

import numpy as np
import pytest

from gscodec.common.binary_format import read_header, write_header
from gscodec.common.types import HAS_MASK_FLAG, HAS_STATIC_ASSET_FLAG
from gscodec.common.v3 import validate_extensions
from gscodec.decoder.chunk_decoder import ChunkDecoder
from gscodec.decoder.sequence_decoder import SequenceDecoder
from gscodec.encoder.chunk_encoder import ChunkEncoder
from gscodec.encoder.config import ChunkConfig
from gscodec.encoder.sequence_encoder import SequenceEncoder
from gscodec.encoder.static_encoder import encode_static_frame


def test_mask_padding_and_legacy_decode(mock_gstensor, mock_quant_ranges):
    frame = mock_gstensor[:8]
    frame.masks[1::2] = False
    encoded, _ = ChunkEncoder(device="cpu", skip_sorting=True).encode(
        [frame], mock_quant_ranges, target_n=12
    )
    atlas = encoded[0].atlas
    decoder = ChunkDecoder(mock_quant_ranges, atlas.shape[1], 12, has_mask=True)
    expected = np.array([1, 0] * 4 + [0] * 4, dtype=bool)
    decoded = decoder.decode_frame(atlas, encoded[0].means_lo)
    np.testing.assert_array_equal(decoded.masks.reshape(-1), expected)
    assert np.isneginf(decoded.opacities[~expected]).all()
    legacy = ChunkDecoder(mock_quant_ranges, atlas.shape[1], 12)
    assert legacy.decode_presence(atlas).all()


@pytest.mark.parametrize("version", [1, 3])
@pytest.mark.parametrize(
    "fault", ["count", "zero_count", "zero_side", "oversized_side", "missing_cells"]
)
def test_reader_rejects_invalid_atlas_geometry(mock_gstensor, version, fault):
    from gscodec.decoder.providers import GSAVFileProvider

    encoder = SequenceEncoder(device="cpu", chunk_config=ChunkConfig(sh_bands=0, lo_snap_k=1))
    data = bytearray(encoder.encode_frames([mock_gstensor[:8]], prune=False))
    side = struct.unpack_from("<H", data, 24)[0]
    struct.pack_into("<I", data, 4, version)
    if version == 1:
        struct.pack_into("<H", data, 30, 0)
    offset, fmt, value = {
        "count": (8, "<I", side * side + 1),
        "zero_count": (8, "<I", 0),
        "zero_side": (24, "<H", 0),
        "oversized_side": (24, "<H", 65535),
        "missing_cells": (26, "<B", 2),
    }[fault]
    struct.pack_into(fmt, data, offset, value)
    with pytest.raises(ValueError, match="Gaussian count|Atlas dimensions"):
        GSAVFileProvider(bytes(data))


@pytest.mark.parametrize("with_static", [False, True])
def test_sequence_roundtrip(tmp_path, monkeypatch, mock_gstensor, with_static):
    frame = mock_gstensor[:32]
    frame.opacities[:] = 4
    frame.scales[:] = -3
    frame.masks[::2] = False
    encoder = SequenceEncoder(device="cpu", chunk_config=ChunkConfig(sh_bands=0, lo_snap_k=1))
    monkeypatch.setattr(encoder, "_load_ply_sequence", lambda _: [frame, frame.clone()])
    monkeypatch.setattr(encoder, "_prune_invisible", lambda frames: frames)
    static = encode_static_frame(frame[:4], sh_bands=0) if with_static else None
    static_path = tmp_path / "scene.gsst"
    if static:
        static_path.write_bytes(static)
    output = tmp_path / "scene.gsav"
    encoder.compress(tmp_path, output, static_asset_path=static_path if static else None)
    decoder = SequenceDecoder.from_file(output)
    header = decoder._provider.header
    assert header["version"] == 3
    assert header["flags"] & HAS_MASK_FLAG
    assert bool(header["flags"] & HAS_STATIC_ASSET_FLAG) == with_static
    assert decoder.get_static_asset() == static
    decoded = decoder.decode_all()
    assert len(decoded) == 2
    assert np.count_nonzero(decoded[0].masks) == 16
    np.testing.assert_array_equal(decoded[0].masks, decoder.decode_frame(0).masks)
    if static:
        assert len(decoder.decode_static_asset().means) == 4
        bad = dict(header, static_asset_size=header["static_asset_size"] + 1)
        with pytest.raises(ValueError, match="section"):
            validate_extensions(bad)
        with pytest.raises(ValueError, match="beyond"):
            validate_extensions(header, header["static_asset_offset"])
    else:
        # Actual v2 wire layout: unchanged offsets, version 1, no v3 flags.
        legacy_bytes = bytearray(output.read_bytes())
        struct.pack_into("<I", legacy_bytes, 4, 1)
        struct.pack_into("<H", legacy_bytes, 30, 0)
        output.write_bytes(legacy_bytes)
        legacy = SequenceDecoder.from_file(output)
        assert legacy.decode_frame(0).masks.all()
        assert legacy.get_static_asset() is None
    packed = io.BytesIO()
    write_header(packed, header)
    assert len(packed.getvalue()) == 128
    packed.seek(0)
    assert read_header(packed) == header
    invalid = dict(header, version=2)
    with pytest.raises(ValueError, match="version"):
        write_header(io.BytesIO(), invalid)
    invalid = dict(header, flags=HAS_STATIC_ASSET_FLAG, static_asset_encoding=0)
    with pytest.raises(ValueError, match="together"):
        validate_extensions(invalid)
    invalid = dict(header, flags=0x40)
    with pytest.raises(ValueError, match="Unsupported"):
        validate_extensions(invalid)
