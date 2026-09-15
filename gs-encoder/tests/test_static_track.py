"""Native static tracks and persistent merged rendering buffers."""

import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gscodec.common.static_track import (
    static_track_from_gsav,
    static_track_to_gsav,
    validate_static_track,
)
from gscodec.decoder.providers import GSAVFileProvider
from gscodec.decoder.sequence_decoder import SequenceDecoder
from gscodec.encoder.config import ChunkConfig
from gscodec.encoder.sequence_encoder import SequenceEncoder
from gscodec.encoder.static_encoder import encode_static_frame


def encoded_scene(frame):
    static = frame[:8]
    static.masks[::2] = False
    track = encode_static_frame(static, sh_bands=0)
    encoder = SequenceEncoder(
        device="cpu", chunk_config=ChunkConfig(size=2, sh_bands=0, lo_snap_k=1)
    )
    dynamic = frame[:16]
    dynamic.masks[:] = True
    following = dynamic.clone()
    following.means += 0.1
    following.masks[::2] = False
    data = encoder.encode_frames(
        [dynamic, following], static_data=track, static_encoding=1, prune=False
    )
    return data, track


def test_static_flat_codec_roundtrip(mock_gstensor):
    data, track = encoded_scene(mock_gstensor)
    assert track[:4] == b"GSST"
    assert len(track) == len(static_track_to_gsav(track)) - 88
    assert static_track_from_gsav(static_track_to_gsav(track)) == track
    decoder = SequenceDecoder(GSAVFileProvider(data))
    assert decoder._provider.header["static_asset_encoding"] == 1
    static = decoder.decode_static_asset()
    assert static is decoder.decode_static_asset()
    assert len(static.means) == 8
    assert np.count_nonzero(static.masks) == 4
    assert len(decoder.decode_frame(1).means) == 16
    bad = bytearray(track)
    struct.pack_into("<I", bad, 48, 0xFFFFFFFF)
    with pytest.raises(ValueError):
        validate_static_track(bytes(bad))
    with pytest.raises(ValueError):
        validate_static_track(track[:-1])
    source = static_track_to_gsav(track)
    # Source adapters must reject truncated ranges and incompatible atlas mappings.
    for offset, fmt, value in ((32, "<I", 200), (26, "<B", 4)):
        malformed = bytearray(source)
        struct.pack_into(fmt, malformed, offset, value)
        with pytest.raises(ValueError):
            static_track_from_gsav(bytes(malformed))


def test_scene_buffer_updates_only_dynamic_and_renders_jointly(mock_gstensor, monkeypatch):
    data, _ = encoded_scene(mock_gstensor)
    decoder = SequenceDecoder(GSAVFileProvider(data))
    scene = decoder.create_scene_buffer()
    pointers = {
        name: getattr(scene, name).data_ptr()
        for name in ("means", "scales", "quats", "opacities", "colors", "presence")
    }
    static_before = {name: getattr(scene, name)[:8].clone() for name in pointers}
    scene.update(0)
    first_dynamic = scene.means[8:].clone()
    scene.update(1)
    assert scene.dynamic_offset == 8 and scene.n_gaussians == 24
    assert not torch.equal(first_dynamic, scene.means[8:])
    assert torch.count_nonzero(scene.opacities[8:]) == 8
    for name, pointer in pointers.items():
        assert getattr(scene, name).data_ptr() == pointer
        torch.testing.assert_close(getattr(scene, name)[:8], static_before[name])
    monkeypatch.setattr(
        decoder, "decode_frame", lambda _: pytest.fail("Repeated frame was decoded")
    )
    scene.update(1)
    calls = []
    monkeypatch.setitem(
        sys.modules, "gsplat", SimpleNamespace(rasterization=lambda **kw: calls.append(kw))
    )
    scene.render(torch.eye(4)[None], torch.eye(3)[None], 32, 32)
    assert len(calls) == 1
    assert calls[0]["means"] is scene.means
    assert calls[0]["means"].shape[0] == 24
    assert calls[0]["colors"] is scene.colors


@pytest.mark.parametrize("flags,encoding", [(0x20, 2), (0x60, 0), (0x60, 1)])
def test_unknown_static_extensions_are_rejected(mock_gstensor, flags, encoding):
    data, _ = encoded_scene(mock_gstensor)
    invalid = bytearray(data)
    struct.pack_into("<H", invalid, 30, flags)
    invalid[93] = encoding
    with pytest.raises(ValueError):
        GSAVFileProvider(bytes(invalid))


def test_static_file_compression(tmp_path, mock_gstensor):
    from gsply import GSData, plywrite

    from gscodec.encoder.static_encoder import encode_static_file

    frame = mock_gstensor[:16]
    arrays = {
        k: getattr(frame, k).numpy() for k in ("means", "scales", "quats", "opacities", "sh0")
    }
    arrays["opacities"] = arrays["opacities"].reshape(-1)
    path = tmp_path / "static.ply"
    plywrite(path, GSData(**arrays, shN=np.empty((16, 0, 3), np.float32)))
    track, encoding = encode_static_file(path)
    assert encoding == 1
    assert validate_static_track(track)[0] == 16
    output = tmp_path / "scene.gsav"
    encoder = SequenceEncoder(device="cpu", chunk_config=ChunkConfig(sh_bands=0, lo_snap_k=1))
    encoder.compress(tmp_path, output, static_asset_path=path)
    decoder = SequenceDecoder.from_file(output)
    assert len(decoder.decode_static_asset().means) == 16
    assert len(decoder.decode_frame(0).means) > 0


def test_static_sh_and_dynamic_dc_merge(mock_gstensor_with_sh):
    static = mock_gstensor_with_sh[:32]
    static.shN = static.shN[:, :3, :].contiguous()
    track = encode_static_frame(static, sh_bands=1)
    dynamic = static.clone()
    dynamic.shN = None
    encoder = SequenceEncoder(device="cpu", chunk_config=ChunkConfig(sh_bands=0, lo_snap_k=1))
    data = encoder.encode_frames([dynamic], static_data=track, static_encoding=1, prune=False)
    decoder = SequenceDecoder(GSAVFileProvider(data))
    assert decoder.decode_static_asset().shN.shape == (32, 3, 3)
    scene = decoder.create_scene_buffer().update(0)
    assert scene.sh_degree == 1
    assert scene.colors.shape == (64, 4, 3)
    assert torch.count_nonzero(scene.colors[32:, 1:]) == 0
    torch.testing.assert_close(
        scene.colors[:32, 1:], torch.from_numpy(decoder.decode_static_asset().shN)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA renderer requires a GPU")
def test_gpu_joint_depth_sorting_and_static_residency(mock_gstensor):
    pytest.importorskip("gsplat")
    static = mock_gstensor[:1]
    static.means[:] = torch.tensor([0.0, 0.0, 2.0])
    static.quats[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    static.scales[:] = np.log(0.15)
    static.opacities[:] = np.log(4.0)
    static.sh0[:] = (torch.tensor([1.0, 0.0, 0.0]) - 0.5) / 0.28209479177387814
    track = encode_static_frame(static, sh_bands=0)
    behind = static.clone()
    behind.means[:, 2] = 3
    behind.sh0[:] = (torch.tensor([0.0, 0.0, 1.0]) - 0.5) / 0.28209479177387814
    front = behind.clone()
    front.means[:, 2] = 1
    disabled = front.clone()
    disabled.masks[:] = False
    encoder = SequenceEncoder(device="cpu", chunk_config=ChunkConfig(sh_bands=0, lo_snap_k=1))
    data = encoder.encode_frames([behind, disabled, front], static_data=track, static_encoding=1, prune=False)
    scene = SequenceDecoder(GSAVFileProvider(data)).create_scene_buffer("cuda")
    static_prefix = scene.means[:1].clone()
    address = scene.means.data_ptr()
    view = torch.eye(4)[None]
    K = torch.tensor([[[60.0, 0, 16], [0, 60.0, 16], [0, 0, 1]]])
    back_image = scene.update(0).render(view, K, 32, 32)[0]
    disabled_image = scene.update(1).render(view, K, 32, 32)[0]
    front_image = scene.update(2).render(view, K, 32, 32)[0]
    torch.cuda.synchronize()
    assert back_image[0, 16, 16, 0] > back_image[0, 16, 16, 2]
    assert front_image[0, 16, 16, 2] > front_image[0, 16, 16, 0]
    assert disabled_image[0, 16, 16, 0] > 0.5
    assert disabled_image[0, 16, 16, 2] < 0.01
    assert scene.means.data_ptr() == address
    torch.testing.assert_close(scene.means[:1], static_prefix)
