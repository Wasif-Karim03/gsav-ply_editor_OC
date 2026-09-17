"""Compaction keeps every visible encoded sample, including chunk boundaries."""

import numpy as np
import pytest
import torch

from gscodec.decoder import SequenceDecoder
from gscodec.decoder.providers import GSAVFileProvider
from gscodec.encoder import ChunkConfig, SequenceEncoder
from gscodec.encoder.crop_source import crop_source


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("degree", [0, 3])
def test_crop_copies_visible_fields_exactly(tmp_path, mock_gstensor, empty, degree, monkeypatch):
    frames = [mock_gstensor[:64].clone() for _ in range(5)]
    for i, frame in enumerate(frames):
        frame.means += i * 0.1
        frame.masks[:] = True
        if degree:
            frame.shN = torch.arange(64 * 45).reshape(64, 15, 3).float() * 0.00001
    encoder = SequenceEncoder(
        device="cpu",
        chunk_config=ChunkConfig(
            identity_mode="preserve",
            size=2,
            sh_bands=degree,
            lo_snap_k=1,
            keyframe_snap=False,
        ),
    )
    source = tmp_path / "source.gsav"
    source.write_bytes(encoder.encode_frames(frames))
    masks = []
    for i in range(5):
        mask = np.zeros(64, dtype=bool)
        if not empty:
            start = (i // 2) * 16
            mask[start + i % 2 : start + 8] = True
        masks.append(mask)

    def forbidden(*args, **kwargs):
        raise AssertionError("Crop-only must not run the frame encoder")

    monkeypatch.setattr(SequenceEncoder, "encode_frames", forbidden)
    result = crop_source(source, masks)
    provider = GSAVFileProvider(result)
    assert provider.n_gaussians == (4 if empty else 8)
    original = SequenceDecoder.from_file(source).decode_all()
    cropped = SequenceDecoder(provider).decode_all()
    for a, b, visible in zip(original, cropped, masks, strict=True):
        assert b.masks.sum() == visible.sum()
        for field in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            np.testing.assert_array_equal(getattr(a, field)[visible], getattr(b, field)[b.masks])
    atlases = provider.decode_all_frames()
    for i in (0, 1, 2, 4):
        np.testing.assert_array_equal(atlases[i], provider.decode_video_frames(i, 1)[0])


def test_crop_rejects_invalid_masks(tmp_path, mock_gstensor):
    frames = [mock_gstensor[:16].clone() for _ in range(2)]
    frames[0].masks[0] = False
    encoder = SequenceEncoder(device="cpu")
    source = tmp_path / "source.gsav"
    source.write_bytes(encoder.encode_frames(frames))
    with pytest.raises(ValueError, match="visibility dimensions"):
        crop_source(source, [np.ones(3, bool)])
    with pytest.raises(ValueError, match="cannot revive"):
        crop_source(source, [np.ones(16, bool) for _ in range(2)])
