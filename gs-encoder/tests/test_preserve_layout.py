"""Version 3 row-preserving encoding contract."""

import numpy as np
import pytest
import torch

from gscodec.decoder import SequenceDecoder
from gscodec.decoder.providers import GSAVFileProvider
from gscodec.encoder import ChunkConfig, SequenceEncoder
from gscodec.encoder.sorting.morton import MortonSortingStrategy
from gscodec.encoder.temporal_matcher import TemporalMatcher


def config(**kwargs):
    values = {"identity_mode": "preserve", "size": 2, "lo_snap_k": 1, "keyframe_snap": False}
    values.update(kwargs)
    return ChunkConfig(**values)


@pytest.mark.parametrize("degree", [0, 3])
@pytest.mark.parametrize("hidden", [False, True])
def test_preserve_never_matches_or_sorts_and_keeps_masks(
    mock_gstensor, monkeypatch, degree, hidden
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preserved rows must never be matched or sorted")

    monkeypatch.setattr(TemporalMatcher, "conform_chunk", forbidden)
    monkeypatch.setattr(MortonSortingStrategy, "sort", forbidden)
    frames = [mock_gstensor[:16].clone() for _ in range(4)]
    for i, frame in enumerate(frames):
        frame.means += i * 0.03
        frame.scales[:] = -3
        frame.opacities[:] = 1
        frame.masks[:] = not hidden
        frame.masks[i] = False
        if degree:
            frame.shN = torch.full((16, 15, 3), 0.02 + i * 0.01)
    messages = []
    encoded = SequenceEncoder(
        device="cpu", chunk_config=config(sh_bands=degree), progress=messages.append
    ).encode_frames(frames)
    decoder = SequenceDecoder(GSAVFileProvider(encoded))
    assert decoder.n_gaussians == 16
    for i in (3, 0, 2, 1):
        decoded = decoder.decode_frame(i)
        np.testing.assert_array_equal(decoded.masks, frames[i].masks.numpy())
        mask = decoded.masks
        np.testing.assert_allclose(decoded.means[mask], frames[i].means.numpy()[mask], atol=0.003)
        if degree:
            np.testing.assert_allclose(decoded.shN[mask], frames[i].shN.numpy()[mask], atol=0.005)
    assert any("original arrangement" in m for m in messages)
    assert "Encoding VP9 video" in messages


@pytest.mark.parametrize(
    "change",
    [
        {"matching_enabled": True},
        {"keyframe_snap": True},
        {"lossy_pruning": True},
        {"lo_snap_k": 3},
    ],
)
def test_preserve_rejects_unsafe_settings(change):
    with pytest.raises(ValueError, match="Preserve mode"):
        SequenceEncoder(device="cpu", chunk_config=config(**change))


def test_preserve_rejects_different_counts(mock_gstensor):
    with pytest.raises(ValueError, match="equal row counts"):
        SequenceEncoder(device="cpu", chunk_config=config()).encode_frames(
            [mock_gstensor[:16], mock_gstensor[:12]]
        )
