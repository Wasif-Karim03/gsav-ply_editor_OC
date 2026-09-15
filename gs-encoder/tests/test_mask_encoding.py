"""Mask-aware statistics, slot reuse, SH training, and GOP independence."""

import numpy as np
import pytest
import torch

from gscodec.decoder.chunk_decoder import ChunkDecoder
from gscodec.decoder.providers import GSAVFileProvider
from gscodec.decoder.sequence_decoder import SequenceDecoder
from gscodec.encoder.chunk_encoder import ChunkEncoder
from gscodec.encoder.config import ChunkConfig
from gscodec.encoder.masks import ATTRIBUTES, sanitize_frame
from gscodec.encoder.sequence_encoder import SequenceEncoder
from gscodec.encoder.temporal_matcher import TemporalMatcher


def lifecycle_frames(source):
    frames = [source[:16].clone() for _ in range(8)]
    for t, frame in enumerate(frames):
        frame.means += t * 0.02
        frame.masks[:] = True
        frame.masks[0] = t in (1, 3, 6)
        frame.masks[1] = False
        for name in ATTRIBUTES:
            value = getattr(frame, name)
            if value is not None:
                value[~frame.masks] = float("nan")
    return frames


def test_inactive_placeholders_do_not_affect_ranges_or_source(mock_gstensor):
    frames = lifecycle_frames(mock_gstensor)
    encoder = SequenceEncoder(device="cpu")
    ranges = encoder._compute_global_ranges(frames)
    compact = [f[f.masks] for f in frames]
    expected = encoder._compute_global_ranges(compact)
    for name in ranges.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(ranges, name), getattr(expected, name))
    clean = sanitize_frame(frames[0])
    assert torch.isnan(frames[0].means[0]).all()
    assert torch.isfinite(clean.means).all()
    frames[0].means[2] = float("nan")
    with pytest.raises(ValueError, match="Active Gaussian means"):
        sanitize_frame(frames[0])


def test_inactive_quantized_reuse_and_activation_safe_snap(mock_gstensor):
    frames = lifecycle_frames(mock_gstensor)[:4]
    encoder = SequenceEncoder(device="cpu")
    ranges = encoder._compute_global_ranges(frames)
    chunk = ChunkEncoder(device="cpu", skip_sorting=True, keyframe_snap=False)
    encoded, _ = chunk.encode(frames, ranges)
    raw, _ = ChunkEncoder(device="cpu", skip_sorting=True, reuse_inactive=False, keyframe_snap=False).encode(frames, ranges)
    decoder = ChunkDecoder(ranges, encoded[0].atlas.shape[1], 16, has_mask=False)
    decoded = [decoder.decode_frame(f.atlas, f.means_lo) for f in encoded]
    original = [decoder.decode_frame(f.atlas, f.means_lo) for f in raw]
    for name in ("means", "scales", "quats", "opacities", "sh0"):
        arrays = [getattr(f, name) for f in decoded]
        np.testing.assert_array_equal(arrays[0][0], arrays[1][0])
        np.testing.assert_array_equal(arrays[2][0], arrays[1][0])
        for t in range(4):
            mask = frames[t].masks.numpy()
            np.testing.assert_array_equal(arrays[t][mask], getattr(original[t], name)[mask])
    # A reactivated slot within the keyframe snap threshold must not snap.
    clean = [mock_gstensor[:16].clone() for _ in range(3)]
    clean[0].scales[0] = -3
    clean[1].masks[0] = False
    clean[2].scales[0] = -3 + 12 / 219
    ranges.scales_min[:] = -6
    ranges.scales_max[:] = 0
    snap, _ = ChunkEncoder(device="cpu", skip_sorting=True, reuse_inactive=False).encode(clean, ranges)
    exact, _ = ChunkEncoder(device="cpu", skip_sorting=True, reuse_inactive=False, keyframe_snap=False).encode(clean, ranges)
    actual = decoder.decode_frame(snap[2].atlas, snap[2].means_lo).scales[0]
    expected = decoder.decode_frame(exact[2].atlas, exact[2].means_lo).scales[0]
    key = decoder.decode_frame(exact[0].atlas, exact[0].means_lo).scales[0]
    assert np.any(expected != key), "Fixture must span a quantization level"
    np.testing.assert_array_equal(actual, expected)
    clean[1].masks[0] = True
    control, _ = ChunkEncoder(device="cpu", skip_sorting=True, reuse_inactive=False).encode(clean, ranges)
    np.testing.assert_array_equal(decoder.decode_frame(control[2].atlas, control[2].means_lo).scales[0], key)



@pytest.mark.parametrize("all_inactive", [False, True])
def test_mask_lifetimes_pruning_and_random_access(mock_gstensor, all_inactive):
    frames = lifecycle_frames(mock_gstensor)
    if all_inactive:
        for frame in frames:
            frame.masks[:] = False
    config = ChunkConfig(size=4, sh_bands=0, lo_snap_k=1, identity_mode="stable")
    encoder = SequenceEncoder(device="cpu", chunk_config=config)
    data = encoder.encode_frames(frames)
    decoder = SequenceDecoder(GSAVFileProvider(data))
    assert decoder.n_gaussians == (1 if all_inactive else 15)
    sequential = decoder.decode_all()
    for index in (7, 0, 4, 3, 6):
        frame = decoder.decode_frame(index)
        np.testing.assert_array_equal(frame.masks, sequential[index].masks)
        np.testing.assert_allclose(frame.means, sequential[index].means)
        assert frame.masks.sum() == frames[index].masks.sum().item()


def test_matching_ignores_inactive_candidates_and_preserves_retired_values(mock_gstensor):
    first = mock_gstensor[:8].clone()
    first.masks[4:] = False
    second = first.clone()
    second.masks[:] = False
    third = first.clone()
    third.masks[:] = True
    out = TemporalMatcher().conform_chunk([first, second, third], 8)
    assert [int(f.masks.sum()) for f in out] == [4, 0, 8]
    torch.testing.assert_close(out[1].means, out[0].means)
    torch.testing.assert_close(out[1].scales, out[0].scales)
    torch.testing.assert_close(out[1].opacities, out[0].opacities)


def test_sh_training_excludes_inactive_and_reuses_labels(monkeypatch):
    import gscodec.encoder.sh_compress as sh

    masks = [np.array([False, True]), np.array([True, False]), np.array([False, True])]
    frames = [torch.full((2, 3, 3), 999.0) for _ in masks]
    for t, mask in enumerate(masks):
        frames[t][mask] = t + 1
    seen = []
    def cluster(vectors, _):
        seen.append(vectors.copy())
        return vectors.copy(), np.arange(len(vectors), dtype=np.uint16)
    monkeypatch.setattr(sh, "cluster_sh_profiles_float", cluster)
    result = sh.encode_sh_global([frames], 2, 1, presence=masks)
    assert seen[0].shape == (3, 9)
    assert seen[0].max() == 3
    np.testing.assert_array_equal(result.labels, [[1, 0], [1, 0], [1, 2]])


def test_all_inactive_sh_has_finite_fallback():
    from gscodec.encoder.sh_compress import encode_sh_global

    result = encode_sh_global([[torch.full((4, 3, 3), float("nan"))]], 4, 1,
                              presence=[np.zeros(4, dtype=bool)])
    assert np.isfinite(result.codebook).all()
    assert result.n_centroids == 1
    assert not result.labels[0].any()


@pytest.mark.parametrize("identity_mode", ["stable", "unstructured"])
@pytest.mark.parametrize("sh_bands", [0, 1])
def test_encoding_accepts_mixed_source_sh_degrees(mock_gstensor, identity_mode, sh_bands):
    frames = [mock_gstensor[:8].clone() for _ in range(2)]
    frames[0].shN = torch.full((8, 15, 3), 0.25)
    frames[1].shN = torch.full((8, 3, 3), 0.5)
    frames[0].masks[0] = False
    config = ChunkConfig(size=4, sh_bands=sh_bands, lo_snap_k=1, identity_mode=identity_mode)
    data = SequenceEncoder(device="cpu", chunk_config=config).encode_frames(frames, prune=False)
    decoder = SequenceDecoder(GSAVFileProvider(data))
    for t, expected in enumerate((0.25, 0.5)):
        frame = decoder.decode_frame(t)
        assert frame.masks.sum() == 7 + t
        if sh_bands:
            np.testing.assert_allclose(frame.shN[frame.masks], expected, atol=0.01)
        else:
            assert frame.shN.size == 0
    assert frames[0].shN.shape == (8, 15, 3)
    assert frames[1].shN.shape == (8, 3, 3)


def test_parallel_matching_accepts_mixed_source_sh_degrees(mock_gstensor):
    frames = [mock_gstensor[:8].clone() for _ in range(4)]
    for t, frame in enumerate(frames):
        frame.shN = torch.full((8, 15 if t % 2 == 0 else 3, 3), (t + 1) / 4)
        frame.masks[0] = t % 2 != 0
    config = ChunkConfig(size=2, sh_bands=1, lo_snap_k=1,
                         identity_mode="unstructured", parallel_chunks=2)
    data = SequenceEncoder(device="cpu", chunk_config=config).encode_frames(frames, prune=False)
    decoder = SequenceDecoder(GSAVFileProvider(data))
    for t in (3, 0, 2, 1):
        frame = decoder.decode_frame(t)
        assert frame.masks.sum() == 7 + t % 2
        np.testing.assert_allclose(frame.shN[frame.masks], (t + 1) / 4, atol=0.01)


@pytest.mark.parametrize("counts", [(16, 8), (8, 16)])
def test_variable_count_padding_includes_every_frame(mock_gstensor, counts):
    frames = [mock_gstensor[:n].clone() for n in counts]
    config = ChunkConfig(size=4, sh_bands=0, lo_snap_k=1)
    data = SequenceEncoder(device="cpu", chunk_config=config).encode_frames(frames, prune=False)
    decoder = SequenceDecoder(GSAVFileProvider(data))
    assert decoder.n_gaussians == 16
    for t, n in enumerate(counts):
        assert decoder.decode_frame(t).masks.sum() == n


def test_requested_sh_band_requires_every_frame(mock_gstensor):
    frames = [mock_gstensor[:8].clone() for _ in range(2)]
    frames[0].shN = torch.ones((8, 3, 3))
    config = ChunkConfig(size=4, sh_bands=1, lo_snap_k=1)
    with pytest.raises(ValueError, match="frame 1.*SH"):
        SequenceEncoder(device="cpu", chunk_config=config).encode_frames(frames, prune=False)


def test_matching_all_inactive_first_frame(mock_gstensor):
    first = mock_gstensor[:8].clone()
    first.masks[:] = False
    first.means[:] = float("nan")
    second = mock_gstensor[:8].clone()
    output = TemporalMatcher().conform_chunk([first, second], 8)
    assert not output[0].masks.any()
    assert output[1].masks.all()
    assert torch.isfinite(output[0].means).all()
    torch.testing.assert_close(output[1].means.sort(dim=0).values, second.means.sort(dim=0).values)


def test_parallel_worker_preserves_per_frame_sh_and_presence(mock_gstensor):
    from pathlib import Path

    from gscodec.encoder.parallel_chunks import (
        ChunkWorkerArgs,
        encode_chunk_worker,
        frame_to_cpu_dict,
    )

    frames = [mock_gstensor[:8].clone() for _ in range(3)]
    for t, frame in enumerate(frames):
        frame.shN = torch.full((8, 3, 3), float(t + 1))
        frame.masks[:] = t != 1
    ranges = SequenceEncoder(device="cpu")._compute_global_ranges(frames)
    result = encode_chunk_worker(ChunkWorkerArgs(
        chunk_idx=0, frame_tensors=[frame_to_cpu_dict(f) for f in frames],
        ranges=ranges, means_min=frames[0].means.min(dim=0).values,
        means_max=frames[0].means.max(dim=0).values, lo_snap_k=1,
        sh_bands=1, sh_max_centroids=16, n_gaussians=8, matcher_k_passes=2,
        faiss_threads=1, device="cpu",
    ))
    try:
        with np.load(result.npz_path) as data:
            np.testing.assert_array_equal(data["presence"].sum(axis=1), [8, 0, 8])
        assert len(result.sh_tensors) == 3
        assert (result.sh_tensors[0] == 1).all()
        assert (result.sh_tensors[2] == 3).all()
    finally:
        Path(result.npz_path).unlink()
