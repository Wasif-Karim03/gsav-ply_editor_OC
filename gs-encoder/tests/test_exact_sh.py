"""Exact palette preservation and equivalence of bounded scalar statistics."""

import numpy as np
import pytest
import torch

from gscodec.encoder.sh_compress import build_scalar_codebook, encode_sh_global
from gscodec.encoder.sh_exact import exact_palette


@pytest.mark.parametrize("seed", range(5))
def test_weighted_codebook_matches_expanded_values(seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=31).astype(np.float32)
    values[0] = 12
    counts = rng.integers(0, 200, size=31, dtype=np.int64)
    compact = build_scalar_codebook(values, 16, value_counts=counts)
    expanded = build_scalar_codebook(np.repeat(values, counts), 16)
    np.testing.assert_allclose(compact, expanded, atol=2e-6, rtol=1e-6)


def test_weighted_codebook_handles_empty_constant_and_huge_counts():
    np.testing.assert_array_equal(
        build_scalar_codebook(np.array([1]), 4, value_counts=np.array([0])), 0
    )
    np.testing.assert_array_equal(
        build_scalar_codebook(np.array([3]), 4, value_counts=np.array([10**10])), 3
    )
    result = build_scalar_codebook(
        np.array([-1, 0, 1]), 8, value_counts=np.array([10**10, 2, 10**10])
    )
    assert np.isfinite(result).all()
    assert np.all(np.diff(result) >= 0)


def test_palette_never_merges_distinct_vectors_and_ignores_absent_rows():
    a = np.array([[1, 2, 3], [1, 2, 3], [99, 99, 99]], np.float32)
    b = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0]], np.float32)
    masks = [np.array([True, True, False]), np.array([True, True, False])]
    result = exact_palette([a, b], masks, 2)
    palette, labels, values, counts = result
    for frame, mask, label in zip([a, b], masks, labels, strict=True):
        np.testing.assert_array_equal(palette[label[mask]], frame[mask])
    assert counts.sum() == 12
    assert exact_palette([a, b], masks, 1) is None
    np.testing.assert_allclose(
        build_scalar_codebook(values, 8, value_counts=counts),
        build_scalar_codebook(np.r_[a[masks[0]].ravel(), b[masks[1]].ravel()], 8),
        atol=1e-6,
    )


def test_exact_sh_path_skips_clustering_and_preserves_active_coefficients(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Repeated vectors that fit the palette must not be clustered")

    monkeypatch.setattr("gscodec.encoder.sh_compress.cluster_sh_profiles_float", forbidden)
    frames = [torch.full((8, 15, 3), value) for value in (0.02, 0.03, 0.02, 0.03)]
    masks = [np.ones(8, bool) for _ in frames]
    masks[0][3] = False
    encoded = encode_sh_global(
        [frames[:2], frames[2:]], 8, 3, chunk_frame_counts=[2, 2], presence=masks, prefer_exact=True
    )
    reconstructed = encoded.codebook[encoded.centroids].reshape(-1, 15, 3)
    for frame, mask, labels in zip(frames, masks, encoded.labels, strict=True):
        np.testing.assert_allclose(reconstructed[labels[mask]], frame.numpy()[mask], atol=1e-6)
