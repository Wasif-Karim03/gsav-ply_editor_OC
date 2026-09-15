"""Tests for SH compression/decompression pipeline."""

import numpy as np
import torch

from gscodec.constants import SH_CODEBOOK_SIZE
from gscodec.decoder.sh_decompress import expand_sh
from gscodec.encoder.sh_compress import (
    assign_frame_labels,
    build_scalar_codebook,
    cluster_sh_profiles_float,
    compute_sh_bands,
    encode_sh_global,
    quantize_to_codebook,
)


class TestComputeShBands:
    def test_none_returns_0(self):
        assert compute_sh_bands(None) == 0

    def test_empty_returns_0(self):
        assert compute_sh_bands(torch.empty(100, 0)) == 0

    def test_sh1(self):
        assert compute_sh_bands(torch.randn(100, 3, 3)) == 1

    def test_sh2(self):
        assert compute_sh_bands(torch.randn(100, 8, 3)) == 2

    def test_sh3(self):
        assert compute_sh_bands(torch.randn(100, 15, 3)) == 3

    def test_sh3_flat(self):
        assert compute_sh_bands(torch.randn(100, 45)) == 3


class TestBuildScalarCodebook:
    def test_shape_and_dtype(self):
        values = np.random.randn(10000).astype(np.float32)
        codebook = build_scalar_codebook(values)
        assert codebook.shape == (SH_CODEBOOK_SIZE,)
        assert codebook.dtype == np.float32

    def test_sorted(self):
        values = np.random.randn(10000).astype(np.float32)
        codebook = build_scalar_codebook(values)
        assert np.all(codebook[1:] >= codebook[:-1])

    def test_covers_range_approximately(self):
        """Codebook should cover most of the value range."""
        values = np.random.randn(10000).astype(np.float32)
        codebook = build_scalar_codebook(values)
        value_range = values.max() - values.min()
        codebook_range = codebook[-1] - codebook[0]
        # Optimal codebook may not reach extreme outliers, but should cover >80%
        assert codebook_range >= value_range * 0.8

    def test_optimal_concentrates_near_density(self):
        """Optimal codebook should place more entries where data is dense."""
        # Bimodal: cluster at 0 and cluster at 10
        values = np.concatenate(
            [
                np.random.randn(5000).astype(np.float32) * 0.1,
                np.random.randn(5000).astype(np.float32) * 0.1 + 10.0,
            ]
        )
        codebook = build_scalar_codebook(values, n_levels=256)

        # Count entries in each cluster's region
        near_zero = np.sum((codebook > -1) & (codebook < 1))
        near_ten = np.sum((codebook > 9) & (codebook < 11))
        in_gap = np.sum((codebook >= 1) & (codebook <= 9))

        # Most entries should be near the two clusters, not in the gap
        assert near_zero + near_ten > in_gap


class TestQuantizeToCodebook:
    def test_roundtrip_error_bounded(self):
        values = np.random.randn(5000).astype(np.float32)
        codebook = build_scalar_codebook(values)
        indices = quantize_to_codebook(values, codebook)
        reconstructed = codebook[indices]
        rmse = np.sqrt(np.mean((values - reconstructed) ** 2))
        # RMSE should be small relative to value range
        value_range = values.max() - values.min()
        assert rmse < value_range * 0.02

    def test_output_dtype(self):
        values = np.random.randn(100).astype(np.float32)
        codebook = build_scalar_codebook(values)
        indices = quantize_to_codebook(values, codebook)
        assert indices.dtype == np.uint8

    def test_shape_preserved(self):
        values = np.random.randn(50, 45).astype(np.float32)
        codebook = build_scalar_codebook(values.ravel())
        indices = quantize_to_codebook(values, codebook)
        assert indices.shape == (50, 45)


class TestClusterShProfilesFloat:
    def test_basic_clustering(self):
        vectors = np.random.randn(500, 9).astype(np.float32)
        centroids, labels = cluster_sh_profiles_float(vectors, max_centroids=50)
        assert centroids.shape[0] <= 50
        assert centroids.shape[1] == 9
        assert centroids.dtype == np.float32
        assert labels.shape == (500,)
        assert labels.dtype == np.uint16
        assert labels.max() < centroids.shape[0]

    def test_uniform_input(self):
        vectors = np.full((100, 9), 0.42, dtype=np.float32)
        centroids, labels = cluster_sh_profiles_float(vectors, max_centroids=50)
        assert labels.shape == (100,)
        assert np.allclose(centroids[labels], 0.42, atol=1e-4)

    def test_deterministic(self):
        """Seeded k-means must give identical results run-to-run (reproducibility)."""
        rng = np.random.default_rng(3)
        vectors = rng.standard_normal((2000, 9)).astype(np.float32)
        c1, l1 = cluster_sh_profiles_float(vectors, max_centroids=64)
        c2, l2 = cluster_sh_profiles_float(vectors, max_centroids=64)
        assert np.array_equal(l1, l2), "SH cluster labels not deterministic"
        assert np.allclose(c1, c2), "SH centroids not deterministic"

    def test_rng_state_preserved(self):
        """Seeding inside k-means must not perturb the caller's global RNG stream."""
        torch.manual_seed(123)
        before = torch.randn(4)
        torch.manual_seed(123)
        cluster_sh_profiles_float(np.random.randn(500, 9).astype(np.float32), max_centroids=32)
        after = torch.randn(4)
        assert torch.allclose(before, after), "k-means leaked RNG state to caller"


class TestAssignFrameLabels:
    def test_basic_assignment(self):
        centroids = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
        frame = np.array(
            [[0.01, 0.01, 0.01], [0.99, 0.99, 0.99], [1.99, 1.99, 1.99]], dtype=np.float32
        )
        labels = assign_frame_labels(frame, centroids)
        assert labels.shape == (3,)
        assert labels[0] == 0
        assert labels[1] == 1
        assert labels[2] == 2


class TestLabelDeltaEncoding:
    def test_xor_roundtrip(self):
        """Labels XOR with keyframe and back should be lossless."""
        keyframe = np.array([0, 100, 5000, 48399], dtype=np.uint16)
        frame = np.array([0, 101, 5000, 48000], dtype=np.uint16)
        delta = np.bitwise_xor(frame, keyframe)
        reconstructed = np.bitwise_xor(keyframe, delta)
        np.testing.assert_array_equal(reconstructed, frame)

    def test_identical_frames_zero_delta(self):
        """Identical labels produce all-zero deltas (compresses well)."""
        labels = np.random.randint(0, 10000, 1000, dtype=np.uint16)
        delta = np.bitwise_xor(labels, labels)
        assert np.all(delta == 0)


class TestEncodeSHGlobal:
    def test_full_pipeline_sh3(self, mock_gstensor_with_sh):
        frames = [mock_gstensor_with_sh] * 3
        # Simulate 1 chunk with 3 frames
        all_frames_shN = [[f.shN for f in frames]]
        sh_data = encode_sh_global(
            all_frames_shN,
            n_gaussians=1024,
            sh_bands=3,
            max_centroids=256,
        )
        assert sh_data.sh_bands == 3
        assert sh_data.codebook.shape == (SH_CODEBOOK_SIZE,)
        assert sh_data.centroids.shape[1] == 45  # 15 coeffs * 3 colors
        assert len(sh_data.labels) == 3
        for lbl in sh_data.labels:
            assert lbl.shape == (1024,)
            assert lbl.max() < sh_data.n_centroids

    def test_multi_chunk(self, mock_gstensor_with_sh):
        """Global encoding across multiple chunks shares centroids."""
        frames = [mock_gstensor_with_sh] * 2
        # 2 chunks, 2 frames each
        all_frames_shN = [
            [f.shN for f in frames],
            [f.shN for f in frames],
        ]
        sh_data = encode_sh_global(
            all_frames_shN,
            n_gaussians=1024,
            sh_bands=3,
            max_centroids=128,
        )
        assert len(sh_data.labels) == 4  # 2 chunks * 2 frames
        # All labels reference same centroid table
        for lbl in sh_data.labels:
            assert lbl.max() < sh_data.n_centroids


class TestExpandSH:
    def test_reconstruction_shape(self):
        codebook = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
        centroids = np.random.randint(0, 256, (100, 45), dtype=np.uint8)
        labels = np.random.randint(0, 100, 500, dtype=np.uint16)

        from gscodec.common.types import SHChunkData

        sh_data = SHChunkData(
            codebook=codebook,
            centroids=centroids,
            labels=[],
            n_centroids=100,
            sh_bands=3,
        )
        result = expand_sh(labels, sh_data)
        assert result.shape == (500, 15, 3)
        assert result.dtype == np.float32
        # Values should be within codebook range
        assert result.min() >= -1.0
        assert result.max() <= 1.0


class TestFullRoundtrip:
    def test_encode_decode_roundtrip(self, mock_gstensor_with_sh):
        """Full encode → decode roundtrip with bounded error."""
        frames = [mock_gstensor_with_sh] * 2
        original_shN = frames[0].shN.cpu().numpy()  # [N, 15, 3]

        # Encode globally (1 chunk, 2 frames)
        sh_data = encode_sh_global(
            [[f.shN for f in frames]],
            n_gaussians=1024,
            sh_bands=3,
            max_centroids=1024,
        )

        # Decode frame 0
        labels = sh_data.labels[0]
        reconstructed = expand_sh(labels, sh_data)  # [N, 15, 3]

        # Check shape
        assert reconstructed.shape == original_shN.shape

        # Error should be bounded — quantization + clustering adds error
        # but it should be reasonable (within ~10% of value range)
        value_range = original_shN.max() - original_shN.min()
        rmse = np.sqrt(np.mean((reconstructed - original_shN) ** 2))
        assert rmse < value_range * 0.15, f"RMSE {rmse:.4f} too high for range {value_range:.4f}"
