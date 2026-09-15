"""Tests for the ChunkEncoder class."""

import math

import numpy as np
import pytest
import torch
from gsply import GSTensor

from gscodec.common.types import QuantRanges
from gscodec.constants import N_ATLAS_COLS, N_ATLAS_ROWS
from gscodec.encoder.chunk_encoder import ChunkEncoder, EncodedFrame


@pytest.fixture
def sample_frames(device):
    """Create sample GSTensor frames for testing."""
    n_gaussians = 100
    n_frames = 5

    frames = []
    for i in range(n_frames):
        means = torch.randn(n_gaussians, 3, device=device) + i * 0.1
        scales = torch.randn(n_gaussians, 3, device=device) - 5
        quats = torch.randn(n_gaussians, 4, device=device)
        quats = quats / quats.norm(dim=-1, keepdim=True)
        opacities = torch.randn(n_gaussians, 1, device=device)
        sh0 = torch.randn(n_gaussians, 3, device=device)
        masks = torch.ones(n_gaussians, dtype=torch.bool, device=device)

        frame = GSTensor(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=None,
            masks=masks,
        )
        frames.append(frame)

    return frames


@pytest.fixture
def sample_ranges():
    """Create sample quantization ranges."""
    return QuantRanges(
        means_min=np.array([-10.0, -10.0, -10.0], dtype=np.float32),
        means_max=np.array([10.0, 10.0, 10.0], dtype=np.float32),
        scales_min=np.array([-15.0, -15.0, -15.0], dtype=np.float32),
        scales_max=np.array([5.0, 5.0, 5.0], dtype=np.float32),
        quats_min=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
        quats_max=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        opacity_min=-6.0,
        opacity_max=12.0,
        sh0_min=np.array([-2.0, -2.0, -2.0], dtype=np.float32),
        sh0_max=np.array([4.0, 4.0, 4.0], dtype=np.float32),
    )


class TestChunkEncoder:
    """Tests for ChunkEncoder (frame-by-frame)."""

    def test_encode_produces_atlases(self, sample_frames, sample_ranges, device):
        """Test that encode returns list of EncodedFrame."""
        encoder = ChunkEncoder(device=device)
        results, sh_data = encoder.encode(sample_frames, sample_ranges)

        n_gaussians = sample_frames[0].means.shape[0]
        assert len(results) == len(sample_frames)
        assert sh_data is None  # No SH when sh_bands=0

        side = math.ceil(math.sqrt(n_gaussians))
        if side % 2 == 1:
            side += 1
        expected_height = N_ATLAS_ROWS * side  # 3 rows
        expected_width = N_ATLAS_COLS * side  # 5 cols

        for ef in results:
            assert isinstance(ef, EncodedFrame)
            assert ef.atlas.dtype == np.uint8
            assert ef.atlas.shape == (expected_height, expected_width)
            assert ef.means_lo.dtype == np.uint8
            assert ef.means_lo.shape == (n_gaussians, 3)

    def test_empty_chunk_raises(self, sample_ranges, device):
        """Test that encoding empty chunk raises error."""
        encoder = ChunkEncoder(device=device)

        with pytest.raises(ValueError, match="empty chunk"):
            encoder.encode([], sample_ranges)

    def test_skip_sorting(self, sample_frames, sample_ranges, device):
        """Test that skip_sorting flag works."""
        encoder = ChunkEncoder(device=device, skip_sorting=True)
        results, _ = encoder.encode(sample_frames, sample_ranges)

        assert len(results) == len(sample_frames)

    def test_scalar_lo_snap_k(self, sample_frames, sample_ranges, device):
        """Scalar lo_snap_k snaps all mean-lo bytes to multiples of K."""
        encoder = ChunkEncoder(device=device, lo_snap_k=8)
        results, _ = encoder.encode(sample_frames, sample_ranges)
        n = sample_frames[0].means.shape[0]
        lo = results[0].means_lo[:n]  # [n, 3] uint8
        assert np.all(lo % 8 == 0), "scalar K-snap: all axes must be multiples of 8"

    def test_tuple_lo_snap_k_per_axis(self, sample_frames, sample_ranges, device):
        """Tuple lo_snap_k applies a different K per axis (no crash; correct snap)."""
        encoder = ChunkEncoder(device=device, lo_snap_k=(8, 4, 2))
        results, _ = encoder.encode(sample_frames, sample_ranges)
        n = sample_frames[0].means.shape[0]
        lo = results[0].means_lo[:n]  # [n, 3] uint8
        assert np.all(lo[:, 0] % 8 == 0), "x axis must snap to 8"
        assert np.all(lo[:, 1] % 4 == 0), "y axis must snap to 4"
        assert np.all(lo[:, 2] % 2 == 0), "z axis must snap to 2"

    def test_all_channels_in_video_safe_range(self, sample_frames, sample_ranges, device):
        """Test that all atlas values are in video-safe range [16, 235]."""
        encoder = ChunkEncoder(device=device)
        results, _ = encoder.encode(sample_frames, sample_ranges)

        n_gaussians = sample_frames[0].means.shape[0]
        atlas = results[0].atlas
        side = atlas.shape[1] // N_ATLAS_COLS

        # 17 active channels (skip channel 17 = padding)
        for ch_idx in range(17):
            atlas_col = ch_idx % N_ATLAS_COLS
            atlas_row = ch_idx // N_ATLAS_COLS
            y_start = atlas_row * side
            x_start = atlas_col * side
            block = atlas[y_start : y_start + side, x_start : x_start + side]
            values = block.flatten()[:n_gaussians]
            assert np.all(values >= 16), f"Channel {ch_idx} has values below 16"
            assert np.all(values <= 235), f"Channel {ch_idx} has values above 235"
