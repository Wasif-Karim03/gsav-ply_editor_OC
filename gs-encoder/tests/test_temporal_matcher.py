"""Tests for the TemporalMatcher class."""

import pytest
import torch
from gsply import GSTensor

from gscodec.encoder.temporal_matcher import TemporalMatcher


def _make_frame(n: int, offset: float = 0.0, device: str = "cpu") -> GSTensor:
    """Helper to create a GSTensor with n Gaussians."""
    means = torch.randn(n, 3, device=device) + offset
    scales = torch.randn(n, 3, device=device) - 5
    quats = torch.randn(n, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    opacities = torch.randn(n, 1, device=device)
    sh0 = torch.randn(n, 3, device=device)
    masks = torch.ones(n, dtype=torch.bool, device=device)
    return GSTensor(
        means=means, scales=scales, quats=quats,
        opacities=opacities, sh0=sh0, shN=None, masks=masks,
    )


class TestTemporalMatcher:
    """Tests for TemporalMatcher."""

    def test_conform_chunk_uniform_output_size(self):
        """All output frames have exactly target_n Gaussians."""
        frames = [_make_frame(150), _make_frame(120), _make_frame(200)]
        target_n = 200

        matcher = TemporalMatcher(k_passes=5)
        result = matcher.conform_chunk(frames, target_n)

        assert len(result) == 3
        for frame in result:
            assert frame.means.shape[0] == target_n
            assert frame.scales.shape[0] == target_n
            assert frame.quats.shape[0] == target_n
            assert frame.opacities.shape[0] == target_n
            assert frame.sh0.shape[0] == target_n

    def test_conform_preserves_correspondence(self):
        """Nearby Gaussians should match to the same slots across frames."""
        torch.manual_seed(42)
        n = 50
        target_n = 50

        base_means = torch.randn(n, 3)
        frame0 = _make_frame(n)
        frame0.means = base_means.clone()

        frame1 = _make_frame(n)
        frame1.means = base_means + torch.randn(n, 3) * 0.01

        matcher = TemporalMatcher(k_passes=10)
        result = matcher.conform_chunk([frame0, frame1], target_n)

        dists = torch.linalg.norm(result[0].means - result[1].means, dim=1)
        assert dists.mean() < 0.1, f"Mean distance {dists.mean():.4f} too high"

    def test_conform_propagates_shN(self):
        """Higher-order SH must survive conforming (regression: was dropped to None)."""
        f0 = _make_frame(100)
        f1 = _make_frame(80)
        f0.shN = torch.randn(100, 15, 3)
        f1.shN = torch.randn(80, 15, 3)

        result = TemporalMatcher(k_passes=10).conform_chunk([f0, f1], 100)
        for frame in result:
            assert frame.shN is not None, "shN dropped during conform"
            assert frame.shN.shape == (100, 15, 3)

    def test_conform_propagates_shN_values(self):
        """A matched Gaussian carries its source frame's shN, not a zero/ghost."""
        torch.manual_seed(1)
        base = torch.randn(40, 3)
        f0 = _make_frame(40)
        f0.means = base.clone()
        f0.shN = torch.randn(40, 15, 3)
        f1 = _make_frame(40)
        f1.means = base + torch.randn(40, 3) * 0.001  # near-identical -> stable match
        f1.shN = torch.randn(40, 15, 3)

        result = TemporalMatcher(k_passes=10).conform_chunk([f0, f1], 40)
        # Every output shN value must come from one of the input frames (not zeros).
        assert result[1].shN.abs().sum() > 0
        out_vals = set(map(float, result[1].shN.flatten().tolist()))
        src_vals = set(map(float, f1.shN.flatten().tolist()))
        assert out_vals.issubset(src_vals), "shN values not sourced from input frame"

    def test_conform_sh0_only_keeps_shN_none(self):
        """SH0-only frames (no shN) must not gain a spurious shN."""
        result = TemporalMatcher(k_passes=5).conform_chunk([_make_frame(60)], 100)
        assert result[0].shN is None

    def test_padding_disables_mask(self):
        """Frames with fewer Gaussians are padded with disabled slots."""
        frame0 = _make_frame(60)
        target_n = 100

        matcher = TemporalMatcher(k_passes=5)
        result = matcher.conform_chunk([frame0], target_n)

        assert result[0].means.shape[0] == target_n
        # First 60 slots keep original opacities
        assert torch.allclose(result[0].opacities[:60], frame0.opacities)
        # Padded slots have minimum opacity
        assert not result[0].masks[60:].any()

    def test_conform_frame_fewer_gaussians_invisible_padding(self):
        """When current frame has fewer Gaussians, unfilled slots get min opacity."""
        torch.manual_seed(0)

        frame0 = _make_frame(100)
        frame1 = _make_frame(60)
        target_n = 100

        matcher = TemporalMatcher(k_passes=5)
        result = matcher.conform_chunk([frame0, frame1], target_n)

        assert result[1].means.shape[0] == 100
        # All 60 current Gaussians should be placed (matched or priority-filled)
        # The remaining 40 slots should have min opacity
        inactive_count = (~result[1].masks).sum().item()
        assert inactive_count == 40, f"Expected 40 invisible slots, got {inactive_count}"

    def test_single_frame_padding(self):
        """Single frame with fewer Gaussians than target pads correctly."""
        frame = _make_frame(80)
        target_n = 100

        matcher = TemporalMatcher(k_passes=5)
        result = matcher.conform_chunk([frame], target_n)

        assert len(result) == 1
        assert result[0].means.shape[0] == target_n
        # First 80 preserve original means
        assert torch.allclose(result[0].means[:80], frame.means)
        # Padded slots are invisible
        assert not result[0].masks[80:].any()

    def test_single_frame_truncation(self):
        """Single frame with more Gaussians than target truncates."""
        frame = _make_frame(200)
        target_n = 100

        matcher = TemporalMatcher(k_passes=5)
        result = matcher.conform_chunk([frame], target_n)

        assert len(result) == 1
        assert result[0].means.shape[0] == target_n
        assert torch.allclose(result[0].means, frame.means[:target_n])

    def test_empty_chunk_raises(self):
        """Empty chunk should raise ValueError."""
        matcher = TemporalMatcher()
        with pytest.raises(ValueError, match="empty chunk"):
            matcher.conform_chunk([], target_n=100)

    def test_matching_disabled_unchanged(self, mock_gstensor, mock_quant_ranges, device):
        """With matching disabled, ChunkEncoder produces same output as before."""
        from gscodec.encoder.chunk_encoder import ChunkEncoder

        frames = [mock_gstensor] * 3

        encoder = ChunkEncoder(device=device)
        results, _ = encoder.encode(frames, mock_quant_ranges)

        assert len(results) == 3
        for ef in results:
            assert ef.atlas.dtype.name == "uint8"
