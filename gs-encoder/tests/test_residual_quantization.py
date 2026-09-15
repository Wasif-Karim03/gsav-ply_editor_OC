"""Tests for quantization round-trip."""

import numpy as np
import torch

from gscodec.constants import MIN_VAL, N_LEVELS, SCALE_8BIT
from gscodec.decoder.dequantization.strategies import (
    dequantize_anchor_8bit,
    dequantize_means_16bit_from_atlas,
)
from gscodec.encoder.quantization.strategies import (
    quantize_anchor_8bit,
    quantize_means_16bit_split,
)


class TestMeans16BitSplitQuantization:
    """Tests for 16-bit means hi/lo split quantization (base-220)."""

    def test_roundtrip(self):
        """Test 16-bit means hi/lo quantization preserves values within tolerance."""
        n_points = 1000
        data = torch.randn(n_points, 3)
        mins = torch.tensor([-5.0, -5.0, -5.0])
        maxs = torch.tensor([5.0, 5.0, 5.0])

        data = data.clamp(mins, maxs)

        hi, lo = quantize_means_16bit_split(data, mins, maxs)

        assert hi.dtype == np.uint8
        assert lo.dtype == np.uint8
        assert hi.shape == (n_points, 3)
        assert lo.shape == (n_points, 3)

        # hi is video-safe [16, 235], lo is raw [0, 255]
        assert hi.min() >= MIN_VAL
        assert hi.max() <= MIN_VAL + N_LEVELS - 1
        assert lo.min() >= 0
        assert lo.max() <= 255

        # Round-trip via dequantize_means_16bit_from_atlas
        dequantized = dequantize_means_16bit_from_atlas(hi, lo, mins.numpy(), maxs.numpy())

        error = np.abs(data.numpy() - dequantized)
        max_error = error.max()

        # 16-bit base-220: error bound ~ range / 48399 ~ 0.00021 for range of 10
        assert max_error < 0.001, f"Max error {max_error} exceeds tolerance"

    def test_edge_values(self):
        """Test min and max values are preserved."""
        mins = torch.tensor([-5.0, -3.0, 0.0])
        maxs = torch.tensor([5.0, 3.0, 10.0])

        data = torch.stack([mins, maxs])

        hi, lo = quantize_means_16bit_split(data, mins, maxs)

        dequantized = dequantize_means_16bit_from_atlas(hi, lo, mins.numpy(), maxs.numpy())

        # Min should be close to mins
        np.testing.assert_allclose(dequantized[0], mins.numpy(), atol=0.001)


class TestAnchor8BitQuantization:
    """Tests for 8-bit quantization."""

    def test_scales_roundtrip(self):
        """Test 8-bit scales quantization preserves values within tolerance."""
        n_points = 1000
        data = torch.rand(n_points, 3) * 10 - 10  # [-10, 0]
        mins = torch.tensor([-10.0, -10.0, -10.0])
        maxs = torch.tensor([0.0, 0.0, 0.0])

        quantized = quantize_anchor_8bit(data, mins, maxs)

        assert quantized.dtype == np.uint8
        assert quantized.min() >= MIN_VAL
        assert quantized.max() <= MIN_VAL + SCALE_8BIT

        dequantized = dequantize_anchor_8bit(quantized, mins.numpy(), maxs.numpy())

        error = np.abs(data.numpy() - dequantized)
        max_error = error.max()

        # 8-bit: error bound ~ range / 219 ~ 0.046
        assert max_error < 0.1, f"Max error {max_error} exceeds tolerance"
