"""Export filtering selects the exact same source rows as the preview."""

import numpy as np
import pytest
import torch
from gsmod import FilterValues
from gsmod.torch import GSTensorPro

from src.gsplay.config.settings import GSPlayConfig
from src.gsplay.gsav_visibility import ExportVisibility


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("invert", [False, True])
def test_visibility_matches_preview_without_mutating_source(device, invert):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(19)
    data = GSTensorPro(
        means=torch.randn(1000, 3, generator=generator).to(device),
        scales=torch.full((1000, 3), -3.0, device=device),
        quats=torch.tensor([1, 0, 0, 0], device=device).float().repeat(1000, 1),
        opacities=torch.randn(1000, generator=generator).to(device),
        sh0=torch.randn(1000, 3, generator=generator).to(device),
        shN=torch.randn(1000, 15, 3, generator=generator).to(device),
    )
    config = GSPlayConfig()
    config.filter_values = FilterValues(sphere_radius=1.1, max_opacity=0.7, invert=invert)
    original = data.clone()
    expected = data.clone().filter(config.filter_values, inplace=True)
    visibility = ExportVisibility()
    visibility.filter_gpu(data, config, None)
    for field in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        actual = getattr(data, field).cpu().numpy()
        np.testing.assert_array_equal(actual, getattr(original, field).cpu().numpy())
        np.testing.assert_array_equal(
            actual[visibility.mask], getattr(expected, field).cpu().numpy()
        )
