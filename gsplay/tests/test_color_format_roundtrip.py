"""Color/activation metadata must survive every renderer/export adapter."""

import numpy as np
import pytest
import torch
from gsply import GSData

from src.domain.data import GaussianData
from src.infrastructure.exporters.ply_exporter import PlyExporter


C0 = 0.28209479177387814


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("rgb", [False, True])
@pytest.mark.parametrize("degree", [0, 3])
@pytest.mark.parametrize("linear", [False, True])
def test_native_format_roundtrip(rgb, degree, linear, device):
    colors = np.array([[0.1, 0.3, 0.8], [0.9, 0.15, 0.4]], dtype=np.float32)
    data = GSData(
        means=np.zeros((2, 3), np.float32),
        scales=np.full((2, 3), -3, np.float32),
        quats=np.array([[1, 0, 0, 0]] * 2, np.float32),
        opacities=np.full(2, -1, np.float32),
        sh0=(colors - 0.5) / C0,
        shN=np.full((2, 15, 3), 0.02, np.float32) if degree else None,
    )
    if linear:
        data = data.denormalize(inplace=False)
    if rgb:
        data = data.to_rgb(inplace=False)
    wrapped = GaussianData.from_gsdata(data)
    tensor = wrapped.to_gstensor(device)
    # Exercise the cached tensor branch as well as GPU -> CPU wrapping.
    for native in (
        wrapped.to_gsdata(),
        tensor,
        wrapped.to_gstensor(device),
        GaussianData.from_gstensor(tensor).to_gsdata(),
    ):
        assert native.is_sh0_rgb == rgb
        assert native.is_scales_ply == (not linear)
        assert native.is_opacities_ply == (not linear)
    np.testing.assert_allclose(GaussianData.from_gstensor(tensor).to_gsdata().sh0, data.sh0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("brightness", [1.0, 0.7])
@pytest.mark.parametrize("degree", [0, 3])
def test_export_preserves_display_color_and_sh(tmp_path, degree, device, brightness):
    colors = np.array([[0.1, 0.3, 0.8], [0.9, 0.15, 0.4]], dtype=np.float32)
    raw = GSData(
        means=np.zeros((2, 3), np.float32),
        scales=np.full((2, 3), -3, np.float32),
        quats=np.array([[1, 0, 0, 0]] * 2, np.float32),
        opacities=np.full(2, -1, np.float32),
        sh0=(colors - 0.5) / C0,
        shN=np.full((2, 15, 3), 0.02, np.float32) if degree else None,
    )
    active = raw.denormalize(inplace=False)
    if not degree:
        active = active.to_rgb(inplace=False)
    tensor = GaussianData.from_gsdata(active).to_gstensor(device)
    tensor = GaussianData.from_gstensor(tensor).to_gstensor(device)
    from gsmod import ColorValues

    from src.gsplay.processing.color import DefaultColorProcessor

    tensor = DefaultColorProcessor().apply_gpu(tensor, ColorValues(brightness=brightness), device)
    output = tmp_path / "color.ply"
    PlyExporter().export_frame(tensor, output)
    saved = GSData.load(str(output))
    np.testing.assert_allclose(saved.sh0, (colors * brightness - 0.5) / C0, atol=1e-6)
    np.testing.assert_allclose(saved.scales, raw.scales, atol=1e-6)
    np.testing.assert_allclose(saved.opacities, raw.opacities, atol=1e-6)
    if degree:
        np.testing.assert_allclose(saved.shN, raw.shN * brightness, atol=1e-6)


def test_export_keeps_fully_transparent_gaussians_invisible(tmp_path):
    raw = GSData(
        means=np.zeros((2, 3), np.float32),
        scales=np.full((2, 3), -3, np.float32),
        quats=np.array([[1, 0, 0, 0]] * 2, np.float32),
        opacities=np.array([-np.inf, -1], np.float32),
        sh0=np.zeros((2, 3), np.float32),
        shN=None,
    )
    active = raw.denormalize(inplace=False).to_rgb(inplace=False)
    output = tmp_path / "transparent.ply"
    PlyExporter().export_frame(GaussianData.from_gsdata(active).to_gstensor("cpu"), output)
    saved = GSData.load(str(output))
    assert np.isneginf(saved.opacities[0])
    np.testing.assert_allclose(saved.opacities[1], -1, atol=1e-6)
