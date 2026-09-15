"""Shared test fixtures for gscodec tests."""

import os
import shutil

import numpy as np
import pytest
import torch
from gsply import GSTensor

from gscodec.common.types import QuantRanges
from gscodec.encoder.config import ChunkConfig, VideoConfig


def _find_ffmpeg_dir() -> str | None:
    """Find the ffmpeg bin directory if not already on PATH."""
    if shutil.which("ffmpeg"):
        return None
    # WinGet install location
    winget_dir = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft",
        "WinGet",
        "Packages",
    )
    if os.path.isdir(winget_dir):
        for entry in os.listdir(winget_dir):
            if "FFmpeg" in entry:
                pkg_dir = os.path.join(winget_dir, entry)
                for sub in os.listdir(pkg_dir):
                    bin_dir = os.path.join(pkg_dir, sub, "bin")
                    if os.path.isfile(os.path.join(bin_dir, "ffmpeg.exe")):
                        return bin_dir
    return None


_ffmpeg_dir = _find_ffmpeg_dir()
if _ffmpeg_dir:
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def mock_gstensor(device):
    """Creates a mock GSTensor for encoder testing."""
    num_points = 1024
    means = torch.randn(num_points, 3, device=device)
    scales = torch.rand(num_points, 3, device=device) - 5  # Log scales typically negative
    quats = torch.randn(num_points, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)  # Normalize
    opacities = torch.rand(num_points, 1, device=device) * 10 - 3  # Logit opacities
    sh0 = torch.randn(num_points, 3, device=device)
    masks = torch.ones(num_points, dtype=torch.bool, device=device)

    return GSTensor(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=None,
        masks=masks,
    )


@pytest.fixture
def video_config():
    return VideoConfig(fps=30)


@pytest.fixture
def chunk_config():
    return ChunkConfig(size=10)


@pytest.fixture
def mock_gstensor_with_sh(device):
    """Creates a mock GSTensor with SH3 data for SH compression testing."""
    num_points = 1024
    means = torch.randn(num_points, 3, device=device)
    scales = torch.rand(num_points, 3, device=device) - 5
    quats = torch.randn(num_points, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    opacities = torch.rand(num_points, 1, device=device) * 10 - 3
    sh0 = torch.randn(num_points, 3, device=device)
    # SH3: 15 coefficients per channel, 3 color channels
    shN = torch.randn(num_points, 15, 3, device=device) * 0.1
    masks = torch.ones(num_points, dtype=torch.bool, device=device)

    return GSTensor(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        masks=masks,
    )


@pytest.fixture
def mock_quant_ranges():
    """Create mock quantization ranges."""
    return QuantRanges(
        means_min=np.array([-5.0, -5.0, -5.0], dtype=np.float32),
        means_max=np.array([5.0, 5.0, 5.0], dtype=np.float32),
        scales_min=np.array([-10.0, -10.0, -10.0], dtype=np.float32),
        scales_max=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        quats_min=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
        quats_max=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        opacity_min=-6.0,
        opacity_max=12.0,
        sh0_min=np.array([-2.0, -2.0, -2.0], dtype=np.float32),
        sh0_max=np.array([4.0, 4.0, 4.0], dtype=np.float32),
    )
