"""PLY I/O utilities for fine-tuning.

Handles loading PLY sequences as GSTensor and saving modified means
back to PLY files while preserving all other attributes.
"""

import logging
import re
from pathlib import Path

import gsply
import numpy as np
import torch
from gsply import GSTensor
from torch import Tensor

logger = logging.getLogger(__name__)


def load_ply_sequence(
    input_dir: Path,
    device: str = "cuda:0",
    max_frames: int = 0,
) -> tuple[list[GSTensor], list[Path]]:
    """Load PLY files as GSTensor list.

    Args:
        input_dir: Directory containing PLY files.
        device: Torch device.
        max_frames: Max frames to load (0 = all).

    Returns:
        Tuple of (frames list, ply file paths).
    """
    ply_files = sorted(
        input_dir.glob("*.ply"),
        key=lambda x: int(re.findall(r"\d+", x.stem)[-1])
        if re.findall(r"\d+", x.stem)
        else x.stem,
    )

    if not ply_files:
        raise ValueError(f"No PLY files found in {input_dir}")

    if max_frames > 0:
        ply_files = ply_files[:max_frames]

    logger.info(f"Loading {len(ply_files)} PLY files from {input_dir}")

    frames: list[GSTensor] = []
    for ply_path in ply_files:
        gsdata = gsply.plyread(ply_path)
        n = gsdata.means.shape[0]

        opacities = gsdata.opacities
        if opacities.ndim == 1:
            opacities = opacities.reshape(-1, 1)

        gstensor = GSTensor(
            means=torch.from_numpy(gsdata.means).float().to(device),
            scales=torch.from_numpy(gsdata.scales).float().to(device),
            quats=torch.from_numpy(gsdata.quats).float().to(device),
            opacities=torch.from_numpy(opacities).float().to(device),
            sh0=torch.from_numpy(gsdata.sh0).float().to(device),
            shN=None,
            masks=torch.ones(n, dtype=torch.bool, device=device),
        )
        frames.append(gstensor)

    logger.info(f"Loaded {len(frames)} frames, {frames[0].means.shape[0]} Gaussians each")
    return frames, ply_files


def save_ply_with_new_means(
    original_path: Path,
    output_path: Path,
    new_means: Tensor,
) -> None:
    """Save a PLY file with modified means, preserving all other attributes.

    Args:
        original_path: Original PLY file to read other attributes from.
        output_path: Output PLY file path.
        new_means: [N, 3] new means tensor.
    """
    gsdata = gsply.plyread(original_path)
    means_np = new_means.detach().cpu().numpy().astype(np.float32)
    gsdata.means = means_np
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gsply.plywrite(output_path, gsdata)


def compute_global_ranges(
    frames: list[GSTensor],
) -> tuple[Tensor, Tensor]:
    """Compute global min/max of log-transformed means across all frames.

    Mirrors SequenceEncoder._compute_global_ranges() for means only.

    Args:
        frames: List of GSTensor frames.

    Returns:
        Tuple of (means_min [3], means_max [3]) on same device as frames.
    """
    device = frames[0].means.device

    means_log = torch.sign(frames[0].means) * torch.log1p(torch.abs(frames[0].means))
    means_min = means_log.amin(dim=0)
    means_max = means_log.amax(dim=0)

    for frame in frames[1:]:
        means_log = torch.sign(frame.means) * torch.log1p(torch.abs(frame.means))
        means_min = torch.minimum(means_min, means_log.amin(dim=0))
        means_max = torch.maximum(means_max, means_log.amax(dim=0))

    return means_min, means_max
