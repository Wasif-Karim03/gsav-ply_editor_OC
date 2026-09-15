"""
Quantization strategies for the GSAV codec.

Provides:
- 16-bit means quantization split into hi/lo uint8 (hi: base-220 video-safe, lo: base-256 raw)
- 8-bit linear quantization for scales, quats, opacity, sh0
"""

import numpy as np
import torch
from torch import Tensor

from gscodec.constants import (
    LO_BASE,
    MIN_VAL,
    N_LEVELS,
    SCALE_8BIT,
    SCALE_16BIT,
)


def quantize_means_16bit_split(
    data: Tensor,
    mins: Tensor,
    maxs: Tensor,
    lo_snap_k: int | tuple[int, int, int] = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize means to 16-bit, split into hi/lo uint8 arrays.

    Maps [mins, maxs] -> [0, 56319].
    Split: hi = val // 256 [0,219] (video-safe), lo = val % 256 [0,255] (raw binary).
    Optional K-snapping rounds lo to nearest multiple of K for better compression.

    Args:
        data: [N, 3] tensor of log-transformed means.
        mins: [3] per-channel minimum values.
        maxs: [3] per-channel maximum values.
        lo_snap_k: Round lo to nearest K (1=off). Scalar applies to all axes.
            Tuple (kx, ky, kz) for per-axis control, e.g. (8, 4, 8).

    Returns:
        Tuple of (hi_u8 [N, 3] video-safe [16,235], lo_u8 [N, 3] raw [0,255]).
    """
    scale = torch.where(maxs > mins, maxs - mins, torch.ones_like(mins))
    normalized = ((data - mins) / scale).clamp(0, 1)
    quantized = (normalized * SCALE_16BIT).floor().long()

    hi = (quantized // LO_BASE).clamp(0, N_LEVELS - 1)
    lo = (quantized % LO_BASE).clamp(0, 255)

    if isinstance(lo_snap_k, (list, tuple)):
        for ax, k in enumerate(lo_snap_k):
            if k > 1:
                lo[:, ax] = ((lo[:, ax] + k // 2) // k * k) % 256
    elif lo_snap_k > 1:
        lo = ((lo + lo_snap_k // 2) // lo_snap_k * lo_snap_k) % 256

    hi_u8 = (hi + MIN_VAL).cpu().numpy().astype(np.uint8)
    lo_u8 = lo.cpu().numpy().astype(np.uint8)

    return hi_u8, lo_u8


def compute_snap_sensitivity(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Compute per-Gaussian visual sensitivity via render gradient magnitude.

    Args:
        means: [N, 3] positions.
        quats: [N, 4] quaternions.
        scales: [N, 3] log-space scales.
        opacities: [N, 1] logit-space opacities.
        sh0: [N, 3] SH DC coefficients.
        viewmats: [C, 4, 4] camera view matrices.
        Ks: [C, 3, 3] camera intrinsics.
        width: Render width.
        height: Render height.

    Returns:
        [N] sensitivity scores (higher = more visually important).
    """
    import gsplat

    means_p = means.clone().requires_grad_(True)
    rendered, _, _ = gsplat.rasterization(
        means=means_p, quats=quats, scales=scales.exp(),
        opacities=torch.sigmoid(opacities.squeeze(-1)),
        colors=0.28209479 * sh0 + 0.5,
        viewmats=viewmats, Ks=Ks, width=width, height=height,
        packed=True, render_mode="RGB",
    )
    rendered.sum().backward()
    return means_p.grad.norm(dim=-1).detach()


def quantize_anchor_8bit(
    data: Tensor,
    mins: Tensor,
    maxs: Tensor,
) -> np.ndarray:
    """Quantize attribute to 8-bit using linear scaling (video-safe range).

    Maps [mins, maxs] -> [16, 235] as u8.

    Args:
        data: [N, C] tensor of attribute values.
        mins: [C] per-channel minimum values.
        maxs: [C] per-channel maximum values.

    Returns:
        [N, C] uint8 array.
    """
    scale = torch.where(maxs > mins, maxs - mins, torch.ones_like(mins))
    normalized = ((data - mins) / scale).clamp(0, 1)
    quantized = (normalized * SCALE_8BIT).round().clamp(0, SCALE_8BIT) + MIN_VAL
    return quantized.to(torch.uint8).cpu().numpy()
