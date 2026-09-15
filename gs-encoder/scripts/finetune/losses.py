"""Loss functions for grid-snap fine-tuning.

Grid-snap loss: penalizes 16-bit quantized means whose lo-byte (v % 256)
is far from 0, making the means_lo payload compressible.

Render loss: L1 + SSIM photometric loss against reference images.
"""

import math

import torch
from torch import Tensor

from gscodec.constants import SCALE_16BIT


def _means_to_v_cont(
    means: Tensor,
    means_min: Tensor,
    means_max: Tensor,
) -> Tensor:
    """Convert raw means to continuous 16-bit quantized values.

    Args:
        means: [N, 3] raw Gaussian positions.
        means_min: [3] global min of log-transformed means.
        means_max: [3] global max of log-transformed means.

    Returns:
        [N, 3] continuous values in [0, 56319].
    """
    means_log = torch.sign(means) * torch.log1p(torch.abs(means))
    scale = (means_max - means_min).clamp(min=1e-8)
    normalized = ((means_log - means_min) / scale).clamp(0.0, 1.0)
    return normalized * SCALE_16BIT


def grid_snap_loss(
    means: Tensor,
    means_min: Tensor,
    means_max: Tensor,
    step: int = 0,
    total_steps: int = 1,
    weights: Tensor | None = None,
) -> Tensor:
    """Cosine grid-snap loss with curriculum scheduling.

    Uses cos(2*pi*v/k) which is smooth, bounded, and naturally periodic —
    no discontinuity at the wrap-around point. Curriculum blends coarse
    (k=64) and fine (k=256) grids for stable convergence.

    Args:
        means: [N, 3] raw Gaussian positions.
        means_min: [3] global min of log-transformed means.
        means_max: [3] global max of log-transformed means.
        step: Current optimization step.
        total_steps: Total optimization steps.
        weights: [N] per-Gaussian importance weights (None = uniform).

    Returns:
        Scalar loss tensor.
    """
    v_cont = _means_to_v_cont(means, means_min, means_max)  # [N, 3]

    # Curriculum: blend coarse (k=64) and fine (k=256)
    progress = min(step / max(total_steps, 1), 1.0)
    w_fine = progress ** 2

    loss_coarse = 0.5 * (1.0 - torch.cos(2.0 * math.pi * v_cont / 64.0))
    loss_fine = 0.5 * (1.0 - torch.cos(2.0 * math.pi * v_cont / 256.0))
    loss = (1.0 - w_fine) * loss_coarse + w_fine * loss_fine  # [N, 3]

    if weights is not None:
        loss = loss * weights.unsqueeze(-1)  # [N, 3]

    return loss.mean()


def temporal_lo_consistency_loss(
    means_t: Tensor,
    means_t_prev: Tensor,
    means_min: Tensor,
    means_max: Tensor,
) -> Tensor:
    """Penalize frame-to-frame lo-byte changes for delta coding.

    Makes delta-coded lo stream more compressible by encouraging temporal
    stability in the lo-byte domain.

    Args:
        means_t: [N, 3] current frame means.
        means_t_prev: [N, 3] previous frame means (detached).
        means_min: [3] global min of log-transformed means.
        means_max: [3] global max of log-transformed means.

    Returns:
        Scalar loss tensor.
    """
    lo_t = torch.fmod(_means_to_v_cont(means_t, means_min, means_max), 256.0)
    lo_prev = torch.fmod(
        _means_to_v_cont(means_t_prev.detach(), means_min, means_max), 256.0,
    )

    # Circular distance in [0, 256)
    diff = lo_t - lo_prev
    diff = diff - 256.0 * torch.round(diff / 256.0)
    return diff.pow(2).mean()


def compute_importance_weights(
    opacities: Tensor,
    scales: Tensor,
) -> Tensor:
    """Compute per-Gaussian importance weights for snap loss.

    Small/transparent Gaussians snap aggressively (low weight),
    large/opaque ones snap gently (high weight inverted in loss).

    Args:
        opacities: [N, 1] logit-space opacities.
        scales: [N, 3] log-space scales.

    Returns:
        [N] normalized inverse-importance weights (higher = snap more).
    """
    alpha = torch.sigmoid(opacities.squeeze(-1))  # [N]
    s_max = scales.exp().max(dim=-1).values  # [N]
    importance = alpha * s_max  # [N]
    # Invert: low importance → high snap weight
    inv = 1.0 / importance.clamp(min=1e-6)
    return inv / inv.mean()


def displacement_quality_proxy(
    means_new: Tensor,
    means_orig: Tensor,
    log_scales: Tensor,
    opacities: Tensor,
    sh0: Tensor,
    n_pixels: int = 512 * 512,
) -> tuple[float, Tensor]:
    """Estimate PSNR degradation without rendering (DSR proxy).

    Args:
        means_new: [N, 3] current positions.
        means_orig: [N, 3] original positions.
        log_scales: [N, 3] log-space scales.
        opacities: [N, 1] logit-space opacities.
        sh0: [N, 3] SH DC coefficients.
        n_pixels: Number of pixels in the rendered image.

    Returns:
        (estimated_psnr_drop_db, per_gaussian_contribution [N]).
    """
    d = (means_new - means_orig).norm(dim=-1)  # [N]
    s_min = log_scales.exp().min(dim=-1).values  # [N]
    alpha = torch.sigmoid(opacities.squeeze(-1))  # [N]
    color_norm = (0.28209479 * sh0 + 0.5).norm(dim=-1)  # [N]

    contrib = alpha.pow(2) * color_norm.pow(2) * d.pow(2) / (
        2 * s_min.pow(2).clamp(min=1e-12)
    )  # [N]
    mse_approx = contrib.sum() / n_pixels

    psnr_drop = 10.0 * torch.log10(mse_approx.clamp(min=1e-10)).item()
    return -psnr_drop, contrib


def render_loss(
    rendered: Tensor,
    target: Tensor,
) -> Tensor:
    """Photometric loss: L1 + 0.2 * (1 - SSIM).

    Args:
        rendered: [C, H, W, 3] rendered images.
        target: [C, H, W, 3] reference images.

    Returns:
        Scalar loss tensor.
    """
    l1 = (rendered - target).abs().mean()
    ssim_val = _ssim(rendered, target)
    return l1 + 0.2 * (1.0 - ssim_val)


def compute_psnr(rendered: Tensor, target: Tensor) -> float:
    """Compute PSNR between rendered and target images.

    Args:
        rendered: [C, H, W, 3] rendered images.
        target: [C, H, W, 3] reference images.

    Returns:
        PSNR in dB.
    """
    mse = (rendered - target).pow(2).mean().item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * torch.log10(torch.tensor(mse)).item()


def compute_psnr_per_view(rendered: Tensor, target: Tensor) -> list[float]:
    """Compute PSNR per camera view.

    Args:
        rendered: [C, H, W, 3] rendered images.
        target: [C, H, W, 3] reference images.

    Returns:
        List of PSNR values, one per view.
    """
    result: list[float] = []
    for c in range(rendered.shape[0]):
        mse = (rendered[c] - target[c]).pow(2).mean().item()
        if mse < 1e-10:
            result.append(100.0)
        else:
            result.append(-10.0 * math.log10(mse))
    return result


def compute_ssim(rendered: Tensor, target: Tensor) -> float:
    """Compute mean SSIM between rendered and target images.

    Args:
        rendered: [C, H, W, 3] rendered images.
        target: [C, H, W, 3] reference images.

    Returns:
        Mean SSIM value.
    """
    return _ssim(rendered, target).item()


def _ssim(
    img1: Tensor,
    img2: Tensor,
    window_size: int = 11,
) -> Tensor:
    """Differentiable SSIM for [C, H, W, 3] batched images.

    Uses a uniform box filter for simplicity (sufficient for quality gating).
    """
    C, H, W, ch = img1.shape
    x = img1.permute(0, 3, 1, 2).reshape(C * ch, 1, H, W)
    y = img2.permute(0, 3, 1, 2).reshape(C * ch, 1, H, W)

    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=x.device)
    kernel = kernel / kernel.numel()

    mu_x = torch.nn.functional.conv2d(x, kernel, padding=pad)
    mu_y = torch.nn.functional.conv2d(y, kernel, padding=pad)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = torch.nn.functional.conv2d(x * x, kernel, padding=pad) - mu_x_sq
    sigma_y_sq = torch.nn.functional.conv2d(y * y, kernel, padding=pad) - mu_y_sq
    sigma_xy = torch.nn.functional.conv2d(x * y, kernel, padding=pad) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    return ssim_map.mean()
