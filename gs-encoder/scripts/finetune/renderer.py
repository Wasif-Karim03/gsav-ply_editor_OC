"""gsplat CUDA rendering wrapper for fine-tuning.

Handles PLY-format -> gsplat convention conversions:
- scales: log-space -> linear (exp)
- opacities: logit -> sigmoid
- sh0: SH DC coefficient -> RGB color
"""

import torch
from torch import Tensor

import gsplat

_SH_C0 = 0.28209479177387814


def render_batch(
    means: Tensor,
    log_scales: Tensor,
    quats: Tensor,
    logit_opacities: Tensor,
    sh0: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
) -> Tensor:
    """Render Gaussians from multiple cameras via gsplat CUDA rasterizer.

    Args:
        means: [N, 3] Gaussian positions.
        log_scales: [N, 3] log-space scales (PLY format).
        quats: [N, 4] quaternions (wxyz).
        logit_opacities: [N, 1] logit-space opacities.
        sh0: [N, 3] SH DC coefficients.
        viewmats: [C, 4, 4] world-to-camera transforms.
        Ks: [C, 3, 3] camera intrinsics.
        width: Image width.
        height: Image height.

    Returns:
        [C, H, W, 3] rendered RGB images in [0, 1].
    """
    scales = torch.exp(log_scales)  # [N, 3]
    opacities = torch.sigmoid(logit_opacities.squeeze(-1))  # [N]
    colors = _SH_C0 * sh0 + 0.5  # [N, 3]

    rendered, _, _ = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        packed=True,
        render_mode="RGB",
    )
    return rendered  # [C, H, W, 3]
