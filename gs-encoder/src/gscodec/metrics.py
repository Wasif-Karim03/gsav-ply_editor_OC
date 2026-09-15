"""Rendering PSNR measurement for Gaussian Splatting quality evaluation.

Renders original and reconstructed Gaussians from multiple orbit cameras
using gsplat, then computes pixel-level PSNR. This is the ground-truth
quality metric for the codec — it measures what the user actually sees.
"""

import math

import numpy as np
import torch
from gsply import GSData, GSTensor

N_CAMERAS = 8
RENDER_SIZE = 512
FOV_DEGREES = 60


def rendering_psnr(
    original: GSTensor,
    decoded: GSData,
    n_cameras: int = N_CAMERAS,
    render_size: int = RENDER_SIZE,
    viewmats: torch.Tensor | None = None,
    Ks: torch.Tensor | None = None,
) -> float:
    """Compute rendering PSNR between original and decoded Gaussians.

    Renders both from orbit cameras and computes mean PSNR across views.
    Pass viewmats/Ks to use fixed cameras (recommended for batch
    consistency); otherwise generates per-frame orbit cameras.

    Args:
        original: Original GSTensor (torch, on GPU).
        decoded: Decoded GSData (numpy, from ChunkDecoder).
        n_cameras: Number of orbit cameras.
        render_size: Render resolution (square).
        viewmats: Optional fixed camera view matrices [C, 4, 4].
        Ks: Optional fixed camera intrinsics [C, 3, 3].

    Returns:
        PSNR in dB (higher = better). Uses [0, 1] pixel range.
    """
    device = original.means.device
    if viewmats is None or Ks is None:
        viewmats, Ks = _make_orbit_cameras(original.means, n_cameras, render_size, device)

    # Render original
    with torch.no_grad():
        img_orig = _render(
            original.means,
            original.quats,
            original.scales,
            original.opacities,
            original.sh0,
            viewmats,
            Ks,
            render_size,
        )

    # Render decoded
    img_dec = _render_gsdata(decoded, viewmats, Ks, render_size, device)

    mse = ((img_orig - img_dec) ** 2).mean().item()
    if mse <= 0:
        return 99.0
    return -10.0 * math.log10(mse)


def rendering_psnr_batch(
    originals: list[GSTensor],
    decoded_frames: list[GSData],
    n_cameras: int = N_CAMERAS,
    render_size: int = RENDER_SIZE,
) -> tuple[float, float, list[float]]:
    """Compute rendering PSNR for a batch of frames with fixed cameras.

    Uses cameras from frame 0 for all frames, ensuring consistent
    measurement that doesn't vary with per-frame centroid drift.

    Args:
        originals: List of original GSTensors (sorted to match decoded order).
        decoded_frames: List of decoded GSData frames.
        n_cameras: Number of orbit cameras.
        render_size: Render resolution.

    Returns:
        Tuple of (avg_psnr, min_psnr, per_frame_psnrs).
    """
    device = originals[0].means.device
    viewmats, Ks = _make_orbit_cameras(
        originals[0].means,
        n_cameras,
        render_size,
        device,
    )

    psnrs = []
    for orig, dec in zip(originals, decoded_frames):
        psnrs.append(
            rendering_psnr(
                orig,
                dec,
                n_cameras,
                render_size,
                viewmats=viewmats,
                Ks=Ks,
            )
        )

    return float(np.mean(psnrs)), float(np.min(psnrs)), psnrs


def _make_orbit_cameras(
    means: torch.Tensor,
    n_cams: int,
    render_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate orbit cameras around the scene centroid.

    Returns:
        Tuple of (viewmats [n_cams, 4, 4], Ks [n_cams, 3, 3]).
    """
    centroid = means.mean(dim=0)
    extent = (means.max(dim=0).values - means.min(dim=0).values).norm()
    radius = extent * 0.3
    golden = (1 + 2.2360679775) / 2

    viewmats = torch.zeros(n_cams, 4, 4, device=device)
    for i in range(n_cams):
        theta = math.acos(1 - 2 * (i + 0.5) / n_cams)
        phi = 2 * math.pi * i / golden
        pos = torch.tensor(
            [
                radius * math.sin(theta) * math.cos(phi) + centroid[0],
                radius * math.sin(theta) * math.sin(phi) + centroid[1],
                radius * math.cos(theta) + centroid[2],
            ],
            device=device,
        )

        fwd = centroid - pos
        fwd = fwd / fwd.norm()
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        if torch.abs(fwd.dot(up)) > 0.99:
            up = torch.tensor([0.0, 0.0, 1.0], device=device)
        right = torch.linalg.cross(fwd, up)
        right = right / right.norm()
        up = torch.linalg.cross(right, fwd)

        viewmats[i, 0, :3] = right
        viewmats[i, 1, :3] = -up
        viewmats[i, 2, :3] = fwd
        viewmats[i, :3, 3] = viewmats[i, :3, :3] @ (-pos)
        viewmats[i, 3, 3] = 1.0

    half = render_size / 2.0
    fx = half / math.tan(math.radians(FOV_DEGREES / 2))
    Ks = torch.zeros(n_cams, 3, 3, device=device)
    Ks[:, 0, 0] = fx
    Ks[:, 1, 1] = fx
    Ks[:, 0, 2] = half
    Ks[:, 1, 2] = half
    Ks[:, 2, 2] = 1.0

    return viewmats, Ks


def _render(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    render_size: int,
) -> torch.Tensor:
    """Render Gaussians from multiple viewpoints. Returns [n_cams, H, W, 3]."""
    import gsplat

    colors = 0.28209479 * sh0 + 0.5
    op = opacities.squeeze(-1) if opacities.ndim > 1 else opacities

    with torch.no_grad():
        rendered, _, _ = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales.exp(),
            opacities=torch.sigmoid(op),
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=render_size,
            height=render_size,
            packed=True,
            render_mode="RGB",
        )
    return rendered.clamp(0, 1)


def _render_gsdata(
    data: GSData,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    render_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Render decoded GSData (numpy) from multiple viewpoints."""

    def t(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).float().to(device)

    return _render(
        t(data.means),
        t(data.quats),
        t(data.scales),
        t(data.opacities).unsqueeze(-1),
        t(data.sh0),
        viewmats,
        Ks,
        render_size,
    )
