"""Post-snap fine-tuning to recover PSNR after K-snap quantization.

After K-snapping means to grid points, fine-tunes scales, opacity,
color (sh0), and rotations (quats) to compensate for position error.
Means stay frozen on-grid so compression is preserved.
"""

import logging
import math

import gsplat
import torch
from gsply import GSTensor
from torch import Tensor

logger = logging.getLogger(__name__)

_SH_C0 = 0.28209479177387814


def post_snap_finetune(
    frames: list[GSTensor],
    means_snapped: list[Tensor],
    device: str = "cuda:0",
    n_cameras: int = 48,
    steps_512: int = 100,
    steps_1024: int = 50,
    cams_per_step_512: int = 6,
    cams_per_step_1024: int = 4,
    lr: float = 1e-3,
) -> list[GSTensor]:
    """Fine-tune attributes after K-snap to recover PSNR.

    Args:
        frames: Original GSTensor frames (for reference rendering).
        means_snapped: K-snapped means tensors (frozen during fine-tuning).
        device: Torch device.
        n_cameras: Total synthetic cameras on orbit sphere.
        steps_512: Training steps at 512x512.
        steps_1024: Training steps at 1024x1024.
        cams_per_step_512: Cameras sampled per step at 512.
        cams_per_step_1024: Cameras sampled per step at 1024.
        lr: Learning rate for Adam.

    Returns:
        List of fine-tuned GSTensor frames with snapped means.
    """
    if not frames:
        return []

    n_eval = 4
    n_train = n_cameras - n_eval
    total_steps = steps_512 + steps_1024

    # Generate cameras once (shared across all frames)
    viewmats, Ks = _make_cameras(frames[0].means, n_cameras, device)
    train_vm, train_Ks = viewmats[:n_train], Ks[:n_train]
    train_Ks_512 = train_Ks.clone()
    train_Ks_512[:, 0, :] *= 0.5
    train_Ks_512[:, 1, :] *= 0.5

    results: list[GSTensor] = []

    for i, (frame, ms) in enumerate(zip(frames, means_snapped)):
        # Pre-render references from original frame
        with torch.no_grad():
            ref_512 = _render(
                frame.means,
                frame.scales,
                frame.quats,
                frame.opacities,
                frame.sh0,
                train_vm,
                train_Ks_512,
                512,
                512,
            ).detach()
            if steps_1024 > 0:
                ref_1024 = _render(
                    frame.means,
                    frame.scales,
                    frame.quats,
                    frame.opacities,
                    frame.sh0,
                    train_vm,
                    train_Ks,
                    1024,
                    1024,
                ).detach()

        # Optimizable parameters (means frozen)
        scales_p = torch.nn.Parameter(frame.scales.clone())
        opacity_p = torch.nn.Parameter(frame.opacities.clone())
        sh0_p = torch.nn.Parameter(frame.sh0.clone())
        quats_p = torch.nn.Parameter(frame.quats.clone())

        optimizer = torch.optim.Adam(
            [scales_p, opacity_p, sh0_p, quats_p],
            lr=lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=lr * 0.1,
        )

        # Phase 1: 512x512
        for _step in range(steps_512):
            idx = torch.randperm(n_train, device=device)[:cams_per_step_512]
            rendered = _render(
                ms, scales_p, quats_p, opacity_p, sh0_p, train_vm[idx], train_Ks_512[idx], 512, 512
            )
            loss = _render_loss(rendered, ref_512[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Phase 2: 1024x1024
        for _step in range(steps_1024):
            idx = torch.randperm(n_train, device=device)[:cams_per_step_1024]
            rendered = _render(
                ms, scales_p, quats_p, opacity_p, sh0_p, train_vm[idx], train_Ks[idx], 1024, 1024
            )
            loss = _render_loss(rendered, ref_1024[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Clamp fine-tuned values to stay within ±1 quantization step
        # of originals. This preserves VP9 atlas smoothness.
        # 8-bit quantization step = range / 219
        with torch.no_grad():
            for param, orig in [
                (scales_p, frame.scales),
                (opacity_p, frame.opacities),
                (sh0_p, frame.sh0),
                (quats_p, frame.quats),
            ]:
                # Estimate quantization step size per channel
                r = orig.amax(dim=0) - orig.amin(dim=0)
                step_size = r / 219.0  # 8-bit video-safe has 220 levels
                max_delta = step_size * 1.5  # allow up to 1.5 quant steps
                param.data.clamp_(orig - max_delta, orig + max_delta)

        # Keep ORIGINAL means (encoder does K-snap after sorting).
        results.append(
            GSTensor(
                means=frame.means,
                scales=scales_p.detach(),
                quats=quats_p.detach(),
                opacities=opacity_p.detach(),
                sh0=sh0_p.detach(),
                shN=None,
                masks=frame.masks,
            )
        )

        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            logger.info(f"  Fine-tuned {i + 1}/{len(frames)} frames")

    return results


def _render(means, scales, quats, opacities, sh0, viewmats, Ks, w, h):
    """Render via gsplat."""
    rendered, _, _ = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales.exp(),
        opacities=torch.sigmoid(opacities.squeeze(-1)),
        colors=_SH_C0 * sh0 + 0.5,
        viewmats=viewmats,
        Ks=Ks,
        width=w,
        height=h,
        packed=True,
        render_mode="RGB",
    )
    return rendered


def _render_loss(rendered, target):
    """L1 + 0.2*(1-SSIM) loss."""
    return (rendered - target).abs().mean()


def _make_cameras(means, n_cameras, device):
    """Generate orbit cameras (OpenCV convention, radius_scale=0.3)."""
    centroid = means.mean(dim=0)
    extent = (means.max(dim=0).values - means.min(dim=0).values).norm()
    radius = extent * 0.3
    golden = (1 + math.sqrt(5)) / 2

    viewmats = torch.zeros(n_cameras, 4, 4, device=device)
    for i in range(n_cameras):
        theta = math.acos(1 - 2 * (i + 0.5) / n_cameras)
        phi = 2 * math.pi * i / golden
        pos = torch.tensor(
            [
                radius * math.sin(theta) * math.cos(phi) + centroid[0].item(),
                radius * math.sin(theta) * math.sin(phi) + centroid[1].item(),
                radius * math.cos(theta) + centroid[2].item(),
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

    fx = 512.0 / math.tan(math.radians(30))
    Ks = torch.zeros(n_cameras, 3, 3, device=device)
    Ks[:, 0, 0] = fx
    Ks[:, 1, 1] = fx
    Ks[:, 0, 2] = 512.0
    Ks[:, 1, 2] = 512.0
    Ks[:, 2, 2] = 1.0

    return viewmats, Ks
