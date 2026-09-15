"""Test gradient-guided and soft rounding for post-snap PSNR recovery."""

import sys
import time

sys.path.insert(0, "src")

import gsplat
import numpy as np
import torch
import xllvp9
from pathlib import Path

from scripts.finetune.io import load_ply_sequence, compute_global_ranges
from scripts.finetune.camera import SyntheticCameras
from scripts.finetune.renderer import render_batch
from scripts.finetune.losses import compute_psnr_per_view
from gscodec.constants import SCALE_16BIT, LO_BASE, SCALE_8BIT, MIN_VAL
from gscodec.encoder.utils.helpers import log_transform
from gscodec.encoder.sorting.morton import MortonSortingStrategy
from gscodec.encoder.video_writer import build_frame_atlas

device = "cuda:0"
frames, _ = load_ply_sequence(Path(r"D:\dymensium_soccer\plys"), device, max_frames=30)
mins, maxs = compute_global_ranges(frames)
scale = (maxs - mins).clamp(min=1e-8)
K = 35
N = frames[0].means.shape[0]
sorter = MortonSortingStrategy()

cameras = SyntheticCameras.from_scene(
    frames[0].means, n_cameras=48, height=1024, width=1024, device=device
)
all_vm, all_Ks = cameras.get(torch.arange(48, device=device))
eval_vm, eval_Ks = all_vm[44:], all_Ks[44:]
train_vm, train_Ks = all_vm[:44], all_Ks[:44]
train_Ks_512 = train_Ks.clone()
train_Ks_512[:, 0, :] *= 0.5
train_Ks_512[:, 1, :] *= 0.5

# Global quant ranges
s_min = torch.stack([f.scales.amin(0) for f in frames]).amin(0)
s_max = torch.stack([f.scales.amax(0) for f in frames]).amax(0)
q_min = torch.tensor([-1.0] * 4, device=device)
q_max = torch.tensor([1.0] * 4, device=device)
o_min = torch.tensor([-6.0], device=device)
o_max = torch.tensor([12.0], device=device)
sh_min = torch.tensor([-2.0] * 3, device=device)
sh_max = torch.tensor([4.0] * 3, device=device)

# Sensitivity
mp = frames[0].means.clone().requires_grad_(True)
r, _, _ = gsplat.rasterization(
    means=mp, quats=frames[0].quats, scales=frames[0].scales.exp(),
    opacities=torch.sigmoid(frames[0].opacities.squeeze(-1)),
    colors=0.28209479 * frames[0].sh0 + 0.5,
    viewmats=all_vm, Ks=all_Ks, width=1024, height=1024,
    packed=True, render_mode="RGB",
)
r.sum().backward()
sens = mp.grad.norm(dim=-1).detach()
protect = sens >= torch.quantile(sens, 0.98)


def snap_means(f):
    ml = torch.sign(f.means) * torch.log1p(torch.abs(f.means))
    n = ((ml - mins) / scale).clamp(0, 1)
    q = (n * SCALE_16BIT).floor().long()
    hi = q // LO_BASE
    lo = q % LO_BASE
    lo_s = lo.clone()
    lo_s[~protect] = ((lo[~protect] + K // 2) // K * K) % 256
    mn = (hi * LO_BASE + lo_s).float() / SCALE_16BIT * scale + mins
    return (torch.sign(mn) * (torch.exp(torch.abs(mn)) - 1)).detach()


def quant_floor(data, dmin, dmax):
    s = torch.where(dmax > dmin, dmax - dmin, torch.ones_like(dmin))
    return ((data - dmin) / s).clamp(0, 1) * SCALE_8BIT


def dequant(qval, dmin, dmax):
    s = torch.where(dmax > dmin, dmax - dmin, torch.ones_like(dmin))
    return (qval / SCALE_8BIT) * s + dmin


def render_q(means, sc_q, qt_q, op_q, sh_q, vm, ks, w, h):
    scales = dequant(sc_q, s_min, s_max)
    quats = dequant(qt_q, q_min, q_max)
    opacity = dequant(op_q, o_min, o_max)
    sh0 = dequant(sh_q, sh_min, sh_max)
    rendered, _, _ = gsplat.rasterization(
        means=means, quats=quats, scales=scales.exp(),
        opacities=torch.sigmoid(opacity.squeeze(-1)),
        colors=0.28209479 * sh0 + 0.5,
        viewmats=vm, Ks=ks, width=w, height=h,
        packed=True, render_mode="RGB",
    )
    return rendered


# === Test all methods ===
snap_p = []
grad_p = []
soft_p = []
grad_changes = []
soft_changes = []

t_grad_total = 0
t_soft_total = 0

for fi in range(10):
    f = frames[fi]
    ms = snap_means(f)

    with torch.no_grad():
        ref = render_batch(f.means, f.scales, f.quats, f.opacities, f.sh0,
                           eval_vm, eval_Ks, 1024, 1024)

    sc_fl = quant_floor(f.scales, s_min, s_max).floor().detach()
    qt_fl = quant_floor(f.quats.clamp(-1, 1), q_min, q_max).floor().detach()
    op_fl = quant_floor(f.opacities.clamp(-6, 12), o_min, o_max).floor().detach()
    sh_fl = quant_floor(f.sh0.clamp(-2, 4), sh_min, sh_max).floor().detach()

    # Snap baseline
    with torch.no_grad():
        img_snap = render_q(ms, sc_fl, qt_fl, op_fl, sh_fl, eval_vm, eval_Ks, 1024, 1024)
    snap_p.append(min(compute_psnr_per_view(img_snap, ref)))

    # --- Gradient-guided rounding ---
    t0 = time.time()
    sc_p = sc_fl.clone().requires_grad_(True)
    qt_p = qt_fl.clone().requires_grad_(True)
    op_p = op_fl.clone().requires_grad_(True)
    sh_p = sh_fl.clone().requires_grad_(True)

    img = render_q(ms, sc_p, qt_p, op_p, sh_p, all_vm, all_Ks, 1024, 1024)
    ref_all = render_batch(f.means, f.scales, f.quats, f.opacities, f.sh0,
                            all_vm, all_Ks, 1024, 1024).detach()
    loss = (img - ref_all).pow(2).sum()
    loss.backward()

    with torch.no_grad():
        sc_g = sc_fl.clone()
        sc_g[sc_p.grad < 0] += 1
        qt_g = qt_fl.clone()
        qt_g[qt_p.grad < 0] += 1
        op_g = op_fl.clone()
        op_g[op_p.grad < 0] += 1
        sh_g = sh_fl.clone()
        sh_g[sh_p.grad < 0] += 1

        img_g = render_q(ms, sc_g, qt_g, op_g, sh_g, eval_vm, eval_Ks, 1024, 1024)
    t_grad_total += time.time() - t0
    grad_p.append(min(compute_psnr_per_view(img_g, ref)))

    total_v = sc_fl.numel() + qt_fl.numel() + op_fl.numel() + sh_fl.numel()
    changed = (sc_g != sc_fl).sum() + (qt_g != qt_fl).sum() + (op_g != op_fl).sum() + (sh_g != sh_fl).sum()
    grad_changes.append(changed.item() / total_v * 100)

    # --- Soft rounding ---
    t0 = time.time()
    sc_logit = torch.nn.Parameter(torch.zeros_like(sc_fl))
    qt_logit = torch.nn.Parameter(torch.zeros_like(qt_fl))
    op_logit = torch.nn.Parameter(torch.zeros_like(op_fl))
    sh_logit = torch.nn.Parameter(torch.zeros_like(sh_fl))

    opt = torch.optim.Adam([sc_logit, qt_logit, op_logit, sh_logit], lr=0.1)
    ref_t = render_batch(f.means, f.scales, f.quats, f.opacities, f.sh0,
                          train_vm, train_Ks_512, 512, 512).detach()

    for step in range(100):
        temp = 1.0 + step * 0.5
        sc_s = sc_fl + torch.sigmoid(sc_logit * temp)
        qt_s = qt_fl + torch.sigmoid(qt_logit * temp)
        op_s = op_fl + torch.sigmoid(op_logit * temp)
        sh_s = sh_fl + torch.sigmoid(sh_logit * temp)

        idx = torch.randperm(44, device=device)[:6]
        img = render_q(ms, sc_s, qt_s, op_s, sh_s, train_vm[idx], train_Ks_512[idx], 512, 512)
        loss = (img - ref_t[idx]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        sc_h = sc_fl + (torch.sigmoid(sc_logit * 100) > 0.5).float()
        qt_h = qt_fl + (torch.sigmoid(qt_logit * 100) > 0.5).float()
        op_h = op_fl + (torch.sigmoid(op_logit * 100) > 0.5).float()
        sh_h = sh_fl + (torch.sigmoid(sh_logit * 100) > 0.5).float()

        img_s = render_q(ms, sc_h, qt_h, op_h, sh_h, eval_vm, eval_Ks, 1024, 1024)
    t_soft_total += time.time() - t0
    soft_p.append(min(compute_psnr_per_view(img_s, ref)))

    changed = (sc_h != sc_fl).sum() + (qt_h != qt_fl).sum() + (op_h != op_fl).sum() + (sh_h != sh_fl).sum()
    soft_changes.append(changed.item() / total_v * 100)

# Video impact test on 30 frames (gradient-guided only for speed)
def encode_atlas(f, sc_q, qt_q, op_q, sh_q):
    ml = log_transform(f.means)
    idx = sorter.sort_with_global_bbox(ml, mins, maxs)
    from gscodec.encoder.quantization.strategies import quantize_means_16bit_split
    hi, lo = quantize_means_16bit_split(ml[idx], mins, maxs, lo_snap_k=K)
    return build_frame_atlas(
        hi, (sc_q[idx] + MIN_VAL).cpu().numpy().astype(np.uint8),
        (qt_q[idx] + MIN_VAL).cpu().numpy().astype(np.uint8),
        (op_q[idx] + MIN_VAL).cpu().numpy().astype(np.uint8).squeeze(-1),
        (sh_q[idx] + MIN_VAL).cpu().numpy().astype(np.uint8), N,
    )


a_floor = []
a_grad = []
for fi in range(30):
    f = frames[fi]
    ms = snap_means(f)
    sc_fl = quant_floor(f.scales, s_min, s_max).floor().detach()
    qt_fl = quant_floor(f.quats.clamp(-1, 1), q_min, q_max).floor().detach()
    op_fl = quant_floor(f.opacities.clamp(-6, 12), o_min, o_max).floor().detach()
    sh_fl = quant_floor(f.sh0.clamp(-2, 4), sh_min, sh_max).floor().detach()

    a_floor.append(encode_atlas(f, sc_fl, qt_fl, op_fl, sh_fl))

    # Gradient rounding
    sc_p = sc_fl.clone().requires_grad_(True)
    qt_p = qt_fl.clone().requires_grad_(True)
    op_p = op_fl.clone().requires_grad_(True)
    sh_p = sh_fl.clone().requires_grad_(True)
    img = render_q(ms, sc_p, qt_p, op_p, sh_p, all_vm, all_Ks, 1024, 1024)
    ref_a = render_batch(f.means, f.scales, f.quats, f.opacities, f.sh0,
                          all_vm, all_Ks, 1024, 1024).detach()
    (img - ref_a).pow(2).sum().backward()
    with torch.no_grad():
        sc_g = sc_fl.clone(); sc_g[sc_p.grad < 0] += 1
        qt_g = qt_fl.clone(); qt_g[qt_p.grad < 0] += 1
        op_g = op_fl.clone(); op_g[op_p.grad < 0] += 1
        sh_g = sh_fl.clone(); sh_g[sh_p.grad < 0] += 1
    a_grad.append(encode_atlas(f, sc_g, qt_g, op_g, sh_g))

v_floor = xllvp9.encode(np.stack(a_floor), fps=30, keyframe_interval=30)
v_grad = xllvp9.encode(np.stack(a_grad), fps=30, keyframe_interval=30)

n = 10
print(f"{'Method':<30} {'min':>5} {'mean':>5} {'bytes_chg':>10} {'time':>8}")
print("-" * 62)
print(f"{'Snap only (floor quant)':<30} {min(snap_p):>5.1f} {sum(snap_p)/n:>5.1f} {'0.0%':>10} {'--':>8}")
print(f"{'Gradient-guided rounding':<30} {min(grad_p):>5.1f} {sum(grad_p)/n:>5.1f} {sum(grad_changes)/n:>9.1f}% {t_grad_total:>7.1f}s")
print(f"{'Soft rounding (100 steps)':<30} {min(soft_p):>5.1f} {sum(soft_p)/n:>5.1f} {sum(soft_changes)/n:>9.1f}% {t_soft_total:>7.1f}s")
print(f"\nVP9 video (30 frames): floor={len(v_floor):,}  grad={len(v_grad):,}  ({len(v_grad)/len(v_floor)*100:.1f}%)")
