"""Test 2% residual correction budget for PSNR improvement."""
import sys
sys.path.insert(0, "src")

import gsplat
import numpy as np
import torch
from pathlib import Path

from scripts.finetune.io import load_ply_sequence, compute_global_ranges
from scripts.finetune.camera import SyntheticCameras
from scripts.finetune.renderer import render_batch
from scripts.finetune.losses import compute_psnr_per_view
from gscodec.constants import SCALE_16BIT, LO_BASE, SCALE_8BIT

device = "cuda:0"
frames, _ = load_ply_sequence(Path(r"D:\dymensium_soccer\plys"), device, max_frames=10)
mins, maxs = compute_global_ranges(frames)
scale = (maxs - mins).clamp(min=1e-8)
K = 51
N = frames[0].means.shape[0]

cameras = SyntheticCameras.from_scene(frames[0].means, n_cameras=48, height=1024, width=1024, device=device)
all_vm, all_Ks = cameras.get(torch.arange(48, device=device))
eval_vm, eval_Ks = all_vm[44:], all_Ks[44:]
train_vm, train_Ks = all_vm[:44], all_Ks[:44]
Ks_512 = train_Ks.clone()
Ks_512[:, 0, :] *= 0.5
Ks_512[:, 1, :] *= 0.5

s_min = torch.stack([f.scales.amin(0) for f in frames]).amin(0)
s_max = torch.stack([f.scales.amax(0) for f in frames]).amax(0)
q_min = torch.tensor([-1.0] * 4, device=device)
q_max = torch.tensor([1.0] * 4, device=device)
o_min = torch.tensor([-6.0], device=device)
o_max = torch.tensor([12.0], device=device)
sh_min = torch.tensor([-2.0] * 3, device=device)
sh_max = torch.tensor([4.0] * 3, device=device)


def qfloor(data, dmin, dmax):
    s = torch.where(dmax > dmin, dmax - dmin, torch.ones_like(dmin))
    return ((data - dmin) / s).clamp(0, 1) * SCALE_8BIT


def dq(qval, dmin, dmax):
    s = torch.where(dmax > dmin, dmax - dmin, torch.ones_like(dmin))
    return (qval / SCALE_8BIT) * s + dmin


def snap_means(f):
    ml = torch.sign(f.means) * torch.log1p(torch.abs(f.means))
    n = ((ml - mins) / scale).clamp(0, 1)
    q = (n * SCALE_16BIT).floor().long()
    hi = q // LO_BASE
    lo = q % LO_BASE
    lo_s = ((lo + K // 2) // K * K) % 256
    mn = (hi * LO_BASE + lo_s).float() / SCALE_16BIT * scale + mins
    return (torch.sign(mn) * (torch.exp(torch.abs(mn)) - 1)).detach()


def rq(ms, sc, qt, op, sh, vm, ks, w, h):
    r, _, _ = gsplat.rasterization(
        means=ms, quats=dq(qt, q_min, q_max), scales=dq(sc, s_min, s_max).exp(),
        opacities=torch.sigmoid(dq(op, o_min, o_max).squeeze(-1)),
        colors=0.28209479 * dq(sh, sh_min, sh_max) + 0.5,
        viewmats=vm, Ks=ks, width=w, height=h, packed=True, render_mode="RGB",
    )
    return r


# 2% of 23.4 MB = 480 KB for 351 frames = 1370 bytes/frame
budget_per_frame = 1370
print(f"Budget: {budget_per_frame} bytes/frame ({budget_per_frame * 8 / N:.2f} bits/Gaussian)")
print()

results_all = {
    "soft_round": [],
    "top100_attr_corr": [],
    "top274_lo_refine": [],
    "top500_lo_refine": [],
}

for fi in range(10):
    f = frames[fi]
    ms = snap_means(f)

    sc_fl = qfloor(f.scales, s_min, s_max).floor().detach()
    qt_fl = qfloor(f.quats.clamp(-1, 1), q_min, q_max).floor().detach()
    op_fl = qfloor(f.opacities.clamp(-6, 12), o_min, o_max).floor().detach()
    sh_fl = qfloor(f.sh0.clamp(-2, 4), sh_min, sh_max).floor().detach()

    # Soft rounding baseline
    sl = torch.nn.Parameter(torch.zeros_like(sc_fl))
    ql = torch.nn.Parameter(torch.zeros_like(qt_fl))
    ol = torch.nn.Parameter(torch.zeros_like(op_fl))
    shl = torch.nn.Parameter(torch.zeros_like(sh_fl))
    opt = torch.optim.Adam([sl, ql, ol, shl], lr=0.1)
    ref_t = render_batch(f.means, f.scales, f.quats, f.opacities, f.sh0,
                         train_vm, Ks_512, 512, 512).detach()
    for step in range(100):
        temp = 1.0 + step * 0.5
        idx = torch.randperm(44, device=device)[:6]
        img = rq(ms, sc_fl + torch.sigmoid(sl * temp), qt_fl + torch.sigmoid(ql * temp),
                 op_fl + torch.sigmoid(ol * temp), sh_fl + torch.sigmoid(shl * temp),
                 train_vm[idx], Ks_512[idx], 512, 512)
        opt.zero_grad()
        (img - ref_t[idx]).abs().mean().backward()
        opt.step()

    with torch.no_grad():
        sc_r = sc_fl + (torch.sigmoid(sl * 100) > 0.5).float()
        qt_r = qt_fl + (torch.sigmoid(ql * 100) > 0.5).float()
        op_r = op_fl + (torch.sigmoid(ol * 100) > 0.5).float()
        sh_r = sh_fl + (torch.sigmoid(shl * 100) > 0.5).float()
        ref = render_batch(f.means, f.scales, f.quats, f.opacities, f.sh0,
                           eval_vm, eval_Ks, 1024, 1024)
        img_soft = rq(ms, sc_r, qt_r, op_r, sh_r, eval_vm, eval_Ks, 1024, 1024)
    results_all["soft_round"].append(min(compute_psnr_per_view(img_soft, ref)))

    # Sensitivity for this frame
    ms_p = ms.clone().requires_grad_(True)
    img_g = rq(ms_p, sc_r, qt_r, op_r, sh_r, eval_vm, eval_Ks, 1024, 1024)
    (img_g - ref).pow(2).sum().backward()
    sensitivity = ms_p.grad.norm(dim=-1).detach()
    sorted_idx = sensitivity.argsort(descending=True)

    # Option A: ±2 STE on top-100 Gaussians (100 * 13 = 1300 bytes)
    n_corr = 100
    top = sorted_idx[:n_corr]
    sc_d = torch.nn.Parameter(torch.zeros(n_corr, 3, device=device))
    qt_d = torch.nn.Parameter(torch.zeros(n_corr, 4, device=device))
    op_d = torch.nn.Parameter(torch.zeros(n_corr, 1, device=device))
    sh_d = torch.nn.Parameter(torch.zeros(n_corr, 3, device=device))
    opt2 = torch.optim.Adam([sc_d, qt_d, op_d, sh_d], lr=0.05)

    for step in range(200):
        sc_try = sc_r.clone()
        qt_try = qt_r.clone()
        op_try = op_r.clone()
        sh_try = sh_r.clone()
        md = 2.0
        sc_try[top] += sc_d.clamp(-md, md).round() + (sc_d.clamp(-md, md) - sc_d.clamp(-md, md).round()).detach()
        qt_try[top] += qt_d.clamp(-md, md).round() + (qt_d.clamp(-md, md) - qt_d.clamp(-md, md).round()).detach()
        op_try[top] += op_d.clamp(-md, md).round() + (op_d.clamp(-md, md) - op_d.clamp(-md, md).round()).detach()
        sh_try[top] += sh_d.clamp(-md, md).round() + (sh_d.clamp(-md, md) - sh_d.clamp(-md, md).round()).detach()
        jdx = torch.randperm(44, device=device)[:6]
        img = rq(ms, sc_try, qt_try, op_try, sh_try, train_vm[jdx], Ks_512[jdx], 512, 512)
        opt2.zero_grad()
        (img - ref_t[jdx]).abs().mean().backward()
        opt2.step()

    with torch.no_grad():
        sc_f = sc_r.clone()
        qt_f = qt_r.clone()
        op_f = op_r.clone()
        sh_f = sh_r.clone()
        sc_f[top] = (sc_r[top] + sc_d.clamp(-2, 2).round()).clamp(0, SCALE_8BIT)
        qt_f[top] = (qt_r[top] + qt_d.clamp(-2, 2).round()).clamp(0, SCALE_8BIT)
        op_f[top] = (op_r[top] + op_d.clamp(-2, 2).round()).clamp(0, SCALE_8BIT)
        sh_f[top] = (sh_r[top] + sh_d.clamp(-2, 2).round()).clamp(0, SCALE_8BIT)
        img_c = rq(ms, sc_f, qt_f, op_f, sh_f, eval_vm, eval_Ks, 1024, 1024)
    results_all["top100_attr_corr"].append(min(compute_psnr_per_view(img_c, ref)))

    # Option B: lo refinement — full-precision lo for top-N (2-byte idx + 3-byte lo = 5 bytes)
    ml = torch.sign(f.means) * torch.log1p(torch.abs(f.means))
    n = ((ml - mins) / scale).clamp(0, 1)
    q = (n * SCALE_16BIT).floor().long()
    lo_full = (q % LO_BASE).clamp(0, 255)
    hi = (q // LO_BASE).clamp(0, 219)
    lo_snapped = ((lo_full + K // 2) // K * K) % 256

    for n_refine, label in [(274, "top274_lo_refine"), (500, "top500_lo_refine")]:
        top_lo = sorted_idx[:n_refine]
        lo_hybrid = lo_snapped.clone()
        lo_hybrid[top_lo] = lo_full[top_lo]
        mn = (hi * LO_BASE + lo_hybrid).float() / SCALE_16BIT * scale + mins
        ms_hybrid = (torch.sign(mn) * (torch.exp(torch.abs(mn)) - 1)).detach()
        with torch.no_grad():
            img_lo = rq(ms_hybrid, sc_r, qt_r, op_r, sh_r, eval_vm, eval_Ks, 1024, 1024)
        results_all[label].append(min(compute_psnr_per_view(img_lo, ref)))

    if (fi + 1) % 5 == 0:
        print(f"  {fi + 1}/10 done")

print()
print(f"{'Method':<35} {'Budget':>8} {'min':>6} {'mean':>6} {'gain':>6}")
print("-" * 65)
base_mean = sum(results_all["soft_round"]) / 10
for label, budget_str in [
    ("soft_round", "--"),
    ("top100_attr_corr", "1300 B"),
    ("top274_lo_refine", "1370 B"),
    ("top500_lo_refine", "2500 B"),
]:
    r = results_all[label]
    gain = sum(r) / 10 - base_mean
    print(f"  {label:<33} {budget_str:>8} {min(r):>6.1f} {sum(r)/10:>6.1f} {gain:>+5.1f}")
