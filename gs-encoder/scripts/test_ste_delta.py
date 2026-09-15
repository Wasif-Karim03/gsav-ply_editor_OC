"""Test STE optimization with varying max_delta + VP9 impact."""
import sys
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
from gscodec.constants import SCALE_16BIT, LO_BASE, SCALE_8BIT
from gscodec.encoder.utils.helpers import log_transform
from gscodec.encoder.sorting.morton import MortonSortingStrategy
from gscodec.encoder.video_writer import build_frame_atlas
from gscodec.encoder.quantization.strategies import quantize_means_16bit_split

device = "cuda:0"
frames, _ = load_ply_sequence(Path(r"D:\dymensium_soccer\plys"), device, max_frames=30)
mins, maxs = compute_global_ranges(frames)
scale = (maxs - mins).clamp(min=1e-8)
K = 51
N = frames[0].means.shape[0]
sorter = MortonSortingStrategy()

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


def dequant(qval, dmin, dmax):
    s = torch.where(dmax > dmin, dmax - dmin, torch.ones_like(dmin))
    return (qval / SCALE_8BIT) * s + dmin


def rq(means, sc, qt, op, sh, vm, ks, w, h):
    r, _, _ = gsplat.rasterization(
        means=means,
        quats=dequant(qt, q_min, q_max),
        scales=dequant(sc, s_min, s_max).exp(),
        opacities=torch.sigmoid(dequant(op, o_min, o_max).squeeze(-1)),
        colors=0.28209479 * dequant(sh, sh_min, sh_max) + 0.5,
        viewmats=vm, Ks=ks, width=w, height=h,
        packed=True, render_mode="RGB",
    )
    return r


def snap_uniform(f):
    ml = torch.sign(f.means) * torch.log1p(torch.abs(f.means))
    n = ((ml - mins) / scale).clamp(0, 1)
    q = (n * SCALE_16BIT).floor().long()
    hi = q // LO_BASE
    lo = q % LO_BASE
    lo_s = ((lo + K // 2) // K * K) % 256
    mn = (hi * LO_BASE + lo_s).float() / SCALE_16BIT * scale + mins
    return (torch.sign(mn) * (torch.exp(torch.abs(mn)) - 1)).detach()


def optimize_frame(f, ms, max_delta, n_steps=200):
    sc_fl = qfloor(f.scales, s_min, s_max).floor().detach()
    qt_fl = qfloor(f.quats.clamp(-1, 1), q_min, q_max).floor().detach()
    op_fl = qfloor(f.opacities.clamp(-6, 12), o_min, o_max).floor().detach()
    sh_fl = qfloor(f.sh0.clamp(-2, 4), sh_min, sh_max).floor().detach()

    sc_d = torch.nn.Parameter(torch.zeros_like(sc_fl))
    qt_d = torch.nn.Parameter(torch.zeros_like(qt_fl))
    op_d = torch.nn.Parameter(torch.zeros_like(op_fl))
    sh_d = torch.nn.Parameter(torch.zeros_like(sh_fl))
    opt = torch.optim.Adam([sc_d, qt_d, op_d, sh_d], lr=0.05)
    ref_t = render_batch(
        f.means, f.scales, f.quats, f.opacities, f.sh0,
        train_vm, Ks_512, 512, 512,
    ).detach()

    md = float(max_delta)
    for step in range(n_steps):
        sc_q = sc_fl + sc_d.clamp(-md, md).round()
        qt_q = qt_fl + qt_d.clamp(-md, md).round()
        op_q = op_fl + op_d.clamp(-md, md).round()
        sh_q = sh_fl + sh_d.clamp(-md, md).round()
        # STE
        sc_ste = sc_fl + sc_d.clamp(-md, md) + (sc_q - sc_fl - sc_d.clamp(-md, md)).detach()
        qt_ste = qt_fl + qt_d.clamp(-md, md) + (qt_q - qt_fl - qt_d.clamp(-md, md)).detach()
        op_ste = op_fl + op_d.clamp(-md, md) + (op_q - op_fl - op_d.clamp(-md, md)).detach()
        sh_ste = sh_fl + sh_d.clamp(-md, md) + (sh_q - sh_fl - sh_d.clamp(-md, md)).detach()

        idx = torch.randperm(44, device=device)[:6]
        img = rq(ms, sc_ste, qt_ste, op_ste, sh_ste, train_vm[idx], Ks_512[idx], 512, 512)
        opt.zero_grad()
        (img - ref_t[idx]).abs().mean().backward()
        opt.step()

    with torch.no_grad():
        sc_r = (sc_fl + sc_d.clamp(-md, md).round()).clamp(0, SCALE_8BIT)
        qt_r = (qt_fl + qt_d.clamp(-md, md).round()).clamp(0, SCALE_8BIT)
        op_r = (op_fl + op_d.clamp(-md, md).round()).clamp(0, SCALE_8BIT)
        sh_r = (sh_fl + sh_d.clamp(-md, md).round()).clamp(0, SCALE_8BIT)
    return sc_r, qt_r, op_r, sh_r, sc_fl, qt_fl, op_fl, sh_fl


# PSNR sweep
print(f"max_delta   min  mean  bytes_chg")
print("-" * 38)

for md in [1, 2, 3]:
    results = []
    total_chg = 0
    total_v = 0
    for fi in range(10):
        f = frames[fi]
        ms = snap_uniform(f)
        sc_r, qt_r, op_r, sh_r, sc_fl, qt_fl, op_fl, sh_fl = optimize_frame(f, ms, md)
        with torch.no_grad():
            ref = render_batch(f.means, f.scales, f.quats, f.opacities, f.sh0, eval_vm, eval_Ks, 1024, 1024)
            img = rq(ms, sc_r, qt_r, op_r, sh_r, eval_vm, eval_Ks, 1024, 1024)
        results.append(min(compute_psnr_per_view(img, ref)))
        total_chg += ((sc_r != sc_fl).sum() + (qt_r != qt_fl).sum() + (op_r != op_fl).sum() + (sh_r != sh_fl).sum()).item()
        total_v += sc_fl.numel() + qt_fl.numel() + op_fl.numel() + sh_fl.numel()
    print(f"  +/-{md}     {min(results):.1f}  {sum(results)/10:.1f}  {total_chg/total_v*100:.1f}%")

# VP9 impact for +/-2
print("\nVP9 video impact (30 frames, +/-2):")
atlases_orig = []
atlases_d2 = []
for fi in range(30):
    f = frames[fi]
    ms = snap_uniform(f)
    ml = log_transform(f.means)
    idx = sorter.sort_with_global_bbox(ml, mins, maxs)
    hi, lo = quantize_means_16bit_split(ml[idx], mins, maxs, lo_snap_k=K)

    sc_fl = qfloor(f.scales, s_min, s_max).floor().detach()
    qt_fl = qfloor(f.quats.clamp(-1, 1), q_min, q_max).floor().detach()
    op_fl = qfloor(f.opacities.clamp(-6, 12), o_min, o_max).floor().detach()
    sh_fl = qfloor(f.sh0.clamp(-2, 4), sh_min, sh_max).floor().detach()

    atlases_orig.append(build_frame_atlas(
        hi,
        (sc_fl[idx] + 16).cpu().numpy().astype(np.uint8),
        (qt_fl[idx] + 16).cpu().numpy().astype(np.uint8),
        (op_fl[idx] + 16).cpu().numpy().astype(np.uint8).squeeze(-1),
        (sh_fl[idx] + 16).cpu().numpy().astype(np.uint8),
        N,
    ))

    sc_r, qt_r, op_r, sh_r, _, _, _, _ = optimize_frame(f, ms, max_delta=2)
    atlases_d2.append(build_frame_atlas(
        hi,
        (sc_r[idx] + 16).cpu().numpy().astype(np.uint8),
        (qt_r[idx] + 16).cpu().numpy().astype(np.uint8),
        (op_r[idx] + 16).cpu().numpy().astype(np.uint8).squeeze(-1),
        (sh_r[idx] + 16).cpu().numpy().astype(np.uint8),
        N,
    ))

v_orig = xllvp9.encode(np.stack(atlases_orig), fps=30, keyframe_interval=30)
v_d2 = xllvp9.encode(np.stack(atlases_d2), fps=30, keyframe_interval=30)
print(f"  Floor only: {len(v_orig):,}")
print(f"  STE +/-2:   {len(v_d2):,}  ({len(v_d2)/len(v_orig)*100:.1f}%)")
