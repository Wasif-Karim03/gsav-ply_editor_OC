"""Optuna TPE sweep for optimal compression parameters.

Optimizes: K value, n_refine, soft rounding config, for best PSNR at smallest size.
Multi-objective: maximize PSNR, minimize file size.
"""
import sys
sys.path.insert(0, "src")

import gsplat
import numpy as np
import optuna
import torch
from math import gcd
from pathlib import Path

from scripts.finetune.io import load_ply_sequence, compute_global_ranges
from scripts.finetune.camera import SyntheticCameras
from scripts.finetune.renderer import render_batch
from scripts.finetune.losses import compute_psnr_per_view
from gscodec.constants import SCALE_16BIT, LO_BASE, SCALE_8BIT
import zstandard as zstd

device = "cuda:0"
print("Loading frames...")
frames, _ = load_ply_sequence(Path(r"D:\dymensium_soccer\plys"), device, max_frames=20)
mins, maxs = compute_global_ranges(frames)
scale = (maxs - mins).clamp(min=1e-8)
N = frames[0].means.shape[0]

cameras = SyntheticCameras.from_scene(
    frames[0].means, n_cameras=48, height=1024, width=1024, device=device,
)
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

# Pre-render all references once
print("Pre-rendering references...")
refs_eval = []
refs_train = []
for f in frames:
    refs_eval.append(
        render_batch(f.means, f.scales, f.quats, f.opacities, f.sh0,
                     eval_vm, eval_Ks, 1024, 1024).detach()
    )
    refs_train.append(
        render_batch(f.means, f.scales, f.quats, f.opacities, f.sh0,
                     train_vm, Ks_512, 512, 512).detach()
    )

# Valid coprime K values
COPRIME_KS = [k for k in range(11, 128) if gcd(k, 256) == 1]


def qfloor(data, dmin, dmax):
    s = torch.where(dmax > dmin, dmax - dmin, torch.ones_like(dmin))
    return ((data - dmin) / s).clamp(0, 1) * SCALE_8BIT


def dq(qval, dmin, dmax):
    s = torch.where(dmax > dmin, dmax - dmin, torch.ones_like(dmin))
    return (qval / SCALE_8BIT) * s + dmin


def rq(ms, sc, qt, op, sh, vm, ks, w, h):
    r, _, _ = gsplat.rasterization(
        means=ms, quats=dq(qt, q_min, q_max),
        scales=dq(sc, s_min, s_max).exp(),
        opacities=torch.sigmoid(dq(op, o_min, o_max).squeeze(-1)),
        colors=0.28209479 * dq(sh, sh_min, sh_max) + 0.5,
        viewmats=vm, Ks=ks, width=w, height=h, packed=True, render_mode="RGB",
    )
    return r


def evaluate(K, n_refine, soft_steps, soft_lr, max_delta, n_attr_corr, n_eval_frames=10):
    """Evaluate a parameter config. Returns (min_psnr, estimated_total_MB)."""
    cctx = zstd.ZstdCompressor(level=3)
    all_psnr = []
    total_lo_compressed = 0

    for fi in range(n_eval_frames):
        f = frames[fi]
        ml = torch.sign(f.means) * torch.log1p(torch.abs(f.means))
        n = ((ml - mins) / scale).clamp(0, 1)
        q = (n * SCALE_16BIT).floor().long()
        hi = q // LO_BASE
        lo_full = q % LO_BASE
        lo_snapped = ((lo_full + K // 2) // K * K) % 256

        mn_snap = (hi * LO_BASE + lo_snapped).float() / SCALE_16BIT * scale + mins
        ms_snap = (torch.sign(mn_snap) * (torch.exp(torch.abs(mn_snap)) - 1)).detach()

        sc_fl = qfloor(f.scales, s_min, s_max).floor().detach()
        qt_fl = qfloor(f.quats.clamp(-1, 1), q_min, q_max).floor().detach()
        op_fl = qfloor(f.opacities.clamp(-6, 12), o_min, o_max).floor().detach()
        sh_fl = qfloor(f.sh0.clamp(-2, 4), sh_min, sh_max).floor().detach()

        # Lo refinement
        if n_refine > 0:
            mp = ms_snap.clone().requires_grad_(True)
            img_s = rq(mp, sc_fl, qt_fl, op_fl, sh_fl, eval_vm, eval_Ks, 1024, 1024)
            (img_s - refs_eval[fi]).pow(2).sum().backward()
            sens = mp.grad.norm(dim=-1).detach()
            top_lo = sens.argsort(descending=True)[:n_refine]

            lo_hybrid = lo_snapped.clone()
            lo_hybrid[top_lo] = lo_full[top_lo]
            mn_h = (hi * LO_BASE + lo_hybrid).float() / SCALE_16BIT * scale + mins
            ms_use = (torch.sign(mn_h) * (torch.exp(torch.abs(mn_h)) - 1)).detach()
        else:
            ms_use = ms_snap

        # Soft rounding / STE with max_delta
        if soft_steps > 0:
            md = float(max_delta)
            if md <= 1.0:
                # Sigmoid soft rounding (±1)
                sl = torch.nn.Parameter(torch.zeros_like(sc_fl))
                ql = torch.nn.Parameter(torch.zeros_like(qt_fl))
                ol = torch.nn.Parameter(torch.zeros_like(op_fl))
                shl = torch.nn.Parameter(torch.zeros_like(sh_fl))
                opt = torch.optim.Adam([sl, ql, ol, shl], lr=soft_lr)
                for step in range(soft_steps):
                    temp = 1.0 + step * 0.5
                    idx = torch.randperm(44, device=device)[:6]
                    img = rq(ms_use, sc_fl + torch.sigmoid(sl * temp),
                             qt_fl + torch.sigmoid(ql * temp),
                             op_fl + torch.sigmoid(ol * temp),
                             sh_fl + torch.sigmoid(shl * temp),
                             train_vm[idx], Ks_512[idx], 512, 512)
                    opt.zero_grad()
                    (img - refs_train[fi][idx]).abs().mean().backward()
                    opt.step()
                with torch.no_grad():
                    sc_r = sc_fl + (torch.sigmoid(sl * 100) > 0.5).float()
                    qt_r = qt_fl + (torch.sigmoid(ql * 100) > 0.5).float()
                    op_r = op_fl + (torch.sigmoid(ol * 100) > 0.5).float()
                    sh_r = sh_fl + (torch.sigmoid(shl * 100) > 0.5).float()
            else:
                # STE with ±max_delta
                sc_d = torch.nn.Parameter(torch.zeros_like(sc_fl))
                qt_d = torch.nn.Parameter(torch.zeros_like(qt_fl))
                op_d = torch.nn.Parameter(torch.zeros_like(op_fl))
                sh_d = torch.nn.Parameter(torch.zeros_like(sh_fl))
                opt = torch.optim.Adam([sc_d, qt_d, op_d, sh_d], lr=soft_lr * 0.5)
                for step in range(soft_steps):
                    sc_q = sc_fl + sc_d.clamp(-md, md).round()
                    qt_q = qt_fl + qt_d.clamp(-md, md).round()
                    op_q = op_fl + op_d.clamp(-md, md).round()
                    sh_q = sh_fl + sh_d.clamp(-md, md).round()
                    sc_ste = sc_fl + sc_d.clamp(-md, md) + (sc_q - sc_fl - sc_d.clamp(-md, md)).detach()
                    qt_ste = qt_fl + qt_d.clamp(-md, md) + (qt_q - qt_fl - qt_d.clamp(-md, md)).detach()
                    op_ste = op_fl + op_d.clamp(-md, md) + (op_q - op_fl - op_d.clamp(-md, md)).detach()
                    sh_ste = sh_fl + sh_d.clamp(-md, md) + (sh_q - sh_fl - sh_d.clamp(-md, md)).detach()
                    idx = torch.randperm(44, device=device)[:6]
                    img = rq(ms_use, sc_ste, qt_ste, op_ste, sh_ste,
                             train_vm[idx], Ks_512[idx], 512, 512)
                    opt.zero_grad()
                    (img - refs_train[fi][idx]).abs().mean().backward()
                    opt.step()
                with torch.no_grad():
                    sc_r = (sc_fl + sc_d.clamp(-md, md).round()).clamp(0, SCALE_8BIT)
                    qt_r = (qt_fl + qt_d.clamp(-md, md).round()).clamp(0, SCALE_8BIT)
                    op_r = (op_fl + op_d.clamp(-md, md).round()).clamp(0, SCALE_8BIT)
                    sh_r = (sh_fl + sh_d.clamp(-md, md).round()).clamp(0, SCALE_8BIT)
        else:
            sc_r = sc_fl
            qt_r = qt_fl
            op_r = op_fl
            sh_r = sh_fl

        # Attribute correction table for top-N Gaussians (±2 STE on top of soft round)
        if n_attr_corr > 0 and soft_steps > 0:
            mp2 = ms_use.clone().requires_grad_(True)
            img_g = rq(mp2, sc_r, qt_r, op_r, sh_r, eval_vm, eval_Ks, 1024, 1024)
            (img_g - refs_eval[fi]).pow(2).sum().backward()
            attr_sens = mp2.grad.norm(dim=-1).detach()
            top_attr = attr_sens.argsort(descending=True)[:n_attr_corr]

            asc = torch.nn.Parameter(torch.zeros(n_attr_corr, 3, device=device))
            aqt = torch.nn.Parameter(torch.zeros(n_attr_corr, 4, device=device))
            aop = torch.nn.Parameter(torch.zeros(n_attr_corr, 1, device=device))
            ash = torch.nn.Parameter(torch.zeros(n_attr_corr, 3, device=device))
            aopt = torch.optim.Adam([asc, aqt, aop, ash], lr=0.05)
            for step in range(100):
                sct = sc_r.clone(); qtt = qt_r.clone(); opt2 = op_r.clone(); sht = sh_r.clone()
                sct[top_attr] += asc.clamp(-2, 2).round() + (asc.clamp(-2, 2) - asc.clamp(-2, 2).round()).detach()
                qtt[top_attr] += aqt.clamp(-2, 2).round() + (aqt.clamp(-2, 2) - aqt.clamp(-2, 2).round()).detach()
                opt2[top_attr] += aop.clamp(-2, 2).round() + (aop.clamp(-2, 2) - aop.clamp(-2, 2).round()).detach()
                sht[top_attr] += ash.clamp(-2, 2).round() + (ash.clamp(-2, 2) - ash.clamp(-2, 2).round()).detach()
                jdx = torch.randperm(44, device=device)[:6]
                img = rq(ms_use, sct, qtt, opt2, sht, train_vm[jdx], Ks_512[jdx], 512, 512)
                aopt.zero_grad()
                (img - refs_train[fi][jdx]).abs().mean().backward()
                aopt.step()
            with torch.no_grad():
                sc_r[top_attr] = (sc_r[top_attr] + asc.clamp(-2, 2).round()).clamp(0, SCALE_8BIT)
                qt_r[top_attr] = (qt_r[top_attr] + aqt.clamp(-2, 2).round()).clamp(0, SCALE_8BIT)
                op_r[top_attr] = (op_r[top_attr] + aop.clamp(-2, 2).round()).clamp(0, SCALE_8BIT)
                sh_r[top_attr] = (sh_r[top_attr] + ash.clamp(-2, 2).round()).clamp(0, SCALE_8BIT)

        with torch.no_grad():
            img_eval = rq(ms_use, sc_r, qt_r, op_r, sh_r,
                          eval_vm, eval_Ks, 1024, 1024)
        all_psnr.append(min(compute_psnr_per_view(img_eval, refs_eval[fi])))

        lo_np = lo_snapped.cpu().numpy().astype(np.uint8)
        total_lo_compressed += len(cctx.compress(lo_np.tobytes()))

    min_psnr = min(all_psnr)
    mean_psnr = sum(all_psnr) / len(all_psnr)

    video_mb = 13.3
    lo_mb = total_lo_compressed / n_eval_frames * 351 / 1e6
    refine_mb = n_refine * 5 * 351 / 1e6 if n_refine > 0 else 0
    # Attr correction: 2-byte index + 11 bytes (delta per channel) = 13 bytes per Gaussian
    attr_corr_mb = n_attr_corr * 13 * 351 / 1e6 if n_attr_corr > 0 else 0
    total_mb = video_mb + lo_mb + refine_mb + attr_corr_mb

    return min_psnr, mean_psnr, total_mb


def objective(trial):
    K = trial.suggest_categorical("K", COPRIME_KS)
    n_refine = trial.suggest_int("n_refine", 0, 1500, step=50)
    soft_steps = trial.suggest_categorical("soft_steps", [0, 50, 100, 150, 200])
    soft_lr = trial.suggest_float("soft_lr", 0.01, 0.3, log=True)
    max_delta = trial.suggest_categorical("max_delta", [1, 2, 3])
    n_attr_corr = trial.suggest_int("n_attr_corr", 0, 500, step=50)

    min_psnr, mean_psnr, total_mb = evaluate(
        K, n_refine, soft_steps, soft_lr, max_delta, n_attr_corr, n_eval_frames=10,
    )

    trial.set_user_attr("mean_psnr", mean_psnr)
    return min_psnr, total_mb


print("Starting Optuna TPE sweep (multi-objective: max PSNR, min size)...")
study = optuna.create_study(
    directions=["maximize", "minimize"],
    sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
)

# Seed with known good configs
study.enqueue_trial({"K": 51, "n_refine": 0, "soft_steps": 100, "soft_lr": 0.1, "max_delta": 1, "n_attr_corr": 0})
study.enqueue_trial({"K": 51, "n_refine": 250, "soft_steps": 100, "soft_lr": 0.1, "max_delta": 1, "n_attr_corr": 0})
study.enqueue_trial({"K": 105, "n_refine": 650, "soft_steps": 150, "soft_lr": 0.257, "max_delta": 1, "n_attr_corr": 0})
study.enqueue_trial({"K": 109, "n_refine": 800, "soft_steps": 100, "soft_lr": 0.05, "max_delta": 2, "n_attr_corr": 0})
study.enqueue_trial({"K": 59, "n_refine": 100, "soft_steps": 100, "soft_lr": 0.03, "max_delta": 2, "n_attr_corr": 100})

study.optimize(objective, n_trials=60, show_progress_bar=True)

# Print Pareto front
print("\n=== Pareto Front (best tradeoffs) ===")
header = f"{'K':>4} {'n_ref':>6} {'soft':>5} {'lr':>6} {'md':>3} {'n_ac':>5} {'min_PSNR':>9} {'mean':>6} {'MB':>6}"
print(header)
print("-" * len(header))

trials = study.best_trials
trials.sort(key=lambda t: t.values[1])
for t in trials:
    p = t.params
    mp = t.user_attrs["mean_psnr"]
    print(f"{p['K']:>4} {p['n_refine']:>6} {p['soft_steps']:>5} {p['soft_lr']:>6.3f} {p['max_delta']:>3} {p['n_attr_corr']:>5} {t.values[0]:>9.1f} {mp:>6.1f} {t.values[1]:>6.1f}")
