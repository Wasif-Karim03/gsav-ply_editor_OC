"""Test smallest-3 quaternion + temporal pixel snapping on VP9 atlas."""
import sys
sys.path.insert(0, "src")

import gsplat
import numpy as np
import torch
import xllvp9
from math import sqrt
from pathlib import Path

from scripts.finetune.io import load_ply_sequence, compute_global_ranges
from scripts.finetune.camera import SyntheticCameras
from scripts.finetune.renderer import render_batch
from scripts.finetune.losses import compute_psnr_per_view
from gscodec.constants import SCALE_16BIT, LO_BASE, SCALE_8BIT, MIN_VAL
from gscodec.encoder.utils.helpers import log_transform
from gscodec.encoder.sorting.morton import MortonSortingStrategy
from gscodec.encoder.video_writer import build_frame_atlas
from gscodec.encoder.quantization.strategies import quantize_means_16bit_split, quantize_anchor_8bit

device = "cuda:0"
frames, _ = load_ply_sequence(Path(r"D:\dymensium_soccer\plys"), device, max_frames=30)
mins, maxs = compute_global_ranges(frames)
scale = (maxs - mins).clamp(min=1e-8)
K = 51
N = frames[0].means.shape[0]
sorter = MortonSortingStrategy()
side = int(np.ceil(np.sqrt(N)))

# Global ranges
s_min = torch.stack([f.scales.amin(0) for f in frames]).amin(0)
s_max = torch.stack([f.scales.amax(0) for f in frames]).amax(0)
q_min_t = torch.tensor([-1.0] * 4, device=device)
q_max_t = torch.tensor([1.0] * 4, device=device)
o_min_t = torch.tensor([-6.0], device=device)
o_max_t = torch.tensor([12.0], device=device)
sh_min_t = torch.tensor([-2.0] * 3, device=device)
sh_max_t = torch.tensor([4.0] * 3, device=device)

# Smallest-3 quat ranges: [-1/sqrt(2), 1/sqrt(2)]
INV_SQRT2 = 1.0 / sqrt(2)
sq3_min = torch.tensor([-INV_SQRT2] * 3, device=device)
sq3_max = torch.tensor([INV_SQRT2] * 3, device=device)

# Camera setup for PSNR measurement
cameras = SyntheticCameras.from_scene(frames[0].means, n_cameras=12, height=1024, width=1024, device=device)
eval_vm, eval_Ks = cameras.get(torch.arange(8, 12, device=device))


def quat_to_smallest3(quats_np):
    """Convert [N, 4] quaternions to smallest-3 representation.

    Returns: (kept_3 [N, 3], drop_idx [N], drop_sign [N])
    """
    abs_q = np.abs(quats_np)
    drop_idx = abs_q.argmax(axis=1)  # which component to drop
    drop_sign = np.sign(quats_np[np.arange(len(quats_np)), drop_idx])

    # Ensure dropped component is positive (negate entire quat if needed)
    flip = drop_sign < 0
    quats_fixed = quats_np.copy()
    quats_fixed[flip] *= -1

    # Extract the 3 kept components
    kept = np.zeros((len(quats_np), 3), dtype=np.float32)
    for i in range(len(quats_np)):
        idx = drop_idx[i]
        kept[i] = np.delete(quats_fixed[i], idx)

    return kept, drop_idx.astype(np.uint8)


def smallest3_to_quat(kept, drop_idx):
    """Reconstruct [N, 4] quaternions from smallest-3."""
    N = len(kept)
    quats = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        idx = drop_idx[i]
        # Insert kept components
        j = 0
        for k in range(4):
            if k == idx:
                continue
            quats[i, k] = kept[i, j]
            j += 1
        # Reconstruct dropped component
        sum_sq = np.sum(kept[i] ** 2)
        quats[i, idx] = np.sqrt(max(0.0, 1.0 - sum_sq))
    return quats


def build_atlas_standard(f, sort_idx):
    """Build standard 3x5 atlas with 4-component quats."""
    ml = log_transform(f.means)[sort_idx]
    hi, lo = quantize_means_16bit_split(ml, mins, maxs, lo_snap_k=K)
    sc_q = quantize_anchor_8bit(f.scales[sort_idx], s_min, s_max)
    qt_q = quantize_anchor_8bit(f.quats[sort_idx].clamp(-1, 1), q_min_t, q_max_t)
    op_q = quantize_anchor_8bit(f.opacities[sort_idx].clamp(-6, 12), o_min_t, o_max_t).squeeze(-1)
    sh_q = quantize_anchor_8bit(f.sh0[sort_idx].clamp(-2, 4), sh_min_t, sh_max_t)
    return build_frame_atlas(hi, sc_q, qt_q, op_q, sh_q, N)


def build_atlas_smallest3(f, sort_idx):
    """Build 3x5 atlas with smallest-3 quats (frees 1 cell)."""
    ml = log_transform(f.means)[sort_idx]
    hi, lo = quantize_means_16bit_split(ml, mins, maxs, lo_snap_k=K)
    sc_q = quantize_anchor_8bit(f.scales[sort_idx], s_min, s_max)

    # Smallest-3 quaternion encoding
    quats_np = f.quats[sort_idx].cpu().numpy()
    kept, drop_idx = quat_to_smallest3(quats_np)
    kept_t = torch.from_numpy(kept).to(device)
    qt_q = quantize_anchor_8bit(kept_t.clamp(-INV_SQRT2, INV_SQRT2), sq3_min, sq3_max)

    op_q = quantize_anchor_8bit(f.opacities[sort_idx].clamp(-6, 12), o_min_t, o_max_t).squeeze(-1)
    sh_q = quantize_anchor_8bit(f.sh0[sort_idx].clamp(-2, 4), sh_min_t, sh_max_t)

    # Build atlas: 3 quat cells instead of 4, freed cell becomes padding
    # New layout Row 1: [scale.z][quat_a][quat_b][quat_c][padding]
    # Pack drop_idx (2 bits) into the freed cell
    h, w = 3 * side, 5 * side
    atlas = np.full((h, w), MIN_VAL, dtype=np.uint8)

    # Row 0: hi.x, hi.y, hi.z, scale.x, scale.y (unchanged)
    for col, data in enumerate([hi[:, 0], hi[:, 1], hi[:, 2], sc_q[:, 0], sc_q[:, 1]]):
        cell = np.full((side, side), MIN_VAL, dtype=np.uint8)
        cell.flat[:N] = data
        atlas[0:side, col * side:(col + 1) * side] = cell

    # Row 1: scale.z, quat_a, quat_b, quat_c, drop_idx (packed)
    row1_data = [sc_q[:, 2], qt_q[:, 0], qt_q[:, 1], qt_q[:, 2]]
    for col, data in enumerate(row1_data):
        cell = np.full((side, side), MIN_VAL, dtype=np.uint8)
        cell.flat[:N] = data
        atlas[side:2 * side, col * side:(col + 1) * side] = cell

    # Drop index: 2 bits per Gaussian, pack as (idx * 73 + 16) to spread in video-safe range
    # 4 possible values → map to {16, 89, 162, 235}
    drop_mapped = (drop_idx * 73 + MIN_VAL).astype(np.uint8)
    cell = np.full((side, side), MIN_VAL, dtype=np.uint8)
    cell.flat[:N] = drop_mapped
    atlas[side:2 * side, 4 * side:5 * side] = cell

    # Row 2: opacity, sh0.r, sh0.g, sh0.b, lo_packed
    row2_data = [op_q, sh_q[:, 0], sh_q[:, 1], sh_q[:, 2]]
    for col, data in enumerate(row2_data):
        cell = np.full((side, side), MIN_VAL, dtype=np.uint8)
        cell.flat[:N] = data
        atlas[2 * side:3 * side, col * side:(col + 1) * side] = cell

    # Lo packed in padding cell (row 2, col 4)
    lo_np = lo
    ix = lo_np[:, 0] // K
    iy = lo_np[:, 1] // K
    iz = lo_np[:, 2] // K
    combined = (ix * 36 + iy * 6 + iz).astype(np.uint8)
    cell = np.full((side, side), MIN_VAL, dtype=np.uint8)
    cell.flat[:N] = combined + MIN_VAL
    atlas[2 * side:3 * side, 4 * side:5 * side] = cell

    return atlas, drop_idx


def temporal_snap_atlas(atlases, threshold=1):
    """Snap atlas pixels within ±threshold of previous frame to exact match."""
    result = [atlases[0].copy()]
    for t in range(1, len(atlases)):
        curr = atlases[t].copy()
        prev = result[-1]
        diff = curr.astype(np.int16) - prev.astype(np.int16)
        close = np.abs(diff) <= threshold
        curr[close] = prev[close]
        result.append(curr)
    return result


def measure_psnr_smallest3(f, sort_idx, drop_idx, atlas):
    """Measure PSNR of smallest-3 quantized quats vs original."""
    # Extract quantized smallest-3 from atlas
    qt_cells = []
    for col in range(1, 4):  # quat_a, quat_b, quat_c in row 1
        cell = atlas[side:2 * side, col * side:(col + 1) * side]
        qt_cells.append(cell.flat[:N])
    qt_q = np.stack(qt_cells, axis=1)  # [N, 3] uint8

    # Dequantize smallest-3
    kept_dq = (qt_q.astype(np.float32) - MIN_VAL) / SCALE_8BIT
    sq3_range = 2 * INV_SQRT2
    kept_dq = kept_dq * sq3_range + (-INV_SQRT2)

    # Reconstruct full quaternion
    quats_recon = smallest3_to_quat(kept_dq, drop_idx)
    quats_t = torch.from_numpy(quats_recon).to(device)

    # Unsort
    unsort = torch.argsort(sort_idx)
    quats_unsorted = quats_t[unsort]

    with torch.no_grad():
        ref = render_batch(f.means, f.scales, f.quats, f.opacities, f.sh0,
                           eval_vm, eval_Ks, 1024, 1024)
        # Use smallest-3 reconstructed quats
        rendered, _, _ = gsplat.rasterization(
            means=f.means, quats=quats_unsorted,
            scales=f.scales.exp(), opacities=torch.sigmoid(f.opacities.squeeze(-1)),
            colors=0.28209479 * f.sh0 + 0.5,
            viewmats=eval_vm, Ks=eval_Ks, width=1024, height=1024,
            packed=True, render_mode="RGB",
        )
    return min(compute_psnr_per_view(rendered, ref))


# === Build atlases ===
print("Building 30-frame atlases...")
atlases_std = []
atlases_sq3 = []
all_sort_idx = []
all_drop_idx = []

for f in frames:
    ml = log_transform(f.means)
    sort_idx = sorter.sort_with_global_bbox(ml, mins, maxs)
    all_sort_idx.append(sort_idx)
    atlases_std.append(build_atlas_standard(f, sort_idx))
    atlas_sq3, drop_idx = build_atlas_smallest3(f, sort_idx)
    atlases_sq3.append(atlas_sq3)
    all_drop_idx.append(drop_idx)

# === Measure smallest-3 PSNR impact ===
print("\nSmallest-3 quaternion PSNR (first 10 frames):")
sq3_psnrs = []
for fi in range(10):
    p = measure_psnr_smallest3(frames[fi], all_sort_idx[fi], all_drop_idx[fi], atlases_sq3[fi])
    sq3_psnrs.append(p)
print(f"  min={min(sq3_psnrs):.1f}  mean={sum(sq3_psnrs)/10:.1f}")

# === VP9 encode comparison ===
print("\nVP9 encoding (30 frames):")

# Standard 4-quat atlas
v_std = xllvp9.encode(np.stack(atlases_std), fps=30, keyframe_interval=30)

# Smallest-3 atlas (same 3x5 size, but freed cell has drop_idx)
v_sq3 = xllvp9.encode(np.stack(atlases_sq3), fps=30, keyframe_interval=30)

# Temporal snapping on standard atlas
atlases_tsnap = temporal_snap_atlas(atlases_std, threshold=1)
v_tsnap = xllvp9.encode(np.stack(atlases_tsnap), fps=30, keyframe_interval=30)

# Temporal snapping on smallest-3 atlas
atlases_sq3_tsnap = temporal_snap_atlas(atlases_sq3, threshold=1)
v_sq3_tsnap = xllvp9.encode(np.stack(atlases_sq3_tsnap), fps=30, keyframe_interval=30)

# Temporal snap T=2 on standard
atlases_tsnap2 = temporal_snap_atlas(atlases_std, threshold=2)
v_tsnap2 = xllvp9.encode(np.stack(atlases_tsnap2), fps=30, keyframe_interval=30)

print(f"  {'Config':<35} {'IVF size':>10} {'vs std':>8}")
print(f"  {'-'*55}")
for label, v in [
    ("Standard (4 quats)", v_std),
    ("Smallest-3 (3 quats)", v_sq3),
    ("Standard + T-snap(1)", v_tsnap),
    ("Standard + T-snap(2)", v_tsnap2),
    ("Smallest-3 + T-snap(1)", v_sq3_tsnap),
]:
    pct = len(v) / len(v_std) * 100
    print(f"  {label:<35} {len(v):>10,} {pct:>7.1f}%")

# Measure PSNR impact of temporal snapping
print("\nTemporal snap PSNR impact (T=1, first 10 frames):")
tsnap_psnrs = []
for fi in range(10):
    f = frames[fi]
    sort_idx = all_sort_idx[fi]
    unsort = torch.argsort(sort_idx)

    # Extract attributes from T-snapped atlas
    atlas_ts = atlases_tsnap[fi]
    # Reconstruct means, scales, quats, opacity, sh0 from atlas
    # (Compare rendering with original vs T-snapped atlas values)

    # For simplicity: measure pixel-level atlas diff
    orig = atlases_std[fi]
    diff = np.abs(orig.astype(int) - atlas_ts.astype(int))
    changed = (diff > 0).sum()
    total = orig.size
    tsnap_psnrs.append(f"changed={changed}/{total} ({changed/total*100:.1f}%)")

print(f"  T=1 atlas pixel changes: {tsnap_psnrs[0]}")

# Extrapolate to 351 frames
v_std_351 = len(v_std) / 30 * 351
v_sq3_tsnap_351 = len(v_sq3_tsnap) / 30 * 351

print(f"\n351-frame estimates:")
print(f"  Standard VP9:              {v_std_351/1e6:.1f} MB")
print(f"  Smallest-3 + T-snap(1):    {v_sq3_tsnap_351/1e6:.1f} MB")
print(f"  Savings:                   {(v_std_351-v_sq3_tsnap_351)/1e6:.1f} MB ({(1-v_sq3_tsnap_351/v_std_351)*100:.1f}%)")


def _parse_ivf_local(ivf_data):
    """Simple IVF parser returning (data, [(offset, size)])."""
    import struct
    entries = []
    offset = 32
    while offset < len(ivf_data):
        if offset + 12 > len(ivf_data):
            break
        frame_size = struct.unpack("<I", ivf_data[offset:offset + 4])[0]
        offset += 12
        entries.append((offset, frame_size))
        offset += frame_size
    return ivf_data, entries
