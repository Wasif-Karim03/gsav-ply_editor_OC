"""Rice/Golomb coding experiment for means_lo (per-frame random access).

Tests whether Rice coding beats zstd for K-snapped lo bytes.
All methods are per-frame — no temporal dependencies.

Pipeline: K-snap lo → spatial delta along Morton → zigzag → Rice (per-axis k)

Run: uv run python scripts/rice_coding_experiment.py
"""

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["PATH"] = (
    r"C:\Users\opsiclear\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.0.1-full_build\bin;" + os.environ.get("PATH", "")
)

import gsply
from gscodec.constants import LO_BASE, SCALE_16BIT
from gscodec.encoder.sorting.morton import MortonSortingStrategy
from gscodec.encoder.utils.helpers import log_transform
from gsply import GSTensor

device = "cuda:0"
PLY_DIR = r"D:\dymensium_soccer\plys"
MAX_FRAMES = 60
K_VALUES = [1, 27, 35, 51, 105]

cctx = zstd.ZstdCompressor(level=3)


def load_lo_bytes(ply_dir: str, max_frames: int, lo_snap_k: int) -> np.ndarray:
    """Load PLYs, Morton sort, quantize, K-snap, return lo bytes [T, N, 3] uint8."""
    ply_files = sorted(Path(ply_dir).glob("*.ply"))[:max_frames]
    print(f"  Loading {len(ply_files)} frames (K={lo_snap_k})...")

    tensors = []
    for f in ply_files:
        gs = gsply.plyread(str(f))
        t = GSTensor(
            means=torch.as_tensor(gs.means, device=device).float(),
            scales=torch.as_tensor(gs.scales, device=device).float(),
            quats=torch.as_tensor(gs.quats, device=device).float(),
            opacities=torch.as_tensor(gs.opacities, device=device).float(),
            sh0=torch.as_tensor(gs.sh0, device=device).float(),
            shN=None,
            masks=torch.ones(len(gs.means), dtype=torch.bool, device=device),
        )
        tensors.append(t)

    ng = max(t.means.shape[0] for t in tensors)
    ng = ((ng + 3) // 4) * 4
    all_ml = torch.cat([log_transform(t.means) for t in tensors])
    g_min, g_max = all_ml.min(0).values, all_ml.max(0).values
    scale = (g_max - g_min).clamp(min=1e-8)
    del all_ml

    sorter = MortonSortingStrategy()
    lo_frames = np.zeros((len(tensors), ng, 3), dtype=np.uint8)

    for i, t in enumerate(tensors):
        ml = log_transform(t.means)
        idx = sorter.sort_with_global_bbox(ml, g_min, g_max)
        ml_sorted = ml[idx]
        n = ml_sorted.shape[0]
        if n < ng:
            ml_sorted = torch.cat([ml_sorted, torch.zeros(ng - n, 3, device=device)])

        norm = ((ml_sorted - g_min) / scale).clamp(0, 1)
        q = (norm * SCALE_16BIT).floor().long()
        lo = (q % LO_BASE).clamp(0, 255)

        if lo_snap_k > 1:
            lo = ((lo + lo_snap_k // 2) // lo_snap_k * lo_snap_k) % 256

        lo_frames[i] = lo.cpu().numpy().astype(np.uint8)

    return lo_frames


# ── Per-Frame Transforms ─────────────────────────────────────────────────


def spatial_delta_u8(frame: np.ndarray) -> np.ndarray:
    """Morton-order spatial delta on uint8 lo. Returns int16 (to avoid wrap ambiguity)."""
    result = np.empty(frame.shape, dtype=np.int16)
    result[0] = frame[0].astype(np.int16)
    result[1:] = frame[1:].astype(np.int16) - frame[:-1].astype(np.int16)
    return result


def zigzag(x: np.ndarray) -> np.ndarray:
    """Signed → unsigned. Small magnitudes → small values."""
    s = x.astype(np.int16)
    return np.where(s >= 0, 2 * s, -2 * s - 1).astype(np.uint16)


def rice_bit_count(values: np.ndarray, k: int) -> int:
    quotients = values.astype(np.int64) >> k
    return int(np.sum(quotients)) + len(values) + k * len(values)


def optimal_rice_k(values: np.ndarray) -> tuple[int, int]:
    best_k, best_bits = 0, rice_bit_count(values, 0)
    for k in range(1, 16):
        bits = rice_bit_count(values, k)
        if bits < best_bits:
            best_bits = bits
            best_k = k
    return best_k, best_bits


# ── Experiments (ALL per-frame random access) ─────────────────────────────


def test_k_snap(lo_snap_k: int):
    lo_frames = load_lo_bytes(PLY_DIR, MAX_FRAMES, lo_snap_k)
    T, N, _ = lo_frames.shape
    raw_bytes = T * N * 3

    unique_vals = len(np.unique(lo_frames))
    entropy_bits = 0
    for v, c in zip(*np.unique(lo_frames, return_counts=True)):
        p = c / lo_frames.size
        if p > 0:
            entropy_bits -= p * np.log2(p)

    print(f"\n  K={lo_snap_k}: {T} frames, {N} Gaussians, {unique_vals} distinct values, "
          f"entropy={entropy_bits:.2f} bits/byte")
    print(f"  Raw: {raw_bytes:,} bytes ({raw_bytes / 1e6:.1f} MB)")

    # ── 1. Per-frame raw zstd (current baseline) ──
    zstd_total = sum(len(cctx.compress(lo_frames[t].tobytes())) for t in range(T))

    # ── 2. Per-frame axis-separated zstd ──
    axis_sep_total = 0
    for t in range(T):
        f = lo_frames[t]
        stream = np.concatenate([f[:, 0], f[:, 1], f[:, 2]]).tobytes()
        axis_sep_total += len(cctx.compress(stream))

    # ── 3. Per-frame spatial delta + zstd ──
    sd_zstd_total = 0
    for t in range(T):
        sd = spatial_delta_u8(lo_frames[t])
        sd_zstd_total += len(cctx.compress(sd.tobytes()))

    # ── 4. Per-frame spatial delta + zigzag + Rice (per-axis k) ──
    rice_total_bits = 0
    rice_overhead = 0
    k_stats = {ax: [] for ax in range(3)}
    for t in range(T):
        sd = spatial_delta_u8(lo_frames[t])
        zz = zigzag(sd)
        for axis in range(3):
            vals = zz[:, axis]
            k, bits = optimal_rice_k(vals)
            rice_total_bits += bits
            rice_overhead += 1
            k_stats[axis].append(k)
    rice_total_bytes = (rice_total_bits + 7) // 8 + rice_overhead

    # ── 5. Per-frame spatial delta + zigzag + byte-plane + zstd ──
    bp_total = 0
    for t in range(T):
        sd = spatial_delta_u8(lo_frames[t])
        zz = zigzag(sd)
        hi = (zz >> 8).astype(np.uint8)
        lo_b = (zz & 0xFF).astype(np.uint8)
        hi_stream = np.concatenate([hi[:, 0], hi[:, 1], hi[:, 2]]).tobytes()
        lo_stream = np.concatenate([lo_b[:, 0], lo_b[:, 1], lo_b[:, 2]]).tobytes()
        bp_total += len(cctx.compress(hi_stream + lo_stream))

    # ── 6. Shannon entropy lower bound ──
    shannon_bits = entropy_bits * raw_bytes
    shannon_bytes = int(shannon_bits / 8)

    # ── Results ──
    print(f"\n  {'Method (all per-frame random access)':<55} {'Bytes':>10} {'%raw':>7} {'vs zstd':>8}")
    print(f"  {'-'*82}")
    methods = [
        ("1. Per-frame raw zstd-3 (current)", zstd_total),
        ("2. Per-frame axis-separated + zstd-3", axis_sep_total),
        ("3. Per-frame spatial delta + zstd-3", sd_zstd_total),
        ("4. Per-frame spatial delta + zigzag + Rice", rice_total_bytes),
        ("5. Per-frame spatial delta + zigzag + byte-plane + zstd-3", bp_total),
        ("Shannon lower bound", shannon_bytes),
    ]
    for name, b in methods:
        pct_raw = b / raw_bytes * 100
        pct_zstd = b / zstd_total * 100 if zstd_total > 0 else 0
        print(f"  {name:<55} {b:>10,} {pct_raw:>6.1f}% {pct_zstd:>7.1f}%")

    # ── Rice k analysis ──
    print(f"\n  Rice k stats (K-snap={lo_snap_k}):")
    for axis, name in enumerate(["X", "Y", "Z"]):
        ks = k_stats[axis]
        print(f"    {name}: mean k={np.mean(ks):.1f}, mode k={max(set(ks), key=ks.count)}")

    # ── Spatial delta distribution (frame 0) ──
    sd = spatial_delta_u8(lo_frames[0])
    zz = zigzag(sd)
    print(f"\n  Spatial delta distribution (frame 0, K-snap={lo_snap_k}):")
    for axis, name in enumerate(["X", "Y", "Z"]):
        vals = zz[:, axis]
        zeros = (vals == 0).mean() * 100
        p50 = int(np.median(vals))
        p95 = int(np.percentile(vals, 95))
        p99 = int(np.percentile(vals, 99))
        print(f"    {name}: {zeros:.1f}% zeros, p50={p50}, p95={p95}, p99={p99}")


def main():
    print("Rice Coding Experiment for means_lo (per-frame random access)")
    print("=" * 80)
    for k in K_VALUES:
        test_k_snap(k)
    print(f"\n{'='*80}")
    print("Done. All methods preserve per-frame random access.")


if __name__ == "__main__":
    main()
