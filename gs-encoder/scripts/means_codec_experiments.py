"""Means codec structured prediction experiments (per-frame random access).

Hard constraint: every frame must be independently decodable (no temporal deps).
All compression comes from WITHIN-FRAME techniques:
  1. Spatial delta along Morton index
  2. Zigzag encoding
  3. Byte-plane splitting (hi/lo separation)
  4. Axis separation
  5. Entropy coding (zstd levels, Rice)

Measures compressed size for each technique in isolation and combined.
Run: uv run python scripts/means_codec_experiments.py
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
from gscodec.encoder.sorting.morton import MortonSortingStrategy
from gscodec.encoder.utils.helpers import log_transform
from gsply import GSTensor

device = "cuda:0"
PLY_DIR = r"D:\dymensium_soccer\plys"
MAX_FRAMES = 351
ZSTD_LEVEL = 3

cctx3 = zstd.ZstdCompressor(level=3)
cctx9 = zstd.ZstdCompressor(level=9)
cctx19 = zstd.ZstdCompressor(level=19)


# ── Data Loading ──────────────────────────────────────────────────────────


def load_and_quantize(ply_dir: str, max_frames: int) -> tuple[np.ndarray, int, int]:
    """Load PLYs, Morton sort per frame, quantize means to uint16 [0, 65535].

    Returns:
        quantized: [T, N, 3] uint16 Morton-sorted quantized means
        n_gaussians: int
        side: int
    """
    ply_files = sorted(Path(ply_dir).glob("*.ply"))[:max_frames]
    print(f"Loading {len(ply_files)} frames...")

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

    n_gaussians = max(t.means.shape[0] for t in tensors)
    n_gaussians = ((n_gaussians + 3) // 4) * 4
    side = math.ceil(math.sqrt(n_gaussians))
    if side % 2 == 1:
        side += 1

    all_means_log = torch.cat([log_transform(t.means) for t in tensors])
    g_min = all_means_log.min(0).values
    g_max = all_means_log.max(0).values
    scale = (g_max - g_min).clamp(min=1e-8)
    del all_means_log

    sorter = MortonSortingStrategy()
    quantized = np.zeros((len(tensors), n_gaussians, 3), dtype=np.uint16)

    for i, t in enumerate(tensors):
        ml = log_transform(t.means)
        idx = sorter.sort_with_global_bbox(ml, g_min, g_max)
        ml_sorted = ml[idx]

        n = ml_sorted.shape[0]
        if n < n_gaussians:
            ml_sorted = torch.cat([ml_sorted, torch.zeros(n_gaussians - n, 3, device=device)])

        norm = ((ml_sorted - g_min) / scale).clamp(0, 1)
        q = (norm * 65535).round().long().clamp(0, 65535)
        quantized[i] = q.cpu().numpy().astype(np.uint16)

    print(f"Loaded {len(tensors)} frames, {n_gaussians} Gaussians, side={side}")
    return quantized, n_gaussians, side


# ── Per-Frame Transforms ─────────────────────────────────────────────────


def spatial_delta(frame: np.ndarray) -> np.ndarray:
    """Morton-order spatial delta: r[0]=raw, r[i]=q[i]-q[i-1]. Returns int16."""
    result = np.empty(frame.shape, dtype=np.int16)
    result[0] = frame[0].astype(np.int16)
    result[1:] = frame[1:].astype(np.int16) - frame[:-1].astype(np.int16)
    return result


def zigzag_16(x: np.ndarray) -> np.ndarray:
    """Signed int16 → unsigned uint16. Small magnitudes → small values."""
    s = x.astype(np.int16).view(np.int16)
    return np.where(s >= 0, 2 * s, -2 * s - 1).astype(np.uint16)


def byte_plane_split(arr_u16: np.ndarray) -> tuple[bytes, bytes]:
    """Split uint16 [N, 3] into hi and lo byte streams, axis-interleaved."""
    hi = (arr_u16 >> 8).astype(np.uint8)
    lo = (arr_u16 & 0xFF).astype(np.uint8)
    hi_stream = np.concatenate([hi[:, 0], hi[:, 1], hi[:, 2]]).tobytes()
    lo_stream = np.concatenate([lo[:, 0], lo[:, 1], lo[:, 2]]).tobytes()
    return hi_stream, lo_stream


def axis_separate(frame: np.ndarray) -> bytes:
    """Reorder [N,3] to axis-contiguous: all X, then Y, then Z."""
    return np.concatenate([frame[:, 0], frame[:, 1], frame[:, 2]]).tobytes()


# ── Compression Helpers ───────────────────────────────────────────────────


_zstd_cache: dict[int, zstd.ZstdCompressor] = {}


def zstd_frame(data: bytes, level: int = ZSTD_LEVEL) -> int:
    """Compress one frame, return compressed size."""
    if level not in _zstd_cache:
        _zstd_cache[level] = zstd.ZstdCompressor(level=level)
    return len(_zstd_cache[level].compress(data))


def rice_bit_count(values: np.ndarray, k: int) -> int:
    """Total Rice-coded bits for unsigned integer array."""
    quotients = values.astype(np.int64) >> k
    return int(np.sum(quotients)) + len(values) + k * len(values)


def optimal_rice_k(values: np.ndarray) -> tuple[int, int]:
    """Find k minimizing total Rice bits."""
    best_k, best_bits = 0, rice_bit_count(values, 0)
    for k in range(1, 16):
        bits = rice_bit_count(values, k)
        if bits < best_bits:
            best_bits = bits
            best_k = k
    return best_k, best_bits


# ── Experiments (all per-frame, random-access safe) ───────────────────────


def exp_raw_zstd(quantized: np.ndarray, level: int = 3) -> int:
    """A: Raw uint16 per-frame zstd."""
    return sum(zstd_frame(quantized[t].tobytes(), level) for t in range(len(quantized)))


def exp_axis_separated_zstd(quantized: np.ndarray, level: int = 3) -> int:
    """B: Axis-separated per-frame zstd (all X, then Y, then Z)."""
    return sum(zstd_frame(axis_separate(quantized[t]), level) for t in range(len(quantized)))


def exp_spatial_delta_zstd(quantized: np.ndarray, level: int = 3) -> int:
    """C: Spatial delta + per-frame zstd."""
    total = 0
    for t in range(len(quantized)):
        sd = spatial_delta(quantized[t])
        total += zstd_frame(sd.tobytes(), level)
    return total


def exp_spatial_delta_axis_sep_zstd(quantized: np.ndarray, level: int = 3) -> int:
    """D: Spatial delta + axis-separated + per-frame zstd."""
    total = 0
    for t in range(len(quantized)):
        sd = spatial_delta(quantized[t])
        total += zstd_frame(axis_separate(sd.view(np.uint16)), level)
    return total


def exp_spatial_zigzag_byteplane_zstd(quantized: np.ndarray, level: int = 3) -> int:
    """E: Spatial delta + zigzag + byte-plane split + per-frame zstd."""
    total = 0
    for t in range(len(quantized)):
        sd = spatial_delta(quantized[t])
        zz = zigzag_16(sd)
        hi, lo = byte_plane_split(zz)
        total += zstd_frame(hi, level) + zstd_frame(lo, level)
    return total


def exp_spatial_zigzag_byteplane_combined_zstd(quantized: np.ndarray, level: int = 3) -> int:
    """F: Same as E but hi+lo concatenated into one zstd frame (less framing overhead)."""
    total = 0
    for t in range(len(quantized)):
        sd = spatial_delta(quantized[t])
        zz = zigzag_16(sd)
        hi, lo = byte_plane_split(zz)
        total += zstd_frame(hi + lo, level)
    return total


def exp_spatial_zigzag_rice(quantized: np.ndarray) -> tuple[int, dict]:
    """G: Spatial delta + zigzag + Rice coding (per-axis auto-k)."""
    total_bits = 0
    overhead = 0  # k params: 1 byte per axis per frame
    k_hist = {ax: [] for ax in range(3)}

    for t in range(len(quantized)):
        sd = spatial_delta(quantized[t])
        zz = zigzag_16(sd)
        for axis in range(3):
            vals = zz[:, axis]
            k, bits = optimal_rice_k(vals)
            total_bits += bits
            overhead += 1
            k_hist[axis].append(k)

    total_bytes = (total_bits + 7) // 8 + overhead
    avg_k = {ax: np.mean(ks) for ax, ks in k_hist.items()}
    return total_bytes, avg_k


def exp_reference_temporal_chunk(quantized: np.ndarray, chunk_size: int = 30) -> int:
    """REF: Chunk temporal delta + zstd (NOT random-access — for comparison only)."""
    total = 0
    T = len(quantized)
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        parts = [quantized[start].tobytes()]
        for t in range(start + 1, end):
            delta = quantized[t].astype(np.int16) - quantized[t - 1].astype(np.int16)
            parts.append(delta.astype(np.int16).view(np.uint8).tobytes())
        total += zstd_frame(b"".join(parts))
    return total


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    quantized, ng, side = load_and_quantize(PLY_DIR, MAX_FRAMES)
    T = quantized.shape[0]
    raw_bytes = T * ng * 3 * 2
    print(f"\nRaw uint16: {raw_bytes:,} bytes ({raw_bytes / 1e6:.1f} MB)")
    print(f"Frames: {T}, Gaussians: {ng}")
    print("=" * 80)
    print("All methods are PER-FRAME (random-access safe) unless marked [NO-RA].")
    print("=" * 80)

    results = []

    def run(name, fn, *args, **kwargs):
        t0 = time.time()
        result = fn(quantized, *args, **kwargs)
        if isinstance(result, tuple):
            b, extra = result
        else:
            b = result
            extra = None
        elapsed = time.time() - t0
        pct = b / raw_bytes * 100
        per_frame_kb = b / T / 1024
        print(f"  {name:<55} {b:>10,} ({pct:>5.1f}%) {per_frame_kb:>6.1f} KB/fr  [{elapsed:.1f}s]")
        if extra:
            print(f"    Rice avg k: X={extra[0]:.1f} Y={extra[1]:.1f} Z={extra[2]:.1f}")
        results.append({"name": name, "bytes": b})

    print(f"\n{'Method':<55} {'Bytes':>10} {'%raw':>7} {'KB/fr':>9}")
    print(f"{'-'*90}")

    # ── Per-frame methods (random access) ──
    run("A. Raw uint16 + zstd-3", exp_raw_zstd, level=3)
    run("B. Axis-separated + zstd-3", exp_axis_separated_zstd, level=3)
    run("C. Spatial delta + zstd-3", exp_spatial_delta_zstd, level=3)
    run("D. Spatial delta + axis-sep + zstd-3", exp_spatial_delta_axis_sep_zstd, level=3)
    run("E. Spatial delta + zigzag + byte-plane + zstd-3", exp_spatial_zigzag_byteplane_zstd, level=3)
    run("F. E combined (hi+lo one blob) + zstd-3", exp_spatial_zigzag_byteplane_combined_zstd, level=3)
    run("G. Spatial delta + zigzag + Rice (per-axis k)", exp_spatial_zigzag_rice)

    # ── zstd level sweep on best per-frame method ──
    print(f"\n  zstd level sweep on best per-frame pipeline (E):")
    for lvl in [1, 3, 6, 9, 12, 19]:
        run(f"  E @ zstd-{lvl}", exp_spatial_zigzag_byteplane_zstd, level=lvl)

    # ── Reference: chunk temporal (NO random access) ──
    print(f"\n  Reference (NO random access — for comparison only):")
    run("[NO-RA] Chunk temporal delta + zstd-3", exp_reference_temporal_chunk)

    # ── Per-frame entropy analysis ──
    print(f"\n{'='*80}")
    print("Per-frame entropy analysis (frame 0 and frame 100):")
    for frame_idx in [0, 100]:
        frame = quantized[frame_idx]
        sd = spatial_delta(frame)
        zz = zigzag_16(sd)
        print(f"\n  Frame {frame_idx}:")
        for axis, name in enumerate(["X", "Y", "Z"]):
            vals = zz[:, axis]
            zeros = (vals == 0).mean() * 100
            p50 = np.median(vals)
            p95 = np.percentile(vals, 95)
            p99 = np.percentile(vals, 99)
            hi_zeros = ((vals >> 8) == 0).mean() * 100
            k, bits = optimal_rice_k(vals)
            bpv = bits / len(vals)
            print(f"    {name}: zeros={zeros:.1f}%, hi=0: {hi_zeros:.1f}%, "
                  f"p50={p50:.0f} p95={p95:.0f} p99={p99:.0f}, "
                  f"Rice k={k} ({bpv:.2f} bits/val)")


if __name__ == "__main__":
    main()
