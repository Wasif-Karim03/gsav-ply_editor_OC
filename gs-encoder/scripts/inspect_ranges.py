"""Diagnostic script to inspect PLY attribute distributions and quantization waste.

Identifies outliers that pollute global quantization ranges, especially scales/quats.
"""

import re
import sys
from pathlib import Path

import gsply
import numpy as np
import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gscodec.constants import OPACITY_CLIP, QUATS_CLIP, SCALE_8BIT, SH0_CLIP
from gscodec.encoder.utils.helpers import log_transform

PERCENTILES = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]


def load_ply_sequence(input_dir: Path, max_frames: int = 0) -> list[dict[str, np.ndarray]]:
    """Load PLY files, return raw numpy arrays."""
    ply_files = sorted(
        input_dir.glob("*.ply"),
        key=lambda x: int(re.findall(r"\d+", x.stem)[-1])
        if re.findall(r"\d+", x.stem)
        else 0,
    )
    if not ply_files:
        raise ValueError(f"No PLY files found in {input_dir}")

    if max_frames > 0:
        ply_files = ply_files[:max_frames]

    frames = []
    for ply_path in ply_files:
        gs = gsply.plyread(ply_path)
        opacities = gs.opacities
        if opacities.ndim == 1:
            opacities = opacities.reshape(-1, 1)
        frames.append({
            "means": gs.means,        # [N, 3]
            "scales": gs.scales,       # [N, 3] (log-space from PLY)
            "quats": gs.quats,         # [N, 4]
            "opacities": opacities,    # [N, 1]
            "sh0": gs.sh0,             # [N, 3]
        })
    return frames


def print_distribution(name: str, data: np.ndarray, clip: tuple[float, float] | None = None):
    """Print distribution stats for a single attribute."""
    flat = data.flatten()
    pcts = np.percentile(flat, PERCENTILES)

    print(f"\n  {name}:")
    print(f"    shape: {data.shape}")
    print(f"    min={flat.min():.6f}  max={flat.max():.6f}  range={flat.max() - flat.min():.6f}")
    print(f"    mean={flat.mean():.6f}  std={flat.std():.6f}")

    pct_str = "    percentiles: " + "  ".join(
        f"p{p}={v:.4f}" for p, v in zip(PERCENTILES, pcts)
    )
    print(pct_str)

    if clip:
        below = (flat < clip[0]).sum()
        above = (flat > clip[1]).sum()
        total = len(flat)
        print(f"    clip [{clip[0]}, {clip[1]}]: {below} below ({below/total*100:.3f}%), "
              f"{above} above ({above/total*100:.3f}%)")

    # Per-channel stats if multi-channel
    if data.ndim == 2 and data.shape[1] > 1:
        for c in range(data.shape[1]):
            col = data[:, c]
            print(f"    ch{c}: min={col.min():.6f}  max={col.max():.6f}  "
                  f"range={col.max() - col.min():.6f}  std={col.std():.6f}")


def quantization_waste_report(
    name: str,
    global_min: np.ndarray,
    global_max: np.ndarray,
    all_values: np.ndarray,
    n_levels: int = 219,
):
    """Show how much quantization precision is wasted by outliers."""
    print(f"\n  === Quantization Waste: {name} ===")
    global_range = global_max - global_min

    if all_values.ndim == 1:
        all_values = all_values.reshape(-1, 1)

    n_channels = all_values.shape[1]
    for c in range(n_channels):
        ch = all_values[:, c]
        ch_range = global_range[c] if hasattr(global_range, '__len__') else float(global_range)
        step = ch_range / n_levels if ch_range > 0 else 0

        # What range would we need without outliers?
        p01 = np.percentile(ch, 0.1)
        p999 = np.percentile(ch, 99.9)
        tight_range = p999 - p01
        tight_step = tight_range / n_levels if tight_range > 0 else 0

        waste = (1 - tight_range / ch_range) * 100 if ch_range > 0 else 0

        print(f"    ch{c}: global_range={ch_range:.6f}  step={step:.6f}")
        print(f"          p0.1-p99.9 range={tight_range:.6f}  step={tight_step:.6f}")
        print(f"          waste={waste:.1f}%  (step {step/tight_step:.2f}x coarser than needed)"
              if tight_step > 0 else f"          waste={waste:.1f}%")


def inspect_sequence(input_dir: Path, max_frames: int = 0):
    """Full inspection of a PLY sequence."""
    print(f"\n{'='*80}")
    print(f"INSPECTING: {input_dir}")
    print(f"{'='*80}")

    frames = load_ply_sequence(input_dir, max_frames)
    n_frames = len(frames)
    n_gaussians = frames[0]["means"].shape[0]
    print(f"\nFrames: {n_frames}  Gaussians: {n_gaussians}")

    # ---- Collect all values across frames ----
    all_means_log = []
    all_scales = []
    all_quats = []
    all_opacities = []
    all_sh0 = []

    for f in frames:
        means_t = torch.from_numpy(f["means"]).float()
        all_means_log.append(log_transform(means_t).numpy())
        all_scales.append(f["scales"])
        all_quats.append(f["quats"])
        all_opacities.append(f["opacities"])
        all_sh0.append(f["sh0"])

    all_means_log = np.concatenate(all_means_log, axis=0)
    all_scales = np.concatenate(all_scales, axis=0)
    all_quats = np.concatenate(all_quats, axis=0)
    all_opacities = np.concatenate(all_opacities, axis=0)
    all_sh0 = np.concatenate(all_sh0, axis=0)

    # ---- Distribution per attribute ----
    print("\n" + "-"*60)
    print("ATTRIBUTE DISTRIBUTIONS (across all frames)")
    print("-"*60)

    print_distribution("means (log-transformed)", all_means_log)
    print_distribution("scales (log-space, RAW - NO CLIP)", all_scales)
    print_distribution("quats", all_quats, clip=QUATS_CLIP)
    print_distribution("opacities", all_opacities, clip=OPACITY_CLIP)
    print_distribution("sh0", all_sh0, clip=SH0_CLIP)

    # ---- Compute global ranges (mimics _compute_global_ranges) ----
    print("\n" + "-"*60)
    print("GLOBAL RANGES (as encoder would compute them)")
    print("-"*60)

    means_min = all_means_log.min(axis=0)
    means_max = all_means_log.max(axis=0)
    scales_min = all_scales.min(axis=0)
    scales_max = all_scales.max(axis=0)
    quats_min = np.clip(all_quats.min(axis=0), QUATS_CLIP[0], None)
    quats_max = np.clip(all_quats.max(axis=0), None, QUATS_CLIP[1])
    opacity_min = max(all_opacities.min(), OPACITY_CLIP[0])
    opacity_max = min(all_opacities.max(), OPACITY_CLIP[1])
    sh0_min = np.clip(all_sh0.min(axis=0), SH0_CLIP[0], None)
    sh0_max = np.clip(all_sh0.max(axis=0), None, SH0_CLIP[1])

    print(f"\n  means_log:  min={means_min}  max={means_max}")
    print(f"  scales:     min={scales_min}  max={scales_max}")
    print(f"  quats:      min={quats_min}  max={quats_max}")
    print(f"  opacity:    min={opacity_min:.6f}  max={opacity_max:.6f}")
    print(f"  sh0:        min={sh0_min}  max={sh0_max}")

    # ---- Quantization step sizes ----
    print("\n" + "-"*60)
    print("QUANTIZATION STEP SIZES (8-bit = 219 levels)")
    print("-"*60)

    def step_sizes(name, mins, maxs, n_levels=SCALE_8BIT):
        ranges = maxs - mins
        steps = ranges / n_levels
        print(f"\n  {name}:")
        if hasattr(ranges, '__len__'):
            for c in range(len(ranges)):
                print(f"    ch{c}: range={ranges[c]:.6f}  step={steps[c]:.6f}")
        else:
            print(f"    range={ranges:.6f}  step={steps:.6f}")

    step_sizes("means (16-bit, 48399 levels)", means_min, means_max, 48399)
    step_sizes("scales (8-bit, NO CLIP!)", scales_min, scales_max)
    step_sizes("quats (8-bit, clipped)", quats_min, quats_max)
    step_sizes("opacity (8-bit, clipped)", opacity_min, opacity_max)
    step_sizes("sh0 (8-bit, clipped)", sh0_min, sh0_max)

    # ---- Quantization waste analysis ----
    print("\n" + "-"*60)
    print("QUANTIZATION WASTE ANALYSIS (outlier impact)")
    print("-"*60)

    quantization_waste_report("scales", scales_min, scales_max, all_scales)
    quantization_waste_report("quats (after clip)", quats_min, quats_max,
                              np.clip(all_quats, QUATS_CLIP[0], QUATS_CLIP[1]))
    quantization_waste_report("means_log", means_min, means_max, all_means_log)
    quantization_waste_report("opacities (after clip)",
                              np.array([opacity_min]), np.array([opacity_max]),
                              np.clip(all_opacities, OPACITY_CLIP[0], OPACITY_CLIP[1]))
    quantization_waste_report("sh0 (after clip)", sh0_min, sh0_max,
                              np.clip(all_sh0, SH0_CLIP[0], SH0_CLIP[1]))

    # ---- Outlier deep-dive for scales ----
    print("\n" + "-"*60)
    print("SCALES OUTLIER DEEP-DIVE")
    print("-"*60)

    for c in range(3):
        ch = all_scales[:, c]
        p01, p999 = np.percentile(ch, [0.1, 99.9])
        p05, p95 = np.percentile(ch, [5, 95])

        outliers_001 = ((ch < p01) | (ch > p999)).sum()
        outliers_05 = ((ch < p05) | (ch > p95)).sum()

        print(f"\n  scales ch{c}:")
        print(f"    p0.1={p01:.4f}  p99.9={p999:.4f}  (0.2% outliers: {outliers_001})")
        print(f"    p5={p05:.4f}  p95={p95:.4f}  (10% outliers: {outliers_05})")

        # Check per-frame: which frames have the worst outliers?
        worst_min_frame = -1
        worst_max_frame = -1
        worst_min_val = ch.min()
        worst_max_val = ch.max()
        for fi, f in enumerate(frames):
            fmin = f["scales"][:, c].min()
            fmax = f["scales"][:, c].max()
            if fmin <= worst_min_val:
                worst_min_val = fmin
                worst_min_frame = fi
            if fmax >= worst_max_val:
                worst_max_val = fmax
                worst_max_frame = fi
        print(f"    worst min: frame {worst_min_frame} val={worst_min_val:.4f}")
        print(f"    worst max: frame {worst_max_frame} val={worst_max_val:.4f}")

    # ---- Per-chunk residual analysis ----
    print("\n" + "-"*60)
    print("PER-CHUNK RESIDUAL ANALYSIS (chunk_size=30)")
    print("-"*60)

    chunk_size = 30
    n_chunks = (n_frames + chunk_size - 1) // chunk_size

    for ci in range(min(n_chunks, 5)):  # Show first 5 chunks
        start = ci * chunk_size
        end = min(start + chunk_size, n_frames)
        chunk_frames = frames[start:end]

        # Compute canonical
        chunk_scales = np.stack([f["scales"] for f in chunk_frames], axis=0)  # [T, N, 3]
        canonical_scales = chunk_scales.mean(axis=0)  # [N, 3]

        # Compute delta max
        deltas = chunk_scales - canonical_scales[None, :, :]  # [T, N, 3]
        delta_max = np.abs(deltas).max(axis=(0, 1))  # [3]

        # Quantization step for residuals: 2*delta_max / 219
        residual_step = 2 * delta_max / SCALE_8BIT

        print(f"\n  Chunk {ci} (frames {start}-{end-1}):")
        print(f"    scales delta_max: {delta_max}")
        print(f"    scales residual_step: {residual_step}")

        # What fraction of Gaussians have delta > 0.5 * delta_max?
        # (these are the outliers that stretch the residual range)
        for c in range(3):
            ch_deltas = np.abs(deltas[:, :, c]).flatten()
            half_max = delta_max[c] * 0.5
            n_big = (ch_deltas > half_max).sum()
            total = len(ch_deltas)
            print(f"    ch{c}: {n_big}/{total} ({n_big/total*100:.2f}%) deltas > 50% of delta_max")

    # ---- Suggested clipping bounds ----
    print("\n" + "-"*60)
    print("SUGGESTED SCALE CLIPPING BOUNDS")
    print("-"*60)

    for pct_lo, pct_hi in [(0.1, 99.9), (0.5, 99.5), (1, 99)]:
        lo = np.percentile(all_scales, pct_lo, axis=0)
        hi = np.percentile(all_scales, pct_hi, axis=0)
        ranges = hi - lo
        steps = ranges / SCALE_8BIT
        print(f"\n  p{pct_lo}-p{pct_hi} clip:")
        print(f"    lo={lo}  hi={hi}")
        print(f"    ranges={ranges}  steps={steps}")
        outliers = ((all_scales < lo).any(axis=1) | (all_scales > hi).any(axis=1)).sum()
        print(f"    Gaussians clipped: {outliers}/{len(all_scales)} ({outliers/len(all_scales)*100:.2f}%)")


def main():
    dirs = [
        Path(r"C:\Users\opsiclear_user\projects\data\gsflow\dymensium_soccer\plys"),
        Path(r"C:\Users\opsiclear_user\projects\data\gsflow\dymensium_cosplay\plys"),
    ]

    max_frames = 0  # 0 = all frames
    if len(sys.argv) > 1:
        max_frames = int(sys.argv[1])
        print(f"Limiting to first {max_frames} frames per sequence")

    for d in dirs:
        if d.exists():
            inspect_sequence(d, max_frames)
        else:
            print(f"\nWARNING: Directory not found: {d}")


if __name__ == "__main__":
    main()
