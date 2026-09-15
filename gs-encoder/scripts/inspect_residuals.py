"""Show the actual residual distribution as the encoder computes it.

Replicates the encoder's sorting + clipping + canonical pipeline,
then shows the delta distribution for scales (and quats for reference).
"""

import math
import re
import sys
from pathlib import Path

import gsply
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gscodec.encoder.sorting.morton import MortonSortingStrategy
from gscodec.encoder.utils.helpers import log_transform
from gscodec.decoder.providers import GSAVFileProvider


def load_frames(ply_dir: Path, n: int, device: str = "cpu"):
    from gsply import GSTensor
    ply_files = sorted(
        ply_dir.glob("*.ply"),
        key=lambda x: int(re.findall(r"\d+", x.stem)[-1])
        if re.findall(r"\d+", x.stem) else 0,
    )[:n]
    frames = []
    for p in ply_files:
        gs = gsply.plyread(p)
        op = gs.opacities
        if op.ndim == 1:
            op = op.reshape(-1, 1)
        frames.append(GSTensor(
            means=torch.from_numpy(gs.means).float().to(device),
            scales=torch.from_numpy(gs.scales).float().to(device),
            quats=torch.from_numpy(gs.quats).float().to(device),
            opacities=torch.from_numpy(op).float().to(device),
            sh0=torch.from_numpy(gs.sh0).float().to(device),
            shN=None,
            masks=torch.ones(gs.means.shape[0], dtype=torch.bool, device=device),
        ))
    return frames


def main():
    gsav_path = Path(r"C:\Users\opsiclear_user\projects\data\gsflow\dymensium_soccer\soccer_lossless.gsav")
    ply_dir = Path(r"C:\Users\opsiclear_user\projects\data\gsflow\dymensium_soccer\plys")

    if len(sys.argv) > 1:
        gsav_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        ply_dir = Path(sys.argv[2])

    # Read stored ranges from GSAV
    provider = GSAVFileProvider(gsav_path)
    h = provider.header
    ranges = provider.ranges
    n_frames = h["n_frames"]
    chunk_size = h["chunk_size"]

    scales_clip_min = torch.from_numpy(ranges.scales_min)
    scales_clip_max = torch.from_numpy(ranges.scales_max)

    print(f"Frames={n_frames}  Gaussians={h['n_gaussians']}  ChunkSize={chunk_size}")
    print(f"Stored scales range: {ranges.scales_min} to {ranges.scales_max}")

    # Load all frames
    frames = load_frames(ply_dir, n_frames)
    n_g = frames[0].means.shape[0]

    sorter = MortonSortingStrategy()
    n_chunks = math.ceil(n_frames / chunk_size)

    PCTS = [0, 0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]

    all_scales_deltas = []  # collect across all chunks
    all_quats_deltas = []

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, n_frames)
        chunk_frames = frames[start:end]

        # Sort (same as encoder)
        _, morton_indices = sorter.sort(chunk_frames[0])
        sorted_frames = [f[morton_indices] for f in chunk_frames]

        # Clip scales
        clipped_scales = torch.stack(
            [f.scales.clamp(min=scales_clip_min, max=scales_clip_max) for f in sorted_frames], dim=0
        )  # [T, N, 3]

        # Canonical
        canonical_scales = clipped_scales.mean(dim=0)  # [N, 3]

        # Deltas
        deltas = clipped_scales - canonical_scales.unsqueeze(0)  # [T, N, 3]
        abs_deltas = deltas.abs()

        # True delta_max (what encoder uses)
        delta_max = abs_deltas.amax(dim=(0, 1))  # [3]

        all_scales_deltas.append(deltas.reshape(-1, 3).numpy())

        # Also do quats for reference
        clipped_quats = torch.stack(
            [f.quats.clamp(-1, 1) for f in sorted_frames], dim=0
        )
        canonical_quats = clipped_quats.mean(dim=0)
        quats_deltas = clipped_quats - canonical_quats.unsqueeze(0)
        all_quats_deltas.append(quats_deltas.reshape(-1, 4).numpy())

        print(f"\nChunk {ci} (frames {start}-{end-1}):")
        print(f"  scales delta_max (true max): {delta_max.numpy()}")
        print(f"  scales |delta| distribution (all 3 channels pooled):")
        flat = abs_deltas.reshape(-1).numpy()
        pcts = np.percentile(flat, PCTS)
        for p, v in zip(PCTS, pcts):
            print(f"    p{p:>5}: {v:.6f}")

        # Per-channel
        for c in range(3):
            ch = abs_deltas[:, :, c].reshape(-1).numpy()
            pcts_ch = np.percentile(ch, PCTS)
            print(f"  scales ch{c} |delta|:")
            for p, v in zip(PCTS, pcts_ch):
                print(f"    p{p:>5}: {v:.6f}")

    # ---- Aggregate across all chunks ----
    print(f"\n{'='*70}")
    print("AGGREGATE ACROSS ALL CHUNKS")
    print(f"{'='*70}")

    all_sd = np.concatenate(all_scales_deltas, axis=0)  # [total, 3]
    all_qd = np.concatenate(all_quats_deltas, axis=0)   # [total, 4]

    abs_sd = np.abs(all_sd)
    abs_qd = np.abs(all_qd)

    print(f"\nScales |delta| (all chunks, all channels pooled):")
    flat = abs_sd.flatten()
    for p in PCTS:
        v = np.percentile(flat, p)
        print(f"  p{p:>5}: {v:.6f}")

    print(f"\n  Per-channel:")
    for c in range(3):
        ch = abs_sd[:, c]
        print(f"  ch{c}: p50={np.percentile(ch, 50):.4f}  p90={np.percentile(ch, 90):.4f}  "
              f"p99={np.percentile(ch, 99):.4f}  p99.9={np.percentile(ch, 99.9):.4f}  "
              f"p99.99={np.percentile(ch, 99.99):.4f}  max={ch.max():.4f}")

    print(f"\nQuats |delta| (for reference):")
    flat_q = abs_qd.flatten()
    for p in PCTS:
        v = np.percentile(flat_q, p)
        print(f"  p{p:>5}: {v:.6f}")

    # ---- What would different percentile caps give us? ----
    print(f"\n{'='*70}")
    print("RESIDUAL_MAX PERCENTILE OPTIONS (scales, per-channel)")
    print(f"{'='*70}")

    for pct in [99.0, 99.5, 99.9, 99.95, 99.99]:
        caps = np.percentile(abs_sd, pct, axis=0)  # [3]
        steps = 2 * caps / 219
        n_clamped = (abs_sd > caps).sum(axis=0)
        total = abs_sd.shape[0]
        print(f"\n  p{pct}:")
        print(f"    caps:    {caps}")
        print(f"    steps:   {steps}")
        print(f"    clamped: {n_clamped} / {total} per channel "
              f"({n_clamped / total * 100}%)")
        # Compare to true max
        true_max = abs_sd.max(axis=0)
        improvement = true_max / caps
        print(f"    vs true max: {improvement}x tighter")


if __name__ == "__main__":
    main()
