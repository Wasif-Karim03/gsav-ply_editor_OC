"""Detect GSFlow chunk boundaries by measuring frame-to-frame difference.

Compares consecutive frames and looks for large jumps at frame 30/60 boundaries.
"""

import re
import sys
from pathlib import Path

import gsply
import numpy as np


def load_scales(ply_dir: Path, max_frames: int = 0) -> list[np.ndarray]:
    ply_files = sorted(
        ply_dir.glob("*.ply"),
        key=lambda x: int(re.findall(r"\d+", x.stem)[-1])
        if re.findall(r"\d+", x.stem) else 0,
    )
    if max_frames > 0:
        ply_files = ply_files[:max_frames]
    return [gsply.plyread(p).scales for p in ply_files]


def frame_diff(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    return {
        "mean": d.mean(),
        "max": d.max(),
        "p99": np.percentile(d, 99),
        "changed": (d > 0.01).mean() * 100,  # % of gaussians that changed
    }


def main():
    ply_dir = Path(r"C:\Users\opsiclear_user\projects\data\gsflow\dymensium_soccer\plys")
    if len(sys.argv) > 1:
        ply_dir = Path(sys.argv[1])

    # Load enough frames to check both boundaries
    scales = load_scales(ply_dir, max_frames=130)
    n = len(scales)
    print(f"Loaded {n} frames from {ply_dir.parent.name}")

    # Show all consecutive diffs
    print(f"\n{'frame':>6s}  {'mean_diff':>10s}  {'p99_diff':>10s}  {'max_diff':>10s}  {'%changed':>8s}  note")
    print("-" * 70)

    for i in range(n - 1):
        d = frame_diff(scales[i], scales[i + 1])
        note = ""
        if (i + 1) % 30 == 0:
            note = " <-- frame 30 boundary"
        if (i + 1) % 60 == 0:
            note = " <-- frame 60 boundary"
        # Only print boundaries and their neighbors, plus a few normal frames
        is_boundary_region = any(abs((i + 1) - b) <= 1 for b in range(30, n, 30))
        is_start = i < 3
        is_sample = i % 10 == 0

        if is_boundary_region or is_start or is_sample:
            print(f"{i:>3}->{i+1:<3} {d['mean']:10.4f}  {d['p99']:10.4f}  {d['max']:10.4f}  {d['changed']:7.2f}%{note}")


if __name__ == "__main__":
    main()
