"""Compare scale distributions across multiple PLY sequences."""

import re
import sys
from pathlib import Path

import gsply
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gscodec.encoder.utils.helpers import log_transform

DIRS = [
    Path(r"C:\Users\opsiclear_user\projects\data\gsflow\dymensium_soccer\plys"),
    Path(r"C:\Users\opsiclear_user\projects\data\gsflow\dymensium_cosplay\plys"),
    Path(r"C:\Users\opsiclear_user\projects\data\gsflow\elly\plys"),
    Path(r"C:\Users\opsiclear_user\projects\data\gsflow\dymensium_soccer_old\plys"),
]

PCTS = [0, 1, 5, 25, 50, 75, 95, 99, 100]


def load_all(input_dir: Path, max_frames: int = 0) -> dict[str, np.ndarray]:
    ply_files = sorted(
        input_dir.glob("*.ply"),
        key=lambda x: int(re.findall(r"\d+", x.stem)[-1])
        if re.findall(r"\d+", x.stem) else 0,
    )
    if max_frames > 0:
        ply_files = ply_files[:max_frames]

    scales_list = []
    means_list = []
    quats_list = []
    opacities_list = []
    for p in ply_files:
        gs = gsply.plyread(p)
        scales_list.append(gs.scales)
        means_list.append(gs.means)
        quats_list.append(gs.quats)
        op = gs.opacities
        if op.ndim == 1:
            op = op.reshape(-1, 1)
        opacities_list.append(op)

    return {
        "scales": np.concatenate(scales_list, axis=0),
        "means": np.concatenate(means_list, axis=0),
        "quats": np.concatenate(quats_list, axis=0),
        "opacities": np.concatenate(opacities_list, axis=0),
        "n_frames": len(ply_files),
        "n_gaussians": scales_list[0].shape[0],
    }


def print_pct_table(name: str, data: np.ndarray):
    """Print percentile table for [N, C] array."""
    print(f"\n  {name}  (shape {data.shape})")
    header = "         " + "".join(f"{'p'+str(p):>10s}" for p in PCTS)
    print(header)
    for c in range(data.shape[1]):
        vals = np.percentile(data[:, c], PCTS)
        row = f"    ch{c}  " + "".join(f"{v:10.4f}" for v in vals)
        print(row)
    # All channels combined
    vals = np.percentile(data.flatten(), PCTS)
    row = f"    all  " + "".join(f"{v:10.4f}" for v in vals)
    print(row)


def main():
    max_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    for d in DIRS:
        label = d.parent.name
        print(f"\n{'='*90}")
        print(f"  {label}  ({d})")
        print(f"{'='*90}")

        if not d.exists():
            print("  NOT FOUND — skipping")
            continue

        data = load_all(d, max_frames)
        print(f"  frames={data['n_frames']}  gaussians={data['n_gaussians']}")

        print_pct_table("scales (log-space)", data["scales"])
        print_pct_table("opacities (logit)", data["opacities"])

        # Show how many are "near-invisible" (opacity logit < -3 => sigmoid < 0.05)
        op = data["opacities"].flatten()
        invisible = (op < -3).sum()
        total = len(op)
        print(f"\n  near-invisible (opacity_logit < -3): {invisible}/{total} ({invisible/total*100:.1f}%)")

        # Cross-reference: scales of near-invisible vs visible Gaussians
        op_flat = data["opacities"].flatten()  # [N*T]
        sc_flat = data["scales"].reshape(-1, 3)  # [N*T, 3]
        invisible_mask = np.repeat(op_flat < -3, 3).reshape(-1, 3)

        if invisible_mask.any():
            invis_scales = sc_flat[invisible_mask[:, 0]]
            vis_scales = sc_flat[~invisible_mask[:, 0]]
            print(f"\n  scales of INVISIBLE Gaussians (opacity_logit < -3):")
            print(f"    min={invis_scales.min(axis=0)}  max={invis_scales.max(axis=0)}")
            vals = np.percentile(invis_scales.flatten(), [1, 50, 99])
            print(f"    p1={vals[0]:.4f}  median={vals[1]:.4f}  p99={vals[2]:.4f}")

            print(f"  scales of VISIBLE Gaussians (opacity_logit >= -3):")
            print(f"    min={vis_scales.min(axis=0)}  max={vis_scales.max(axis=0)}")
            vals = np.percentile(vis_scales.flatten(), [1, 50, 99])
            print(f"    p1={vals[0]:.4f}  median={vals[1]:.4f}  p99={vals[2]:.4f}")


if __name__ == "__main__":
    main()
