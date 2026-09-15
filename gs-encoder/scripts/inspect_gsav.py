"""Stage-by-stage error breakdown: original PLYs vs decoded frames.

Compares original PLY data against decoded GSAV data, splitting
analysis by visible vs invisible Gaussians.
"""

import re
import sys
from pathlib import Path

import gsply
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gscodec.decoder.chunk_decoder import FrameDecoder
from gscodec.decoder.providers import GSAVFileProvider


def load_originals(ply_dir: Path, n: int) -> list[dict[str, np.ndarray]]:
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
        frames.append({"scales": gs.scales, "opacities": op, "quats": gs.quats})
    return frames


def pct_line(arr: np.ndarray) -> str:
    flat = np.abs(arr).flatten()
    ps = np.percentile(flat, [50, 90, 99, 99.9, 100])
    return (f"mean={flat.mean():.4f}  p50={ps[0]:.4f}  p90={ps[1]:.4f}  "
            f"p99={ps[2]:.4f}  p99.9={ps[3]:.4f}  max={ps[4]:.4f}")


def main():
    gsav_path = Path(r"C:\Users\opsiclear_user\projects\data\gsflow\dymensium_soccer\soccer_lossless.gsav")
    ply_dir = Path(r"C:\Users\opsiclear_user\projects\data\gsflow\dymensium_soccer\plys")

    if len(sys.argv) > 1:
        gsav_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        ply_dir = Path(sys.argv[2])

    provider = GSAVFileProvider(gsav_path)
    h = provider.header
    ranges = provider.ranges
    n_frames, n_g = h["n_frames"], h["n_gaussians"]
    chunk_size = h["chunk_size"]

    print(f"GSAV: {gsav_path}")
    print(f"Frames={n_frames}  Gaussians={n_g}  ChunkSize={chunk_size}")
    print(f"Stored scales range: min={ranges.scales_min}  max={ranges.scales_max}")
    sr = ranges.scales_max - ranges.scales_min
    print(f"  range={sr}  8-bit step={sr/219}")

    originals = load_originals(ply_dir, n_frames)
    print(f"Loaded {len(originals)} original frames")

    # Decode all frames
    decoder = FrameDecoder(ranges, provider.atlas_width, n_g, provider.n_atlas_cols)
    atlases = provider.decode_all_frames()

    # Use frame 0 opacity as proxy for visibility
    op0 = originals[0]["opacities"].flatten()
    vis0 = op0 >= -3.0

    print(f"\n{'='*80}")
    print("AGGREGATE ACROSS ALL FRAMES")
    print(f"{'='*80}")

    all_total_err = []
    all_total_err_vis = []
    all_total_err_invis = []

    for fi in range(n_frames):
        decoded = decoder.decode_frame(atlases[fi])
        err = decoded.scales - originals[fi]["scales"]
        all_total_err.append(err)
        all_total_err_vis.append(err[vis0])
        all_total_err_invis.append(err[~vis0])

    all_err = np.concatenate(all_total_err)
    all_vis = np.concatenate(all_total_err_vis)
    all_invis = np.concatenate(all_total_err_invis)

    print(f"\n  Total scale error (decoded - original):")
    print(f"    ALL:       {pct_line(all_err)}")
    print(f"    VISIBLE:   {pct_line(all_vis)}")
    print(f"    INVISIBLE: {pct_line(all_invis)}")

    for thresh in [0.05, 0.1, 0.5, 1.0]:
        n_all = (np.abs(all_err) > thresh).sum()
        n_vis = (np.abs(all_vis) > thresh).sum()
        print(f"    |err|>{thresh}: all={n_all/all_err.size*100:.1f}%  "
              f"visible={n_vis/all_vis.size*100:.1f}%")


if __name__ == "__main__":
    main()
