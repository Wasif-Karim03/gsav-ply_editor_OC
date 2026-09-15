"""Codec pre-conditioning experiment: does lossy->decode->lossless shrink lossless size?

Tests whether lossy AV1 encode -> decode -> lossless VP9 re-encode produces
smaller lossless output than direct lossless VP9 encode.

No fine-tuning (gsplat) — just pure codec pre-conditioning.
Tests full 351-frame soccer sequence with VP9 lossless (xllvp9).

Run: uv run python scripts/codec_precondition_experiment.py
"""

import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["PATH"] = (
    r"C:\Users\opsiclear\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.0.1-full_build\bin;" + os.environ.get("PATH", "")
)

import gsply
import xllvp9
from gscodec.common.types import QuantRanges
from gscodec.constants import MIN_VAL, N_ATLAS_COLS, N_ATLAS_ROWS, SCALE_8BIT
from gscodec.encoder.quantization.strategies import quantize_anchor_8bit, quantize_means_16bit_split
from gscodec.encoder.sorting.morton import MortonSortingStrategy
from gscodec.encoder.utils.helpers import log_transform
from gscodec.encoder.video_writer import build_frame_atlas, encode_to_ivf
from gsply import GSTensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = "cuda:0"
PLY_DIR = r"D:\dymensium_soccer\plys"
MAX_FRAMES = 351
FPS = 30
GOP = 30
CRF_VALUES = [1, 2, 4, 8, 16, 24]


# ── Loading + Quantization ────────────────────────────────────────────────


def load_and_build_atlases(
    ply_dir: str, max_frames: int,
) -> tuple[list[np.ndarray], int]:
    """Load PLYs, sort, quantize, build 3x5 atlases."""
    ply_files = sorted(Path(ply_dir).glob("*.ply"))[:max_frames]
    logger.info(f"Loading {len(ply_files)} frames...")

    tensors = []
    for f in ply_files:
        gs = gsply.plyread(str(f))
        op = gs.opacities
        if op.ndim == 1:
            op = op.reshape(-1, 1)
        tensors.append(GSTensor(
            means=torch.as_tensor(gs.means, device=device).float(),
            scales=torch.as_tensor(gs.scales, device=device).float(),
            quats=torch.as_tensor(gs.quats, device=device).float(),
            opacities=torch.as_tensor(op, device=device).float(),
            sh0=torch.as_tensor(gs.sh0, device=device).float(),
            shN=None,
            masks=torch.ones(len(gs.means), dtype=torch.bool, device=device),
        ))

    ng = max(t.means.shape[0] for t in tensors)
    ng = ((ng + 3) // 4) * 4

    # Global ranges
    all_ml = torch.cat([log_transform(t.means) for t in tensors])
    g_min, g_max = all_ml.amin(0), all_ml.amax(0)
    scale_m = (g_max - g_min).clamp(min=1e-8)
    del all_ml

    all_s = torch.cat([t.scales for t in tensors])
    s_min = torch.quantile(all_s, 0.01, dim=0)
    s_max = torch.quantile(all_s, 0.99, dim=0)
    del all_s

    q_min = torch.tensor([-1, -1, -1, -1], device=device)
    q_max = torch.tensor([1, 1, 1, 1], device=device)
    o_min = torch.tensor([-6.0], device=device)
    o_max = torch.tensor([12.0], device=device)
    sh_min = torch.tensor([-2, -2, -2], device=device, dtype=torch.float32)
    sh_max = torch.tensor([4, 4, 4], device=device, dtype=torch.float32)

    sorter = MortonSortingStrategy()
    atlases = []

    for i, t in enumerate(tensors):
        ml = log_transform(t.means)
        idx = sorter.sort_with_global_bbox(ml, g_min, g_max)
        sf = t[idx]

        # Pad
        n = sf.means.shape[0]
        if n < ng:
            def _pad(a, fill=0.0):
                return torch.cat([a, torch.full((ng - n, *a.shape[1:]), fill, device=device)])
            sf = GSTensor(
                means=_pad(sf.means), scales=_pad(sf.scales, -10),
                quats=_pad(sf.quats), opacities=_pad(sf.opacities, -6),
                sh0=_pad(sf.sh0), shN=None,
                masks=torch.ones(ng, dtype=torch.bool, device=device),
            )

        # Quantize
        hi, lo = quantize_means_16bit_split(log_transform(sf.means), g_min, g_max)
        sc_q = quantize_anchor_8bit(
            sf.scales.clamp(min=s_min, max=s_max), s_min, s_max)
        qt_q = quantize_anchor_8bit(sf.quats.clamp(-1, 1), q_min, q_max)
        op_q = quantize_anchor_8bit(
            sf.opacities.clamp(-6, 12), o_min, o_max).squeeze(-1)
        sh_q = quantize_anchor_8bit(sf.sh0.clamp(-2, 4), sh_min, sh_max)

        atlas = build_frame_atlas(hi, sc_q, qt_q, op_q, sh_q, ng)
        atlases.append(atlas)

        if (i + 1) % 50 == 0:
            logger.info(f"  Built {i + 1}/{len(tensors)} atlases")

    logger.info(f"Built {len(atlases)} atlases, {ng} Gaussians")
    return atlases, ng


# ── Lossy Encode -> Decode ─────────────────────────────────────────────────


def lossy_encode_decode_ffmpeg(
    atlases: list[np.ndarray], crf: int, codec: str = "libsvtav1",
) -> list[np.ndarray]:
    """Lossy encode with FFmpeg, decode back to Y-plane."""
    height, width = atlases[0].shape
    n_frames = len(atlases)

    with tempfile.TemporaryDirectory() as tmpdir:
        ivf_path = Path(tmpdir) / "lossy.ivf"

        if codec == "libvpx-vp9":
            enc_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-s", f"{width}x{height}", "-r", str(FPS),
                "-color_range", "tv", "-i", "pipe:0",
                "-c:v", "libvpx-vp9",
                "-crf", str(crf), "-b:v", "0",
                "-deadline", "realtime", "-cpu-used", "8",
                "-row-mt", "1", "-auto-alt-ref", "0", "-lag-in-frames", "0",
                "-g", str(GOP),
                "-pix_fmt", "yuv420p", "-color_range", "tv",
                "-f", "ivf", str(ivf_path),
            ]
        else:
            enc_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-s", f"{width}x{height}", "-r", str(FPS),
                "-color_range", "tv", "-i", "pipe:0",
                "-c:v", codec,
                "-crf", str(crf), "-b:v", "0", "-preset", "8",
                "-g", str(GOP),
                "-pix_fmt", "yuv420p", "-color_range", "tv",
                "-f", "ivf", str(ivf_path),
            ]

        proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
        uv_w, uv_h = width // 2, height // 2
        uv_bytes = np.full((uv_h, uv_w), 128, dtype=np.uint8).tobytes()
        for atlas in atlases:
            proc.stdin.write(atlas.tobytes())
            proc.stdin.write(uv_bytes)  # U
            proc.stdin.write(uv_bytes)  # V
        proc.stdin.close()
        proc.wait()

        lossy_size = ivf_path.stat().st_size

        # Decode
        dec_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-color_range", "tv", "-i", str(ivf_path),
            "-vf", "extractplanes=y",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-color_range", "pc", "pipe:1",
        ]
        result = subprocess.run(dec_cmd, capture_output=True, check=True)
        raw = np.frombuffer(result.stdout, dtype=np.uint8).reshape(n_frames, height, width)

    return [raw[i].copy() for i in range(n_frames)], lossy_size


def vp9_lossless_size(atlases: list[np.ndarray]) -> int:
    """Encode losslessly with xllvp9, return total OBU bytes."""
    frames = np.stack(atlases)
    ivf_data = xllvp9.encode(frames, fps=FPS, keyframe_interval=GOP)
    # Parse IVF to get raw OBU size
    offset = 32
    total = 0
    while offset < len(ivf_data):
        if offset + 12 > len(ivf_data):
            break
        import struct
        frame_size = struct.unpack("<I", ivf_data[offset:offset+4])[0]
        offset += 12 + frame_size
        total += frame_size
    return total


def av1_lossless_size(atlases: list[np.ndarray]) -> int:
    """Encode losslessly with libaom-av1, return total OBU bytes."""
    raw_bytes, entries = encode_to_ivf(atlases, fps=FPS, codec="libaom-av1", gop_size=GOP)
    return len(raw_bytes)


# ── Atlas Comparison ──────────────────────────────────────────────────────


def atlas_psnr(original: list[np.ndarray], decoded: list[np.ndarray]) -> float:
    """Compute average PSNR across all frames (Y-channel)."""
    mse_sum = 0.0
    for o, d in zip(original, decoded):
        mse_sum += np.mean((o.astype(np.float32) - d.astype(np.float32)) ** 2)
    mse_avg = mse_sum / len(original)
    if mse_avg < 1e-10:
        return 999.0
    return float(10 * np.log10(255**2 / mse_avg))


def count_changed_pixels(original: list[np.ndarray], decoded: list[np.ndarray]) -> tuple[float, float]:
    """Return (fraction changed, mean absolute delta of changed pixels)."""
    total_px = 0
    changed = 0
    delta_sum = 0
    for o, d in zip(original, decoded):
        diff = np.abs(o.astype(np.int16) - d.astype(np.int16))
        mask = diff > 0
        total_px += o.size
        changed += mask.sum()
        delta_sum += diff[mask].sum()
    pct = changed / total_px
    mean_delta = delta_sum / max(changed, 1)
    return float(pct), float(mean_delta)


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    atlases, ng = load_and_build_atlases(PLY_DIR, MAX_FRAMES)
    T = len(atlases)

    print(f"\n{T} frames, {ng} Gaussians, atlas {atlases[0].shape[1]}x{atlases[0].shape[0]}")
    print("=" * 95)

    # Baselines
    logger.info("Encoding VP9 lossless baseline...")
    t0 = time.time()
    vp9_base = vp9_lossless_size(atlases)
    vp9_time = time.time() - t0
    print(f"VP9 lossless baseline:  {vp9_base:>12,} bytes ({vp9_base/1e6:.2f} MB) [{vp9_time:.1f}s]")

    logger.info("Encoding AV1 lossless baseline...")
    t0 = time.time()
    av1_base = av1_lossless_size(atlases)
    av1_time = time.time() - t0
    print(f"AV1 lossless baseline:  {av1_base:>12,} bytes ({av1_base/1e6:.2f} MB) [{av1_time:.1f}s]")

    print(f"\n{'CRF':>4} {'Codec':>8} {'Lossy IVF':>12} {'->VP9 LL':>12} {'vs VP9base':>11} "
          f"{'->AV1 LL':>12} {'vs AV1base':>11} {'PSNR':>7} {'%changed':>9} {'avgD':>5}")
    print("-" * 95)

    for crf in CRF_VALUES:
        for lossy_codec, codec_name in [("libsvtav1", "AV1"), ("libvpx-vp9", "VP9")]:
            t0 = time.time()

            # Lossy encode -> decode
            decoded, lossy_ivf_size = lossy_encode_decode_ffmpeg(atlases, crf, lossy_codec)

            # Measure distortion
            psnr = atlas_psnr(atlases, decoded)
            pct_changed, avg_delta = count_changed_pixels(atlases, decoded)

            # Re-encode decoded frames losslessly
            vp9_reenc = vp9_lossless_size(decoded)
            av1_reenc = av1_lossless_size(decoded)

            elapsed = time.time() - t0
            vp9_pct = (vp9_reenc - vp9_base) / vp9_base * 100
            av1_pct = (av1_reenc - av1_base) / av1_base * 100

            print(f"{crf:>4} {codec_name:>8} {lossy_ivf_size:>12,} {vp9_reenc:>12,} {vp9_pct:>+10.1f}% "
                  f"{av1_reenc:>12,} {av1_pct:>+10.1f}% {psnr:>6.1f} {pct_changed:>8.1%} {avg_delta:>5.1f}")

    print(f"\nVP9 lossless baseline:  {vp9_base:>12,}")
    print(f"AV1 lossless baseline:  {av1_base:>12,}")


if __name__ == "__main__":
    main()
