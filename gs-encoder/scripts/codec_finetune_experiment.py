"""Codec-aware fine-tuning experiment.

Hypothesis: lossy encode → decode → fine-tune (gsplat) → lossless re-encode
produces SMALLER lossless video than direct lossless encode, because the
lossy codec pre-conditions values to be smoother/more predictable.

Pipeline:
  1. Quantize attributes to 8-bit [16,235] (standard)
  2. Build 3x5 atlas, encode LOSSY (CRF=N) with FFmpeg
  3. Decode back → get codec-distorted attribute values
  4. Fine-tune scales/quats/opacity/sh0 with gsplat (means_hi frozen)
  5. Re-quantize fine-tuned attributes to 8-bit
  6. Build atlas, encode LOSSLESSLY
  7. Compare: lossless size of fine-tuned vs original

Skips means_hi — only non-means cells are affected by CRF distortion.

Run: uv run python scripts/codec_finetune_experiment.py
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

import gsplat
import gsply
from gscodec.common.types import QuantRanges
from gscodec.constants import MIN_VAL, N_ATLAS_COLS, N_ATLAS_ROWS, SCALE_8BIT
from gscodec.encoder.quantization.strategies import quantize_anchor_8bit, quantize_means_16bit_split
from gscodec.encoder.sorting.morton import MortonSortingStrategy
from gscodec.encoder.utils.helpers import log_transform
from gscodec.encoder.video_writer import build_frame_atlas, _morton_2d_order, encode_to_ivf
from gsply import GSTensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = "cuda:0"
PLY_DIR = r"D:\dymensium_soccer\plys"
MAX_FRAMES = 30  # one chunk — enough to measure the effect
FPS = 30
CRF_VALUES = [1, 2, 4, 8, 16]
FINETUNE_STEPS = 80  # per frame
N_CAMERAS = 32
SH_C0 = 0.28209479177387814


# ── Loading ───────────────────────────────────────────────────────────────


def load_frames(ply_dir: str, max_frames: int) -> tuple[list[GSTensor], int]:
    ply_files = sorted(Path(ply_dir).glob("*.ply"))[:max_frames]
    logger.info(f"Loading {len(ply_files)} frames...")
    tensors = []
    for f in ply_files:
        gs = gsply.plyread(str(f))
        op = gs.opacities
        if op.ndim == 1:
            op = op.reshape(-1, 1)
        t = GSTensor(
            means=torch.as_tensor(gs.means, device=device).float(),
            scales=torch.as_tensor(gs.scales, device=device).float(),
            quats=torch.as_tensor(gs.quats, device=device).float(),
            opacities=torch.as_tensor(op, device=device).float(),
            sh0=torch.as_tensor(gs.sh0, device=device).float(),
            shN=None,
            masks=torch.ones(len(gs.means), dtype=torch.bool, device=device),
        )
        tensors.append(t)
    ng = max(t.means.shape[0] for t in tensors)
    ng = ((ng + 3) // 4) * 4
    return tensors, ng


def compute_global_ranges(frames: list[GSTensor]) -> QuantRanges:
    all_ml = torch.cat([log_transform(f.means) for f in frames])
    means_min = all_ml.amin(0).cpu().numpy()
    means_max = all_ml.amax(0).cpu().numpy()
    del all_ml
    all_s = torch.cat([f.scales for f in frames])
    scales_min = torch.quantile(all_s, 0.01, dim=0).cpu().numpy()
    scales_max = torch.quantile(all_s, 0.99, dim=0).cpu().numpy()
    del all_s
    return QuantRanges(
        means_min=means_min, means_max=means_max,
        scales_min=scales_min, scales_max=scales_max,
        quats_min=np.array([-1, -1, -1, -1], dtype=np.float32),
        quats_max=np.array([1, 1, 1, 1], dtype=np.float32),
        opacity_min=-6.0, opacity_max=12.0,
        sh0_min=np.array([-2, -2, -2], dtype=np.float32),
        sh0_max=np.array([4, 4, 4], dtype=np.float32),
    )


# ── Quantize + Sort ──────────────────────────────────────────────────────


def sort_and_quantize(
    frames: list[GSTensor], ranges: QuantRanges, n_gaussians: int,
) -> list[dict]:
    """Sort, clip, quantize each frame. Returns list of per-frame dicts."""
    sorter = MortonSortingStrategy()
    g_min = torch.as_tensor(ranges.means_min, device=device)
    g_max = torch.as_tensor(ranges.means_max, device=device)
    rt = {
        "scales_min": torch.as_tensor(ranges.scales_min, device=device),
        "scales_max": torch.as_tensor(ranges.scales_max, device=device),
        "quats_min": torch.as_tensor(ranges.quats_min, device=device),
        "quats_max": torch.as_tensor(ranges.quats_max, device=device),
        "opacity_min": torch.tensor([ranges.opacity_min], device=device),
        "opacity_max": torch.tensor([ranges.opacity_max], device=device),
        "sh0_min": torch.as_tensor(ranges.sh0_min, device=device),
        "sh0_max": torch.as_tensor(ranges.sh0_max, device=device),
    }

    results = []
    for frame in frames:
        ml = log_transform(frame.means)
        idx = sorter.sort_with_global_bbox(ml, g_min, g_max)
        sf = frame[idx]

        # Pad
        n = sf.means.shape[0]
        if n < n_gaussians:
            def _pad(t, fill=0.0):
                return torch.cat([t, torch.full((n_gaussians - n, *t.shape[1:]), fill, device=device)])
            sf = GSTensor(
                means=_pad(sf.means), scales=_pad(sf.scales, -10),
                quats=_pad(sf.quats), opacities=_pad(sf.opacities, -6),
                sh0=_pad(sf.sh0), shN=None,
                masks=torch.ones(n_gaussians, dtype=torch.bool, device=device),
            )

        # Clip
        scales_c = sf.scales.clamp(min=rt["scales_min"], max=rt["scales_max"])
        quats_c = sf.quats.clamp(-1, 1)
        opacity_c = sf.opacities.clamp(-6, 12)
        sh0_c = sf.sh0.clamp(-2, 4)

        # Quantize
        means_hi, means_lo = quantize_means_16bit_split(
            log_transform(sf.means),
            torch.as_tensor(ranges.means_min, device=device),
            torch.as_tensor(ranges.means_max, device=device),
        )
        scales_q = quantize_anchor_8bit(scales_c, rt["scales_min"], rt["scales_max"])
        quats_q = quantize_anchor_8bit(quats_c, rt["quats_min"], rt["quats_max"])
        opacity_q = quantize_anchor_8bit(opacity_c, rt["opacity_min"], rt["opacity_max"]).squeeze(-1)
        sh0_q = quantize_anchor_8bit(sh0_c, rt["sh0_min"], rt["sh0_max"])

        results.append({
            "frame": sf, "means_hi": means_hi, "means_lo": means_lo,
            "scales_q": scales_q, "quats_q": quats_q,
            "opacity_q": opacity_q, "sh0_q": sh0_q,
        })
    return results


# ── Atlas / Video Encode ──────────────────────────────────────────────────


def build_atlases(frame_data: list[dict], n_gaussians: int) -> list[np.ndarray]:
    atlases = []
    for fd in frame_data:
        atlas = build_frame_atlas(
            fd["means_hi"], fd["scales_q"], fd["quats_q"],
            fd["opacity_q"], fd["sh0_q"], n_gaussians,
        )
        atlases.append(atlas)
    return atlases


def lossless_video_size(atlases: list[np.ndarray]) -> int:
    """Encode losslessly, return total OBU bytes."""
    raw_bytes, entries = encode_to_ivf(atlases, fps=FPS, codec="libaom-av1", gop_size=len(atlases))
    return len(raw_bytes)


def lossy_encode_decode(atlases: list[np.ndarray], crf: int) -> list[np.ndarray]:
    """Encode lossy AV1, decode back, return decoded Y-plane atlases."""
    if not atlases:
        return []
    height, width = atlases[0].shape
    n_frames = len(atlases)

    with tempfile.TemporaryDirectory() as tmpdir:
        ivf_path = Path(tmpdir) / "lossy.ivf"

        enc_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{width}x{height}", "-r", str(FPS),
            "-color_range", "tv", "-i", "pipe:0",
            "-c:v", "libsvtav1",
            "-crf", str(crf), "-b:v", "0", "-preset", "6",
            "-g", str(n_frames),
            "-pix_fmt", "yuv420p", "-color_range", "tv",
            "-f", "ivf", str(ivf_path),
        ]
        proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
        uv_w, uv_h = width // 2, height // 2
        u_plane = np.full((uv_h, uv_w), 128, dtype=np.uint8).tobytes()
        v_plane = np.full((uv_h, uv_w), 128, dtype=np.uint8).tobytes()
        for atlas in atlases:
            proc.stdin.write(atlas.tobytes())
            proc.stdin.write(u_plane)
            proc.stdin.write(v_plane)
        proc.stdin.close()
        proc.wait()

        # Decode back
        dec_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-color_range", "tv", "-i", str(ivf_path),
            "-vf", "extractplanes=y",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-color_range", "pc", "pipe:1",
        ]
        result = subprocess.run(dec_cmd, capture_output=True, check=True)
        raw = np.frombuffer(result.stdout, dtype=np.uint8).reshape(n_frames, height, width)
        return [raw[i].copy() for i in range(n_frames)]


# ── Extract Channels from Atlas ──────────────────────────────────────────


def extract_non_means_from_atlas(
    atlas: np.ndarray, n_gaussians: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract scales/quats/opacity/sh0 from decoded atlas (channels 3-13)."""
    side = atlas.shape[1] // N_ATLAS_COLS
    inv_morton = np.argsort(
        _compute_morton_codes(side)
    ).astype(np.int64)

    channels = []
    for ch_idx in range(15):
        atlas_col = ch_idx % N_ATLAS_COLS
        atlas_row = ch_idx // N_ATLAS_COLS
        y0 = atlas_row * side
        x0 = atlas_col * side
        block = atlas[y0:y0 + side, x0:x0 + side]
        channels.append(block.flat[inv_morton[:n_gaussians]])

    scales_q = np.stack([channels[3], channels[4], channels[5]], axis=-1)
    quats_q = np.stack([channels[6], channels[7], channels[8], channels[9]], axis=-1)
    opacity_q = channels[10]
    sh0_q = np.stack([channels[11], channels[12], channels[13]], axis=-1)
    return scales_q, quats_q, opacity_q, sh0_q


def _compute_morton_codes(side: int) -> np.ndarray:
    ys, xs = np.meshgrid(np.arange(side, dtype=np.uint32),
                         np.arange(side, dtype=np.uint32), indexing="ij")
    def part1by1(n):
        n = n & np.uint32(0x0000FFFF)
        n = (n | (n << 8)) & np.uint32(0x00FF00FF)
        n = (n | (n << 4)) & np.uint32(0x0F0F0F0F)
        n = (n | (n << 2)) & np.uint32(0x33333333)
        n = (n | (n << 1)) & np.uint32(0x55555555)
        return n
    return part1by1(xs.ravel()) | (part1by1(ys.ravel()) << np.uint32(1))


# ── Dequantize (8-bit → float) ──────────────────────────────────────────


def dequant_8bit(q: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> torch.Tensor:
    """Dequantize uint8 [16,235] back to float32."""
    norm = (q.astype(np.float32) - MIN_VAL) / SCALE_8BIT
    scale = (vmax - vmin).astype(np.float32)
    result = norm * scale + vmin.astype(np.float32)
    return torch.as_tensor(result, device=device, dtype=torch.float32)


# ── Fine-Tuning ──────────────────────────────────────────────────────────


def finetune_frame(
    original: GSTensor,
    decoded_scales: torch.Tensor,
    decoded_quats: torch.Tensor,
    decoded_opacity: torch.Tensor,
    decoded_sh0: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    steps: int = FINETUNE_STEPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fine-tune decoded attributes to recover visual quality.

    Means stay frozen. Optimizes from codec-decoded starting point.
    """
    # Reference renders from original
    n_train = viewmats.shape[0] - 4
    train_vm, train_Ks = viewmats[:n_train], Ks[:n_train]
    Ks_512 = train_Ks.clone()
    Ks_512[:, 0, :] *= 0.5
    Ks_512[:, 1, :] *= 0.5

    with torch.no_grad():
        ref = gsplat.rasterization(
            means=original.means, quats=original.quats, scales=original.scales.exp(),
            opacities=torch.sigmoid(original.opacities.squeeze(-1)),
            colors=SH_C0 * original.sh0 + 0.5,
            viewmats=train_vm, Ks=Ks_512, width=512, height=512,
            packed=True, render_mode="RGB",
        )[0].detach()

    # Start from codec-decoded values
    scales_p = torch.nn.Parameter(decoded_scales.clone())
    quats_p = torch.nn.Parameter(decoded_quats.clone())
    opacity_p = torch.nn.Parameter(decoded_opacity.clone())
    sh0_p = torch.nn.Parameter(decoded_sh0.clone())

    optimizer = torch.optim.Adam([scales_p, quats_p, opacity_p, sh0_p], lr=1e-3)

    for step in range(steps):
        idx = torch.randperm(n_train, device=device)[:4]
        rendered = gsplat.rasterization(
            means=original.means, quats=quats_p, scales=scales_p.exp(),
            opacities=torch.sigmoid(opacity_p.squeeze(-1)),
            colors=SH_C0 * sh0_p + 0.5,
            viewmats=train_vm[idx], Ks=Ks_512[idx], width=512, height=512,
            packed=True, render_mode="RGB",
        )[0]
        loss = (rendered - ref[idx]).abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return scales_p.detach(), quats_p.detach(), opacity_p.detach(), sh0_p.detach()


# ── Requantize ────────────────────────────────────────────────────────────


def requantize_8bit(data: torch.Tensor, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    """Quantize float tensor back to uint8 [16,235]."""
    mn = torch.as_tensor(vmin, device=device)
    mx = torch.as_tensor(vmax, device=device)
    return quantize_anchor_8bit(data, mn, mx)


# ── Main Experiment ───────────────────────────────────────────────────────


def main():
    frames, n_gaussians = load_frames(PLY_DIR, MAX_FRAMES)
    ranges = compute_global_ranges(frames)
    frame_data = sort_and_quantize(frames, ranges, n_gaussians)

    # Build original atlases and measure lossless baseline
    logger.info("Building original atlases...")
    original_atlases = build_atlases(frame_data, n_gaussians)

    logger.info("Encoding lossless baseline...")
    t0 = time.time()
    baseline_size = lossless_video_size(original_atlases)
    baseline_time = time.time() - t0
    logger.info(f"Baseline lossless: {baseline_size:,} bytes ({baseline_size/1024:.1f} KB) [{baseline_time:.1f}s]")

    # Generate cameras for fine-tuning
    from gscodec.encoder.post_snap_finetune import _make_cameras
    viewmats, Ks = _make_cameras(frames[0].means, N_CAMERAS, device)

    print(f"\n{'CRF':>4} {'Lossy':>10} {'FT+LL':>10} {'vs base':>8} {'PSNR_orig':>10} {'PSNR_ft':>10} {'Time':>6}")
    print("-" * 70)

    for crf in CRF_VALUES:
        t0 = time.time()

        # Step 1: Lossy encode → decode
        lossy_decoded = lossy_encode_decode(original_atlases, crf)

        # Measure lossy-only size for reference
        lossy_raw, lossy_entries = encode_to_ivf(
            original_atlases, fps=FPS, codec="libaom-av1", gop_size=len(original_atlases),
            lossy=True, crf=crf,
        )
        lossy_size = len(lossy_raw)

        # Step 2: Extract decoded non-means channels, dequantize, fine-tune, re-quantize
        finetuned_data = []
        psnr_before_list = []
        psnr_after_list = []

        for i, (fd, decoded_atlas) in enumerate(zip(frame_data, lossy_decoded)):
            # Extract decoded attribute values
            sc_dec, qt_dec, op_dec, sh_dec = extract_non_means_from_atlas(decoded_atlas, n_gaussians)

            # Measure atlas-level PSNR of lossy decode vs original
            orig_nonmeans = np.concatenate([
                fd["scales_q"].ravel(), fd["quats_q"].ravel(),
                fd["opacity_q"].ravel(), fd["sh0_q"].ravel(),
            ]).astype(np.float32)
            dec_nonmeans = np.concatenate([
                sc_dec.ravel(), qt_dec.ravel(), op_dec.ravel(), sh_dec.ravel(),
            ]).astype(np.float32)
            mse_before = np.mean((orig_nonmeans - dec_nonmeans) ** 2)
            psnr_before = 10 * np.log10(255**2 / max(mse_before, 1e-10))
            psnr_before_list.append(psnr_before)

            # Dequantize decoded values to float
            sc_float = dequant_8bit(sc_dec, ranges.scales_min, ranges.scales_max)
            qt_float = dequant_8bit(qt_dec, ranges.quats_min, ranges.quats_max)
            op_float = dequant_8bit(op_dec.reshape(-1, 1),
                                    np.array([ranges.opacity_min]),
                                    np.array([ranges.opacity_max]))
            sh_float = dequant_8bit(sh_dec, ranges.sh0_min, ranges.sh0_max)

            # Fine-tune
            sc_ft, qt_ft, op_ft, sh_ft = finetune_frame(
                fd["frame"], sc_float, qt_float, op_float, sh_float,
                viewmats, Ks, steps=FINETUNE_STEPS,
            )

            # Clip to valid ranges
            sc_ft = sc_ft.clamp(min=torch.as_tensor(ranges.scales_min, device=device),
                                max=torch.as_tensor(ranges.scales_max, device=device))
            qt_ft = qt_ft.clamp(-1, 1)
            op_ft = op_ft.clamp(-6, 12)
            sh_ft = sh_ft.clamp(-2, 4)

            # Re-quantize
            sc_rq = requantize_8bit(sc_ft, ranges.scales_min, ranges.scales_max)
            qt_rq = requantize_8bit(qt_ft, ranges.quats_min, ranges.quats_max)
            op_rq = requantize_8bit(op_ft, np.array([ranges.opacity_min]),
                                     np.array([ranges.opacity_max])).squeeze(-1)
            sh_rq = requantize_8bit(sh_ft, ranges.sh0_min, ranges.sh0_max)

            # Measure how close finetuned values are to codec-decoded values
            ft_nonmeans = np.concatenate([
                sc_rq.ravel(), qt_rq.ravel(), op_rq.ravel(), sh_rq.ravel(),
            ]).astype(np.float32)
            mse_after = np.mean((orig_nonmeans - ft_nonmeans) ** 2)
            psnr_after = 10 * np.log10(255**2 / max(mse_after, 1e-10))
            psnr_after_list.append(psnr_after)

            finetuned_data.append({
                **fd,
                "scales_q": sc_rq, "quats_q": qt_rq,
                "opacity_q": op_rq, "sh0_q": sh_rq,
            })

        # Step 3: Build finetuned atlases, encode losslessly
        finetuned_atlases = build_atlases(finetuned_data, n_gaussians)
        ft_lossless_size = lossless_video_size(finetuned_atlases)

        elapsed = time.time() - t0
        pct = (ft_lossless_size - baseline_size) / baseline_size * 100
        avg_psnr_before = np.mean(psnr_before_list)
        avg_psnr_after = np.mean(psnr_after_list)

        print(f"{crf:>4} {lossy_size:>10,} {ft_lossless_size:>10,} {pct:>+7.1f}% "
              f"{avg_psnr_before:>9.1f}dB {avg_psnr_after:>9.1f}dB {elapsed:>5.0f}s")

    print(f"\nBaseline lossless: {baseline_size:,} bytes")
    print("PSNR_orig = atlas PSNR of lossy decode vs original quantized")
    print("PSNR_ft = atlas PSNR of fine-tuned re-quantized vs original quantized")
    print("FT+LL = lossless encode of fine-tuned atlas")


if __name__ == "__main__":
    main()
