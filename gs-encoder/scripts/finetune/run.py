"""Grid-snap fine-tuning for Gaussian Splatting compression.

Fine-tunes PLY positions so that the lo-byte of 16-bit quantized means
clusters near 0, making the means_lo payload highly compressible via zstd.

Quality is maintained via gsplat CUDA rendering + worst-view PSNR gating.

Usage:
    uv run python -m scripts.finetune.run --input-dir D:/dymensium_soccer/plys
"""

import logging
import math
import sys
import time
from pathlib import Path

import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scripts.finetune.alpha_controller import AlphaController, Checkpoint
from scripts.finetune.camera import SyntheticCameras
from scripts.finetune.compression_bench import (
    delta_lo_entropy,
    snap_rate,
    zstd_compressed_size,
)
from scripts.finetune.config import FinetuneConfig
from scripts.finetune.io import (
    compute_global_ranges,
    load_ply_sequence,
    save_ply_with_new_means,
)
from scripts.finetune.losses import (
    compute_importance_weights,
    compute_psnr_per_view,
    compute_ssim,
    grid_snap_loss,
    render_loss,
    temporal_lo_consistency_loss,
)
from scripts.finetune.renderer import render_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _render_frame(
    means: torch.Tensor,
    frame_data: dict[str, torch.Tensor],
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Render one frame from given cameras. Returns [C, H, W, 3]."""
    return render_batch(
        means=means,
        log_scales=frame_data["scales"],
        quats=frame_data["quats"],
        logit_opacities=frame_data["opacities"],
        sh0=frame_data["sh0"],
        viewmats=viewmats, Ks=Ks, width=width, height=height,
    )


def _scale_intrinsics(
    Ks: torch.Tensor,
    src_res: tuple[int, int],
    dst_res: tuple[int, int],
) -> torch.Tensor:
    """Scale intrinsics for a different resolution."""
    Ks_out = Ks.clone()
    Ks_out[:, 0, :] *= dst_res[1] / src_res[1]
    Ks_out[:, 1, :] *= dst_res[0] / src_res[0]
    return Ks_out


class TSVLogger:
    """Simple TSV logger."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w")
        self._wrote_header = False

    def log(self, row: dict[str, float | int | str]) -> None:
        if not self._wrote_header:
            self._f.write("\t".join(row.keys()) + "\n")
            self._wrote_header = True
        self._f.write("\t".join(str(v) for v in row.values()) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def finetune_chunk(
    chunk_frames: list,
    ply_paths: list[Path],
    output_dir: Path,
    chunk_idx: int,
    means_min: torch.Tensor,
    means_max: torch.Tensor,
    cameras: SyntheticCameras,
    cfg: FinetuneConfig,
    tsv: TSVLogger,
) -> dict[str, float]:
    """Fine-tune one chunk of frames."""
    device = cfg.device
    n_frames = len(chunk_frames)
    train_H, train_W = cfg.train_resolution
    eval_H, eval_W = cfg.eval_resolution

    # Camera split
    n_eval = min(cfg.n_eval_cameras, cameras.n_cameras)
    eval_indices = torch.arange(cameras.n_cameras - n_eval, cameras.n_cameras, device=device)
    train_indices = torch.arange(cameras.n_cameras - n_eval, device=device)
    eval_viewmats, eval_Ks_base = cameras.get(eval_indices)
    train_viewmats_all, train_Ks_base = cameras.get(train_indices)

    # Scale intrinsics for train resolution
    train_Ks = _scale_intrinsics(train_Ks_base, cfg.eval_resolution, cfg.train_resolution)
    eval_Ks = eval_Ks_base

    # Per-frame frozen attributes
    frame_data_list: list[dict[str, torch.Tensor]] = []
    for f in chunk_frames:
        frame_data_list.append({
            "scales": f.scales.detach(), "quats": f.quats.detach(),
            "opacities": f.opacities.detach(), "sh0": f.sh0.detach(),
        })

    # Pre-render references
    logger.info(f"  Chunk {chunk_idx}: pre-rendering references...")
    ref_eval: list[torch.Tensor] = []
    ref_train: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(n_frames):
            ref_eval.append(_render_frame(
                chunk_frames[i].means, frame_data_list[i],
                eval_viewmats, eval_Ks, eval_W, eval_H,
            ).detach())
            ref_train.append(_render_frame(
                chunk_frames[i].means, frame_data_list[i],
                train_viewmats_all, train_Ks, train_W, train_H,
            ).detach())

    # Optimizable means
    means_params = [torch.nn.Parameter(f.means.clone()) for f in chunk_frames]

    # Importance weights
    imp_w: list[torch.Tensor | None] = [None] * n_frames
    if cfg.use_importance_weights:
        for i in range(n_frames):
            imp_w[i] = compute_importance_weights(
                frame_data_list[i]["opacities"], frame_data_list[i]["scales"],
            )

    optimizer = torch.optim.Adam([{"params": means_params, "lr": cfg.lr_means}], amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_steps, eta_min=cfg.lr_min,
    )
    alpha_ctrl = AlphaController(
        alpha_max=cfg.alpha_max, warmup_start=cfg.alpha_warmup_start,
        warmup_end=cfg.alpha_warmup_end, max_halvings=cfg.max_alpha_halvings,
    )

    # Baseline quality (worst-view PSNR)
    with torch.no_grad():
        baseline_psnrs: list[float] = []
        baseline_ssim = 0.0
        for i in range(n_frames):
            rendered = _render_frame(
                means_params[i], frame_data_list[i],
                eval_viewmats, eval_Ks, eval_W, eval_H,
            )
            baseline_psnrs.extend(compute_psnr_per_view(rendered, ref_eval[i]))
            baseline_ssim += compute_ssim(rendered, ref_eval[i])
    baseline_worst = min(baseline_psnrs)
    baseline_mean = sum(baseline_psnrs) / len(baseline_psnrs)
    baseline_ssim /= n_frames
    logger.info(f"  Chunk {chunk_idx}: baseline worst_PSNR={baseline_worst:.1f}, mean={baseline_mean:.1f}")

    checkpoint = Checkpoint()
    checkpoint.save(means_params, baseline_worst, baseline_ssim)

    with torch.no_grad():
        sr0 = sum(snap_rate(p, means_min, means_max) for p in means_params) / n_frames
    logger.info(f"  Chunk {chunk_idx}: initial snap_rate={sr0:.3f}")

    snap_history: list[float] = [sr0]
    best_snap = sr0
    rng = torch.Generator(device="cpu")
    rng.manual_seed(chunk_idx * 1000)
    fps = min(cfg.frames_per_step, n_frames) if cfg.frames_per_step > 0 else n_frames

    for step in range(cfg.n_steps):
        alpha = alpha_ctrl.get_alpha(step)
        beta = cfg.beta_temporal * alpha if cfg.use_temporal_loss else 0.0

        # Sample cameras and frames
        n_sample = min(cfg.n_train_cameras, len(train_indices))
        cam_perm = torch.randperm(len(train_indices), generator=rng)[:n_sample]
        sample_vm = train_viewmats_all[cam_perm]
        sample_Ks = train_Ks[cam_perm]
        frame_perm = torch.randperm(n_frames, generator=rng)[:fps]

        total_loss = torch.tensor(0.0, device=device)
        total_render = 0.0
        total_snap_v = 0.0

        for fi in frame_perm:
            i = fi.item()
            rendered = _render_frame(means_params[i], frame_data_list[i],
                                     sample_vm, sample_Ks, train_W, train_H)
            target = ref_train[i][cam_perm]

            l_render = render_loss(rendered, target)
            l_snap = grid_snap_loss(
                means_params[i], means_min, means_max,
                step=step, total_steps=cfg.n_steps, weights=imp_w[i],
            )
            frame_loss = l_render + alpha * l_snap
            total_render += l_render.item()
            total_snap_v += l_snap.item()

            if beta > 0 and i > 0:
                l_temp = temporal_lo_consistency_loss(
                    means_params[i], means_params[max(0, i - 1)].data,
                    means_min, means_max,
                )
                frame_loss += beta * l_temp

            total_loss += frame_loss

        total_loss = total_loss / fps
        optimizer.zero_grad()
        total_loss.backward()

        # Per-Gaussian norm clipping
        for p in means_params:
            if p.grad is not None:
                gn = p.grad.norm(dim=-1, keepdim=True)
                p.grad.mul_(cfg.grad_clip_max / gn.clamp(min=cfg.grad_clip_max))

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            sr = sum(snap_rate(p, means_min, means_max) for p in means_params) / n_frames

        # Periodic eval
        if (step + 1) % cfg.eval_interval == 0:
            with torch.no_grad():
                all_psnrs: list[float] = []
                for i in range(n_frames):
                    rendered = _render_frame(
                        means_params[i], frame_data_list[i],
                        eval_viewmats, eval_Ks, eval_W, eval_H,
                    )
                    all_psnrs.extend(compute_psnr_per_view(rendered, ref_eval[i]))
                worst_psnr = min(all_psnrs)
                mean_psnr = sum(all_psnrs) / len(all_psnrs)
                d_entropy = delta_lo_entropy(
                    [p.data for p in means_params], means_min, means_max,
                )

            worst_drop = baseline_worst - worst_psnr
            logger.info(
                f"  Chunk {chunk_idx} step {step+1}: "
                f"worst_PSNR={worst_psnr:.1f} (Δ={worst_drop:+.1f}), "
                f"snap={sr:.3f}, δ_ent={d_entropy:.2f}, α={alpha:.1e}"
            )

            tsv.log({
                "chunk": chunk_idx, "step": step + 1,
                "snap_rate": f"{sr:.4f}", "worst_psnr": f"{worst_psnr:.1f}",
                "mean_psnr": f"{mean_psnr:.1f}", "delta_entropy": f"{d_entropy:.2f}",
                "alpha": f"{alpha:.2e}", "L_render": f"{total_render/fps:.4f}",
                "L_snap": f"{total_snap_v/fps:.4f}",
            })

            # Quality gate: absolute worst-view PSNR floor
            if worst_psnr < cfg.psnr_floor_db:
                logger.warning(f"  Chunk {chunk_idx}: PSNR gate at step {step+1}! drop={worst_drop:.2f}")
                checkpoint.restore(means_params)
                if not alpha_ctrl.quality_violation():
                    logger.warning(f"  Chunk {chunk_idx}: max halvings, stopping")
                    break
                logger.info(f"  Chunk {chunk_idx}: halved alpha → {alpha_ctrl.alpha_max:.1e}")
            elif sr > best_snap:
                best_snap = sr
                checkpoint.save(means_params, worst_psnr, 1.0)

            # Early stop
            snap_history.append(sr)
            if len(snap_history) >= cfg.snap_patience:
                if snap_history[-1] - snap_history[-cfg.snap_patience] < cfg.min_snap_gain:
                    logger.info(f"  Chunk {chunk_idx}: snap plateaued at {sr:.3f}")
                    break

        # Compression bench
        if (step + 1) % cfg.bench_interval == 0:
            with torch.no_grad():
                zb = zstd_compressed_size([p.data for p in means_params], means_min, means_max)
                raw = sum(p.numel() for p in means_params)
            logger.info(f"  Chunk {chunk_idx} step {step+1}: zstd={zb:,} ({raw/max(zb,1):.1f}x)")

    # Final metrics
    with torch.no_grad():
        final_sr = sum(snap_rate(p, means_min, means_max) for p in means_params) / n_frames
        final_de = delta_lo_entropy([p.data for p in means_params], means_min, means_max)
        final_zstd = zstd_compressed_size([p.data for p in means_params], means_min, means_max)
        final_psnrs: list[float] = []
        for i in range(n_frames):
            rendered = _render_frame(means_params[i], frame_data_list[i],
                                     eval_viewmats, eval_Ks, eval_W, eval_H)
            final_psnrs.extend(compute_psnr_per_view(rendered, ref_eval[i]))
        final_worst = min(final_psnrs)
        disp = sum((means_params[i] - chunk_frames[i].means).norm(dim=-1).mean().item()
                    for i in range(n_frames)) / n_frames

    logger.info(
        f"  Chunk {chunk_idx} final: worst_PSNR={final_worst:.1f} "
        f"(Δ={baseline_worst-final_worst:+.1f}), snap={final_sr:.3f}, "
        f"δ_ent={final_de:.2f}, zstd={final_zstd:,}, disp={disp:.6f}"
    )

    for i in range(n_frames):
        save_ply_with_new_means(ply_paths[i], output_dir / ply_paths[i].name, means_params[i].data)

    return {
        "worst_psnr_drop": baseline_worst - final_worst,
        "snap_rate": final_sr, "delta_lo_entropy": final_de,
        "zstd_bytes": final_zstd, "displacement": disp,
    }


def main(cfg: FinetuneConfig) -> None:
    """Run grid-snap fine-tuning."""
    t0 = time.time()
    logger.info(f"Grid-snap fine-tuning: {cfg.input_dir} → {cfg.output_dir}")
    logger.info(f"Config: steps={cfg.n_steps}, lr={cfg.lr_means}, alpha={cfg.alpha_max}")

    frames, ply_paths = load_ply_sequence(cfg.input_dir, cfg.device, cfg.max_frames)
    means_min, means_max = compute_global_ranges(frames)
    logger.info(f"Global ranges: min={means_min.cpu().numpy()}, max={means_max.cpu().numpy()}")

    cameras = SyntheticCameras.from_scene(
        frames[0].means, n_cameras=cfg.n_cameras,
        height=cfg.eval_resolution[0], width=cfg.eval_resolution[1], device=cfg.device,
    )
    logger.info(f"Generated {cameras.n_cameras} cameras (radius_scale=0.5)")

    with torch.no_grad():
        init_zstd = zstd_compressed_size(
            [f.means for f in frames[:cfg.chunk_size]], means_min, means_max)
        init_de = delta_lo_entropy(
            [f.means for f in frames[:cfg.chunk_size]], means_min, means_max)
        logger.info(f"Initial chunk0: zstd={init_zstd:,}, δ_entropy={init_de:.2f}")

    tsv = TSVLogger(cfg.log_dir / "finetune.tsv")
    n_chunks = math.ceil(len(frames) / cfg.chunk_size)
    all_metrics: list[dict[str, float]] = []

    for ci in range(n_chunks):
        s, e = ci * cfg.chunk_size, min((ci + 1) * cfg.chunk_size, len(frames))
        logger.info(f"Chunk {ci}/{n_chunks} (frames {s}-{e-1})...")
        metrics = finetune_chunk(
            frames[s:e], ply_paths[s:e], cfg.output_dir, ci,
            means_min, means_max, cameras, cfg, tsv,
        )
        all_metrics.append(metrics)
        torch.cuda.empty_cache()

    tsv.close()
    elapsed = time.time() - t0
    avg_drop = sum(m["worst_psnr_drop"] for m in all_metrics) / len(all_metrics)
    avg_snap = sum(m["snap_rate"] for m in all_metrics) / len(all_metrics)
    avg_de = sum(m["delta_lo_entropy"] for m in all_metrics) / len(all_metrics)
    total_zstd = sum(m["zstd_bytes"] for m in all_metrics)

    logger.info("=" * 60)
    logger.info(f"Done in {elapsed:.0f}s | PSNR drop: {avg_drop:.2f} dB | "
                f"snap: {avg_snap:.3f} | δ_entropy: {avg_de:.2f} | zstd: {total_zstd:,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    cfg = tyro.cli(FinetuneConfig)
    main(cfg)
