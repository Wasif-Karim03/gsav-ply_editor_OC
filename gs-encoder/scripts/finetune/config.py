"""Configuration for grid-snap fine-tuning."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FinetuneConfig:
    """Hyperparameters for grid-snap fine-tuning.

    Attributes:
        input_dir: Directory containing PLY files.
        output_dir: Directory for fine-tuned PLY output.
        device: Torch device.
        chunk_size: Frames per chunk (matches encoder GOP).
        n_steps: Optimization steps per chunk.
        lr_means: Learning rate for means parameters.
        lr_min: Minimum LR for cosine annealing.
        lr_secondary: Learning rate for scales/opacity (0 = frozen).
        alpha_max: Maximum grid-snap loss weight.
        alpha_warmup_start: Step to begin alpha warm-up.
        alpha_warmup_end: Step to reach alpha_max.
        beta_temporal: Weight for temporal consistency loss relative to alpha.
        psnr_budget_db: Max allowable PSNR drop from baseline.
        ssim_budget: Max allowable SSIM drop from baseline.
        max_alpha_halvings: Max times to halve alpha before stopping.
        n_cameras: Total synthetic cameras for reference rendering.
        n_train_cameras: Cameras sampled per training step.
        n_eval_cameras: Cameras for periodic quality checks.
        train_resolution: Resolution (H, W) for training renders.
        eval_resolution: Resolution (H, W) for eval renders.
        frames_per_step: Frames sampled per gradient step (0 = all).
        eval_interval: Steps between quality checks.
        bench_interval: Steps between compression benchmarks.
        grad_clip_max: Max gradient norm per Gaussian.
        snap_patience: Eval intervals without improvement before early stop.
        min_snap_gain: Minimum snap_rate improvement per patience window.
        max_frames: Max frames to process (0 = all).
        use_importance_weights: Weight snap loss by inverse Gaussian importance.
        use_temporal_loss: Add temporal lo consistency loss.
        log_dir: Directory for TSV experiment logs.
    """

    input_dir: Path = field(default_factory=lambda: Path("D:/dymensium_soccer/plys"))
    output_dir: Path = field(default_factory=lambda: Path("D:/dymensium_soccer/plys_finetuned"))
    device: str = "cuda:0"

    # Chunking
    chunk_size: int = 30

    # Optimization
    n_steps: int = 800
    lr_means: float = 1e-3
    lr_min: float = 1e-4
    lr_secondary: float = 0.0
    alpha_max: float = 10.0
    alpha_warmup_start: int = 10
    alpha_warmup_end: int = 100
    beta_temporal: float = 0.3

    # Quality gates (absolute floor, not relative drop)
    psnr_floor_db: float = 40.0
    max_alpha_halvings: int = 3

    # Camera setup
    n_cameras: int = 16
    n_train_cameras: int = 4
    n_eval_cameras: int = 4
    train_resolution: tuple[int, int] = (512, 512)
    eval_resolution: tuple[int, int] = (512, 512)

    # Temporal sampling
    frames_per_step: int = 8

    # Logging intervals
    eval_interval: int = 50
    bench_interval: int = 200

    # Gradient clipping
    grad_clip_max: float = 0.5

    # Early stopping
    snap_patience: int = 5
    min_snap_gain: float = 0.005

    # Limits
    max_frames: int = 0

    # Feature flags
    use_importance_weights: bool = True
    use_temporal_loss: bool = True

    # Logging
    log_dir: Path = field(default_factory=lambda: Path("D:/dymensium_soccer/finetune_logs"))
