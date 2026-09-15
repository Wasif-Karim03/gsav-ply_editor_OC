"""Adaptive alpha controller with cosine warm-up and quality gating."""

import math

import torch
from torch import Tensor


class AlphaController:
    """Controls the grid-snap loss weight with warm-up and quality gating.

    Phases:
        1. Warm-up: alpha ramps from 0 to alpha_max via cosine schedule.
        2. Active: alpha stays at current level; halved if quality drops.
        3. Stopped: alpha frozen after max halvings.

    Attributes:
        alpha_max: Current maximum alpha (reduced by halvings).
        n_halvings: Number of times alpha has been halved.
    """

    def __init__(
        self,
        alpha_max: float,
        warmup_start: int,
        warmup_end: int,
        max_halvings: int = 3,
    ):
        """Initialize alpha controller.

        Args:
            alpha_max: Initial maximum alpha value.
            warmup_start: Step to begin warm-up (alpha=0 before this).
            warmup_end: Step to reach alpha_max.
            max_halvings: Max times to halve alpha before stopping.
        """
        self.alpha_max = alpha_max
        self.warmup_start = warmup_start
        self.warmup_end = warmup_end
        self.max_halvings = max_halvings
        self.n_halvings = 0
        self._stopped = False

    def get_alpha(self, step: int) -> float:
        """Get alpha value for the given step.

        Args:
            step: Current optimization step.

        Returns:
            Alpha weight for grid-snap loss.
        """
        if self._stopped:
            return 0.0

        if step < self.warmup_start:
            return 0.0

        if step >= self.warmup_end:
            return self.alpha_max

        # Cosine warm-up
        progress = (step - self.warmup_start) / max(self.warmup_end - self.warmup_start, 1)
        return self.alpha_max * 0.5 * (1.0 - math.cos(math.pi * progress))

    def quality_violation(self) -> bool:
        """Called when quality gate is violated. Halves alpha.

        Returns:
            True if still active (more halvings available), False if stopped.
        """
        self.n_halvings += 1
        if self.n_halvings >= self.max_halvings:
            self._stopped = True
            return False

        self.alpha_max *= 0.5
        return True

    @property
    def is_stopped(self) -> bool:
        return self._stopped


class Checkpoint:
    """Stores a parameter checkpoint for quality-gated rollback."""

    def __init__(self) -> None:
        self.state: dict[int, Tensor] | None = None
        self.psnr: float = 0.0
        self.ssim: float = 0.0

    def save(self, params: list[Tensor], psnr: float, ssim: float) -> None:
        """Save current parameter state.

        Args:
            params: List of means parameter tensors.
            psnr: Current PSNR.
            ssim: Current SSIM.
        """
        self.state = {i: p.data.clone() for i, p in enumerate(params)}
        self.psnr = psnr
        self.ssim = ssim

    def restore(self, params: list[Tensor]) -> None:
        """Restore parameters from checkpoint.

        Args:
            params: List of means parameter tensors to restore into.
        """
        if self.state is None:
            return
        for i, p in enumerate(params):
            if i in self.state:
                p.data.copy_(self.state[i])
