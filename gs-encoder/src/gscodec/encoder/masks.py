"""Mask-aware encoder preparation; active samples are never replaced."""

import numpy as np
import torch
from gsply import GSTensor

ATTRIBUTES = ("means", "scales", "quats", "opacities", "sh0", "shN")


def sanitize_frame(frame: GSTensor) -> GSTensor:
    """Validate active attributes and replace inactive placeholders on a copy."""
    mask = frame.masks.reshape(-1).bool()
    if len(mask) != len(frame.means):
        raise ValueError("Mask length must match Gaussian count")
    for name in ATTRIBUTES:
        values = getattr(frame, name)
        if values is not None and not torch.isfinite(values[mask]).all():
            raise ValueError(f"Active Gaussian {name} must be finite")
    if mask.all():
        return frame
    result = frame.clone()
    for name in ATTRIBUTES:
        values = getattr(result, name)
        if values is not None:
            values[~mask] = 0
    result.quats[~mask, 0] = 1
    return result


def active_frames(frames: list[GSTensor]) -> list[GSTensor]:
    """Samples for range/precision statistics, with a finite all-inactive fallback."""
    selected = [f[f.masks.reshape(-1).bool()] for f in frames if f.masks.any()]
    if selected:
        return selected
    return [sanitize_frame(frames[0])[:1]]


def first_active_sort_frame(frames: list[GSTensor]) -> GSTensor:
    """Position-only representative for Morton sorting within one GOP.

    Other attributes are not representatives and must not be encoded from this
    frame. In particular, source SH degrees may differ between frames.
    """
    representative = sanitize_frame(frames[0]).clone()
    found = frames[0].masks.reshape(-1).bool().clone()
    for frame in frames[1:]:
        take = frame.masks.reshape(-1).bool() & ~found
        representative.means[take] = frame.means[take]
        found |= take
    representative.masks = found
    return representative


def reuse_inactive_rows(values, presence: list[np.ndarray]) -> None:
    """Reuse quantized rows/labels in-place inside a single GOP."""
    missing = ~presence[0].copy()
    for t in range(1, len(presence)):
        first = missing & presence[t]
        values[0][first] = values[t][first]
        missing &= ~presence[t]
    for t in range(1, len(presence)):
        values[t][~presence[t]] = values[t - 1][~presence[t]]
