"""Bounded candidate matching with explicit rejection of ambiguous links.

Accepted links are estimates, not verified physical identities. Row numbers and
chunk boundaries carry no correspondence information in this prototype.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class MatchSettings:
    radius: float
    candidates: int = 8
    ratio: float = 0.7
    color_limit: float = 0.25
    log_scale_limit: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Matching radius must be finite and positive")
        if not 2 <= self.candidates <= 32 or not 0 < self.ratio < 1:
            raise ValueError("Invalid candidate count or ambiguity ratio")
        if not all(np.isfinite(x) and x > 0 for x in (self.color_limit, self.log_scale_limit)):
            raise ValueError("Feature limits must be finite and positive")


@dataclass
class Features:
    positions: np.ndarray
    colors: np.ndarray
    log_scales: np.ndarray

    def validate(self):
        shape = (len(self.positions), 3)
        for values in (self.positions, self.colors, self.log_scales):
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError("Expected finite Nx3 tracking features")


def _best(source, target, source_positions, target_positions, settings):
    """Local spatial candidates, then appearance/shape scoring and ambiguity gate."""
    count = len(source_positions)
    if not count or not len(target_positions):
        return np.full(count, -1, np.int64)
    tree = cKDTree(target_positions)
    distances, candidates = tree.query(
        source_positions,
        k=settings.candidates,
        distance_upper_bound=settings.radius,
        workers=1,
    )
    valid = candidates < len(target_positions)
    safe = np.minimum(candidates, len(target_positions) - 1)
    color = np.linalg.norm(source.colors[:, None] - target.colors[safe], axis=2)
    # Sorted axes reduce sensitivity to equivalent rotated scale parameterizations.
    scale = np.max(np.abs(source.log_scales[:, None] - target.log_scales[safe]), axis=2)
    valid &= (color <= settings.color_limit) & (scale <= settings.log_scale_limit)
    costs = (distances / settings.radius) ** 2
    costs += (color / settings.color_limit) ** 2 + 0.25 * (scale / settings.log_scale_limit) ** 2
    costs[~valid] = np.inf
    ordering = np.argsort(costs, axis=1)
    rows = np.arange(count)
    best = costs[rows, ordering[:, 0]]
    second = costs[rows, ordering[:, 1]]
    accepted = np.isfinite(best) & (best < settings.ratio * second)
    # Equal perfect candidates are ambiguous too (0 < 0 is false).
    return np.where(accepted, candidates[rows, ordering[:, 0]], -1)


def match(previous, current, velocity, settings):
    """Return current-row -> previous-row links; -1 means no confident estimate."""
    previous.validate()
    current.validate()
    if velocity.shape != previous.positions.shape or not np.isfinite(velocity).all():
        raise ValueError("Invalid velocity array")
    predicted = previous.positions + velocity
    forward = _best(previous, current, predicted, current.positions, settings)
    reverse = _best(current, previous, current.positions, predicted, settings)
    links = np.full(len(current.positions), -1, np.int64)
    rows = np.flatnonzero(forward >= 0)
    mutual = reverse[forward[rows]] == rows
    links[forward[rows[mutual]]] = rows[mutual]
    return links


class Tracker:
    """Consecutive-frame tracks; breaks instead of inventing occlusion recovery."""

    def __init__(self, settings):
        self.settings = settings
        self.previous = None
        self.ids = np.empty(0, np.int32)
        self.velocity = np.empty((0, 3), np.float32)
        self.next_id = 0

    def update(self, current):
        current.validate()
        links = (
            match(self.previous, current, self.velocity, self.settings)
            if self.previous is not None
            else np.full(len(current.positions), -1, np.int64)
        )
        linked = links >= 0
        result = np.empty(len(links), np.int32)
        result[linked] = self.ids[links[linked]]
        births = int((~linked).sum())
        if self.next_id + births >= np.iinfo(np.int32).max:
            raise ValueError("Too many track fragments")
        result[~linked] = np.arange(self.next_id, self.next_id + births, dtype=np.int32)
        self.next_id += births
        velocity = np.zeros_like(current.positions)
        if linked.any():
            velocity[linked] = current.positions[linked] - self.previous.positions[links[linked]]
        self.previous, self.ids, self.velocity = current, result, velocity
        return result, links
