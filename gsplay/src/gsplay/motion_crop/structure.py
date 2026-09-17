"""Local shape evidence for motion correspondence, independent of storage rows."""

import numpy as np
from scipy.spatial import cKDTree


def descriptors(frame, settings):
    if len(frame.positions) < 5:
        return None
    distance, _ = cKDTree(frame.positions).query(frame.positions, k=5, workers=1)
    shape = np.log(np.maximum(distance[:, 1:], settings.radius * 1e-6) / settings.radius)
    return np.concatenate(
        (shape, frame.colors / settings.color_limit, frame.log_scales / settings.log_scale_limit),
        axis=1,
    )


def structural_links(previous, current, settings):
    """Mutual descriptor proposals, accepted only with local rigidity support."""
    a, b = descriptors(previous, settings), descriptors(current, settings)
    result = np.full(len(current.positions), -1, np.int64)
    if a is None or b is None:
        return result

    # Unique shape/appearance groups avoid high-dimensional nearest-neighbor
    # searches. Shifted grids reduce bin-edge loss; collisions are rejected.
    forward = np.full(len(a), -1, np.int64)
    conflict = np.zeros(len(a), bool)
    for shift in (0.0, 0.5):
        bins = np.ascontiguousarray(np.floor(np.concatenate((a, b)) / 0.05 + shift), dtype=np.int64)
        keys = bins.view(np.dtype((np.void, bins.dtype.itemsize * bins.shape[1]))).reshape(-1)
        _, inverse = np.unique(keys, return_inverse=True)
        size = int(inverse.max()) + 1
        ca = np.bincount(inverse[: len(a)], minlength=size)
        cb = np.bincount(inverse[len(a) :], minlength=size)
        target = np.full(size, -1, np.int64)
        target[inverse[len(a) :]] = np.arange(len(b))
        groups = inverse[: len(a)]
        rows = np.flatnonzero((ca[groups] == 1) & (cb[groups] == 1))
        candidates = target[groups[rows]]
        conflict[rows] |= (forward[rows] >= 0) & (forward[rows] != candidates)
        forward[rows] = candidates
    forward[conflict] = -1
    rows = np.flatnonzero(forward >= 0)
    used = np.bincount(forward[rows], minlength=len(b))
    rows = rows[used[forward[rows]] == 1]
    candidate = forward[rows]
    movement = np.linalg.norm(previous.positions[rows] - current.positions[candidate], axis=1)
    rows, candidate = rows[movement <= settings.radius], candidate[movement <= settings.radius]
    provisional = np.full(len(previous.positions), -1, np.int64)
    provisional[rows] = candidate
    if not len(rows):
        return result
    d0, neighbors = cKDTree(previous.positions).query(previous.positions[rows], k=5, workers=1)
    mapped = provisional[neighbors[:, 1:]]
    valid = mapped >= 0
    d1 = np.linalg.norm(
        current.positions[candidate, None] - current.positions[np.maximum(mapped, 0)], axis=2
    )
    consistent = valid & (
        np.abs(d1 - d0[:, 1:]) <= 0.05 * np.maximum(d0[:, 1:], settings.radius * 0.001)
    )
    supported = consistent.sum(axis=1) >= 3
    result[candidate[supported]] = rows[supported]
    return result


def appearance_distinctive(frame, settings):
    """Conservative gate for position-based fallback in indistinct neighborhoods."""
    if len(frame.positions) < 2:
        return np.ones(len(frame.positions), bool)
    k = min(8, len(frame.positions))
    distance, neighbors = cKDTree(frame.positions).query(frame.positions, k=k, workers=1)
    color = np.linalg.norm(frame.colors[:, None] - frame.colors[neighbors[:, 1:]], axis=2)
    scale = np.max(np.abs(frame.log_scales[:, None] - frame.log_scales[neighbors[:, 1:]]), axis=2)
    ambiguous = (distance[:, 1:] <= settings.radius) & (color < 0.05) & (scale < 0.15)
    return ~ambiguous.any(axis=1)
