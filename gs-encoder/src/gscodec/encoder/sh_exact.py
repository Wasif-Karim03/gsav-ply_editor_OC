"""Exact repeated-vector palette and bounded weighted scalar statistics."""

import numpy as np


def weighted_histogram(values, multiplicities, n_levels, alpha):
    """Match the expanded scalar quantizer histogram without repeating values.

    A bin boundary is found by binary searching its rank in the conceptual
    sorted expanded array. This retains duplicate counts and adaptive rank bins.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    multiplicities = np.asarray(multiplicities, dtype=np.int64).ravel()
    if values.shape != multiplicities.shape or np.any(multiplicities < 0):
        raise ValueError("Invalid scalar multiplicities")
    keep = multiplicities > 0
    order = np.argsort(values[keep])
    values, multiplicities = values[keep][order], multiplicities[keep][order]
    if not len(values):
        return np.zeros(n_levels, dtype=np.float32), None
    cumulative = np.cumsum(multiplicities)
    total = int(cumulative[-1])
    vmin, vmax = values[0], values[-1]
    span = vmax - vmin
    if span < 1e-20:
        return np.full(n_levels, vmin, dtype=np.float32), None

    def value_at(rank):
        return values[np.searchsorted(cumulative, rank, side="right")]

    bins_count = min(n_levels * 2, total)
    iqr = value_at(int(total * 0.75)) - value_at(int(total * 0.25))
    beta = max(0.5, min(0.999, 1.0 - iqr / span))
    cuts = [0]
    for boundary in range(1, bins_count):
        lo, hi = cuts[-1], total
        while lo < hi:
            mid = (lo + hi) // 2
            blended = beta * (mid / total) + (1.0 - beta) * ((value_at(mid) - vmin) / span)
            if min(bins_count - 1, int(np.floor(bins_count * blended))) < boundary:
                lo = mid + 1
            else:
                hi = mid
        cuts.append(lo)
    cuts.append(total)
    cuts = np.asarray(cuts, dtype=np.int64)
    counts = np.diff(cuts).astype(np.float64)
    prefix = np.r_[0.0, np.cumsum(values * multiplicities)]
    indices = np.searchsorted(cumulative, cuts, side="right")
    prior_counts = np.r_[0, cumulative][indices]
    partial = np.where(indices < len(values), values[np.minimum(indices, len(values) - 1)], 0)
    prefix_at_cut = prefix[indices] + (cuts - prior_counts) * partial
    sums = np.diff(prefix_at_cut)
    safe_counts = np.where(counts > 0, counts, 1.0)
    centers = np.where(
        counts > 0, sums / safe_counts, vmin + (np.arange(bins_count) + 0.5) / bins_count * span
    )
    weights = np.where(counts > 0, np.power(counts, alpha), 0.0)
    return None, (centers, weights)


def exact_palette(flat_frames, presence, limit, progress=None):
    """Return all distinct active vectors and exact labels, or None if too many.

    Float32 bit patterns are keys; no tolerance, rounding, sampling or merging.
    Temporary storage is bounded by one frame plus the palette and output labels.
    """
    lookup, vectors, frequencies, labels = {}, [], [], []
    dims = int(np.prod(flat_frames[0].shape[1:]))
    for index, (frame, mask) in enumerate(zip(flat_frames, presence, strict=True)):
        active = np.ascontiguousarray(frame[mask], dtype=np.float32).reshape(-1, dims)
        keys = active.view(np.dtype((np.void, dims * 4))).reshape(-1)
        unique, first, inverse, counts = np.unique(
            keys, return_index=True, return_inverse=True, return_counts=True
        )
        ids = np.empty(len(unique), dtype=np.uint16)
        for j, vector_key in enumerate(unique):
            key = vector_key.tobytes()
            slot = lookup.get(key)
            if slot is None:
                if len(vectors) >= limit:
                    return None
                slot = len(vectors)
                lookup[key] = slot
                vectors.append(active[first[j]].copy())
                frequencies.append(0)
            ids[j] = slot
            frequencies[slot] += int(counts[j])
        frame_labels = np.zeros(len(mask), dtype=np.uint16)
        frame_labels[mask] = ids[inverse]
        labels.append(frame_labels)
        if progress is not None and (index == 0 or (index + 1) % 30 == 0):
            progress(f"Checking exact SH palette: {index + 1}/{len(flat_frames)} frames")
    if not vectors:
        vectors, frequencies = [np.zeros(dims, np.float32)], [1]
    palette = np.stack(vectors)
    scalar_values, scalar_inverse = np.unique(palette.ravel(), return_inverse=True)
    scalar_counts = np.bincount(
        scalar_inverse, weights=np.repeat(np.asarray(frequencies, np.int64), dims)
    ).astype(np.int64)
    return palette, labels, scalar_values, scalar_counts
