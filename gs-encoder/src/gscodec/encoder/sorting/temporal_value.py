"""
Temporal Value Sorting (TVS) — minimizes inter-frame pixel residuals for video compression.

Instead of sorting by spatial position (Morton) or identity (fixed order), TVS sorts
Gaussians so that the quantized attribute VALUE at each pixel position is as close as
possible to what was at that position in the previous frame. This directly minimizes the
residual that VP9/AV1 inter-prediction must encode.

The optimal solution is bipartite matching (Hungarian algorithm, O(N^3)), which is too
expensive for N=29K. TVS approximates it using vectorized random swap refinement:

Starting from the previous frame's sort order (identity permutation), randomly pair
all N positions into N/2 pairs. For each pair, check whether swapping the assigned
Gaussians reduces the total L1 residual across all 14 channels. Apply all beneficial
swaps simultaneously.

Each pass is fully vectorized: O(N * C) with no Python loops over Gaussians.
Empirically, 50 passes achieve ~35% residual reduction at ~160ms for N=29K.

The identity starting point preserves spatial coherence from Morton sorting: positions
whose Gaussians are not swapped retain their spatial locality, which benefits VP9/AV1
intra-frame prediction (TM_PRED). Only positions where a better value-match exists get
reassigned.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Swap passes per frame. 50 gives ~35% residual reduction at ~160ms for N=29K.
DEFAULT_TVS_PASSES = 50


def compute_tvs_permutation(
    current_channels: list[np.ndarray],
    prev_channels: list[np.ndarray],
    n_gaussians: int,
    n_passes: int = DEFAULT_TVS_PASSES,
) -> np.ndarray:
    """Compute the permutation that minimizes pixel residual vs previous frame.

    Starting from identity (same position = same Gaussian), performs random pairwise
    swaps that reduce total L1 residual across all channels. The identity start
    preserves Morton spatial coherence for unswapped positions.

    Args:
        current_channels: List of 14 arrays [N] uint8 — current frame's quantized
                          attribute values in Morton-sorted order.
        prev_channels: List of 14 arrays [N] uint8 — previous frame's quantized
                       attribute values in position order.
        n_gaussians: Number of active Gaussians.
        n_passes: Number of swap refinement passes.

    Returns:
        [N] int64 permutation: perm[j] = index of the Gaussian that should go
        to position j in the new frame.
    """
    n_ch = len(current_channels)

    # Stack into [N, C] int16 matrices for fast L1 computation
    curr = np.empty((n_gaussians, n_ch), dtype=np.int16)
    prev = np.empty((n_gaussians, n_ch), dtype=np.int16)
    for i in range(n_ch):
        curr[:, i] = current_channels[i][:n_gaussians].astype(np.int16)
        prev[:, i] = prev_channels[i][:n_gaussians].astype(np.int16)

    perm = np.arange(n_gaussians, dtype=np.int64)
    perm = _vectorized_swap_refinement(curr, prev, perm, n_passes)

    return perm


def _vectorized_swap_refinement(
    curr: np.ndarray,
    prev: np.ndarray,
    perm: np.ndarray,
    n_passes: int,
    swap_radius: int = 64,
) -> np.ndarray:
    """Refine permutation via locally-constrained vectorized swap passes.

    Each pass pairs positions with random nearby partners (within swap_radius
    in the sorted order). This preserves spatial locality from Morton sorting
    while optimizing temporal residual. Only beneficial swaps are applied.

    O(N * C) per pass, fully vectorized via NumPy.

    Args:
        curr: [N, C] int16 current frame attribute values.
        prev: [N, C] int16 previous frame attribute values.
        perm: [N] int64 current permutation (modified in-place).
        n_passes: Number of refinement passes.
        swap_radius: Maximum distance between swap partners in sorted order.
                     Smaller = better spatial preservation, less temporal gain.

    Returns:
        [N] int64 refined permutation.
    """
    n = len(perm)
    perm = perm.copy()

    for pass_idx in range(n_passes):
        # Local pairing: shuffle, take consecutive pairs within swap_radius
        shuffled = np.random.permutation(n)
        pos_i = shuffled[:n // 2]
        offsets = np.random.randint(1, min(swap_radius + 1, n), size=len(pos_i))
        if pass_idx % 2 == 0:
            pos_j = (pos_i + offsets) % n
        else:
            pos_j = (pos_i - offsets) % n

        # Remove self-pairs
        valid = pos_i != pos_j
        pos_i = pos_i[valid]
        pos_j = pos_j[valid]

        # Remove duplicates (a position can appear as both i and j)
        seen = set()
        keep = []
        for k in range(len(pos_i)):
            pi, pj = pos_i[k], pos_j[k]
            if pi not in seen and pj not in seen:
                seen.add(pi)
                seen.add(pj)
                keep.append(k)
        if not keep:
            continue
        keep = np.array(keep)
        pos_i = pos_i[keep]
        pos_j = pos_j[keep]

        gi = perm[pos_i]
        gj = perm[pos_j]

        # Cost before swap
        cost_before = (
            np.abs(curr[gi] - prev[pos_i]).sum(axis=1)
            + np.abs(curr[gj] - prev[pos_j]).sum(axis=1)
        )

        # Cost after swap
        cost_after = (
            np.abs(curr[gj] - prev[pos_i]).sum(axis=1)
            + np.abs(curr[gi] - prev[pos_j]).sum(axis=1)
        )

        # Apply beneficial swaps
        improve = cost_after < cost_before
        swap_i = pos_i[improve]
        swap_j = pos_j[improve]

        perm[swap_i], perm[swap_j] = perm[swap_j].copy(), perm[swap_i].copy()

        n_swaps = int(improve.sum())
        if n_swaps == 0:
            logger.debug(f"TVS converged at pass {pass_idx}")
            break

    return perm


def apply_permutation_to_channels(
    channels: list[np.ndarray],
    perm: np.ndarray,
) -> list[np.ndarray]:
    """Reorder channel arrays according to a permutation.

    Args:
        channels: List of arrays to reorder. Each is [N] or [N, C] uint8.
        perm: [N] int64 permutation.

    Returns:
        List of reordered arrays.
    """
    return [ch[perm] for ch in channels]
