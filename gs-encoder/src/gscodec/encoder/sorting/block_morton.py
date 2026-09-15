"""
Temporally-damped hierarchical Morton sort for VP9/AV1 video compression.

Partitions the 2D atlas grid into coarse blocks aligned with VP9 superblock
boundaries (64x64). Between frames, Gaussians are assigned to the same coarse
block as the previous frame (temporal stability). Within each block, Gaussians
are Morton-sorted by their current 3D positions (spatial coherence).

This gives VP9 the best of both worlds:
- Inter prediction works because each superblock's population is stable
- TM_PRED works because within each block, Gaussians are spatially sorted

VP9 lossless uses 4x4 Walsh-Hadamard Transform on residuals after prediction.
TM_PRED (predicted = left + above - above_left) excels when spatially nearby
Gaussians have similar attribute values. By maintaining spatial coherence within
each superblock-aligned block, we minimize TM_PRED residual entropy.

Simultaneously, by keeping the same Gaussians in the same superblock across
frames, VP9's inter-frame motion search at 64x64/32x32 block level finds
low-cost matches, reducing P-frame overhead.
"""

import functools
import logging
import math
from typing import Any

import numpy as np
import torch
from gsply import GSTensor
from torch import Tensor

from gscodec.encoder.sorting.base import SortingStrategy
from gscodec.encoder.sorting.morton import MortonSortingStrategy

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def _morton_2d_positions(side: int) -> np.ndarray:
    """Compute 2D grid positions for each linear Gaussian index via Morton curve.

    Returns an array of shape [side*side, 2] where entry i gives the (row, col)
    grid position of the i-th Gaussian in sorted order.

    Args:
        side: Side length of the square grid.

    Returns:
        Array [side*side, 2] of (row, col) positions.
    """
    ys, xs = np.meshgrid(
        np.arange(side, dtype=np.uint32),
        np.arange(side, dtype=np.uint32),
        indexing="ij",
    )

    def part1by1(val: np.ndarray) -> np.ndarray:
        val = val & np.uint32(0x0000FFFF)
        val = (val | (val << 8)) & np.uint32(0x00FF00FF)
        val = (val | (val << 4)) & np.uint32(0x0F0F0F0F)
        val = (val | (val << 2)) & np.uint32(0x33333333)
        val = (val | (val << 1)) & np.uint32(0x55555555)
        return val

    codes = part1by1(xs.ravel()) | (part1by1(ys.ravel()) << np.uint32(1))
    morton_order = np.argsort(codes)  # [side*side]

    # morton_order[i] = flat grid index for Gaussian i
    positions = np.empty((side * side, 2), dtype=np.int64)
    positions[:, 0] = morton_order // side  # row
    positions[:, 1] = morton_order % side   # col
    return positions


class BlockMortonSortingStrategy(SortingStrategy):
    """Hierarchical block-stable Morton sort for video codec spatial coherence.

    Divides the 2D atlas grid into coarse blocks aligned with VP9 superblock
    boundaries. Frame 0 establishes the reference assignment. Subsequent frames
    reassign Gaussians to coarse blocks via 3D centroid nearest-neighbor, then
    Morton-sort within each block for spatial coherence.

    Args:
        block_side: Coarse block side length in pixels. Should match VP9
            superblock size (64) or a divisor thereof.
    """

    def __init__(self, block_side: int = 64) -> None:
        self.block_side = block_side
        self._morton_sorter = MortonSortingStrategy()
        # Temporal state
        self._prev_block_assignments: Tensor | None = None
        self._prev_means: Tensor | None = None
        self._grid_side: int = 0
        self._n_blocks_y: int = 0
        self._n_blocks_x: int = 0
        self._block_capacities: Tensor | None = None

    def reset(self) -> None:
        """Reset temporal state (call at chunk boundaries)."""
        self._prev_block_assignments = None
        self._prev_means = None
        self._block_capacities = None

    def sort(
        self,
        splats: GSTensor,
        **kwargs: Any,
    ) -> tuple[GSTensor, Tensor]:
        """Sort splats using block-stable hierarchical Morton ordering.

        On first call (or after reset), establishes reference ordering via
        standard Morton sort partitioned into coarse blocks. On subsequent
        calls, reassigns Gaussians to coarse blocks based on 3D centroid
        proximity to previous frame's blocks, then Morton-sorts within blocks.

        Args:
            splats: GSTensor with means attribute. [N, 3]
            **kwargs: Unused.

        Returns:
            Tuple of (sorted GSTensor, sorted indices tensor [N]).
        """
        if splats.means is None:
            raise ValueError("BlockMorton sorting requires 'means' in the GSTensor.")

        means: Tensor = splats.means  # [N, 3]
        n: int = means.shape[0]
        device = means.device

        if n == 0:
            return splats, torch.tensor([], dtype=torch.long, device=device)

        if self._prev_block_assignments is None:
            sorted_indices = self._init_reference(splats)
        else:
            sorted_indices = self._sort_subsequent(means)

        sorted_splats: GSTensor = splats[sorted_indices]
        return sorted_splats, sorted_indices

    def _compute_grid_layout(self, n: int) -> None:
        """Compute grid and block layout parameters.

        Args:
            n: Number of Gaussians.
        """
        self._grid_side = math.ceil(math.sqrt(n))
        if self._grid_side % 2 == 1:
            self._grid_side += 1

        self._n_blocks_y = math.ceil(self._grid_side / self.block_side)
        self._n_blocks_x = math.ceil(self._grid_side / self.block_side)

    def _compute_block_capacities(self, device: torch.device) -> Tensor:
        """Compute pixel capacity for each coarse block (vectorized).

        Edge blocks may have fewer pixels than block_side^2.

        Args:
            device: Torch device.

        Returns:
            Capacities tensor [n_blocks_y * n_blocks_x].
        """
        # Block heights: full blocks get block_side, last row may be smaller
        block_heights = torch.full(
            (self._n_blocks_y,), self.block_side, dtype=torch.long, device=device
        )
        last_h = self._grid_side - (self._n_blocks_y - 1) * self.block_side
        block_heights[-1] = last_h

        block_widths = torch.full(
            (self._n_blocks_x,), self.block_side, dtype=torch.long, device=device
        )
        last_w = self._grid_side - (self._n_blocks_x - 1) * self.block_side
        block_widths[-1] = last_w

        # Outer product gives capacity of each block [n_blocks_y, n_blocks_x]
        capacities = block_heights.unsqueeze(1) * block_widths.unsqueeze(0)
        return capacities.flatten()

    def _init_reference(self, splats: GSTensor) -> Tensor:
        """Establish reference ordering from first frame via Morton sort.

        Assigns each sorted Gaussian to a coarse block based on its 2D Morton
        grid position (vectorized).

        Args:
            splats: First frame GSTensor.

        Returns:
            Sorted indices [N].
        """
        means = splats.means
        n = means.shape[0]
        device = means.device

        self._compute_grid_layout(n)
        self._block_capacities = self._compute_block_capacities(device)

        # Morton sort the full frame
        _, morton_indices = self._morton_sorter.sort(splats)

        # Assign block IDs based on 2D Morton-curve grid positions (vectorized)
        positions = _morton_2d_positions(self._grid_side)  # [side*side, 2] numpy
        rows = torch.from_numpy(positions[:n, 0]).to(device)  # [N]
        cols = torch.from_numpy(positions[:n, 1]).to(device)  # [N]

        block_y = (rows // self.block_side).clamp(max=self._n_blocks_y - 1)
        block_x = (cols // self.block_side).clamp(max=self._n_blocks_x - 1)
        block_assignments = block_y * self._n_blocks_x + block_x  # [N]

        sorted_means = means[morton_indices]
        self._prev_block_assignments = block_assignments
        self._prev_means = sorted_means

        return morton_indices

    def _sort_subsequent(self, means: Tensor) -> Tensor:
        """Sort a subsequent frame with block-stable assignment.

        1. Compute 3D centroid of each coarse block from previous frame
        2. Assign each Gaussian to nearest centroid (capacity-constrained)
        3. Within each block, Morton-sort by current 3D means

        Args:
            means: Current frame 3D means [N, 3].

        Returns:
            Sorted indices [N].
        """
        # Step 1: Compute previous frame's block centroids
        centroids = self._compute_block_centroids()  # [n_blocks, 3]

        # Step 2: Assign Gaussians to blocks via nearest centroid
        block_assignments = self._assign_to_blocks(means, centroids)  # [N]

        # Step 3: Within each block, Morton-sort by current 3D means
        sorted_indices = self._sort_within_blocks(means, block_assignments)

        # Update state for next frame
        self._prev_block_assignments = block_assignments[sorted_indices]
        self._prev_means = means[sorted_indices]

        return sorted_indices

    def _compute_block_centroids(self) -> Tensor:
        """Compute 3D centroid of each coarse block from previous frame.

        Uses scatter_add for fully vectorized centroid computation.

        Returns:
            Centroids tensor [n_blocks, 3].
        """
        n_blocks = self._n_blocks_y * self._n_blocks_x
        device = self._prev_means.device

        centroids = torch.zeros(n_blocks, 3, device=device)
        counts = torch.zeros(n_blocks, 1, device=device)

        block_ids = self._prev_block_assignments  # [N]
        centroids.scatter_add_(0, block_ids.unsqueeze(1).expand(-1, 3), self._prev_means)
        counts.scatter_add_(
            0,
            block_ids.unsqueeze(1),
            torch.ones(block_ids.shape[0], 1, device=device),
        )
        centroids /= counts.clamp(min=1)
        return centroids

    def _assign_to_blocks(self, means: Tensor, centroids: Tensor) -> Tensor:
        """Assign Gaussians to coarse blocks via nearest centroid (capacity-constrained).

        Uses iterative eviction: start with unconstrained nearest-centroid
        assignment, then for over-capacity blocks, evict the farthest Gaussians
        to their next-nearest available block. With ~9 blocks (3x3 grid),
        this converges in very few iterations.

        Args:
            means: Current frame 3D means [N, 3].
            centroids: Block centroids [n_blocks, 3].

        Returns:
            Block assignments [N].
        """
        n = means.shape[0]
        device = means.device
        n_blocks = centroids.shape[0]
        capacities = self._block_capacities.clone()  # [n_blocks]

        # Distance from each Gaussian to each centroid [N, n_blocks]
        dists = torch.cdist(means, centroids)

        # For each Gaussian, rank blocks by distance [N, n_blocks]
        sorted_block_indices = torch.argsort(dists, dim=1)

        # Start with unconstrained nearest-centroid assignment
        assignments = sorted_block_indices[:, 0].clone()  # [N]

        # Track each Gaussian's current preference rank (0 = nearest)
        pref_rank = torch.zeros(n, dtype=torch.long, device=device)

        # Iteratively resolve over-capacity blocks
        max_iters = n_blocks * 2  # Generous upper bound
        for _ in range(max_iters):
            # Count Gaussians per block
            counts = torch.zeros(n_blocks, dtype=torch.long, device=device)
            counts.scatter_add_(0, assignments, torch.ones(n, dtype=torch.long, device=device))

            # Find over-capacity blocks
            overflow = counts - capacities  # [n_blocks]
            if (overflow <= 0).all():
                break

            # For each over-capacity block, evict farthest Gaussians
            for block_id in torch.where(overflow > 0)[0]:
                block_id_val = block_id.item()
                n_evict = overflow[block_id_val].item()

                # Find Gaussians assigned to this block
                in_block = torch.where(assignments == block_id_val)[0]

                # Evict the farthest ones
                block_dists = dists[in_block, block_id_val]
                _, dist_order = block_dists.sort(descending=True)
                evict_indices = in_block[dist_order[:n_evict]]

                # Move evicted Gaussians to their next preference
                pref_rank[evict_indices] += 1
                new_ranks = pref_rank[evict_indices].clamp(max=n_blocks - 1)
                assignments[evict_indices] = sorted_block_indices[
                    evict_indices, new_ranks
                ]

        return assignments

    def _sort_within_blocks(self, means: Tensor, block_assignments: Tensor) -> Tensor:
        """Morton-sort Gaussians within each coarse block by current 3D means.

        Groups Gaussians by block, Morton-sorts within each group, and
        concatenates in block order to produce the global ordering.

        Args:
            means: Current frame 3D means [N, 3].
            block_assignments: Block assignment for each Gaussian [N].

        Returns:
            Global sorted indices [N].
        """
        n_blocks = self._n_blocks_y * self._n_blocks_x
        all_sorted: list[Tensor] = []

        for block_idx in range(n_blocks):
            mask = block_assignments == block_idx
            gaussian_indices = torch.where(mask)[0]

            if gaussian_indices.shape[0] == 0:
                continue

            if gaussian_indices.shape[0] == 1:
                all_sorted.append(gaussian_indices)
                continue

            block_means = means[gaussian_indices]  # [K, 3]

            # Normalize to [0, 1023] and compute Morton codes
            min_bounds = block_means.min(dim=0).values
            max_bounds = block_means.max(dim=0).values
            ranges = max_bounds - min_bounds
            ranges[ranges == 0] = 1e-8

            scaled = (
                ((block_means - min_bounds) / ranges * 1023.0).clamp(0, 1023).floor().long()
            )
            morton_codes = MortonSortingStrategy._encode_morton3_vec_30bit(
                scaled[:, 0], scaled[:, 1], scaled[:, 2]
            )
            local_order = torch.argsort(morton_codes)
            all_sorted.append(gaussian_indices[local_order])

        return torch.cat(all_sorted)
