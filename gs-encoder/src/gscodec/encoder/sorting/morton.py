"""
This module provides a deterministic sorting algorithm for Gaussian Splatting
data based on Morton codes (also known as Z-order curves).

Morton sorting maps 3D coordinates to a 1D curve, which helps preserve spatial
locality. This can improve the performance of subsequent compression steps by
grouping nearby splats together in memory. It is a very fast and deterministic
alternative to the more complex PLAS sorting.
"""

from typing import Any

import torch
from gsply import GSTensor
from torch import Tensor

from gscodec.encoder.sorting.base import SortingStrategy


class MortonSortingStrategy(SortingStrategy):
    """
    Implements the deterministic Morton sorting strategy for Gaussian Splatting data.

    Morton codes (or Z-order curves) map multi-dimensional coordinates to a
    single dimension, preserving spatial locality. This makes them suitable
    for ordering 3D Gaussians in a way that can improve memory access patterns
    and, consequently, compression efficiency for spatially-aware encoding schemes.
    This strategy is deterministic and relatively fast.
    """

    def sort(
        self,
        splats: GSTensor,
        **kwargs: Any,
    ) -> tuple[GSTensor, Tensor]:
        """
        Sorts the input `GSTensor` of splats using the Morton sort algorithm.

        This method generates a 1D ordering of splats based on their 3D mean
        positions by converting coordinates to Morton codes and then sorting
        these codes. The sorting is performed recursively to handle large numbers
        of splats efficiently.

        Args:
            splats: A `GSTensor` object containing the Gaussian data to be sorted.
                    Requires the `means` attribute to be present.
            **kwargs: Additional keyword arguments. Currently unused for this strategy.

        Returns:
            A tuple containing:
                - `sorted_splats`: A new `GSTensor` with all its attributes
                                   reordered according to the Morton sort.
                - `sorted_indices`: A 1D `torch.Tensor` of the sorted global indices,
                                    mapping original positions to their new sorted positions.

        Raises:
            ValueError: If the `splats` `GSTensor` does not contain `means`.
        """
        if splats.means is None:
            raise ValueError("Morton sorting requires 'means' in the splats GSTensor.")

        means: Tensor = splats.means
        num_splats: int = means.shape[0]
        if num_splats == 0:
            # If no splats, return an empty GSTensor and an empty index tensor.
            return splats, torch.tensor([], dtype=torch.long, device=means.device)

        # Initialize the indices tensor [0, 1, 2, ..., N-1]
        # This tensor will be modified in-place to hold the sorted global indices.
        sorted_indices: Tensor = torch.arange(num_splats, device=means.device)

        # Start the recursive sorting process on the full set of indices
        self._generate_ordering_recursive(means, sorted_indices)

        # Apply the final sorted indices to all splat tensors within the GSTensor.
        # This reorders all attributes consistently.
        sorted_splats: GSTensor = splats[sorted_indices]

        return sorted_splats, sorted_indices

    def sort_with_global_bbox(
        self,
        means: Tensor,
        global_min: Tensor,
        global_max: Tensor,
        secondary: Tensor | None = None,
        secondary_min: Tensor | None = None,
        secondary_max: Tensor | None = None,
        pos_bits: int = 10,
        sec_bits: int = 0,
    ) -> Tensor:
        """Sort by Morton code using a global bounding box for consistent quantization.

        Using a global bbox (shared across all frames) ensures stationary Gaussians
        get the same Morton code every frame, providing temporal stability for
        inter-frame video compression without constraining spatial quality.

        When secondary attributes are provided, uses a composite Morton code:
        high bits = position (coarse spatial locality), low bits = secondary attribute
        (fine-grained tiebreaking for attribute smoothness). This improves compression
        of non-position attributes (quats, scales) without significantly degrading
        position smoothness.

        Args:
            means: [N, 3] tensor of 3D positions.
            global_min: [3] global minimum across all frames.
            global_max: [3] global maximum across all frames.
            secondary: [N, 3+] optional secondary attribute for tiebreaking.
            secondary_min: [3+] min of secondary attribute.
            secondary_max: [3+] max of secondary attribute.
            pos_bits: Bits per axis for position (default 10 = full precision).
            sec_bits: Bits per axis for secondary attribute (default 0 = none).

        Returns:
            [N] sort indices.
        """
        ranges = (global_max - global_min).clamp(min=1e-8)

        if secondary is None or sec_bits == 0:
            scaled = ((means - global_min) / ranges * 1023.0).clamp(0, 1023).long()
            codes = self._encode_morton3_vec_30bit(scaled[:, 0], scaled[:, 1], scaled[:, 2])
            return torch.argsort(codes)

        # Composite Morton: position in high bits, secondary in low bits
        pos_max = (1 << pos_bits) - 1
        sec_max = (1 << sec_bits) - 1

        pos_scaled = ((means - global_min) / ranges * pos_max).clamp(0, pos_max).long()
        pos_code = self._encode_morton3_vec_30bit(pos_scaled[:, 0], pos_scaled[:, 1], pos_scaled[:, 2])

        sec_ranges = (secondary_max - secondary_min).clamp(min=1e-8)
        sec_attr = secondary[:, :3] if secondary.shape[1] > 3 else secondary
        sec_min3 = secondary_min[:3] if secondary_min.shape[0] > 3 else secondary_min
        sec_ranges3 = sec_ranges[:3] if sec_ranges.shape[0] > 3 else sec_ranges
        sec_scaled = ((sec_attr - sec_min3) / sec_ranges3 * sec_max).clamp(0, sec_max).long()
        sec_code = self._encode_morton3_vec_30bit(sec_scaled[:, 0], sec_scaled[:, 1], sec_scaled[:, 2])

        combined = (pos_code << (3 * sec_bits)) | sec_code
        return torch.argsort(combined)

    @staticmethod
    def _part1by2_vec_10bit(x: torch.Tensor) -> torch.Tensor:
        """
        Spreads the bits of a 10-bit integer for Morton encoding.

        This is a helper function used in the generation of Morton codes.
        It takes a tensor of 10-bit integers and inserts two zero bits
        between each bit of the input, effectively "spreading" the bits.
        This operation is crucial for interleaving bits from multiple
        coordinates (x, y, z) to form a single Morton code.

        Args:
            x: A `torch.Tensor` of 10-bit integer values (or convertible to long).

        Returns:
            A `torch.Tensor` where the bits of the input `x` have been spread.
        """
        if x.dtype != torch.long:
            x = x.long()

        x = x & 0x3FF  # Mask for 10 bits (0 to 1023)

        x_int32: torch.Tensor = x.to(torch.int32)

        x_int32 = torch.bitwise_and(torch.bitwise_xor(x_int32, (x_int32 << 16)), 0xFF0000FF)
        x_int32 = torch.bitwise_and(torch.bitwise_xor(x_int32, (x_int32 << 8)), 0x0300F00F)
        x_int32 = torch.bitwise_and(torch.bitwise_xor(x_int32, (x_int32 << 4)), 0x030C30C3)
        x_int32 = torch.bitwise_and(torch.bitwise_xor(x_int32, (x_int32 << 2)), 0x09249249)

        # Cast back to long for the final combination step in the calling function.
        return x_int32.long()

    @staticmethod
    def _encode_morton3_vec_30bit(
        x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Encodes three 10-bit integer coordinates (x, y, z) into a single 30-bit Morton code.

        This function interleaves the bits of the three input coordinates
        (each assumed to be 10-bit, i.e., in the range [0, 1023]) to form
        a 30-bit Morton code. The `_part1by2_vec_10bit` helper is used
        to spread the bits of each coordinate before interleaving.

        Args:
            x: A `torch.Tensor` of 10-bit integer x-coordinates.
            y: A `torch.Tensor` of 10-bit integer y-coordinates.
            z: A `torch.Tensor` of 10-bit integer z-coordinates.

        Returns:
            A `torch.Tensor` of 30-bit Morton codes.
        """
        return (
            (MortonSortingStrategy._part1by2_vec_10bit(z) << 2)  # Z bits at positions 2, 5, 8...
            + (MortonSortingStrategy._part1by2_vec_10bit(y) << 1)  # Y bits at positions 1, 4, 7...
            + MortonSortingStrategy._part1by2_vec_10bit(x)  # X bits at positions 0, 3, 6...
        )

    @staticmethod
    def _encode_morton2_vec_20bit(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Encodes two 10-bit integer coordinates (x, y) into a single 20-bit Morton code.

        This function interleaves the bits of the two input coordinates
        (each assumed to be 10-bit, i.e., in the range [0, 1023]) to form
        a 20-bit Morton code. The `_part1by2_vec_10bit` helper is used
        to spread the bits of each coordinate before interleaving.

        Args:
            x: A `torch.Tensor` of 10-bit integer x-coordinates.
            y: A `torch.Tensor` of 10-bit integer y-coordinates.

        Returns:
            A `torch.Tensor` of 20-bit Morton codes.
        """
        return (
            (MortonSortingStrategy._part1by2_vec_10bit(y) << 1)  # Y bits at positions 1, 3, 5...
            + MortonSortingStrategy._part1by2_vec_10bit(x)  # X bits at positions 0, 2, 4...
        )

    @staticmethod
    def get_grid_morton_sorted_indices(height: int, width: int, device: str) -> Tensor:
        """
        Generates 2D grid indices sorted by Morton (Z-order) codes.

        This method creates a 2D grid of coordinates (y, x), converts them
        into 20-bit Morton codes, and then returns the indices that would
        sort this grid according to these Morton codes. This is useful for
        arranging 2D data (like pixels in an image or Gaussians in a grid)
        in a spatially coherent order.

        Args:
            height: The height of the 2D grid.
            width: The width of the 2D grid.
            device: The `torch.device` on which to create the tensors.

        Returns:
            A 1D `torch.Tensor` of indices that, when applied to a flattened
            2D grid, would arrange its elements in Morton order.
        """
        # Create flattened 1D tensors of x and y coordinates for the grid
        slot_coords_y: Tensor = (
            torch.arange(height, device=device).view(-1, 1).repeat(1, width).flatten()
        )
        slot_coords_x: Tensor = (
            torch.arange(width, device=device).view(1, -1).repeat(height, 1).flatten()
        )

        # Encode these (x, y) pairs into 20-bit Morton codes
        morton_codes_2d: Tensor = MortonSortingStrategy._encode_morton2_vec_20bit(
            slot_coords_x, slot_coords_y
        )

        # Return the indices that sort these Morton codes
        return torch.argsort(morton_codes_2d)

    def _generate_ordering_recursive(self, all_means: Tensor, indices_to_sort: Tensor) -> None:
        """
        Recursively sorts a subset of splat indices using 10-bit Morton codes.

        This function implements the core recursive logic of the Morton sort.
        It takes a subset of `all_means` (identified by `indices_to_sort`),
        scales their coordinates to a 10-bit range, generates Morton codes,
        and sorts the `indices_to_sort` based on these codes.
        For large clusters of splats that share the same Morton code prefix
        (i.e., they are spatially very close), it recursively calls itself
        to further sort within those clusters.

        This function modifies `indices_to_sort` in-place.

        Args:
            all_means: A `torch.Tensor` containing the 3D mean positions of all splats.
            indices_to_sort: A 1D `torch.Tensor` (long type) representing the global
                             indices of the splats that need to be sorted in the
                             current recursive call. This tensor is modified in-place.
        """
        num_indices: int = len(indices_to_sort)
        if num_indices == 0:
            return

        # 1. Get the subset of means for the current sorting task
        current_means: Tensor = all_means[indices_to_sort]

        # 2. Scale coordinates to the 10-bit integer range [0, 1023]
        # This normalization ensures that the Morton codes are generated consistently
        # within the spatial extent of the current subset of splats.
        min_bounds: Tensor = current_means.min(dim=0).values
        max_bounds: Tensor = current_means.max(dim=0).values

        ranges: Tensor = max_bounds - min_bounds
        # Add a small epsilon to avoid division by zero for flat distributions
        ranges[ranges == 0] = 1e-8

        # Normalize coordinates to be in [0, 1023] for Morton encoding
        scaled_centers: Tensor = (
            ((current_means - min_bounds) / ranges * 1023.0).clamp(0, 1023).floor().long()
        )
        x, y, z = scaled_centers[:, 0], scaled_centers[:, 1], scaled_centers[:, 2]

        # 3. Generate 30-bit Morton codes from the scaled 10-bit coordinates
        morton_codes: Tensor = self._encode_morton3_vec_30bit(x, y, z)

        # 4. Sort the current indices based on the generated Morton codes
        sort_order: Tensor = torch.argsort(morton_codes)

        # Create a clone to correctly reorder `indices_to_sort` in-place
        original_indices_in_slice: Tensor = indices_to_sort.clone()
        sorted_morton_codes: Tensor = morton_codes[sort_order]  # For identifying clusters

        # Reorder the `indices_to_sort` tensor IN-PLACE
        indices_to_sort[:] = original_indices_in_slice[sort_order]

        # 5. Find dense clusters (groups of splats with identical Morton codes) and recurse
        # Identify points where the Morton code changes, marking the boundaries of clusters.
        change_points: Tensor = torch.where(torch.diff(sorted_morton_codes) != 0)[0] + 1

        # Define block boundaries for recursion
        starts: Tensor = torch.cat([torch.tensor([0], device=morton_codes.device), change_points])
        ends: Tensor = torch.cat(
            [change_points, torch.tensor([num_indices], device=morton_codes.device)]
        )

        block_sizes: Tensor = ends - starts

        # Identify large blocks (clusters) that need further recursive sorting.
        # A threshold of 256 is used to limit recursion depth and balance performance.
        large_block_indices: Tensor = torch.where(block_sizes > 256)[0]

        # Recurse on these large blocks
        for i in large_block_indices:
            start = int(starts[i].item())
            end = int(ends[i].item())
            block_indices_slice: Tensor = indices_to_sort[start:end]

            # Check to prevent infinite recursion on a block that is not shrinking,
            # though this scenario is unlikely with the current logic.
            if len(block_indices_slice) < num_indices:
                self._generate_ordering_recursive(all_means, block_indices_slice)
