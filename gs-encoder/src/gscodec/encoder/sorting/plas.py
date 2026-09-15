"""
This module implements the Parallel Linear Assignment Sorting (PLAS) algorithm,
a sophisticated method for sorting Gaussian Splats to improve temporal coherence
in dynamic scenes.

PLAS works by rearranging the splats in each frame to match a low-pass filtered
(blurred) version of the frame's attributes. This process is iterative and
operates on blocks of the data at multiple scales, resulting in a smooth and
coherent ordering of splats from one frame to the next. This is crucial for
achieving high compression ratios in video formats.

The module also provides a variant for sorting a frame against a static target
grid from a previous frame or chunk, further enhancing temporal stability.
"""

import functools
import itertools
import logging
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import torch.nn.functional as f
from gsply import GSTensor
from torch import Tensor

from gscodec.encoder.sorting.base import SortingStrategy
from gscodec.encoder.utils.helpers import log_transform

logger = logging.getLogger(__name__)


class PLASSortingStrategy(SortingStrategy):
    """
    Implements the Parallel Linear Assignment Sorting (PLAS) strategy.

    PLAS is an iterative, multi-scale sorting algorithm designed to rearrange
    Gaussian splats within a frame to improve their spatial and temporal coherence.
    It works by repeatedly assigning splats to positions in a target grid
    (either a blurred version of the frame itself or a static target from
    a previous frame) to minimize a cost function. This results in a smooth
    and consistent ordering over time, which is beneficial for video compression.
    """

    def sort(
        self,
        splats: GSTensor,
        **kwargs: Any,
    ) -> tuple[GSTensor, Tensor]:
        """
        Sorts the input `GSTensor` of splats using the PLAS algorithm.

        This method prepares the splat data, applies the core PLAS sorting logic
        (via `sort_splats`), and then reorders the original `GSTensor` based on
        the resulting sorted indices. It specifically handles a log transformation
        of `means` for sorting purposes and restores the original scale for the output.

        Args:
            splats: A `GSTensor` object containing the Gaussian data to be sorted.
                    Requires the `means` attribute to be present.
            **kwargs: Additional arguments for PLAS sorting, which are passed
                      directly to the `sort_splats` helper function. These can include:
                      - `prev_sorted_grid` (`Tensor | None`): A target grid from
                        a previous frame for temporally-anchored sorting.
                      - `seed` (`int | None`): A random seed for reproducibility.
                      - `initial_indices` (`Tensor | None`): An optional set of
                        indices to start the sorting from.
                      - `improvement_break` (`float`): Threshold for stopping
                        iterative optimization.
                      - `sort_keys` (`list[str]`): Attributes to use for calculating
                        sorting distances (defaults to `["means"]`).

        Returns:
            A tuple containing:
                - `sorted_splats`: A new `GSTensor` with all its attributes
                                   reordered according to the PLAS sort.
                - `final_indices`: A 1D `torch.Tensor` of the sorted global indices,
                                   mapping original positions to their new sorted positions.
        """

        # Clone the GSTensor to avoid modifying the original input
        splats_for_sorting: GSTensor = self._clone_gstensor(splats)

        # Apply log transformation to means for sorting. This is often done
        # to better represent scale changes in visual space.
        if splats_for_sorting.means is not None:
            splats_for_sorting.means = log_transform(splats_for_sorting.means)

        # Perform the actual PLAS sorting
        _, final_indices = sort_splats(splats_for_sorting, **kwargs)

        # Create a new GSTensor with all original attributes reordered by `final_indices`
        sorted_splats: GSTensor = splats[final_indices]

        return sorted_splats, final_indices

    def _clone_gstensor(self, splats: GSTensor) -> GSTensor:
        """
        Creates a shallow copy of the `GSTensor` structure with cloned tensors.

        This utility method is used to create a mutable copy of `GSTensor` data
        when transformations or calculations need to modify attribute tensors
        without affecting the original `GSTensor` object passed to `sort`.
        It specifically clones the tensors for relevant attributes used in sorting
        ("means", "scales", "quats", "opacities", "sh0", "shN") and the `masks` attribute.

        Args:
            splats: The `GSTensor` object to be cloned.

        Returns:
            A new `GSTensor` instance with cloned tensor attributes.
        """
        new_data: dict[str, Tensor | None] = {}
        # Attributes we care about for sorting/cloning
        keys: list[str] = ["means", "scales", "quats", "opacities", "sh0", "shN"]
        for key in keys:
            val = getattr(splats, key, None)
            if val is not None:
                new_data[key] = val.clone()
            else:
                new_data[key] = None

        new_gs = GSTensor(**new_data)
        if splats.masks is not None:
            new_gs.masks = splats.masks.clone()
        return new_gs


def sort_splats(
    splats: GSTensor,
    prev_sorted_grid: Tensor | None = None,
    seed: int | None = None,
    initial_indices: Tensor | None = None,
    improvement_break: float = 1e-4,
    sort_keys: list[str] | None = None,
) -> tuple[GSTensor, Tensor]:
    """
    Sorts Gaussian Splats using the Parallel Linear Assignment Sorting (PLAS) algorithm.

    This function orchestrates the PLAS sorting process. It can either sort a frame's
    splats to match its own blurred version (for a keyframe) or sort them to match
    a target grid from a previous frame (for temporal coherence).

    The main steps are:
    1.  **Preparation**: Extracts and concatenates specified attributes (`sort_keys`)
        from the `GSTensor` into a single tensor for sorting. Ensures the number
        of Gaussians forms a perfect square grid.
    2.  **Initialization**: Shuffles the Gaussian indices to randomize the starting
        arrangement, or uses provided `initial_indices`.
    3.  **PLAS Execution**:
        *   If `prev_sorted_grid` is provided, it performs a temporally-anchored
            PLAS sort (`_sort_with_static_target`) to align current splats
            with the previous frame's structure.
        *   Otherwise, it performs a self-sorting PLAS (`_sort_with_plas`)
            where splats are sorted to match a blurred version of themselves.
    4.  **Final Reordering**: Applies the resulting sorted indices to the
        original `GSTensor` attributes to produce the final sorted `GSTensor`.

    Args:
        splats: A `GSTensor` object containing the Gaussian data to be sorted.
                Requires the `means` attribute to be present.
        prev_sorted_grid: An optional `torch.Tensor` representing a target grid
                          from a previous frame. If provided, the current `splats`
                          will be sorted to align with this grid.
        seed: An optional integer seed for `torch.manual_seed` to ensure
              reproducibility of random permutations.
        initial_indices: An optional 1D `torch.Tensor` specifying an initial
                         ordering of splats. If `None`, a random permutation is used.
        improvement_break: A float threshold. The iterative PLAS optimization stops
                           when the relative improvement in the cost function falls
                           below this value.
        sort_keys: A list of strings, specifying which attributes of the `GSTensor`
                   (e.g., "means", "scales", "sh0") should be used to calculate
                   distances during the sorting process. Defaults to `["means"]`.

    Returns:
        A tuple containing:
            - `sorted_splats`: A new `GSTensor` object with all its attributes
                               reordered according to the PLAS sort.
            - `final_indices`: A 1D `torch.Tensor` of the sorted global indices,
                               mapping original positions to their new sorted positions.

    Raises:
        ValueError: If `splats` does not contain `means`, or if `sort_keys`
                    contain attributes missing from `GSTensor`, or if the number
                    of Gaussians is not a perfect square.
    """
    if sort_keys is None:
        sort_keys = ["means"]

    if seed is not None:
        torch.manual_seed(seed)

    if splats.means is None:
        raise ValueError("PLAS sorting requires 'means' in the splats GSTensor.")

    n_gs: int = len(splats.means)
    n_sidelen: int = int(n_gs**0.5)
    # Ensure the number of Gaussians forms a perfect square for grid operations
    assert n_sidelen**2 == n_gs, "The number of Gaussians must be a perfect square for PLAS."

    # Remove 'shN' from sort_keys if present, as it's typically handled by VQ
    if "shN" in sort_keys:
        sort_keys.remove("shN")

    # Extract and concatenate tensors for sorting based on `sort_keys`
    tensors_to_sort: list[Tensor] = []
    for k in sort_keys:
        val = getattr(splats, k, None)
        if val is None:
            raise ValueError(f"Sort key '{k}' is missing in GSTensor.")
        tensors_to_sort.append(val.reshape(n_gs, -1))

    # Concatenate all chosen attributes into a single tensor for distance calculations
    params_to_sort: Tensor = torch.cat(tensors_to_sort, dim=-1)

    # Initialize shuffled indices: either provided or a random permutation
    shuffled_indices: Tensor
    if initial_indices is not None:
        shuffled_indices = initial_indices
    else:
        shuffled_indices = torch.randperm(params_to_sort.shape[0], device=params_to_sort.device)

    # Apply initial shuffle
    params_to_sort_shuffled: Tensor = params_to_sort[shuffled_indices]

    # Reshape into a 3D grid (channels, height, width) for PLAS processing
    grid: Tensor = params_to_sort_shuffled.reshape((n_sidelen, n_sidelen, -1)).permute(2, 0, 1)

    # Choose between self-sorting PLAS or static-target PLAS
    if prev_sorted_grid is not None:
        logger.debug("Using temporally-anchored sorting against static target.")

        # Interpolate the previous grid to match the current grid's dimensions
        target_grid: Tensor = f.interpolate(
            prev_sorted_grid.unsqueeze(0), size=(n_sidelen, n_sidelen), mode="nearest"
        ).squeeze(0)

        # Perform sorting against the static target grid
        _, sorted_indices_grid = _sort_with_static_target(
            grid,
            target_grid,
            improvement_break=improvement_break,
            seed=seed,
        )
        # Flatten the 2D grid indices to a 1D relative order
        sorted_indices_relative = sorted_indices_grid.squeeze().flatten()
    else:
        # Perform self-sorting PLAS
        _, plas_sorted_indices = _sort_with_plas(
            grid, improvement_break=improvement_break, seed=seed
        )
        # Flatten the 2D grid indices to a 1D relative order
        sorted_indices_relative = plas_sorted_indices.squeeze().flatten()

    # Combine the initial shuffle with the relative sorting to get final global indices
    final_indices: Tensor = shuffled_indices[sorted_indices_relative]

    # Apply these final indices to the original GSTensor to get the sorted result
    sorted_splats: GSTensor = splats[final_indices]

    return sorted_splats, final_indices


def _create_gaussian_kernel_2d(
    kernel_size_y: int, kernel_size_x: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """
    Creates a 2D Gaussian kernel tensor.

    This kernel is used for applying Gaussian blur in the `_low_pass_filter` function.
    The standard deviation (sigma) for the Gaussian is automatically determined
    based on the kernel size to ensure a reasonable blur.

    Args:
        kernel_size_y: The height of the Gaussian kernel.
        kernel_size_x: The width of the Gaussian kernel.
        device: The `torch.device` on which to create the kernel tensor.
        dtype: The `torch.dtype` of the kernel tensor.

    Returns:
        A 2D `torch.Tensor` representing the normalized Gaussian kernel.
    """
    # Calculate sigma based on kernel size, typically 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    sigma_y: float = 0.3 * ((kernel_size_y - 1) * 0.5 - 1) + 0.8
    sigma_x: float = 0.3 * ((kernel_size_x - 1) * 0.5 - 1) + 0.8

    # Create 1D Gaussian kernels for Y and X dimensions
    coords_y: Tensor = torch.arange(kernel_size_y, dtype=dtype, device=device)
    coords_y -= (kernel_size_y - 1) / 2.0
    kernel_y: Tensor = torch.exp(-(coords_y**2) / (2 * sigma_y**2))

    coords_x: Tensor = torch.arange(kernel_size_x, dtype=dtype, device=device)
    coords_x -= (kernel_size_x - 1) / 2.0
    kernel_x: Tensor = torch.exp(-(coords_x**2) / (2 * sigma_x**2))

    # Combine 1D kernels to form a 2D kernel using an outer product
    kernel: Tensor = torch.outer(kernel_y, kernel_x)
    # Normalize the kernel so its elements sum to 1
    kernel /= torch.sum(kernel)
    return kernel


def _low_pass_filter(
    img: Tensor,
    filter_size_x: int,
    filter_size_y: int,
    border_type_x: str,
    border_type_y: str,
) -> Tensor:
    """
    Applies a 2D Gaussian blur (low-pass filter) to an input image tensor.

    This function uses a custom 2D Gaussian kernel created by `_create_gaussian_kernel_2d`
    and `torch.nn.functional.conv2d` to blur the input `img`. Padding is applied
    using `reflect` mode to handle border effects. This is a core component
    of PLAS, used to generate the "blurred" target grids.

    Args:
        img: The input image tensor with shape `(C, H, W)` or `(C, S, S)`
             where C is channels, H is height, W is width (or S for square).
        filter_size_x: The desired width of the Gaussian filter.
        filter_size_y: The desired height of the Gaussian filter.
        border_type_x: The border padding mode for the x-axis (e.g., "circular", "reflect").
                       Currently, this function hardcodes "reflect" for both axes due
                       to `F.pad`'s limitations, so this argument is mostly for
                       conceptual consistency.
        border_type_y: The border padding mode for the y-axis (e.g., "reflect").
                       See `border_type_x` note.

    Returns:
        A new `torch.Tensor` representing the blurred image.
    """
    C, H, W = img.shape  # noqa: N806

    # Adjust kernel sizes to be odd if they are even, as is common for convolution kernels.
    ky = filter_size_y + 1 if filter_size_y % 2 == 0 else filter_size_y
    kx = filter_size_x + 1 if filter_size_x % 2 == 0 else filter_size_x

    # Create the 2D Gaussian kernel
    kernel: Tensor = _create_gaussian_kernel_2d(ky, kx, device=img.device, dtype=img.dtype)
    # Reshape kernel for conv2d: (out_channels, in_channels/groups, kH, kW)
    kernel = kernel.view(1, 1, ky, kx).repeat(C, 1, 1, 1)

    # Calculate padding amounts
    padding_y: int = (ky - 1) // 2
    padding_x: int = (kx - 1) // 2

    # Apply reflection padding manually. F.conv2d's padding_mode applies to all sides.
    img_padded: Tensor = f.pad(
        img.unsqueeze(0), (padding_x, padding_x, padding_y, padding_y), mode="reflect"
    )

    # Perform 2D convolution with the Gaussian kernel
    blurred: Tensor = f.conv2d(img_padded, kernel, groups=C)

    return blurred.squeeze(0)


@functools.cache
def _get_permutations(device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Generates and caches all 24 permutations for a 4-element set, along with
    their one-hot encoded representations.

    This function is primarily used by `_solve_assignments_batch_dim` to efficiently
    test all possible assignments within 2x2 blocks during the PLAS algorithm.
    The result is cached for performance.

    Args:
        device: The `torch.device` on which to create the tensors.

    Returns:
        A tuple containing:
            - `perms`: A `torch.Tensor` of shape `(24, 4)` representing all
                       permutations of `[0, 1, 2, 3]`.
            - `perms_one_hot`: A `torch.Tensor` of shape `(24, 4, 4)` representing
                               the one-hot encoding of each permutation.
    """
    # Generate all permutations of [0, 1, 2, 3]
    perms: Tensor = torch.tensor(list(itertools.permutations(range(4))), device=device)
    # Convert permutations to one-hot encoding for efficient matrix multiplication later
    perms_one_hot: Tensor = torch.nn.functional.one_hot(perms, num_classes=4).float()
    return perms, perms_one_hot


def _solve_assignments_batch_dim(cand_dists: Tensor, device: torch.device) -> Tensor:
    """
    Solves the optimal assignment problem for a batch of 2x2 blocks by testing
    all 24 permutations.

    In the PLAS algorithm, this function is called repeatedly to determine the
    best permutation of elements within small blocks (typically 2x2 blocks)
    to minimize a cost function (distance). It leverages pre-computed permutations
    to quickly find the optimal assignment.

    Args:
        cand_dists: A `torch.Tensor` of shape `(batch_size, num_channels, 4, 4)`
                    representing the cost matrix for each 2x2 assignment problem
                    in the batch. `num_channels` here corresponds to the number
                    of attributes being sorted. The last two dimensions (4, 4)
                    represent the distances between candidates (rows) and target
                    positions (columns) for a 2x2 block.
        device: The `torch.device` on which to perform calculations.

    Returns:
        A `torch.Tensor` of shape `(batch_size, 4)` where each row represents
        the optimal permutation of `[0, 1, 2, 3]` for the corresponding 2x2
        block in the input batch.
    """
    perms, perms_one_hot = _get_permutations(device)

    # Calculate the total distance for each permutation for each batch element
    # `perms_dist` will have shape (batch_size, num_channels, 24)
    # The sum over the last two dimensions effectively calculates the cost
    # of applying each permutation to the candidate distances.
    perms_dist: Tensor = torch.einsum("bsce,pce->bsp", cand_dists, perms_one_hot)

    # Find the index of the permutation that yields the minimum distance for each batch element
    best_perms_idx: Tensor = torch.argmin(perms_dist.sum(dim=1), dim=-1)

    # Return the actual optimal permutations
    return perms[best_perms_idx]


def _params_to_blocky(
    params: Tensor,
    block_size: int,
    block_divisor: int,
    num_pixel_blocks: int,
    shift_y: int,
    shift_x: int,
) -> tuple[Tensor, Tensor]:
    """
    Rearranges a grid of parameters into a "blocky" representation.

    This function takes a 3D tensor representing a grid of parameters (channels, H, W)
    and transforms it into a batched, flattened representation where each "block"
    is treated as a separate element in a batch. This is a crucial step for
    applying local operations (like the 2x2 assignment problem) in parallel
    across the entire grid in PLAS.

    The process involves:
    1.  **Rolling**: Shifting the grid by `shift_y` and `shift_x` to create
        overlapping blocks effectively.
    2.  **Truncating**: Cropping the rolled grid to ensure it's divisible by `block_size`.
    3.  **PixelUnshuffle**: Applying a pixel unshuffle operation to create
        the batched blocky structure.
    4.  **Reshaping**: Further reshaping to flatten the internal structure
        of each block for processing.

    Args:
        params: The input parameter grid `(C, H, W)`.
        block_size: The side length of each block (e.g., 2 for 2x2 blocks).
        block_divisor: The divisor used in `PixelUnshuffle` (typically `block_size`).
        num_pixel_blocks: The number of blocks along each dimension.
        shift_y: The vertical shift applied to the grid.
        shift_x: The horizontal shift applied to the grid.

    Returns:
        A tuple containing:
            - `params_rolled`: The input `params` tensor after being rolled and
                               potentially truncated. This is returned to be used
                               by `_blocky_to_params` for reconstruction.
            - `params_blocky_flat`: The transformed tensor in its blocky, flattened
                                    representation, ready for block-wise processing.
    """
    # 1. Roll (shift) the parameters to create overlapping blocks
    params_rolled: Tensor = torch.roll(params, (shift_y, shift_x), dims=(1, 2))
    # 2. Truncate to ensure dimensions are divisible by block_size
    params_truncated: Tensor = params_rolled[
        :, : num_pixel_blocks * block_size, : num_pixel_blocks * block_size
    ]
    params_c_hw: Tensor = params_truncated.unsqueeze(1)
    # 3. Apply PixelUnshuffle to break into blocks
    params_unshuffled_cbhw: Tensor = torch.nn.PixelUnshuffle(block_divisor)(params_c_hw)
    # Rearrange dimensions for further reshaping
    params_unshuffled: Tensor = params_unshuffled_cbhw.permute(1, 0, 2, 3)
    # Reshape into a structure that groups blocks
    params_unshuffled_blocky_inline: Tensor = params_unshuffled.reshape(
        -1, params_unshuffled.shape[1], num_pixel_blocks, block_size, num_pixel_blocks, block_size
    )
    params_unshuffled_blocky_first: Tensor = params_unshuffled_blocky_inline.permute(
        0, 2, 4, 1, 3, 5
    )
    # Flatten the blocks for batch processing
    params_blocky_batch_flat: Tensor = params_unshuffled_blocky_first.reshape(
        -1,
        params_unshuffled_blocky_first.shape[3],
        params_unshuffled_blocky_first.shape[4],
        params_unshuffled_blocky_first.shape[5],
    )
    # Final flattening of individual blocks' content
    params_blocky_flat: Tensor = params_blocky_batch_flat.flatten(start_dim=2)
    return params_rolled, params_blocky_flat


def _blocky_to_params(
    params_rolled: Tensor,
    params_blocky_flat: Tensor,
    block_size: int,
    block_divisor: int,
    num_pixel_blocks: int,
    shift_y: int,
    shift_x: int,
) -> Tensor:
    """
    Reassembles a grid of parameters from a "blocky" representation back into
    its original grid-like structure.

    This function reverses the operation performed by `_params_to_blocky`,
    taking the processed blocky representation and reconstructing the 3D
    parameter grid (`C, H, W`). It involves reshaping, applying a PixelShuffle
    operation, and rolling the grid back to its original position.

    Args:
        params_rolled: The initially rolled tensor from `_params_to_blocky`,
                       used as a template for the final grid dimensions.
        params_blocky_flat: The processed blocky, flattened representation of
                            the parameters.
        block_size: The side length of each block.
        block_divisor: The divisor used in `PixelShuffle`.
        num_pixel_blocks: The number of blocks along each dimension.
        shift_y: The vertical shift that was originally applied.
        shift_x: The horizontal shift that was originally applied.

    Returns:
        A `torch.Tensor` representing the reassembled 3D parameter grid
        `(C, H, W)`.
    """
    # Reverse the flattening and reshaping
    params_unshuffled_blocky_first: Tensor = params_blocky_flat.reshape(
        block_divisor**2,
        num_pixel_blocks,
        num_pixel_blocks,
        params_blocky_flat.shape[1],  # Original channels dimension
        block_size,
        block_size,
    )
    params_unshuffled_blocky_inline: Tensor = params_unshuffled_blocky_first.permute(
        0, 3, 1, 4, 2, 5
    )
    params_unshuffled: Tensor = params_unshuffled_blocky_inline.reshape(
        params_unshuffled_blocky_inline.shape[0],
        params_unshuffled_blocky_inline.shape[1],
        params_unshuffled_blocky_inline.shape[2] * block_size,
        params_unshuffled_blocky_inline.shape[4] * block_size,
    )
    params_unshuffled_cbhw: Tensor = params_unshuffled.permute(1, 0, 2, 3)

    # Apply PixelShuffle to reconstruct the grid from blocks
    params_shuffled: Tensor = torch.nn.PixelShuffle(block_divisor)(params_unshuffled_cbhw)
    params_unshuffled = params_shuffled.squeeze(1)

    # Assign the unshuffled (reconstructed) part back to the original rolled tensor's relevant section
    params_rolled[:, : num_pixel_blocks * block_size, : num_pixel_blocks * block_size] = (
        params_unshuffled
    )
    # Roll back the tensor to its original (un-shifted) position
    return torch.roll(params_rolled, (-shift_y, -shift_x), dims=(1, 2))


def _reorder_blocky_shuffled(
    params_blocky_flat: Tensor,
    grid_indices_blocky_flat: Tensor,
    target_blocky_flat: Tensor,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    """
    Performs the core reordering step of PLAS within each block by solving
    many small assignment problems (typically 2x2).

    This function operates on the "blocky" flattened representations of the
    parameters, their corresponding grid indices, and the target parameters.
    For each small block, it determines the optimal permutation of elements
    to minimize the distance to the target, and then applies this permutation
    to both the parameters and their grid indices.

    Args:
        params_blocky_flat: A `torch.Tensor` representing the current parameters
                            in their blocky, flattened form.
        grid_indices_blocky_flat: A `torch.Tensor` representing the current grid
                                  indices in their blocky, flattened form.
        target_blocky_flat: A `torch.Tensor` representing the target parameters
                            in their blocky, flattened form (e.g., blurred version).
        block_size: The side length of each block (e.g., 2).

    Returns:
        A tuple containing:
            - `params_blocky_flat`: The parameters tensor after block-wise reordering.
            - `grid_indices_blocky_flat`: The grid indices tensor after block-wise reordering.
    """
    # Create random permutations of indices within each block
    shuffled_block_indices: Tensor = torch.randperm(
        block_size * block_size, device=params_blocky_flat.device
    )
    # Reshape to group 4 elements, assuming 2x2 blocks for permutation solving
    shuffled_block_indices_cand: Tensor = shuffled_block_indices.reshape(-1, 4)

    # Extract blockwise shuffled parameters and target values
    blockwise_shuffled_params: Tensor = params_blocky_flat[:, :, shuffled_block_indices_cand]
    blockwise_shuffled_target: Tensor = target_blocky_flat[:, :, shuffled_block_indices_cand]

    # Calculate cost matrix C: Squared Euclidean distance between candidates and targets
    # C will have shape (batch_size, num_channels, 4, 4) where the last two
    # dimensions represent the 2x2 assignment problem within each block.
    # Calculate cost matrix C: Squared Euclidean distance between candidates and targets
    # C will have shape (batch_size, num_chunks, 4, 4)
    C: Tensor = torch.pow(  # noqa: N806
        blockwise_shuffled_params.unsqueeze(-1) - blockwise_shuffled_target.unsqueeze(-2), 2
    ).sum(dim=1)

    # Reshape C to treat each chunk as a separate batch element for solving
    # New shape: (batch_size * num_chunks, 1, 4, 4)
    batch_size, num_chunks = C.shape[:2]
    c_flat = C.reshape(batch_size * num_chunks, 1, 4, 4)

    # Solve the assignment problem for each block
    # best_perm_indices_flat shape: (batch_size * num_chunks, 4)
    best_perm_indices_flat: Tensor = _solve_assignments_batch_dim(
        c_flat, device=target_blocky_flat.device
    )

    # Reshape back to (batch_size, num_chunks, 4)
    best_perm_indices: Tensor = best_perm_indices_flat.reshape(batch_size, num_chunks, 4)

    # Expand shuffled_block_indices_cand to match the batch size
    # shuffled_block_indices_cand shape: (num_chunks, 4) -> (batch_size, num_chunks, 4)
    shuffled_block_indices_cand_exp: Tensor = shuffled_block_indices_cand.unsqueeze(0).expand(
        batch_size, -1, -1
    )

    # Get the best positions (indices) for reordering within each block
    best_positions: Tensor = torch.gather(shuffled_block_indices_cand_exp, -1, best_perm_indices)
    best_positions_flat: Tensor = best_positions.flatten(start_dim=1)

    # Expand best_positions_flat to match the number of channels for gathering values
    best_positions_exp: Tensor = best_positions_flat.unsqueeze(1).expand_as(params_blocky_flat)

    # Gather the best values (parameters) according to the optimal permutations
    best_values: Tensor = torch.gather(params_blocky_flat, -1, best_positions_exp)

    # Gather the corresponding grid indices
    best_grid_indices: Tensor = torch.gather(
        grid_indices_blocky_flat, -1, best_positions_flat.unsqueeze(1)
    )

    # Update the blocky flattened tensors with the reordered values and indices
    # We need to flatten the indices to match the flattened last dimension of params_blocky_flat
    # But wait, params_blocky_flat last dim is (num_chunks * 4).
    # shuffled_block_indices_cand_flat is just (num_chunks * 4).
    # We need to scatter/assign into the correct positions.
    # Actually, since we are updating the WHOLE tensor, we can just assign directly?
    # No, shuffled_block_indices_cand was a subset/permutation of indices.
    # But here we are replacing the *values* at those positions.
    # Since we gathered *all* values into best_values, we can just overwrite?
    # Wait, shuffled_block_indices_cand covers the *entire* blocky_flat last dimension?
    # shuffled_block_indices was randperm(block_size^2).
    # shuffled_block_indices_cand is reshaped to (-1, 4).
    # So yes, it covers all indices.
    # So we can just assign best_values to the positions indicated by shuffled_block_indices_cand_flat?
    # Yes, but we need to handle the batch dimension.

    shuffled_block_indices_cand_flat: Tensor = shuffled_block_indices_cand.flatten()

    # Actually, simpler: we can just scatter best_values back.
    # Or since shuffled_block_indices is a permutation of ALL indices,
    # we can just reorder best_values to the "canonical" order and assign?
    # No, we want to put best_values[b, c, i] into params_blocky_flat[b, c, original_index[i]]

    # Let's stick to the original logic pattern but fix dimensions
    # shuffled_block_indices_cand_flat is 1D indices into the last dim.

    # We need to scatter best_values into params_blocky_flat at indices shuffled_block_indices_cand_flat
    # params_blocky_flat.scatter_(-1, index_tensor, best_values)

    # Construct index tensor for scatter
    scatter_indices = shuffled_block_indices_cand_flat.view(1, 1, -1).expand_as(params_blocky_flat)
    params_blocky_flat.scatter_(-1, scatter_indices, best_values)
    grid_indices_blocky_flat.scatter_(
        -1, scatter_indices.select(1, 0).unsqueeze(1), best_grid_indices
    )

    return params_blocky_flat, grid_indices_blocky_flat.contiguous()


def _reorder_plas(
    params: Tensor,
    grid_indices: Tensor,
    min_block_size: int,
    filter_size_x: int,
    filter_size_y: int,
    border_type_x: str,
    border_type_y: str,
    improvement_break: float,
    pbar: Any,  # TQDM progress bar object, can be None
    static_target: Tensor | None = None,
) -> tuple[Tensor, Tensor, int]:
    """
    Manages the iterative reordering process for a single scale (blur radius)
    within the PLAS algorithm.

    This function performs multiple passes of block-wise reordering for a given
    `params` grid. In each pass, it shifts the grid to create different block
    configurations, then reorders elements within those blocks to minimize
    the distance to a `target` (either a blurred version of `params` itself
    or a `static_target`). The iteration continues until the improvement falls
    below `improvement_break` or a maximum number of block configurations is reached.

    Args:
        params: The `torch.Tensor` representing the current parameters grid (C, H, W).
        grid_indices: The `torch.Tensor` representing the grid indices (1, H, W).
        min_block_size: The minimum size of blocks to consider for reordering.
        filter_size_x: The horizontal size of the filter used for blurring the target.
        filter_size_y: The vertical size of the filter used for blurring the target.
        border_type_x: The border padding type for the x-axis of the blur filter.
        border_type_y: The border padding type for the y-axis of the blur filter.
        improvement_break: The threshold for relative improvement in cost to stop iterating.
        pbar: A `tqdm` progress bar object (or similar) for displaying progress, or `None`.
        static_target: An optional `torch.Tensor` representing a fixed target grid
                       to sort against. If `None`, `params` will be blurred to
                       create its own target.

    Returns:
        A tuple containing:
            - `params`: The reordered parameters grid after iterative refinement.
            - `grid_indices`: The reordered grid indices after iterative refinement.
            - `num_reorders`: The total number of block reordering operations performed.
    """
    # Determine the target for sorting: either a static target or a blurred version of params
    target: Tensor
    if static_target is not None:
        target = static_target
    else:
        target = _low_pass_filter(
            params, filter_size_x, filter_size_y, border_type_x, border_type_y
        )

    sidelen: int = params.shape[1]  # Assumes square grid for now (H=W=sidelen)
    block_size: int = min(
        filter_size_x + 1, sidelen
    )  # Block size should not exceed grid dimensions
    block_size = block_size // 2 * 2  # Ensure block size is even
    block_size = max(block_size, min_block_size)  # Ensure minimum block size
    block_divisor: int = 1  # Used for PixelUnshuffle/Shuffle (currently fixed)
    num_pixel_blocks: int = sidelen // block_size

    if pbar:
        pbar.set_description(f"filter_size={filter_size_x} - {block_size=}")

    num_reorders: int = 0
    block_config: int = 0  # Counter for block shift configurations tried

    # Iterate through different block shift configurations
    while True:
        # Randomly choose a shift for the blocks
        shift_y, shift_x = np.random.randint(0, block_size, 2)

        # Convert parameters, grid indices, and target to blocky format
        params_rolled, params_blocky_flat = _params_to_blocky(
            params, block_size, block_divisor, num_pixel_blocks, shift_y, shift_x
        )
        _, target_blocky_flat = _params_to_blocky(
            target, block_size, block_divisor, num_pixel_blocks, shift_y, shift_x
        )
        grid_indices_rolled, grid_indices_blocky_flat = _params_to_blocky(
            grid_indices, block_size, block_divisor, num_pixel_blocks, shift_y, shift_x
        )

        # Initial distance (cost) before reordering
        prev_dist: float = torch.pow(params_blocky_flat - target_blocky_flat, 2).sum().item()
        i: int = 0  # Iteration counter for improvement loop
        has_improved: bool = False

        # Iteratively reorder within blocks until convergence or small improvement
        while True:
            num_reorders += 1
            params_blocky_flat, grid_indices_blocky_flat = _reorder_blocky_shuffled(
                params_blocky_flat, grid_indices_blocky_flat, target_blocky_flat, block_size
            )
            cur_dist: float = torch.pow(params_blocky_flat - target_blocky_flat, 2).sum().item()
            improvement_factor: float = 1 - (cur_dist / prev_dist) if prev_dist != 0 else 0

            if pbar:
                pbar.set_postfix(
                    {
                        "it": f"{i:04d}",
                        "dist": f"{cur_dist:.2E}",
                        "dist_factor": f"{improvement_factor:+.2E}",
                    }
                )

            # Break condition: if improvement is too small
            if improvement_factor < improvement_break:
                break
            prev_dist = cur_dist
            i += 1
            has_improved = True

        # Reassemble the grid from the blocky representation after reordering
        params = _blocky_to_params(
            params_rolled,
            params_blocky_flat,
            block_size,
            block_divisor,
            num_pixel_blocks,
            shift_y,
            shift_x,
        ).contiguous()
        grid_indices = _blocky_to_params(
            grid_indices_rolled,
            grid_indices_blocky_flat,
            block_size,
            block_divisor,
            num_pixel_blocks,
            shift_y,
            shift_x,
        ).contiguous()

        # If no improvement was made in this block configuration and we've tried enough, break
        if not has_improved and block_config >= 3:
            break
        block_config += 1

    return params, grid_indices, num_reorders


def _radius_seq(max_radius: float, min_radius: float, radius_update: float) -> Iterator[int]:
    """
    Generates a sequence of decreasing blur radii for the multi-scale PLAS algorithm.

    PLAS operates at multiple scales (different blur radii) to refine the sorting
    from coarse to fine. This generator yields integer radii, starting from `max_radius`
    and progressively decreasing by `radius_update` until `min_radius` is reached.

    Args:
        max_radius: The starting (largest) blur radius.
        min_radius: The minimum blur radius to generate.
        radius_update: The multiplicative factor by which the radius is decreased
                       in each step (e.g., 0.95 to reduce by 5%).

    Yields:
        An integer representing the current blur radius in the sequence.
    """
    radius: float = max_radius
    while True:
        yield int(radius)
        radius *= radius_update
        if radius < min_radius:
            break


def _sort_with_plas(
    params: Tensor,
    min_block_size: int = 16,
    min_blur_radius: int = 1,
    improvement_break: float = 1e-5,
    border_type_x: str = "circular",
    border_type_y: str = "reflect",
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Performs the full multi-scale PLAS algorithm, sorting `params` to match
    a blurred version of itself.

    This function orchestrates the self-sorting variant of PLAS. It iterates
    through a sequence of decreasing blur radii, from coarse to fine. At each
    scale, it calls `_reorder_plas` to iteratively reorder the `params` grid
    to match a version of itself blurred with the current radius.

    Args:
        params: The `torch.Tensor` representing the parameter grid to be sorted (C, H, W).
        min_block_size: The minimum size of blocks to consider for reordering.
        min_blur_radius: The minimum blur radius to use in the multi-scale process.
        improvement_break: The threshold for relative improvement to stop iterating
                           within a single scale.
        border_type_x: The border padding type for the x-axis of the blur filter.
        border_type_y: The border padding type for the y-axis of the blur filter.
        seed: An optional integer seed for reproducibility.

    Returns:
        A tuple containing:
            - `params`: The sorted parameters grid `(C, H, W)`.
            - `grid_indices`: The final sorted grid indices `(1, H, W)`.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    H, W = params.shape[1:]  # noqa: N806
    assert H == W, "PLAS requires a square grid (H==W)."

    start_time: float = time.time()
    # Generate the sequence of blur radii for the multi-scale approach
    radii: list[int] = list(
        _radius_seq(max_radius=max(H, W) / 2 - 1, min_radius=min_blur_radius, radius_update=0.95)
    )
    total_num_reorders: int = 0
    with torch.inference_mode():
        # Initialize grid indices to a simple ascending order
        grid_indices: Tensor = (
            torch.arange(0, H * W, dtype=torch.int32, device=params.device)
            .reshape(H, W)
            .unsqueeze(0)
        )
        # Iterate through blur radii from coarse to fine
        for radius in radii:
            filter_size_x: int = min(W - 1, int(2 * radius + 1))
            filter_size_y: int = min(H - 1, int(2 * radius + 1))

            # Perform iterative reordering for the current scale
            params, grid_indices, num_reorders = _reorder_plas(
                params=params,
                grid_indices=grid_indices,
                min_block_size=min_block_size,
                filter_size_x=filter_size_x,
                filter_size_y=filter_size_y,
                border_type_x=border_type_x,
                border_type_y=border_type_y,
                improvement_break=improvement_break,
                pbar=None,
                static_target=None,  # No static target in self-sorting mode
            )
            total_num_reorders += num_reorders
    duration: float = time.time() - start_time
    logger.debug(
        f"\nSorted {params.shape[1]}x{params.shape[2]} Gaussians in {duration:.3f}s ({total_num_reorders / duration:.3f} reorders/s)"
    )
    return params, grid_indices


def _sort_with_static_target(
    params: Tensor,
    target_grid: Tensor,
    min_block_size: int = 16,
    min_blur_radius: int = 1,
    improvement_break: float = 1e-4,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Performs the multi-scale PLAS algorithm, but sorts `params` to match an
    explicit `target_grid` instead of a self-blurred version.

    This function orchestrates the temporally-anchored variant of PLAS.
    It iterates through a sequence of decreasing blur radii, from coarse to fine.
    At each scale, it blurs the *static* `target_grid` and then calls `_reorder_plas`
    to iteratively reorder the `params` grid to match this blurred static target.

    Args:
        params: The `torch.Tensor` representing the parameter grid to be sorted (C, H, W).
        target_grid: A static `torch.Tensor` representing the target grid from
                     a previous frame, to which `params` will be aligned.
        min_block_size: The minimum size of blocks to consider for reordering.
        min_blur_radius: The minimum blur radius to use in the multi-scale process.
        improvement_break: The threshold for relative improvement to stop iterating
                           within a single scale.
        seed: An optional integer seed for reproducibility.

    Returns:
        A tuple containing:
            - `params`: The sorted parameters grid `(C, H, W)`.
            - `grid_indices`: The final sorted grid indices `(1, H, W)`.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    start_time: float = time.time()
    H, W = params.shape[1:]  # noqa: N806
    assert H == W, "PLAS requires a square grid (H==W)."
    # Initialize grid indices to a simple ascending order
    grid_indices: Tensor = (
        torch.arange(0, H * W, dtype=torch.int32, device=params.device).reshape(H, W).unsqueeze(0)
    )
    # Generate the sequence of blur radii
    radii: list[int] = list(
        _radius_seq(max_radius=max(H, W) / 2 - 1, min_radius=min_blur_radius, radius_update=0.95)
    )
    total_num_reorders: int = 0
    with torch.inference_mode():
        # Iterate through blur radii from coarse to fine
        for radius in radii:
            filter_size_x: int = min(W - 1, int(2 * radius + 1))
            filter_size_y: int = min(H - 1, int(2 * radius + 1))

            # Blur the STATIC target grid for the current scale
            blurred_static_target: Tensor = _low_pass_filter(
                target_grid, filter_size_x, filter_size_y, "circular", "reflect"
            )
            # Perform iterative reordering for the current scale
            params, grid_indices, num_reorders = _reorder_plas(
                params=params,
                grid_indices=grid_indices,
                min_block_size=min_block_size,
                filter_size_x=filter_size_x,
                filter_size_y=filter_size_y,
                border_type_x="circular",
                border_type_y="reflect",
                improvement_break=improvement_break,
                pbar=None,
                static_target=blurred_static_target,
            )
            total_num_reorders += num_reorders
    duration: float = time.time() - start_time
    logger.debug(
        f"\nSorted against static target in {duration:.3f}s with {total_num_reorders} reorders."
    )
    return params, grid_indices
