"""
SH decompression for the GSAV codec.

Reconstructs higher-order SH coefficients from:
1. Labels (from binary payload, uint16) -> palette entry index
2. Centroids (from binary payload) -> uint8 codebook indices
3. Codebook (from binary payload) -> float SH values
"""

import numpy as np
from numpy.typing import NDArray

from gscodec.common.types import SHChunkData
from gscodec.constants import SH_COEFFS


def expand_sh(
    labels: NDArray[np.uint16],
    sh_chunk_data: SHChunkData,
) -> NDArray[np.float32]:
    """Reconstruct SH float values from labels via centroid lookup + codebook.

    Args:
        labels: [N] uint16 palette indices.
        sh_chunk_data: Chunk SH data with codebook and centroids.

    Returns:
        [N, coeffs_per_channel, 3] float32 SH coefficients.
    """
    coeffs = SH_COEFFS[sh_chunk_data.sh_bands]

    # Look up centroid for each gaussian: [N, coeffs*3] uint8
    centroid_indices = sh_chunk_data.centroids[labels]

    # Map uint8 codebook indices to floats: [N, coeffs*3] float32
    sh_flat = sh_chunk_data.codebook[centroid_indices]

    # Reshape to [N, coeffs_per_channel, 3]
    return sh_flat.reshape(-1, coeffs, 3)
