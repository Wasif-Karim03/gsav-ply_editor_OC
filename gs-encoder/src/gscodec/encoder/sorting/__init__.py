"""
Sorting strategies for arranging Gaussian Splats to improve compression.

The `SortingStrategy` abstract base class defines the common interface.
`MortonSortingStrategy` is the primary strategy — uses Morton codes (Z-order curves)
for spatial locality that VP9 exploits for temporal prediction.
"""

from .base import SortingStrategy
from .morton import MortonSortingStrategy

__all__ = [
    "SortingStrategy",
    "MortonSortingStrategy",
]
