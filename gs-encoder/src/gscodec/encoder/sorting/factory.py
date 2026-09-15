"""
This module provides a factory for instantiating different Gaussian Splatting
sorting strategies.

The `SortingStrategyFactory` allows for dynamic selection of sorting algorithms
based on a string identifier, promoting modularity and extensibility.
"""

from gscodec.encoder.sorting.base import SortingStrategy
from gscodec.encoder.sorting.block_morton import BlockMortonSortingStrategy
from gscodec.encoder.sorting.morton import MortonSortingStrategy
from gscodec.encoder.sorting.plas import PLASSortingStrategy


class SortingStrategyFactory:
    """A factory for creating sorting strategy instances."""

    _strategies: dict[str, type[SortingStrategy]] = {
        "morton": MortonSortingStrategy,
        "plas": PLASSortingStrategy,
        "block_morton": BlockMortonSortingStrategy,
    }

    @staticmethod
    def create(strategy_name: str) -> SortingStrategy:
        """
        Creates a sorting strategy instance based on its name.

        Args:
            strategy_name: The name of the sorting strategy.

        Returns:
            An instance of the specified sorting strategy.

        Raises:
            ValueError: If the strategy_name is unknown.
        """
        strategy_name = strategy_name.lower()
        if strategy_name not in SortingStrategyFactory._strategies:
            raise ValueError(f"Unknown sorting strategy: {strategy_name}")

        return SortingStrategyFactory._strategies[strategy_name]()
