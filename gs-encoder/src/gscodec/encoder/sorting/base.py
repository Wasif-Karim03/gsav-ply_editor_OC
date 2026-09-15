"""Defines the abstract base class for sorting strategies."""

from abc import ABC, abstractmethod
from typing import Any

from gsply import GSTensor
from torch import Tensor


class SortingStrategy(ABC):
    """Abstract base class for all sorting strategies."""

    @abstractmethod
    def sort(
        self,
        splats: GSTensor,
        **kwargs: Any,
    ) -> tuple[GSTensor, Tensor]:
        """
        Sorts a GSTensor of splats.

        Args:
            splats (GSTensor): The splat data to sort.
            **kwargs: Additional strategy-specific arguments.

        Returns:
            A tuple containing:
                - A new GSTensor with all splat tensors sorted.
                - A tensor of the sorted indices.
        """
        pass
