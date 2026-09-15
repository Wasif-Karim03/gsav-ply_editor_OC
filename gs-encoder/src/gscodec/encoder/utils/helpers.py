"""
This module provides a collection of shared, stateless mathematical utilities
designed for handling Gaussian Splatting data. These functions are used across
different parts of the processing pipeline, such as in encoding and decoding.
"""

import torch


def log_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Applies a log transform to the input tensor: sign(x) * log(1 + abs(x)).
    This can be useful for handling data with heavy-tailed distributions.

    Args:
        x: The input tensor.

    Returns:
        The transformed tensor.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))
