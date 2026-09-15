import numpy as np
from numpy.typing import NDArray


def inverse_log_transform(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Applies the inverse of the log transform: `sign(x) * (exp(abs(x)) - 1)`.

    This function reverses the `log_transform` applied during encoding, which is
    used to map a wide dynamic range of values (like positions) to a more
    compact and perceptually uniform space before quantization.

    The use of `abs()` and `sign()` ensures that the transform is symmetric
    and can handle both positive and negative values correctly.

    Args:
        x: The input NumPy array, representing the log-transformed data.

    Returns:
        The transformed NumPy array, with values restored to their original
        (pre-transform) scale.
    """
    result: NDArray[np.floating] = np.sign(x) * (np.exp(np.abs(x)) - 1)
    return result


def inverse_sigmoid(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Applies the inverse of the sigmoid function (logit).

    This function is used to convert values from a probability-like range `[0, 1]`
    back into an unbounded logit space. This is typically used for attributes
    like opacity, which are often modeled as probabilities.

    The input `x` is clipped to a safe range just inside `(0, 1)` to avoid
    numerical instability issues like division by zero or `log(0)` when `x` is
    exactly 0 or 1, which can occur due to quantization artifacts.

    Args:
        x: The input NumPy array, with values expected to be in the range `[0, 1]`.

    Returns:
        The transformed NumPy array, representing the data in logit space.
    """
    # Clip the input to a small epsilon away from 0 and 1 to avoid infinity.
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))
