"""Color processing using gsmod's native Triton-accelerated operations.

Uses GSTensorPro.color() which internally uses Triton kernels for
brightness/saturation adjustments when available.
"""

from __future__ import annotations

from gsmod import ColorValues, GSDataPro
from gsmod.torch import GSTensorPro

from src.domain.entities import GSData, GSTensor

from .protocols import ColorProcessor


class DefaultColorProcessor(ColorProcessor):
    """Color processor using gsmod's native Triton-accelerated operations.

    GPU: GSTensorPro.color() with Triton kernels for brightness/saturation
    CPU: GSDataPro.color() with Numba LUT kernels
    """

    def apply_gpu(
        self,
        gaussians: GSTensor,
        color_values: ColorValues,
        device: str,
    ) -> GSTensor:
        """Apply color adjustments using native GSTensorPro.color().

        Uses inplace=False to let gsmod handle copying internally.
        """
        if color_values.is_neutral():
            return gaussians

        # Wrap GSTensor as GSTensorPro if needed (preserves format state)
        if not isinstance(gaussians, GSTensorPro):
            gaussians = GSTensorPro.from_gstensor(gaussians)

        # gsply can clone into interleaved PLY storage (row stride > 3).
        # gsmod's RGB Triton kernels require densely packed color arrays.
        # Clone first: color(inplace=False) would repack our contiguous inputs.
        edited = gaussians.clone()
        edited.sh0 = edited.sh0.contiguous()
        if edited.shN is not None:
            edited.shN = edited.shN.contiguous()
        edited._base = None
        return edited.color(color_values, inplace=True)

    def apply_cpu(
        self,
        data: GSData,
        color_values: ColorValues,
    ) -> GSData:
        """Apply color adjustments using native GSDataPro.color()."""
        if color_values.is_neutral():
            return data

        # Ensure we have GSDataPro for native API
        if not isinstance(data, GSDataPro):
            data = GSDataPro.from_gsdata(data)

        # Native API with inplace=False handles copying internally
        edited = data.color(color_values, inplace=False)
        # Native edits may replace fields without updating packed storage.
        edited._base = None
        return edited
