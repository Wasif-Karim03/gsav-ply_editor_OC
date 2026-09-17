"""Apply a validated source-row mask consistently on CPU and GPU."""

import numpy as np
import torch


class FixedMaskFilter:
    def __init__(self, mask, capture=None):
        self.mask = np.asarray(mask)
        self.capture = capture

    def _apply(self, data):
        if self.mask.dtype != np.bool_ or self.mask.shape != (len(data.means),):
            raise ValueError("Crop mask does not match this source frame")
        if self.capture is not None:
            self.capture.mask = self.mask
            return data
        mask = self.mask
        if torch.is_tensor(data.means):
            mask = torch.as_tensor(mask, device=data.means.device)
        # gsply slicing preserves attribute formats and clears stale packed buffers.
        return data[mask]

    def filter_cpu(self, data, config, scene_bounds):
        return self._apply(data)

    def filter_gpu(self, data, config, scene_bounds):
        return self._apply(data)
