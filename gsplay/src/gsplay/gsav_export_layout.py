"""Conservative eligibility checks for GSAV exports that retain source rows."""

import os

import numpy as np
import torch

from src.models.gsav import GsavModel


class SourceLayout:
    """Track source slots outside editable Gaussian attributes; never infer identities."""

    fields = ("means", "scales", "quats", "opacities")

    def __init__(self, model, times, fps, enabled=True):
        self.model = model
        self.reason = "Source or frame selection requires ordinary export"
        self.valid = False
        self.chunk_size = 0
        self.count = None
        if not enabled or os.environ.get("GSPLAY_FAST_EXPORT", "1") == "0":
            self.reason = "Fast export disabled"
            return
        if not isinstance(model, GsavModel):
            return
        n = model.get_total_frames()
        meta = model.gsav_metadata
        size = meta.get("chunk_size", 0)
        if (
            list(times) != list(range(n))
            or fps != meta.get("fps")
            or not isinstance(size, int)
            or size <= 0
        ):
            return
        expected = [[i, min(i + size, n)] for i in range(0, n, size)]
        if meta.get("chunks") != expected:
            self.reason = "Source has unsupported chunk boundaries"
            return
        self.chunk_size = size
        self.valid = True
        self.reason = "Original GSAV arrangement preserved"

    def before(self, frame):
        if not self.valid:
            return None
        return {key: getattr(frame, key).detach().cpu().numpy().copy() for key in self.fields}

    def after(self, index, reference, edited):
        if not self.valid:
            return None
        for key, original in reference.items():
            value = getattr(edited, key)
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            if not np.array_equal(original, value):
                self.valid = False
                self.reason = "Geometry, opacity, filtering or row order changed"
                return None
        presence = self.model._raw(index).get("presence")
        n = len(reference["means"])
        if (
            presence is None
            or presence.shape != (n,)
            or (self.count is not None and self.count != n)
            or not np.array_equal(presence, reference["opacities"].reshape(-1) > 0)
        ):
            self.valid = False
            self.reason = "Source presence or row count cannot be preserved"
            return None
        self.count = n
        return presence.copy()
