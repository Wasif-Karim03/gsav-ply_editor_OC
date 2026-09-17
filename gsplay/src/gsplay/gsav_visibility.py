"""Export-only filtering that retains the original GSAV row identities."""

from types import SimpleNamespace

import numpy as np
import torch
from gsmod.filter.apply import compute_filter_mask
from gsmod.torch import GSTensorPro

from src.gsplay.processing.volume_filter import VolumeFilterService


class ExportVisibility(VolumeFilterService):
    """Apply the existing filter criteria as visibility, without compacting rows."""

    def __init__(self):
        self.mask = None
        self.crop_only = False

    def with_mask(self, mask):
        from src.gsplay.processing.fixed_mask import FixedMaskFilter

        return FixedMaskFilter(mask, capture=self)

    def _capture(self, data, config):
        def numpy(value):
            return value.detach().cpu().numpy() if torch.is_tensor(value) else value

        values = SimpleNamespace(
            **{name: numpy(getattr(data, name)) for name in ("means", "scales", "opacities")},
            is_opacities_ply=data.is_opacities_ply,
            is_scales_ply=data.is_scales_ply,
        )
        self.mask = (
            np.ones(len(data.means), dtype=bool)
            if config.filter_values.is_neutral()
            else compute_filter_mask(values, config.filter_values)
        )

    def filter_cpu(self, data, config, scene_bounds):
        self._capture(data, config)
        return data

    def filter_gpu(self, gaussians, config, scene_bounds):
        # Ask the exact preview filter to select explicit source indices. This
        # scratch tensor carries IDs in its unused color field; real colors and
        # geometry are never modified. Float32 IDs are exact below 2**24 rows.
        n = len(gaussians.means)
        if n >= 2**24:
            raise ValueError("Too many source rows for exact filter identities")
        indices = torch.arange(n, dtype=torch.float32, device=gaussians.device)
        carrier = GSTensorPro(
            means=gaussians.means,
            scales=gaussians.scales,
            opacities=gaussians.opacities,
            quats=gaussians.quats,
            sh0=indices[:, None].expand(-1, 3),
            shN=None,
        )
        carrier._format = gaussians._format.copy()
        selected = carrier.filter(config.filter_values, inplace=True).sh0[:, 0].long()
        mask = torch.zeros(n, dtype=torch.bool, device=gaussians.device)
        mask[selected] = True
        self.mask = mask.cpu().numpy()
        return None


def create_export_manager(config, device, model, times, fps, output_format):
    """Use visibility only when geometry reuse can be verified for the whole source."""
    from src.gsplay.core.container import build_default_processing_providers, create_edit_manager
    from src.gsplay.gsav_export_layout import SourceLayout

    visibility = None
    providers = build_default_processing_providers()
    if (
        output_format == "GSAV"
        and config.edits_active
        and not config.filter_values.is_neutral()
        and config.transform_values.is_neutral()
        and config.alpha_scaler == 1
        and SourceLayout(model, times, fps).valid
    ):
        visibility = ExportVisibility()
        visibility.crop_only = config.color_values.is_neutral()
        providers.volume_filter = visibility
    return create_edit_manager(config, device, providers), visibility
