"""Persistent combined Gaussian buffers for joint visibility and depth sorting."""

from typing import TYPE_CHECKING

import numpy as np
import torch
from gsply import GSData

if TYPE_CHECKING:
    from gscodec.decoder.sequence_decoder import SequenceDecoder


class SceneBuffer:
    """Static prefix initialized once; dynamic suffix updated in place.

    Tensors are in rendering space: linear scales/alpha, unit quaternions,
    and SH coefficients. ``render`` submits the entire scene in one call so
    static and dynamic primitives participate in the same depth ordering.
    The object is mutable and must not be updated concurrently with rendering.
    """

    def __init__(self, decoder: "SequenceDecoder", device: str = "cpu"):
        self.decoder = decoder
        self.device = torch.device(device)
        static = decoder.decode_static_asset()
        self.n_static = 0 if static is None else len(static.means)
        self.n_dynamic = decoder.n_gaussians
        self.dynamic_offset = self.n_static
        self.n_gaussians = self.n_static + self.n_dynamic
        static_coeffs = 0 if static is None or static.shN is None else static.shN.shape[1]
        dynamic_coeffs = (decoder._provider.sh_bands + 1) ** 2 - 1
        self.sh_degree = int(np.sqrt(max(static_coeffs, dynamic_coeffs) + 1)) - 1
        n, k = self.n_gaussians, (self.sh_degree + 1) ** 2
        self.means = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.scales = torch.zeros_like(self.means)
        self.quats = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        self.quats[:, 0] = 1
        self.opacities = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.colors = torch.zeros((n, k, 3), dtype=torch.float32, device=self.device)
        self.presence = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.frame_index = None
        if static is not None:
            self._write(static, 0, self.n_static)

    @torch.no_grad()
    def _write(self, frame: GSData, offset: int, count: int) -> None:
        if len(frame.means) != count:
            raise ValueError("Decoded Gaussian count changed within a track")
        region = slice(offset, offset + count)
        for name in ("means", "scales", "quats"):
            getattr(self, name)[region].copy_(torch.from_numpy(getattr(frame, name)))
        self.scales[region].exp_()
        self.opacities[region].copy_(torch.from_numpy(frame.opacities.reshape(-1)))
        self.opacities[region].sigmoid_()
        if frame.masks is None:
            self.presence[region].fill_(True)
        else:
            self.presence[region].copy_(torch.from_numpy(frame.masks.reshape(-1)))
        self.opacities[region].masked_fill_(~self.presence[region], 0)
        self.colors[region].zero_()
        self.colors[region, 0].copy_(torch.from_numpy(frame.sh0))
        if frame.shN is not None and frame.shN.size:
            self.colors[region, 1 : 1 + frame.shN.shape[1]].copy_(torch.from_numpy(frame.shN))

    def update(self, frame_index: int) -> "SceneBuffer":
        """Decode and upload only the dynamic suffix; repeated requests are no-ops."""
        if frame_index != self.frame_index:
            dynamic = self.decoder.decode_frame(frame_index)
            self._write(dynamic, self.dynamic_offset, self.n_dynamic)
            self.frame_index = frame_index
        return self

    @torch.no_grad()
    def render(self, viewmats: torch.Tensor, Ks: torch.Tensor, width: int, height: int):
        """Rasterize one merged scene, including view-dependent static SH."""
        if self.frame_index is None:
            raise ValueError("Call update(frame_index) before rendering")
        import gsplat

        return gsplat.rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            sh_degree=self.sh_degree,
            viewmats=viewmats.to(self.device),
            Ks=Ks.to(self.device),
            width=width,
            height=height,
            packed=True,
            render_mode="RGB",
        )
