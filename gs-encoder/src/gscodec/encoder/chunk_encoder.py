"""
Chunk encoder for frame-by-frame GSAV codec.

Each chunk processes a group of frames. Every frame is fully quantized
independently using global ranges and packed into a 3x5 atlas.
means_lo is returned separately as raw binary data.
Chunks exist only for GOP/I-frame alignment.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from gsply import GSTensor

from gscodec.common.types import QuantRanges
from gscodec.encoder.masks import first_active_sort_frame, reuse_inactive_rows, sanitize_frame
from gscodec.encoder.sorting.morton import MortonSortingStrategy
from gscodec.encoder.utils.helpers import log_transform
from gscodec.encoder.video_writer import build_frame_atlas

logger = logging.getLogger(__name__)


@dataclass
class EncodedFrame:
    """Result of encoding a single frame.

    Attributes:
        atlas: [H, W] uint8 frame atlas (3x5 or 3x6 layout).
        means_lo: [N, 3] uint8 lo-bytes.
    """

    atlas: np.ndarray
    means_lo: np.ndarray
    presence: np.ndarray | None = None


class ChunkEncoder:
    """Encodes a chunk of frames into per-frame atlases."""

    def __init__(
        self,
        device: str = "cpu",
        skip_sorting: bool = False,
        global_means_min: torch.Tensor | None = None,
        global_means_max: torch.Tensor | None = None,
        lo_snap_k: int | tuple[int, int, int] = 1,
        sh_bands: int = 0,
        sh_max_centroids: int = 48400,
        identity_mode: str = "auto",
        reuse_inactive: bool = True,
        keyframe_snap: bool = True,
    ):
        """Initialize chunk encoder.

        Args:
            device: Torch device for computation.
            skip_sorting: If True, assume frames are already sorted.
            global_means_min: [3] global min of log-transformed means (for per-frame sort).
            global_means_max: [3] global max of log-transformed means (for per-frame sort).
            lo_snap_k: Round lo bytes to nearest K (1=off). Use coprime with 256 (e.g. 35).
                Scalar applies to all axes; a (kx, ky, kz) tuple snaps per-axis.
            sh_bands: SH band level (0=DC only, 1-3=higher bands).
            sh_max_centroids: Maximum SH palette entries.
        """
        self.device = device
        self.skip_sorting = skip_sorting
        self.sorter = MortonSortingStrategy()
        self.global_means_min = global_means_min
        self.global_means_max = global_means_max
        self.lo_snap_k = lo_snap_k
        self.sh_bands = sh_bands
        self.sh_max_centroids = sh_max_centroids
        self.identity_mode = identity_mode
        self.reuse_inactive = reuse_inactive
        self.keyframe_snap = keyframe_snap

    def encode(
        self,
        frames: list[GSTensor],
        global_ranges: QuantRanges,
        target_n: int | None = None,
    ) -> tuple[list[EncodedFrame], list[torch.Tensor] | None]:
        """Encode a chunk of frames into per-frame atlases + means arrays.

        Adaptively chooses single-sort (for stable correspondence) or
        per-frame composite sort (for dynamic/unstructured sequences).
        Padding to target_n (if needed) happens AFTER sorting so that
        padding gaussians stay at the end of the sorted order.

        Args:
            frames: List of GSTensor frames in this chunk.
            global_ranges: Global quantization ranges.
            target_n: Pad to this count after sorting. None = no padding.

        Returns:
            Tuple of (list of EncodedFrame per frame, raw shN tensors or None).
        """
        if not frames:
            raise ValueError("Cannot encode empty chunk")
        frames = [sanitize_frame(f) for f in frames]
        if self.identity_mode == "stable" and any(len(f.means) != len(frames[0].means) for f in frames):
            raise ValueError("Stable identities require a constant count within each GOP")

        if self.skip_sorting:
            sorted_frames = frames
        elif self.global_means_min is not None and self.global_means_max is not None:
            stable = self.identity_mode == "stable" or self._has_stable_correspondence(frames)
            if stable:
                _, morton_indices = self.sorter.sort(first_active_sort_frame(frames))
                sorted_frames = [f[morton_indices] for f in frames]
                logger.info("Stable correspondence detected — using single sort")
            else:
                quats_min = torch.from_numpy(global_ranges.quats_min).to(frames[0].means.device)
                quats_max = torch.from_numpy(global_ranges.quats_max).to(frames[0].means.device)
                sorted_frames = []
                for frame in frames:
                    means_log = log_transform(frame.means)
                    idx = self.sorter.sort_with_global_bbox(
                        means_log,
                        self.global_means_min,
                        self.global_means_max,
                        secondary=frame.quats.clamp(-1, 1),
                        secondary_min=quats_min,
                        secondary_max=quats_max,
                        pos_bits=4,
                        sec_bits=4,
                    )
                    sorted_frames.append(frame[idx])
        else:
            _, morton_indices = self.sorter.sort(first_active_sort_frame(frames))
            sorted_frames = [f[morton_indices] for f in frames]

        # Pad AFTER sorting — padding stays at the end of sorted order
        if target_n is not None:
            sorted_frames = [self._pad_to(f, target_n) for f in sorted_frames]

        n_gaussians = sorted_frames[0].means.shape[0]
        range_tensors = self._prepare_range_tensors(global_ranges, sorted_frames[0].means.device)

        return self._batch_quantize_and_build(sorted_frames, range_tensors, n_gaussians)

    @staticmethod
    def _pad_to(frame: GSTensor, target_n: int) -> GSTensor:
        """Pad frame by repeating the last gaussian to target_n."""
        n = frame.means.shape[0]
        if n >= target_n:
            return frame[:target_n]
        pad = target_n - n
        means = torch.cat([frame.means, frame.means[-1:].expand(pad, -1)])
        scales = torch.cat([frame.scales, frame.scales[-1:].expand(pad, -1)])
        quats = torch.cat([frame.quats, frame.quats[-1:].expand(pad, -1)])
        opacities = torch.cat([frame.opacities, frame.opacities[-1:].expand(pad, -1)])
        sh0 = torch.cat([frame.sh0, frame.sh0[-1:].expand(pad, -1)])
        shN = None
        if frame.shN is not None and frame.shN.numel() > 0:
            shN = torch.cat([frame.shN, frame.shN[-1:].expand(pad, *frame.shN.shape[1:])])
        masks = torch.zeros(target_n, dtype=torch.bool, device=frame.means.device)
        masks[:n] = frame.masks
        return GSTensor(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            masks=masks,
        )

    @staticmethod
    def _has_stable_correspondence(frames: list[GSTensor]) -> bool:
        """Detect if frames have stable Gaussian correspondence."""
        if len(frames) < 2:
            return False
        f0, f1 = frames[0], frames[min(1, len(frames) - 1)]

        # If the number of Gaussians changes, it's not stable
        if f0.means.shape != f1.means.shape:
            return False

        for frame in frames[1:]:
            if frame.means.shape != f0.means.shape:
                return False
            active = f0.masks.reshape(-1).bool() & frame.masks.reshape(-1).bool()
            if not active.any():
                return False
            diff = (f0.means[active] - frame.means[active]).abs().mean().item()
            extent = max(f0.means[active].abs().max().item(), 1e-6)
            if diff >= extent * 0.01:
                return False
        return True

    def _batch_quantize_and_build(
        self,
        sorted_frames: list[GSTensor],
        rt: dict[str, torch.Tensor],
        n_gaussians: int,
    ) -> tuple[list[EncodedFrame], list[torch.Tensor] | None]:
        """Batch quantize all frames on GPU, then build atlases on CPU."""
        T = len(sorted_frames)

        # Stack all attributes into batched tensors on GPU — [T, N, C]
        all_means_log = torch.stack([log_transform(f.means) for f in sorted_frames])
        all_scales = torch.stack(
            [f.scales.clamp(min=rt["scales_min"], max=rt["scales_max"]) for f in sorted_frames]
        )
        all_quats = torch.stack([f.quats.clamp(-1, 1) for f in sorted_frames])
        # Arcsine transform: redistributes quantization precision to match
        # the quat distribution (dense near 0, sparse near ±1). Produces
        # more uniform quantized values → lower VP9 TM_PRED residuals.
        # Decoder applies inverse: sin(x * π/2). Cost: 1 GPU sin() per component.
        all_quats = torch.asin(all_quats.clamp(-0.9999, 0.9999)) / (3.14159265 / 2)
        all_opacities = torch.stack([f.opacities.clamp(-6, 12) for f in sorted_frames])
        all_sh0 = torch.stack([
            f.sh0 if self.identity_mode == "preserve" else f.sh0.clamp(-2, 4)
            for f in sorted_frames
        ])
        # SH0 RGB→YCbCr decorrelation: concentrates energy in Y channel.
        _ycbcr_mat = torch.tensor(
            [
                [0.299, 0.587, 0.114],
                [-0.169, -0.331, 0.500],
                [0.500, -0.419, -0.081],
            ],
            device=all_sh0.device,
            dtype=all_sh0.dtype,
        )
        all_sh0 = torch.einsum("...c,dc->...d", all_sh0, _ycbcr_mat)

        # Batch 16-bit means quantization — [T, N, 3]
        scale_m = torch.where(
            rt["means_max"] > rt["means_min"],
            rt["means_max"] - rt["means_min"],
            torch.ones_like(rt["means_min"]),
        )
        norm_m = ((all_means_log - rt["means_min"]) / scale_m).clamp(0, 1)
        from gscodec.constants import LO_BASE, N_LEVELS, SCALE_16BIT

        quant_m = (norm_m * SCALE_16BIT).floor().long()
        hi_all = (quant_m // LO_BASE).clamp(0, N_LEVELS - 1)
        lo_all = (quant_m % LO_BASE).clamp(0, 255)

        # K-snap lo (scalar applies to all axes; tuple gives per-axis (kx, ky, kz))
        if isinstance(self.lo_snap_k, tuple | list):
            for ax, k in enumerate(self.lo_snap_k):
                if k > 1:
                    lo_all[..., ax] = ((lo_all[..., ax] + k // 2) // k * k) % 256
        elif self.lo_snap_k > 1:
            k = self.lo_snap_k
            lo_all = ((lo_all + k // 2) // k * k) % 256

        # Batch 8-bit quantization with nearest rounding (+2-5 dB vs floor)
        from gscodec.constants import MIN_VAL as MV
        from gscodec.constants import SCALE_8BIT

        def batch_q8(data: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor) -> np.ndarray:
            s = torch.where(vmax > vmin, vmax - vmin, torch.ones_like(vmin))
            n = ((data - vmin) / s).clamp(0, 1)
            return (
                (n * SCALE_8BIT).round().clamp(0, SCALE_8BIT).add_(MV).to(torch.uint8).cpu().numpy()
            )

        hi_np = (hi_all + MV).to(torch.uint8).cpu().numpy()  # [T, N, 3]
        lo_np = lo_all.to(torch.uint8).cpu().numpy()  # [T, N, 3]
        sc_np = batch_q8(all_scales, rt["scales_min"], rt["scales_max"])
        qt_np = batch_q8(all_quats, rt["quats_min"], rt["quats_max"])
        op_np = batch_q8(all_opacities, rt["opacity_min"], rt["opacity_max"])
        sh_np = batch_q8(all_sh0, rt["sh0_min"], rt["sh0_max"])

        del all_means_log, all_scales, all_quats, all_opacities, all_sh0
        del hi_all, lo_all, quant_m, norm_m

        # Snap ≤1-level changes to keyframe (frame 0) — VP9 encodes
        # zero-diff for free. Snaps to keyframe (not previous frame) to
        # prevent drift accumulation across the chunk.
        # EXCLUDE means_hi: 1 hi level = 256 lo-bytes of position drift.
        # Per-attribute keyframe snap thresholds tuned for quality/compression:
        # scales ≤2, quats ≤2, opacity ≤3, sh0 ≤3.
        presence = [f.masks.detach().cpu().numpy().reshape(-1).astype(bool) for f in sorted_frames]
        if T > 1 and self.keyframe_snap:
            continuous = np.stack(presence[1:]) & np.stack(presence[:-1]) & presence[0]
            for arr, thresh in [(sc_np, 2), (qt_np, 2), (op_np, 3), (sh_np, 3)]:
                kf_i16 = arr[0].astype(np.int16)
                mask = np.abs(arr[1:].astype(np.int16) - kf_i16) <= thresh
                mask &= continuous[..., None]
                arr[1:][mask] = np.broadcast_to(arr[0], arr[1:].shape)[mask]

        if self.reuse_inactive:
            for arr in (hi_np, lo_np, sc_np, qt_np, op_np, sh_np):
                reuse_inactive_rows(arr, presence)

        # Collect raw SH tensors (global encoding happens in sequence_encoder)
        sh_tensors: list[torch.Tensor] | None = None

        if self.sh_bands > 0:
            frames_shN = [f.shN for f in sorted_frames if f.shN is not None and f.shN.numel() > 0]
            if frames_shN:
                sh_tensors = frames_shN

        # Build atlases on CPU
        results: list[EncodedFrame] = []
        for t in range(T):
            op_1d = op_np[t].squeeze(-1) if op_np[t].ndim > 1 else op_np[t]
            atlas = build_frame_atlas(
                hi_np[t],
                sc_np[t],
                qt_np[t],
                op_1d,
                sh_np[t],
                n_gaussians,
                presence=sorted_frames[t].masks.detach().cpu().numpy().reshape(-1),
            )
            results.append(EncodedFrame(atlas=atlas, means_lo=lo_np[t], presence=presence[t]))

        return results, sh_tensors

    def _prepare_range_tensors(
        self,
        global_ranges: QuantRanges,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        """Convert QuantRanges to device tensors (computed once per chunk)."""
        return {
            "means_min": torch.from_numpy(global_ranges.means_min).to(device),
            "means_max": torch.from_numpy(global_ranges.means_max).to(device),
            "scales_min": torch.from_numpy(global_ranges.scales_min).to(device),
            "scales_max": torch.from_numpy(global_ranges.scales_max).to(device),
            "quats_min": torch.from_numpy(global_ranges.quats_min).to(device),
            "quats_max": torch.from_numpy(global_ranges.quats_max).to(device),
            "opacity_min": torch.tensor([global_ranges.opacity_min], device=device),
            "opacity_max": torch.tensor([global_ranges.opacity_max], device=device),
            "sh0_min": torch.from_numpy(global_ranges.sh0_min).to(device),
            "sh0_max": torch.from_numpy(global_ranges.sh0_max).to(device),
        }
