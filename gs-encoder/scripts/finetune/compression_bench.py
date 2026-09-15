"""Compression benchmarking for grid-snap fine-tuning.

Three tiers of measurement:
- Proxy (every step): snap_rate — % of lo bytes within ±2 of 0. GPU-side, free.
- Cheap (periodic): delta_lo_entropy — Shannon entropy of delta-coded lo. GPU, ~2ms.
- Medium (periodic): zstd compress extracted lo bytes. CPU, ~50ms.
"""

import numpy as np
import torch
import zstandard as zstd
from torch import Tensor

from gscodec.constants import SCALE_16BIT


def snap_rate(
    means: Tensor,
    means_min: Tensor,
    means_max: Tensor,
    threshold: int = 2,
) -> float:
    """Compute fraction of lo bytes within ±threshold of 0.

    GPU-side proxy metric — effectively free.

    Args:
        means: [N, 3] raw Gaussian positions.
        means_min: [3] global min of log-transformed means.
        means_max: [3] global max of log-transformed means.
        threshold: Distance threshold for "snapped" classification.

    Returns:
        Fraction in [0, 1].
    """
    lo = _extract_lo_continuous(means, means_min, means_max)
    dist = torch.minimum(lo, 256.0 - lo)
    return (dist <= threshold).float().mean().item()


def lo_entropy(
    means: Tensor,
    means_min: Tensor,
    means_max: Tensor,
) -> float:
    """Compute Shannon entropy of quantized lo bytes.

    Args:
        means: [N, 3] raw Gaussian positions.
        means_min: [3] global min of log-transformed means.
        means_max: [3] global max of log-transformed means.

    Returns:
        Entropy in bits/byte.
    """
    lo_bytes = _extract_lo_discrete(means, means_min, means_max)
    counts = torch.bincount(lo_bytes.flatten().long(), minlength=256).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -(probs * probs.log2()).sum().item()


def delta_lo_entropy(
    means_list: list[Tensor],
    means_min: Tensor,
    means_max: Tensor,
) -> float:
    """Shannon entropy of delta-coded lo bytes across a chunk.

    Matches the actual zstd input pipeline: first frame is raw,
    subsequent frames are (lo_t - lo_{t-1}) % 256.

    Args:
        means_list: List of [N, 3] means tensors (one per frame).
        means_min: [3] global min of log-transformed means.
        means_max: [3] global max of log-transformed means.

    Returns:
        Entropy in bits/byte. Predicts zstd_size ≈ raw_bytes * entropy / 8.
    """
    all_deltas: list[Tensor] = []
    prev_lo: Tensor | None = None

    for means in means_list:
        lo = _extract_lo_discrete(means, means_min, means_max)
        if prev_lo is not None:
            delta = (lo.long() - prev_lo.long()) % 256
            all_deltas.append(delta)
        else:
            all_deltas.append(lo)
        prev_lo = lo

    combined = torch.cat([d.flatten() for d in all_deltas])
    counts = torch.bincount(combined.long(), minlength=256).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -(probs * probs.log2()).sum().item()


def zstd_compressed_size(
    means_list: list[Tensor],
    means_min: Tensor,
    means_max: Tensor,
    level: int = 3,
) -> int:
    """Compress lo bytes from multiple frames with zstd.

    Applies chunk-delta coding (frame-to-frame delta) before compression.

    Args:
        means_list: List of [N, 3] means tensors (one per frame).
        means_min: [3] global min of log-transformed means.
        means_max: [3] global max of log-transformed means.
        level: Zstd compression level.

    Returns:
        Compressed size in bytes.
    """
    cctx = zstd.ZstdCompressor(level=level)

    all_lo: list[np.ndarray] = []
    prev_lo: np.ndarray | None = None

    for means in means_list:
        lo = _extract_lo_discrete(means, means_min, means_max)
        lo_np = lo.cpu().numpy().astype(np.uint8)

        if prev_lo is not None:
            delta = (lo_np.astype(np.int16) - prev_lo.astype(np.int16)) % 256
            all_lo.append(delta.astype(np.uint8))
        else:
            all_lo.append(lo_np)
        prev_lo = lo_np

    raw = b"".join(a.tobytes() for a in all_lo)
    return len(cctx.compress(raw))


def _extract_lo_continuous(
    means: Tensor,
    means_min: Tensor,
    means_max: Tensor,
) -> Tensor:
    """Extract continuous lo values (for snap_rate proxy).

    Returns:
        [N, 3] tensor with lo values in [0, 256).
    """
    means_log = torch.sign(means) * torch.log1p(torch.abs(means))
    scale = (means_max - means_min).clamp(min=1e-8)
    normalized = ((means_log - means_min) / scale).clamp(0.0, 1.0)
    v_cont = normalized * SCALE_16BIT
    return torch.fmod(v_cont, 256.0)


def _extract_lo_discrete(
    means: Tensor,
    means_min: Tensor,
    means_max: Tensor,
) -> Tensor:
    """Extract discrete lo bytes (matches encoder exactly).

    Returns:
        [N, 3] tensor with integer lo values in [0, 255].
    """
    means_log = torch.sign(means) * torch.log1p(torch.abs(means))
    scale = (means_max - means_min).clamp(min=1e-8)
    normalized = ((means_log - means_min) / scale).clamp(0.0, 1.0)
    quantized = (normalized * SCALE_16BIT).floor().long()
    return (quantized % 256).clamp(0, 255)
