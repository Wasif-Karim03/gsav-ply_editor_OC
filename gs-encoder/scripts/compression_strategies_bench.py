"""Benchmark compression strategies for means_lo payload.

Tests each strategy on real PLY data and reports compression ratios.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gscodec.constants import LO_BASE, MIN_VAL, N_LEVELS, SCALE_16BIT
from scripts.finetune.io import compute_global_ranges, load_ply_sequence


def quantize_frame(means: torch.Tensor, mins: torch.Tensor, maxs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Quantize means to hi/lo uint8. Returns (hi [N,3], lo [N,3])."""
    means_log = torch.sign(means) * torch.log1p(torch.abs(means))
    scale = torch.where(maxs > mins, maxs - mins, torch.ones_like(mins))
    normalized = ((means_log - mins) / scale).clamp(0, 1)
    quantized = (normalized * SCALE_16BIT).floor().long()
    hi = (quantized // LO_BASE).clamp(0, N_LEVELS - 1)
    lo = (quantized % LO_BASE).clamp(0, 255)
    return (hi + MIN_VAL).cpu().numpy().astype(np.uint8), lo.cpu().numpy().astype(np.uint8)


def raw_size(lo_frames: list[np.ndarray]) -> int:
    return sum(lo.nbytes for lo in lo_frames)


def strategy_raw_zstd(lo_frames: list[np.ndarray], level: int = 3) -> int:
    """Baseline: raw lo bytes compressed with zstd."""
    cctx = zstd.ZstdCompressor(level=level)
    raw = b"".join(lo.tobytes() for lo in lo_frames)
    return len(cctx.compress(raw))


def strategy_temporal_delta_zstd(lo_frames: list[np.ndarray], level: int = 3) -> int:
    """Temporal delta coding: lo[t] - lo[t-1] mod 256, then zstd."""
    cctx = zstd.ZstdCompressor(level=level)
    parts: list[bytes] = [lo_frames[0].tobytes()]
    for i in range(1, len(lo_frames)):
        delta = (lo_frames[i].astype(np.int16) - lo_frames[i - 1].astype(np.int16)) % 256
        parts.append(delta.astype(np.uint8).tobytes())
    return len(cctx.compress(b"".join(parts)))


def strategy_spatial_delta_zstd(lo_frames: list[np.ndarray], level: int = 3) -> int:
    """Spatial delta coding: lo[i] - lo[i-1] along Morton order, then zstd."""
    cctx = zstd.ZstdCompressor(level=level)
    parts: list[bytes] = []
    for lo in lo_frames:
        delta = np.zeros_like(lo)
        delta[0] = lo[0]
        delta[1:] = (lo[1:].astype(np.int16) - lo[:-1].astype(np.int16)) % 256
        parts.append(delta.astype(np.uint8).tobytes())
    return len(cctx.compress(b"".join(parts)))


def strategy_spatial_temporal_delta_zstd(lo_frames: list[np.ndarray], level: int = 3) -> int:
    """Temporal delta first, then spatial delta, then zstd."""
    cctx = zstd.ZstdCompressor(level=level)
    parts: list[bytes] = []
    prev_lo: np.ndarray | None = None
    for lo in lo_frames:
        if prev_lo is not None:
            temp_delta = (lo.astype(np.int16) - prev_lo.astype(np.int16)) % 256
        else:
            temp_delta = lo.astype(np.int16)
        # Spatial delta on the temporal residual
        spatial = np.zeros_like(temp_delta)
        spatial[0] = temp_delta[0]
        spatial[1:] = (temp_delta[1:] - temp_delta[:-1]) % 256
        parts.append(spatial.astype(np.uint8).tobytes())
        prev_lo = lo
    return len(cctx.compress(b"".join(parts)))


def strategy_ksnap_zstd(lo_frames: list[np.ndarray], K: int, level: int = 3) -> int:
    """K-snap: round lo to nearest K, then zstd."""
    cctx = zstd.ZstdCompressor(level=level)
    parts: list[bytes] = []
    for lo in lo_frames:
        snapped = ((lo.astype(np.int16) + K // 2) // K * K) % 256
        parts.append(snapped.astype(np.uint8).tobytes())
    return len(cctx.compress(b"".join(parts)))


def strategy_ksnap_temporal_delta_zstd(lo_frames: list[np.ndarray], K: int, level: int = 3) -> int:
    """K-snap + temporal delta + zstd."""
    cctx = zstd.ZstdCompressor(level=level)
    snapped_frames = [((lo.astype(np.int16) + K // 2) // K * K % 256).astype(np.uint8) for lo in lo_frames]
    parts: list[bytes] = [snapped_frames[0].tobytes()]
    for i in range(1, len(snapped_frames)):
        delta = (snapped_frames[i].astype(np.int16) - snapped_frames[i - 1].astype(np.int16)) % 256
        parts.append(delta.astype(np.uint8).tobytes())
    return len(cctx.compress(b"".join(parts)))


def strategy_ksnap_spatial_delta_zstd(lo_frames: list[np.ndarray], K: int, level: int = 3) -> int:
    """K-snap + spatial delta + zstd."""
    cctx = zstd.ZstdCompressor(level=level)
    parts: list[bytes] = []
    for lo in lo_frames:
        snapped = ((lo.astype(np.int16) + K // 2) // K * K % 256).astype(np.uint8)
        delta = np.zeros_like(snapped)
        delta[0] = snapped[0]
        delta[1:] = (snapped[1:].astype(np.int16) - snapped[:-1].astype(np.int16)) % 256
        parts.append(delta.astype(np.uint8).tobytes())
    return len(cctx.compress(b"".join(parts)))


def strategy_ksnap_spatial_temporal_delta_zstd(lo_frames: list[np.ndarray], K: int, level: int = 3) -> int:
    """K-snap + temporal delta + spatial delta + zstd."""
    cctx = zstd.ZstdCompressor(level=level)
    snapped_frames = [((lo.astype(np.int16) + K // 2) // K * K % 256).astype(np.uint8) for lo in lo_frames]
    parts: list[bytes] = []
    prev: np.ndarray | None = None
    for snap in snapped_frames:
        if prev is not None:
            temp = (snap.astype(np.int16) - prev.astype(np.int16)) % 256
        else:
            temp = snap.astype(np.int16)
        spatial = np.zeros_like(temp)
        spatial[0] = temp[0]
        spatial[1:] = (temp[1:] - temp[:-1]) % 256
        parts.append(spatial.astype(np.uint8).tobytes())
        prev = snap
    return len(cctx.compress(b"".join(parts)))


def compute_max_error(K: int, mins: torch.Tensor, maxs: torch.Tensor) -> float:
    """Max position error from K-snapping in world units."""
    scale = (maxs - mins).cpu().numpy()
    max_lo_err = K / 2  # max error in lo-byte units
    max_norm_err = max_lo_err / SCALE_16BIT
    return float(np.max(max_norm_err * scale))


def main() -> None:
    device = "cuda:0"
    input_dir = Path("D:/dymensium_soccer/plys")

    print("Loading PLYs...")
    frames, _ = load_ply_sequence(input_dir, device, max_frames=30)
    mins, maxs = compute_global_ranges(frames)
    print(f"Loaded {len(frames)} frames, {frames[0].means.shape[0]} Gaussians each")

    # Quantize all frames
    print("Quantizing...")
    lo_frames: list[np.ndarray] = []
    for f in frames:
        _, lo = quantize_frame(f.means, mins, maxs)
        lo_frames.append(lo)

    raw = raw_size(lo_frames)
    print(f"\nRaw lo size: {raw:,} bytes ({raw / 1024 / 1024:.2f} MB)")
    print(f"{'Strategy':<50} {'Size':>10} {'Ratio':>7} {'Savings':>8} {'Max err':>10}")
    print("-" * 90)

    results: list[tuple[str, int, str]] = []

    # Baseline
    sz = strategy_raw_zstd(lo_frames)
    results.append(("raw + zstd", sz, ""))
    print(f"{'raw + zstd':<50} {sz:>10,} {raw/sz:>7.2f}x {(1-sz/raw)*100:>7.1f}% {'—':>10}")

    # Temporal delta
    sz = strategy_temporal_delta_zstd(lo_frames)
    results.append(("temporal_delta + zstd", sz, ""))
    print(f"{'temporal_delta + zstd':<50} {sz:>10,} {raw/sz:>7.2f}x {(1-sz/raw)*100:>7.1f}% {'lossless':>10}")

    # Spatial delta
    sz = strategy_spatial_delta_zstd(lo_frames)
    results.append(("spatial_delta + zstd", sz, ""))
    print(f"{'spatial_delta + zstd':<50} {sz:>10,} {raw/sz:>7.2f}x {(1-sz/raw)*100:>7.1f}% {'lossless':>10}")

    # Spatial + temporal delta
    sz = strategy_spatial_temporal_delta_zstd(lo_frames)
    results.append(("spatial+temporal_delta + zstd", sz, ""))
    print(f"{'spatial+temporal_delta + zstd':<50} {sz:>10,} {raw/sz:>7.2f}x {(1-sz/raw)*100:>7.1f}% {'lossless':>10}")

    # K-snap variants
    for K in [4, 8, 16]:
        max_err = compute_max_error(K, mins, maxs)

        sz = strategy_ksnap_zstd(lo_frames, K)
        label = f"K-snap K={K} + zstd"
        print(f"{label:<50} {sz:>10,} {raw/sz:>7.2f}x {(1-sz/raw)*100:>7.1f}% {max_err:>10.6f}")

        sz = strategy_ksnap_temporal_delta_zstd(lo_frames, K)
        label = f"K-snap K={K} + temporal_delta + zstd"
        print(f"{label:<50} {sz:>10,} {raw/sz:>7.2f}x {(1-sz/raw)*100:>7.1f}% {max_err:>10.6f}")

        sz = strategy_ksnap_spatial_delta_zstd(lo_frames, K)
        label = f"K-snap K={K} + spatial_delta + zstd"
        print(f"{label:<50} {sz:>10,} {raw/sz:>7.2f}x {(1-sz/raw)*100:>7.1f}% {max_err:>10.6f}")

        sz = strategy_ksnap_spatial_temporal_delta_zstd(lo_frames, K)
        label = f"K-snap K={K} + spatial+temporal_delta + zstd"
        print(f"{label:<50} {sz:>10,} {raw/sz:>7.2f}x {(1-sz/raw)*100:>7.1f}% {max_err:>10.6f}")


if __name__ == "__main__":
    main()
