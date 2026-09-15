"""Process-based parallel chunk encoding for the GSAV encoder.

Each chunk (Morton sort -> temporal match -> quantize -> atlas) is independent:
it conforms to its own keyframe, so chunks share no state beyond the global
quantization ranges and config. Encoding them in separate processes overlaps the
CPU-bound FAISS matching of one chunk with the GPU-bound atlas encode of another.

The single GPU saturates around four worker processes (~2x wall-clock vs serial).
Atlases are returned via temp ``.npz`` files rather than pickled through the pool
queue, so the large per-frame arrays don't bottleneck IPC.
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
from gsply import GSTensor

from gscodec.common.types import QuantRanges


@dataclass
class ChunkWorkerArgs:
    """Self-contained inputs for encoding one chunk in a worker process.

    Frame attributes are passed as CPU tensors (picklable under spawn); the
    worker reconstructs ``GSTensor`` and moves to GPU itself.
    """

    chunk_idx: int
    frame_tensors: list[dict[str, torch.Tensor]]  # per-frame attribute dicts
    ranges: QuantRanges
    means_min: torch.Tensor  # [3] CPU
    means_max: torch.Tensor  # [3] CPU
    lo_snap_k: int | tuple[int, int, int]
    sh_bands: int
    sh_max_centroids: int
    n_gaussians: int
    matcher_k_passes: int
    faiss_threads: int
    device: str
    reuse_inactive: bool = True
    keyframe_snap: bool = True


@dataclass
class ChunkWorkerResult:
    """Outputs from one chunk worker, ordered back into sequence by ``chunk_idx``."""

    chunk_idx: int
    npz_path: str  # temp file with atlases [F, H, W, 3] + means_lo [F, N, 3]
    n_frames: int
    sh_tensors: list[torch.Tensor] | None  # per-frame shN (CPU)


def frame_to_cpu_dict(frame: GSTensor) -> dict[str, torch.Tensor]:
    """Extract a GSTensor's attributes as CPU tensors for pickling to a worker.

    ``shN`` is omitted (rather than stored as ``None``) when absent, so every
    value in the dict is a real tensor.
    """
    out = {
        "means": frame.means.detach().cpu(),
        "scales": frame.scales.detach().cpu(),
        "quats": frame.quats.detach().cpu(),
        "opacities": frame.opacities.detach().cpu(),
        "sh0": frame.sh0.detach().cpu(),
        "masks": frame.masks.detach().cpu(),
    }
    if frame.shN is not None:
        out["shN"] = frame.shN.detach().cpu()
    return out


def _dict_to_gstensor(d: dict[str, torch.Tensor], device: str) -> GSTensor:
    shN = d.get("shN")
    return GSTensor(
        means=d["means"].to(device),
        scales=d["scales"].to(device),
        quats=d["quats"].to(device),
        opacities=d["opacities"].to(device),
        sh0=d["sh0"].to(device),
        shN=shN.to(device) if shN is not None else None,
        masks=d["masks"].to(device),
    )


def encode_chunk_worker(args: ChunkWorkerArgs) -> ChunkWorkerResult:
    """Encode one chunk end-to-end in a worker process.

    Mirrors the serial chunk body in ``SequenceEncoder.compress``: Morton sort,
    temporal conform, then ``ChunkEncoder.encode``. Atlases and means_lo are
    stacked and written to a temp ``.npz``; only its path is returned.
    """
    import faiss

    from gscodec.encoder.chunk_encoder import ChunkEncoder
    from gscodec.encoder.sorting import MortonSortingStrategy
    from gscodec.encoder.temporal_matcher import TemporalMatcher

    if args.faiss_threads > 0:
        faiss.omp_set_num_threads(args.faiss_threads)

    device = args.device
    frames = [_dict_to_gstensor(d, device) for d in args.frame_tensors]

    sorter = MortonSortingStrategy()
    matcher = TemporalMatcher(k_passes=args.matcher_k_passes, device=device)
    chunk_encoder = ChunkEncoder(
        device=device,
        skip_sorting=True,  # matching path always pre-sorts then conforms
        global_means_min=args.means_min.to(device),
        global_means_max=args.means_max.to(device),
        lo_snap_k=args.lo_snap_k,
        sh_bands=args.sh_bands,
        sh_max_centroids=args.sh_max_centroids,
        reuse_inactive=args.reuse_inactive,
        keyframe_snap=args.keyframe_snap,
    )

    sorted_chunk = [f[sorter.sort(f)[1]] for f in frames]
    conformed = matcher.conform_chunk(sorted_chunk, args.n_gaussians)
    pad_target = args.n_gaussians if conformed[0].means.shape[0] != args.n_gaussians else None
    encoded_frames, sh_tensors = chunk_encoder.encode(conformed, args.ranges, target_n=pad_target)

    atlases = np.stack([ef.atlas for ef in encoded_frames])  # [F, H, W, 3]
    means_lo = np.stack([ef.means_lo for ef in encoded_frames])  # [F, N, 3]
    fd, npz_path = tempfile.mkstemp(suffix=f"_chunk{args.chunk_idx:04d}.npz")
    os.close(fd)
    np.savez(npz_path, atlases=atlases, means_lo=means_lo, presence=np.stack([ef.presence for ef in encoded_frames]))

    sh_out: list[torch.Tensor] | None = None
    if sh_tensors is not None:
        sh_out = [sh.cpu() for sh in sh_tensors]

    return ChunkWorkerResult(
        chunk_idx=args.chunk_idx,
        npz_path=npz_path,
        n_frames=len(encoded_frames),
        sh_tensors=sh_out,
    )


def resolve_worker_count(parallel_chunks: int) -> int:
    """Resolve the configured worker count (0 = auto) to a concrete value."""
    if parallel_chunks == 0:
        return max(1, min(4, (os.cpu_count() or 8) // 8))
    return max(1, parallel_chunks)


def run_parallel_chunks(
    args_list: list[ChunkWorkerArgs], n_workers: int
) -> list[ChunkWorkerResult]:
    """Encode chunks across ``n_workers`` spawned processes, ordered by chunk_idx.

    Args:
        args_list: Per-chunk worker inputs.
        n_workers: Number of worker processes.

    Returns:
        Results sorted by ``chunk_idx`` so the parent can concatenate in sequence.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        results = list(pool.map(encode_chunk_worker, args_list))
    results.sort(key=lambda r: r.chunk_idx)
    return results
