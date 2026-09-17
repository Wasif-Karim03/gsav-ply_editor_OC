"""Compact encoded source slots without inferring physical Gaussian identities."""

import numpy as np

from gscodec.common.types import SHChunkData
from gscodec.encoder.video_writer import _precompute_atlas_scatter, compute_atlas_dimensions


def compact_source(atlases, provider, visibility, sh_data, chunk_size):
    """Keep the union of visible slots per GOP, copying quantized bytes exactly.

    A slot may represent different objects over time. This operation makes no
    identity assumption: every visible sample retains all its original fields.
    The format has one global row count, so shorter chunks get invisible padding.
    """
    rows = provider.n_gaussians
    if len(visibility) != len(atlases) or any(
        mask.dtype != np.bool_ or mask.shape != (rows,) for mask in visibility
    ):
        raise ValueError("Invalid visibility for compaction")
    selections = [
        np.flatnonzero(np.logical_or.reduce(visibility[start : start + chunk_size]))
        for start in range(0, len(atlases), chunk_size)
    ]
    count = max(len(indices) for indices in selections)
    # Preserve WebGPU buffer alignment, including an entirely invisible scene.
    count = max(4, ((count + 3) // 4) * 4)
    if count >= rows:
        return rows, provider.atlas_side, None, sh_data
    side, _, _ = compute_atlas_dimensions(count)
    old_scatter = _precompute_atlas_scatter(provider.atlas_side, rows)
    scatter = _precompute_atlas_scatter(side, count)
    lows, labels = [], []
    for index, atlas in enumerate(atlases):
        selected = selections[index // chunk_size]
        n = len(selected)
        compact = np.full((side * 3, side * 5), 16, dtype=np.uint8)
        compact[side * 2 : side * 3, side * 4 : side * 5] = 235
        compact.flat[scatter[:, :n]] = atlas.flat[old_scatter[:, selected]]
        atlases[index] = compact
        low = np.zeros((count, 3), dtype=np.uint8)
        low[:n] = provider.get_means_lo(index)[selected]
        lows.append(low)
        if sh_data is not None:
            frame_labels = np.zeros(count, dtype=np.uint16)
            frame_labels[:n] = sh_data.labels[index][selected]
            labels.append(frame_labels)
    if sh_data is not None:
        used = np.unique(np.concatenate(labels))
        remap = np.zeros(sh_data.n_centroids, dtype=np.uint16)
        remap[used] = np.arange(len(used), dtype=np.uint16)
        sh_data = SHChunkData(
            codebook=sh_data.codebook,
            centroids=sh_data.centroids[used],
            labels=[remap[label] for label in labels],
            n_centroids=len(used),
            sh_bands=sh_data.sh_bands,
        )
    return count, side, lows, sh_data
