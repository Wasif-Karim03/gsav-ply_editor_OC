"""
SH compression for the GSAV codec.

Compresses higher-order spherical harmonics (SH1-SH3) using a three-level
codebook + palette + label approach:

1. Scalar codebook (256 floats): optimal Lloyd-Max quantizer mapping uint8 -> float.
2. Palette centroids (up to 65,535 entries): K-means clustered SH profiles
   across ALL frames globally, stored as uint8 codebook indices.
3. Labels (per gaussian per frame): 16-bit palette index.
"""

import logging

import numpy as np
import torch
from numpy.typing import NDArray

from gscodec.common.types import SHChunkData
from gscodec.constants import MAX_SH_CENTROIDS, SH_CODEBOOK_SIZE, SH_COEFFS

logger = logging.getLogger(__name__)

# Fixed seed for SH k-means so encodes are reproducible and serial == parallel.
SH_KMEANS_SEED = 0


def compute_sh_bands(shN: torch.Tensor | None) -> int:
    """Auto-detect SH band level from shN tensor shape.

    Args:
        shN: [N, C, 3] or [N, C*3] tensor of SH coefficients, or None.

    Returns:
        SH band level (0-3). 0 if shN is None or empty.
    """
    if shN is None:
        return 0

    if shN.ndim == 2:
        total = shN.shape[1]
    elif shN.ndim == 3:
        total = shN.shape[1] * shN.shape[2]
    else:
        return 0

    if total == 0:
        return 0

    coeffs_per_channel = total // 3
    for bands, coeffs in SH_COEFFS.items():
        if coeffs == coeffs_per_channel:
            return bands

    logger.warning(f"Cannot determine SH bands from {total} values, skipping SH compression")
    return 0


def build_scalar_codebook(
    all_values: NDArray[np.float32],
    n_levels: int = SH_CODEBOOK_SIZE,
    alpha: float = 0.5,
    *,
    value_counts: NDArray[np.int64] | None = None,
) -> NDArray[np.float32]:
    """Build an optimal scalar codebook via DP on a weighted histogram.

    Uses dynamic programming to find n_levels centroids that minimize weighted
    sum-of-squared-errors. Sub-linear density weighting (weight = count^alpha)
    prevents dense near-zero bins from dominating, giving tail values adequate
    representation in the codebook.

    Histogram bins use adaptive blended positioning (uniform + quantile) based
    on the data's IQR-to-range ratio, handling outlier distributions gracefully.

    Args:
        all_values: Flat array of all SH float values.
        n_levels: Number of codebook entries (default 256).
        alpha: Density weight exponent. 0 = uniform weight per bin,
            0.5 = sqrt (balanced), 1.0 = standard MSE. Default 0.5.

    Returns:
        [n_levels] float32 sorted codebook.
    """
    if value_counts is not None:
        from gscodec.encoder.sh_exact import weighted_histogram

        constant, histogram = weighted_histogram(all_values, value_counts, n_levels, alpha)
        if constant is not None:
            return constant
        return _codebook_from_hist(*histogram, n_levels)

    sorted_data = np.sort(all_values.ravel().astype(np.float64))
    N = len(sorted_data)

    if N == 0:
        return np.zeros(n_levels, dtype=np.float32)

    vmin, vmax = sorted_data[0], sorted_data[-1]
    vrange = vmax - vmin

    # Degenerate: all values identical
    if vrange < 1e-20:
        return np.full(n_levels, vmin, dtype=np.float32)

    # Build histogram — H >= n_levels gives the DP enough bins to work with,
    # but more bins than 2×n_levels adds cost (O(k×H²)) without benefit.
    H = min(n_levels * 2, N)

    # Adaptive blend: extreme outliers (small IQR) → more quantile positioning
    iqr = sorted_data[int(N * 0.75)] - sorted_data[int(N * 0.25)]
    beta = max(0.5, min(0.999, 1.0 - iqr / vrange))

    counts = np.zeros(H, dtype=np.float64)
    sums = np.zeros(H, dtype=np.float64)

    uniform_pos = (sorted_data - vmin) / vrange
    quantile_pos = np.arange(N, dtype=np.float64) / N
    blended = beta * quantile_pos + (1.0 - beta) * uniform_pos
    bins = np.minimum(H - 1, np.floor(H * blended).astype(np.int64))

    np.add.at(counts, bins, 1)
    np.add.at(sums, bins, sorted_data)

    safe_counts = np.where(counts > 0, counts, 1.0)
    centers = np.where(counts > 0, sums / safe_counts, vmin + (np.arange(H) + 0.5) / H * vrange)

    # Sub-linear density weighting
    weights = np.where(counts > 0, np.power(counts, alpha), 0.0)
    return _codebook_from_hist(centers, weights, n_levels)


def _codebook_from_hist(centers, weights, n_levels):
    H = len(weights)

    # Prefix sums for O(1) range cost queries
    prefW = np.zeros(H + 1, dtype=np.float64)
    prefWX = np.zeros(H + 1, dtype=np.float64)
    prefWXX = np.zeros(H + 1, dtype=np.float64)
    prefW[1:] = np.cumsum(weights)
    prefWX[1:] = np.cumsum(weights * centers)
    prefWXX[1:] = np.cumsum(weights * centers * centers)

    def range_cost_vec(a_arr: NDArray, b: int) -> NDArray:
        """Vectorized range cost for multiple start positions to fixed end."""
        w = prefW[b + 1] - prefW[a_arr]
        wx = prefWX[b + 1] - prefWX[a_arr]
        wxx = prefWXX[b + 1] - prefWXX[a_arr]
        safe_w = np.where(w > 0, w, 1.0)
        return np.where(w > 0, wxx - (wx * wx) / safe_w, 0.0)

    def range_mean(a: int, b: int) -> float:
        w = prefW[b + 1] - prefW[a]
        if w <= 0:
            return (centers[a] + centers[b]) * 0.5
        return (prefWX[b + 1] - prefWX[a]) / w

    non_empty = int(np.sum(weights > 0))
    effective_k = min(n_levels, non_empty)

    # DP: dp[m][j] = min weighted SSE of quantizing bins 0..j into m centroids
    # Vectorized: for each j, compute all candidate split costs at once
    INF = 1e30
    dp_prev = np.full(H, INF, dtype=np.float64)
    split_table: list[NDArray[np.int32]] = [np.zeros(0, dtype=np.int32)]

    # Base case: m = 1
    split1 = np.full(H, -1, dtype=np.int32)
    a_arr = np.zeros(H, dtype=np.int64)
    dp_prev = range_cost_vec(a_arr, np.arange(H))  # cost(0, j) for all j
    # Recompute properly per-j
    for j in range(H):
        dp_prev[j] = range_cost_vec(np.array([0]), j)[0]
    split_table.append(split1)

    # Fill DP for m = 2..effective_k (vectorized inner loop)
    for m in range(2, effective_k + 1):
        dp_curr = np.full(H, INF, dtype=np.float64)
        split_m = np.zeros(H, dtype=np.int32)

        for j in range(m - 1, H):
            s_range = np.arange(m - 2, j, dtype=np.int64)
            costs = dp_prev[s_range] + range_cost_vec(s_range + 1, j)
            best_idx = np.argmin(costs)
            dp_curr[j] = costs[best_idx]
            split_m[j] = s_range[best_idx]

        split_table.append(split_m)
        dp_prev = dp_curr

    # Backtrack to find centroid values
    centroid_values = np.zeros(effective_k, dtype=np.float64)
    j = H - 1
    for m in range(effective_k, 0, -1):
        s = int(split_table[m][j]) if m > 1 else -1
        centroid_values[m - 1] = range_mean(s + 1, j)
        j = s

    centroid_values.sort()

    # Pad to n_levels if needed
    codebook = np.empty(n_levels, dtype=np.float32)
    codebook[:effective_k] = centroid_values.astype(np.float32)
    if effective_k < n_levels:
        codebook[effective_k:] = centroid_values[-1]

    return codebook


def quantize_to_codebook(
    values: NDArray[np.float32],
    codebook: NDArray[np.float32],
) -> NDArray[np.uint8]:
    """Map each float value to nearest codebook index.

    Uses searchsorted for O(N log K) lookup.

    Args:
        values: Float values to quantize (any shape).
        codebook: [K] sorted float32 codebook.

    Returns:
        Same shape as values, dtype uint8.
    """
    flat = values.ravel()
    # searchsorted finds insertion point; compare with left and right neighbors
    idx = np.searchsorted(codebook, flat, side="left")
    idx = np.clip(idx, 0, len(codebook) - 1)

    # Check if the left neighbor is closer
    left = np.clip(idx - 1, 0, len(codebook) - 1)
    dist_right = np.abs(flat - codebook[idx])
    dist_left = np.abs(flat - codebook[left])
    idx = np.where(dist_left < dist_right, left, idx)

    return idx.astype(np.uint8).reshape(values.shape)


def _flash_kmeans(
    vectors: NDArray[np.float32],
    n_clusters: int,
    max_iters: int = 20,
) -> tuple[NDArray[np.float32], NDArray[np.uint16]]:
    """K-means clustering via flash-kmeans (Triton GPU kernel).

    Pads D to the next power-of-2 >= 16 (Triton requirement), runs
    clustering, then trims centroids back to the original dimension.

    Args:
        vectors: [N, D] float32 array.
        n_clusters: Number of clusters.
        max_iters: Max Lloyd iterations.

    Returns:
        (centroids [K, D] float32, labels [N] uint16).
    """
    from flash_kmeans import batch_kmeans_Euclid

    n_samples, d_real = vectors.shape
    # Pad D to next power-of-2 >= 16 (Triton kernel constraint)
    d_pad = max(16, 1 << (d_real - 1).bit_length())

    x = torch.zeros(1, n_samples, d_pad, dtype=torch.float32, device="cuda")
    x[0, :, :d_real] = torch.from_numpy(np.ascontiguousarray(vectors))

    # flash-kmeans seeds its k-means++ init from torch's global RNG. Seed it
    # deterministically (saving/restoring state so the rest of the pipeline is
    # unaffected) — otherwise SH encoding varies run-to-run and serial/parallel
    # chunk encoding produce different (though equivalent) output.
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(SH_KMEANS_SEED)
        cluster_ids, centers, _ = batch_kmeans_Euclid(x, n_clusters, max_iters=max_iters)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)

    centroids = centers[0, :, :d_real].float().cpu().numpy()  # [K, D]
    labels = cluster_ids[0].cpu().numpy().astype(np.uint16)  # [N]
    return centroids, labels


def cluster_sh_profiles_float(
    float_vectors: NDArray[np.float32],
    max_centroids: int = MAX_SH_CENTROIDS,
    max_iters: int = 20,
) -> tuple[NDArray[np.float32], NDArray[np.uint16]]:
    """K-means cluster raw float SH vectors via flash-kmeans (Triton GPU).

    Args:
        float_vectors: [total_gaussians, coeffs*3] float32.
        max_centroids: Maximum number of palette entries.
        max_iters: Max Lloyd iterations.

    Returns:
        Tuple of (centroids [K, coeffs*3] float32, labels [N] uint16).
    """
    n_samples, n_dims = float_vectors.shape
    n_clusters = min(max_centroids, n_samples)

    logger.info(
        f"  SH clustering: {n_samples} vectors, {n_dims} dims, " f"targeting {n_clusters} centroids"
    )

    if n_clusters <= 1:
        centroids = float_vectors[:1].copy()
        labels = np.zeros(n_samples, dtype=np.uint16)
        return centroids, labels

    return _flash_kmeans(float_vectors, n_clusters, max_iters)


def assign_frame_labels(
    frame_vectors: NDArray[np.float32],
    centroids: NDArray[np.float32],
) -> NDArray[np.uint16]:
    """Assign each gaussian to the nearest centroid via GPU chunked matmul.

    Uses flash-kmeans euclid_assign_torch_native_chunked: pure PyTorch,
    no power-of-2 D constraint, OOM-safe via internal N/K chunking.

    Args:
        frame_vectors: [N, D] float32.
        centroids: [K, D] float32 palette entries.

    Returns:
        [N] uint16 label per gaussian.
    """
    from flash_kmeans.interface import euclid_assign_torch_native_chunked

    x = torch.from_numpy(np.ascontiguousarray(frame_vectors)).cuda().unsqueeze(0)  # [1, N, D]
    c = torch.from_numpy(np.ascontiguousarray(centroids)).cuda().unsqueeze(0)  # [1, K, D]
    x_sq = (x * x).sum(dim=2)  # [1, N]

    labels = euclid_assign_torch_native_chunked(x, c, x_sq)  # [1, N] int32
    return labels[0].cpu().numpy().astype(np.uint16)


def _compute_palette_size(n_unique: int, max_centroids: int = MAX_SH_CENTROIDS) -> int:
    """Compute palette size using splat-transform's power-of-2 formula.

    Scales in 1024-unit blocks: min(64K, 1024 × 2^floor(log2(N/1024))).
    For 1K-2K vectors → 1024, 2K-4K → 2048, ..., 64K+ → 65535.

    Args:
        n_unique: Number of unique SH vectors to cluster.
        max_centroids: Hard cap.

    Returns:
        Palette size (number of centroids).
    """
    if n_unique < 1024:
        return min(max(64, n_unique), max_centroids)
    import math

    size = min(64, 2 ** math.floor(math.log2(n_unique / 1024))) * 1024
    return min(size, max_centroids, 65535)


def encode_sh_global(
    all_frames_shN: list[list[torch.Tensor]],
    n_gaussians: int,
    sh_bands: int,
    max_centroids: int = MAX_SH_CENTROIDS,
    chunk_frame_counts: list[int] | None = None,
    device: str = "cpu",
    presence: list[NDArray[np.bool_]] | None = None,
    reuse_inactive: bool = True,
    prefer_exact: bool = False,
    progress=None,
) -> SHChunkData:
    """Full SH compression pipeline across all frames globally.

    SH is static within a chunk, so unique data = n_gaussians × n_chunks.
    Clustering and assignment both use one frame per chunk (cluster keyframe).
    Labels are replicated to all frames in each chunk.

    For stable-correspondence sequences, the caller passes only the keyframe
    shN per chunk (len(all_frames_shN[i]) == 1) plus chunk_frame_counts for
    label replication. For independent-per-frame sequences, all frames are
    passed and chunk_frame_counts == [len(c) for c in all_frames_shN].

    Pipeline:
    1. Flatten stored frames, take one per chunk for clustering
    2. Float K-means on unique SH vectors -> float32 centroids
    3. Build optimal codebook from all SH scalars (full data range)
    4. Quantize float centroids to uint8 via codebook
    5. Assign cluster keyframes to quantized centroids (one per chunk)
    6. Replicate labels to all frames per chunk

    Args:
        all_frames_shN: Nested list [chunks][stored_frames] of [N, coeffs, 3] tensors.
            For stable sequences, stored_frames=1 (keyframe only).
        n_gaussians: Number of gaussians per frame.
        sh_bands: SH band level (1-3).
        max_centroids: Maximum palette entries.
        chunk_frame_counts: Actual frame count per chunk for label replication.
            Defaults to [len(c) for c in all_frames_shN] if None.
        device: Torch device.

    Returns:
        SHChunkData with global codebook, centroids, and flat list of per-frame labels.
    """
    if sh_bands <= 0:
        sh_bands = compute_sh_bands(all_frames_shN[0][0])
    coeffs = SH_COEFFS[sh_bands]

    # Step 1: Flatten all frames to [N, coeffs*3] float32
    flat_frames: list[NDArray[np.float32]] = []
    for chunk_frames in all_frames_shN:
        for shN in chunk_frames:
            arr = shN.detach().cpu().float().numpy()
            if arr.ndim == 2:
                arr = arr.reshape(arr.shape[0], -1, 3)
            arr = arr[:, :coeffs, :]  # truncate to target band
            # Keep strided PLY SH views until each exact-palette frame is read.
            # Flattening all transposed views here would copy the entire sequence.
            flat_frames.append(arr if prefer_exact and presence is not None
                               else arr.reshape(arr.shape[0], -1))

    n_chunks = len(all_frames_shN)
    chunk_sizes = [len(chunk) for chunk in all_frames_shN]
    actual_frame_counts = chunk_frame_counts if chunk_frame_counts is not None else chunk_sizes

    if presence is not None:
        if len(presence) != len(flat_frames) or sum(actual_frame_counts) != len(flat_frames):
            raise ValueError("Mask-aware SH encoding requires every frame and its presence mask")
        if prefer_exact:
            from gscodec.encoder.sh_exact import exact_palette

            active_count = sum(int(mask.sum()) for mask in presence)
            limit = min(max_centroids, max(64, active_count // 10), 65535)
            exact = exact_palette(flat_frames, presence, limit, progress)
            if exact is not None:
                centers, labels, values, counts = exact
                if progress is not None:
                    progress(f"Encoding exact SH palette: {len(centers)} distinct vectors")
                if len(values) <= SH_CODEBOOK_SIZE:
                    # A decoded source (including uniform brightness edits) can
                    # already fit exactly. Histogram fitting would merge rare
                    # values unnecessarily and introduce another lossy step.
                    codebook = np.pad(values, (0, SH_CODEBOOK_SIZE - len(values)), mode="edge")
                else:
                    codebook = build_scalar_codebook(values, SH_CODEBOOK_SIZE, value_counts=counts)
                quantized = quantize_to_codebook(centers, codebook)
                if reuse_inactive:
                    from gscodec.encoder.masks import reuse_inactive_rows

                    start = 0
                    for count in actual_frame_counts:
                        reuse_inactive_rows(labels[start:start + count], presence[start:start + count])
                        start += count
                return SHChunkData(codebook=codebook, centroids=quantized, labels=labels,
                                   n_centroids=len(quantized), sh_bands=sh_bands)
            if torch.cuda.is_available():
                padded_dims = max(16, 1 << (coeffs * 3 - 1).bit_length())
                required = active_count * padded_dims * 4
                free, _ = torch.cuda.mem_get_info()
                if required > free * 0.75:
                    raise ValueError(
                        "SH vectors exceed the exact palette capacity and full clustering "
                        f"needs at least {required / 1024**3:.1f} GiB of GPU input memory. "
                        "This GPU cannot encode this edit at full quality in one sequence."
                    )
        active_vectors = [f[m].reshape(int(m.sum()), coeffs * 3)
                          for f, m in zip(flat_frames, presence, strict=True)]
        training = np.concatenate(active_vectors, axis=0)
        if len(training) == 0:
            training = np.zeros((1, coeffs * 3), dtype=np.float32)
        effective_max = min(max_centroids, max(64, len(training) // 10), 65535)
        centers, active_labels = cluster_sh_profiles_float(training, effective_max)
        codebook = build_scalar_codebook(training.ravel(), SH_CODEBOOK_SIZE)
        quantized = quantize_to_codebook(centers, codebook)
        labels = []
        offset = 0
        for mask in presence:
            frame_labels = np.zeros(n_gaussians, dtype=np.uint16)
            count = int(mask.sum())
            frame_labels[mask] = active_labels[offset:offset + count]
            offset += count
            labels.append(frame_labels)
        if reuse_inactive:
            from gscodec.encoder.masks import reuse_inactive_rows

            start = 0
            for count in actual_frame_counts:
                reuse_inactive_rows(labels[start:start + count], presence[start:start + count])
                start += count
        return SHChunkData(codebook=codebook, centroids=quantized, labels=labels,
                           n_centroids=len(quantized), sh_bands=sh_bands)

    # If the caller provided fewer frames than the final output frames, it means
    # they are passing stable keyframes and relying on us to replicate labels.
    is_stable_sequence = sum(chunk_sizes) < sum(actual_frame_counts)

    if is_stable_sequence:
        # Take one frame per chunk for clustering (SH is static within a chunk)
        cluster_indices: list[int] = []
        frame_offset = 0
        for cs in chunk_sizes:
            cluster_indices.append(frame_offset)  # first frame of each chunk
            frame_offset += cs
        cluster_flat = np.concatenate([flat_frames[i] for i in cluster_indices], axis=0)
        n_unique = n_gaussians * n_chunks
    else:
        # Dynamic sequence: use all frames for optimal clustering
        cluster_flat = np.concatenate(flat_frames, axis=0)
        n_unique = n_gaussians * len(flat_frames)

    # Step 2: Float K-means — 1:10 ratio of unique vectors to centroids
    effective_max = min(max_centroids, max(64, n_unique // 10), 65535)

    logger.info(
        f"  SH global clustering: {cluster_flat.shape[0]} unique vectors "
        f"({n_unique // n_gaussians} frames × {n_gaussians} gaussians), "
        f"targeting {effective_max} centroids"
    )
    float_centroids, kf_labels = cluster_sh_profiles_float(
        cluster_flat,
        effective_max,
    )
    logger.info(f"  SH centroids: {len(float_centroids)} palette entries (float)")

    # Step 3: Build optimal codebook from all SH scalars (full data range)
    all_scalars = cluster_flat.ravel()
    codebook = build_scalar_codebook(all_scalars, SH_CODEBOOK_SIZE)
    logger.info(
        f"  SH codebook: range [{codebook[0]:.4f}, {codebook[-1]:.4f}], "
        f"{len(codebook)} entries (optimal)"
    )

    # Step 4: Quantize float centroids to uint8 codebook indices
    quantized_centroids = quantize_to_codebook(float_centroids, codebook)

    # Step 5: Labels come directly from K-means — no separate assignment pass.
    # kf_labels are w.r.t. float centroids; minor drift after uint8 quantization
    # is acceptable for compression. Eliminates the entire assignment step.
    total_output_frames = sum(actual_frame_counts)
    logger.info(
        f"  SH labels: reusing K-means assignments "
        f"({cluster_flat.shape[0]:,} vectors → {total_output_frames} output frames)"
    )

    # Step 6: Assign labels to output frames
    frame_labels: list[NDArray[np.uint16]] = []

    if is_stable_sequence:
        # Replicate each chunk's keyframe labels to all frames in that chunk
        for i, actual_cs in enumerate(actual_frame_counts):
            chunk_gaussian_labels = kf_labels[i * n_gaussians : (i + 1) * n_gaussians]
            for _ in range(actual_cs):
                frame_labels.append(chunk_gaussian_labels)
    else:
        # Use exact assignments for every frame
        for i in range(len(flat_frames)):
            frame_labels.append(kf_labels[i * n_gaussians : (i + 1) * n_gaussians])

    return SHChunkData(
        codebook=codebook,
        centroids=quantized_centroids,
        labels=frame_labels,
        n_centroids=len(quantized_centroids),
        sh_bands=sh_bands,
    )
