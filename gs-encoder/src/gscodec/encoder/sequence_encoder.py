"""
Sequence encoder for frame-by-frame GSAV codec.

Orchestrates the full encoding pipeline:
1. Load PLY sequence
2. Compute global quantization ranges
3. Process frames in chunks (per-frame encoding)
4. Write GSAV container file

GSAV Layout:
    Header (128B) -> Ranges (112B) -> ChunkIndex (16B x nChunks)
    -> FrameIndex (8B x nFrames) -> MeansLoPayload -> VideoPayload
"""

import io
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gsply
import numpy as np
import torch
import zstandard as zstd
from gsply import GSTensor
from tqdm import tqdm

from gscodec.common.binary_format import (
    CHUNK_ENTRY_SIZE,
    FRAME_ENTRY_SIZE,
    GSAV_MAGIC,
    GSAV_VERSION,
    HEADER_SIZE,
    RANGES_SIZE,
    make_frame_entry,
    write_chunk_index,
    write_frame_index,
    write_header,
    write_ranges,
)
from gscodec.common.types import (
    HAS_AUDIO_FLAG,
    HAS_SH_FLAG,
    ChunkEntry,
    GSAVHeader,
    QuantRanges,
    SHChunkData,
)
from gscodec.constants import (
    N_ATLAS_COLS,
    N_ATLAS_ROWS,
    OPACITY_CLIP,
    QUATS_CLIP,
    SH_COEFFS,
)
from gscodec.encoder.audio import transcode_to_opus
from gscodec.encoder.chunk_encoder import ChunkEncoder
from gscodec.encoder.config import ChunkConfig, VideoConfig
from gscodec.encoder.gsflow_metadata import GSFlowMetadata
from gscodec.encoder.masks import active_frames, sanitize_frame
from gscodec.encoder.parallel_chunks import (
    ChunkWorkerArgs,
    frame_to_cpu_dict,
    resolve_worker_count,
    run_parallel_chunks,
)
from gscodec.encoder.sorting.morton import MortonSortingStrategy
from gscodec.encoder.temporal_matcher import TemporalMatcher
from gscodec.encoder.utils.helpers import log_transform
from gscodec.encoder.video_writer import (
    compute_atlas_dimensions,
    encode_to_ivf,
)

logger = logging.getLogger(__name__)

# VP9 Level 5.1 (120 Mbps) — Quest 3 4K decode
_VP9_WEBCODECS_CODEC = "vp09.00.51.08"


class SequenceEncoder:
    """Encodes a sequence of Gaussian Splatting frames to GSAV format."""

    def __init__(
        self,
        video_config: VideoConfig | None = None,
        chunk_config: ChunkConfig | None = None,
        device: str = "cuda:0",
        progress=None,
    ):
        """Initialize sequence encoder.

        Args:
            video_config: Video encoding configuration.
            chunk_config: Chunk processing configuration.
            device: Torch device for computation.
        """
        self.progress = progress
        self.video_config = video_config or VideoConfig()
        self.chunk_config = chunk_config or ChunkConfig()
        if self.chunk_config.identity_mode not in ("auto", "stable", "unstructured", "preserve"):
            raise ValueError("Unknown identity_mode")
        if self.chunk_config.identity_mode == "preserve" and (
            self.chunk_config.matching_enabled or self.chunk_config.lossy_pruning
            or self.chunk_config.keyframe_snap or self.chunk_config.lo_snap_k != 1
            or self.chunk_config.gsflow_metadata is not None
        ):
            raise ValueError("Preserve mode requires matching/pruning/snapping disabled and lo_snap_k=1")
        if self.chunk_config.identity_mode == "stable" and self.chunk_config.matching_enabled:
            raise ValueError("Stable source identities cannot be combined with matching")
        matching = self.chunk_config.matching_enabled or self.chunk_config.identity_mode == "unstructured"
        self.device = device
        self._skip_sorting = (
            self.chunk_config.gsflow_metadata is not None or matching
            or self.chunk_config.identity_mode == "preserve"
        )
        # ChunkEncoder is created after global ranges are computed
        # so we can pass the global means bbox for per-frame sorting
        self.chunk_encoder: ChunkEncoder | None = None
        self.temporal_matcher: TemporalMatcher | None = None
        if matching:
            self.temporal_matcher = TemporalMatcher(
                k_passes=self.chunk_config.matcher_k_passes,
                device=device,
            )
            self.sorter = MortonSortingStrategy()

    def _progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def compress(
        self,
        input_dir: str | Path,
        output: str | Path,
        audio_path: str | Path | None = None,
        static_asset_path: str | Path | None = None,
        geometry_source: str | Path | None = None,
        *,
        visibility: list[np.ndarray] | None = None,
        compact_visibility: bool = False,
    ) -> None:
        """Compress a PLY or SPZ sequence to GSAV format.

        Args:
            input_dir: Directory containing per-frame PLY or SPZ files.
            output: Output GSAV file path.
            audio_path: Optional audio file to embed (any format, transcoded to Opus).
            geometry_source: Source GSAV for caller-verified color-only edits with
                unchanged rows and timeline. Its geometry and audio are retained.
            visibility: Optional per-frame masks for filtering original rows.
                Requires geometry_source; does not compact or reorder the rows.
            compact_visibility: Remove slots never visible within each chunk,
                preserving all visible encoded samples and retained row order.
        """
        input_path = Path(input_dir)
        output_path = Path(output)
        static_data = b""
        static_encoding = 0
        if static_asset_path is not None:
            from gscodec.encoder.static_encoder import encode_static_file
            static_data, static_encoding = encode_static_file(static_asset_path, device=self.device)

        if output_path.suffix != ".gsav":
            output_path = output_path.with_suffix(".gsav")

        self._progress("Reading prepared frames")
        data = self.encode_frames(self._load_ply_sequence(input_path), audio_path=audio_path,
                                  static_data=static_data, static_encoding=static_encoding,
                                  geometry_source=geometry_source, visibility=visibility,
                                  compact_visibility=compact_visibility)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        logger.info("Wrote %d bytes to %s", len(data), output_path)

    def encode_frames(
        self, frames: list[GSTensor], *, audio_path: str | Path | None = None,
        static_data: bytes = b"", static_encoding: int = 0, prune: bool = True,
        geometry_source: str | Path | None = None,
        visibility: list[np.ndarray] | None = None,
        compact_visibility: bool = False,
    ) -> bytes:
        """Encode frames in memory; shared by dynamic and single-frame static tracks.

        geometry_source is reserved for caller-verified color-only edits. The
        preserve identity mode alone does not prove that geometry was unchanged.
        """
        if not frames or any(f.means.shape[0] == 0 for f in frames):
            raise ValueError("Frames must contain at least one Gaussian")
        source_geometry = None
        if compact_visibility and (visibility is None or geometry_source is None):
            raise ValueError("Compaction requires source geometry and visibility")
        if visibility is not None and geometry_source is None:
            raise ValueError("Visibility overrides require verified source geometry")
        if geometry_source is not None:
            if self.chunk_config.identity_mode != "preserve" or static_data:
                raise ValueError("Source geometry reuse requires verified preserved rows")
            from gscodec.encoder.source_geometry import SourceGeometry

            source_geometry = SourceGeometry(geometry_source)
        if self.chunk_config.identity_mode == "preserve":
            if len({len(f.means) for f in frames}) != 1:
                raise ValueError("Preserve mode requires equal row counts")
            prune = False
        frames = [sanitize_frame(f) for f in frames]
        n_frames = len(frames)

        # Lifetime pruning requires explicit stable identities. Opacity/scale
        # thresholds additionally require the lossy_pruning option.
        if prune:
            frames = self._prune_invisible(frames)

        counts = {f.means.shape[0] for f in frames}
        if len(counts) == 1:
            n_gaussians = counts.pop()
            logger.info(f"Loaded {n_frames} frames with {n_gaussians} Gaussians each")
        else:
            raw_max = max(counts)
            # WebGPU writeBuffer requires means_lo (n*3 bytes) to be 4-byte aligned
            n_gaussians = ((raw_max + 3) // 4) * 4
            logger.info(
                f"Loaded {n_frames} frames (counts: min={min(counts)}, max={max(counts)}). "
                f"Target n_gaussians={n_gaussians}"
            )

        if self.chunk_config.gsflow_metadata is not None:
            sqrt_n = int(math.sqrt(n_gaussians))
            if sqrt_n * sqrt_n != n_gaussians:
                logger.warning(
                    f"GSFlow metadata provided but n_gaussians ({n_gaussians}) is not a perfect square. "
                    f"Atlas will have padding. Consider using {sqrt_n * sqrt_n} or {(sqrt_n + 1) ** 2} gaussians."
                )

        logger.info("Computing global quantization ranges...")
        self._progress("Computing quantization ranges")
        global_ranges = self._compute_global_ranges(frames)

        # Create chunk encoder with global means bbox for per-frame sorting
        global_means_min = torch.from_numpy(global_ranges.means_min).to(self.device)
        global_means_max = torch.from_numpy(global_ranges.means_max).to(self.device)

        # Auto-K: compute optimal K from scene statistics
        lo_snap_k = self.chunk_config.lo_snap_k
        if lo_snap_k == 0:
            lo_snap_k = self._compute_auto_k(frames, global_means_min, global_means_max)

        # Detect SH bands from data or config
        from gscodec.encoder.sh_compress import compute_sh_bands

        sh_bands = self.chunk_config.sh_bands
        if sh_bands < 0:  # auto-detect
            sh_bands = compute_sh_bands(frames[0].shN)
            logger.info(f"Auto-detected SH bands: {sh_bands}")
        if sh_bands > 0:
            logger.info(f"SH compression enabled: band level {sh_bands}")
        frames = self._prepare_sh_frames(frames, sh_bands)

        self.chunk_encoder = ChunkEncoder(
            device=self.device,
            skip_sorting=self._skip_sorting,
            global_means_min=global_means_min,
            global_means_max=global_means_max,
            lo_snap_k=lo_snap_k,
            sh_bands=sh_bands,
            sh_max_centroids=self.chunk_config.sh_max_centroids,
            identity_mode=self.chunk_config.identity_mode,
            reuse_inactive=self.chunk_config.reuse_inactive,
            keyframe_snap=self.chunk_config.keyframe_snap,
        )

        chunk_size = self.chunk_config.size
        chunk_boundaries = self._compute_chunk_boundaries(n_frames)
        n_chunks = len(chunk_boundaries)
        logger.info(f"Processing {n_chunks} chunks (GOP size = {chunk_size})...")

        atlas_side, atlas_width, atlas_height = compute_atlas_dimensions(n_gaussians)
        n_atlas_cols = N_ATLAS_COLS
        n_atlas_rows = N_ATLAS_ROWS
        logger.info(f"Atlas dimensions: {atlas_width}x{atlas_height} (side={atlas_side})")

        # Process chunks — collect frame atlases and means arrays
        all_atlases: list[np.ndarray] = []
        all_means_lo: list[np.ndarray] = []
        all_sh_tensors: list[list[torch.Tensor]] = []
        all_presence: list[np.ndarray] = []
        chunk_frame_counts = [end - start for start, end in chunk_boundaries]

        # Preserved rows need no chunk permutation or padding. Build their SH
        # palette from CPU input before allocating GPU chunks, retaining labels
        # instead of a second full sequence of SH tensors.
        sh_global_data: SHChunkData | None = None
        if sh_bands > 0 and self.chunk_config.identity_mode == "preserve":
            from gscodec.encoder.sh_compress import encode_sh_global

            self._progress("Checking repeated SH coefficients before chunk encoding")
            sh_global_data = encode_sh_global(
                [[f.shN for f in frames[start:end]] for start, end in chunk_boundaries],
                n_gaussians=n_gaussians, sh_bands=sh_bands,
                max_centroids=self.chunk_config.sh_max_centroids,
                chunk_frame_counts=chunk_frame_counts, device=self.device,
                presence=[f.masks.reshape(-1).cpu().numpy() for f in frames],
                reuse_inactive=self.chunk_config.reuse_inactive,
                prefer_exact=True, progress=self._progress,
            )

        n_workers = resolve_worker_count(self.chunk_config.parallel_chunks)
        use_parallel = (
            self.temporal_matcher is not None and n_workers > 1 and len(chunk_boundaries) > 1
        )

        if use_parallel:
            self._encode_chunks_parallel(
                frames,
                chunk_boundaries,
                global_ranges,
                global_means_min,
                global_means_max,
                lo_snap_k,
                sh_bands,
                n_gaussians,
                n_workers,
                all_atlases,
                all_means_lo,
                all_sh_tensors,
                all_presence,
            )
        else:
            for chunk_number, (start, end) in enumerate(tqdm(chunk_boundaries, desc="Encoding chunks"), 1):
                self._progress(f"Encoding chunk {chunk_number}/{n_chunks}" + (" (original arrangement)" if self.chunk_config.identity_mode == "preserve" else " (matching enabled)"))
                chunk_frames = frames[start:end]

                # Move to GPU for processing
                chunk_frames_gpu = []
                for f in chunk_frames:
                    chunk_frames_gpu.append(
                        GSTensor(
                            means=f.means.to(self.device),
                            scales=f.scales.to(self.device),
                            quats=f.quats.to(self.device),
                            opacities=f.opacities.to(self.device),
                            sh0=f.sh0.to(self.device),
                            shN=f.shN.to(self.device) if f.shN is not None else None,
                            masks=f.masks.to(self.device),
                        )
                    )

                if self.temporal_matcher is not None:
                    sorted_chunk = []
                    for f in chunk_frames_gpu:
                        _, m_idx = self.sorter.sort(f)
                        sorted_chunk.append(f[m_idx])
                    chunk_frames_gpu = self.temporal_matcher.conform_chunk(
                        sorted_chunk, n_gaussians
                    )

                # Pass target_n so chunk encoder pads AFTER sorting
                encoded_frames, sh_tensors = self.chunk_encoder.encode(
                    chunk_frames_gpu,
                    global_ranges,
                    target_n=n_gaussians,
                )
                for ef in encoded_frames:
                    all_atlases.append(ef.atlas)
                    all_means_lo.append(ef.means_lo)
                    all_presence.append(ef.presence)
                if sh_tensors is not None and sh_global_data is None:
                    all_sh_tensors.append([sh.cpu() for sh in sh_tensors])
                # This list is owned by encode_frames (created by sanitization).
                # Caller-owned tensors remain untouched; release processed slots.
                frames[start:end] = [None] * (end - start)

        # Global SH encoding (once, across all chunks)
        if all_sh_tensors:
            self._progress("Compressing SH coefficients (this may take several minutes)")
            from gscodec.encoder.sh_compress import encode_sh_global

            logger.info(
                f"SH global encoding: {len(all_sh_tensors)} chunks, "
                f"{sum(chunk_frame_counts)} total frames "
                f"({sum(len(c) for c in all_sh_tensors)} shN frames stored)"
            )
            sh_global_data = encode_sh_global(
                all_sh_tensors,
                n_gaussians=n_gaussians,
                sh_bands=sh_bands,
                max_centroids=self.chunk_config.sh_max_centroids,
                chunk_frame_counts=chunk_frame_counts,
                device=self.device,
                presence=all_presence,
                reuse_inactive=self.chunk_config.reuse_inactive,
            )

        if source_geometry is not None:
            self._progress("Preserving source geometry; updating colors and visibility")
            source_geometry.merge(
                all_atlases, global_ranges, rows=n_gaussians,
                fps=self.video_config.fps, chunk_size=chunk_size,
                visibility=visibility,
            )

        compact_lows = None
        if compact_visibility:
            from gscodec.encoder.compact import compact_source

            self._progress("Removing source slots that are never visible within each chunk")
            original_rows = n_gaussians
            n_gaussians, atlas_side, compact_lows, sh_global_data = compact_source(
                all_atlases, source_geometry.provider, visibility, sh_global_data, chunk_size
            )
            self._progress(f"Compacted storage: {original_rows} -> {n_gaussians} rows")

        # Encode video with GOP size = chunk_size
        logger.info(f"Encoding {len(all_atlases)} frame atlases (GOP={chunk_size})...")
        self._progress("Encoding VP9 with native xllvp9")
        video_data, raw_frame_entries = encode_to_ivf(
            all_atlases,
            fps=self.video_config.fps,
            gop_size=chunk_size,
        )

        self._progress("Writing GSAV container")
        # Build frame entries with keyframe flags
        frame_entries = []
        frame_in_chunk = 0
        chunk_idx = 0

        for raw_entry in raw_frame_entries:
            is_keyframe = frame_in_chunk == 0

            frame_entries.append(
                make_frame_entry(
                    offset=raw_entry["obu_offset"],
                    size=raw_entry["obu_size"],
                    keyframe=is_keyframe,
                )
            )

            frame_in_chunk += 1
            if (
                chunk_idx < len(chunk_frame_counts)
                and frame_in_chunk >= chunk_frame_counts[chunk_idx]
            ):
                frame_in_chunk = 0
                chunk_idx += 1

        # Build chunk index entries (no anchor payload)
        chunk_entries: list[ChunkEntry] = []
        for start, end in chunk_boundaries:
            chunk_entries.append(
                ChunkEntry(
                    start_frame=start,
                    end_frame=end - 1,  # inclusive
                    offset=0,
                    size=0,
                )
            )

        means_lo_data = (
            self._compress_means_lo(compact_lows) if compact_lows is not None
            else source_geometry.means_lo if source_geometry is not None
            else self._compress_means_lo(all_means_lo)
        )

        # Serialize SH payload if present (global centroids + per-chunk labels)
        sh_payload_data = b""
        if sh_global_data is not None:
            sh_payload_data = self._compress_sh_payload(sh_global_data, chunk_frame_counts)

        # Transcode audio if provided
        audio_data = b""
        if source_geometry is not None:
            audio_data = source_geometry.audio
        elif audio_path is not None:
            target_duration = n_frames / self.video_config.fps
            audio_data = transcode_to_opus(Path(audio_path), target_duration)

        # Compute file layout:
        # Header -> Ranges -> ChunkIndex -> FrameIndex -> MeansLo -> SHPayload -> Video -> Audio
        ranges_offset = HEADER_SIZE
        chunk_index_offset = ranges_offset + RANGES_SIZE
        frame_index_offset = chunk_index_offset + CHUNK_ENTRY_SIZE * n_chunks
        means_lo_payload_offset = frame_index_offset + FRAME_ENTRY_SIZE * n_frames
        static_asset_offset = means_lo_payload_offset if static_data else 0
        means_lo_payload_offset += len(static_data)
        means_hi_payload_offset = 0
        sh_payload_offset = means_lo_payload_offset + len(means_lo_data) if sh_payload_data else 0
        video_payload_offset = means_lo_payload_offset + len(means_lo_data) + len(sh_payload_data)
        audio_payload_offset = video_payload_offset + len(video_data) if audio_data else 0
        audio_payload_size = len(audio_data)

        # Map encoder codec name → WebCodecs codec string
        codec_string = _VP9_WEBCODECS_CODEC

        from gscodec.common.types import HAS_MASK_FLAG, HAS_STATIC_ASSET_FLAG

        flags = source_geometry.mask_flag if source_geometry is not None else HAS_MASK_FLAG
        if static_data:
            from gscodec.common.static_track import validate_static_track
            if static_encoding != 1:
                raise ValueError("Unknown static asset encoding")
            validate_static_track(static_data)
            flags |= HAS_STATIC_ASSET_FLAG
        if audio_data:
            flags |= HAS_AUDIO_FLAG
        if sh_payload_data:
            flags |= HAS_SH_FLAG

        header = GSAVHeader(
            magic=GSAV_MAGIC,
            version=GSAV_VERSION,
            n_gaussians=n_gaussians,
            n_frames=n_frames,
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            atlas_side=atlas_side,
            n_atlas_cols=n_atlas_cols,
            n_atlas_rows=n_atlas_rows,
            fps=self.video_config.fps,
            flags=flags,
            ranges_offset=ranges_offset,
            chunk_index_offset=chunk_index_offset,
            frame_index_offset=frame_index_offset,
            means_lo_payload_offset=means_lo_payload_offset,
            video_payload_offset=video_payload_offset,
            codec=codec_string,
            audio_payload_offset=audio_payload_offset,
            audio_payload_size=audio_payload_size,
            means_hi_payload_offset=means_hi_payload_offset,
            sh_bands=sh_bands,
            sh_payload_offset=sh_payload_offset,
            static_asset_offset=static_asset_offset,
            static_asset_size=len(static_data),
            static_asset_encoding=static_encoding,
        )

        with io.BytesIO() as f:
            write_header(f, header)
            write_ranges(f, global_ranges)
            write_chunk_index(f, chunk_entries)
            write_frame_index(f, frame_entries)
            f.write(static_data)
            f.write(means_lo_data)
            if sh_payload_data:
                f.write(sh_payload_data)
            f.write(video_data)
            if audio_data:
                f.write(audio_data)
            result = f.getvalue()

        file_size = len(result)
        video_size = len(video_data)
        means_lo_size = len(means_lo_data)
        index_size = means_lo_payload_offset

        logger.info(f"Encoded {file_size:,} bytes")
        logger.info(
            f"  Index (header+ranges+chunk_index+frame_index): {index_size:,} bytes ({index_size/file_size*100:.1f}%)"
        )
        logger.info(
            f"  Means Lo payload: {means_lo_size:,} bytes ({means_lo_size/file_size*100:.1f}%)"
        )
        if sh_payload_data:
            sh_size = len(sh_payload_data)
            logger.info(f"  SH payload: {sh_size:,} bytes ({sh_size/file_size*100:.1f}%)")
        logger.info(f"  Video payload: {video_size:,} bytes ({video_size/file_size*100:.1f}%)")
        if audio_payload_size > 0:
            logger.info(
                f"  Audio payload: {audio_payload_size:,} bytes ({audio_payload_size/file_size*100:.1f}%)"
            )

        return result

    @staticmethod
    def _prepare_sh_frames(frames: list[GSTensor], sh_bands: int) -> list[GSTensor]:
        """Select one SH layout before matching or dispatching parallel workers.

        Wrappers share attribute tensors with the source; only SH views change.
        Encoding must not change the source frame's SH degree.
        """
        if sh_bands not in SH_COEFFS:
            raise ValueError("SH bands must be between 0 and 3")
        coeffs = SH_COEFFS[sh_bands]
        prepared = []
        for t, frame in enumerate(frames):
            sh = frame.shN
            if coeffs:
                if sh is None or sh.numel() < len(frame.means) * coeffs * 3:
                    raise ValueError(f"frame {t} has insufficient SH coefficients for band {sh_bands}")
                if sh.ndim == 2 and sh.shape[1] % 3 == 0:
                    sh = sh.reshape(len(frame.means), -1, 3)
                if sh.ndim != 3 or sh.shape[0] != len(frame.means) or sh.shape[2] != 3:
                    raise ValueError(f"frame {t} SH must have shape [N, coefficients, 3]")
                sh = sh[:, :coeffs, :]
            else:
                sh = None
            prepared.append(GSTensor(
                means=frame.means, scales=frame.scales, quats=frame.quats,
                opacities=frame.opacities, sh0=frame.sh0, shN=sh, masks=frame.masks,
            ))
        return prepared

    def _encode_chunks_parallel(
        self,
        frames: list[GSTensor],
        chunk_boundaries: list[tuple[int, int]],
        global_ranges: QuantRanges,
        global_means_min: torch.Tensor,
        global_means_max: torch.Tensor,
        lo_snap_k: int | tuple[int, int, int],
        sh_bands: int,
        n_gaussians: int,
        n_workers: int,
        all_atlases: list[np.ndarray],
        all_means_lo: list[np.ndarray],
        all_sh_tensors: list[list[torch.Tensor]],
        all_presence: list[np.ndarray],
    ) -> None:
        """Encode chunks across worker processes, appending results in order.

        Mutates ``all_atlases`` / ``all_means_lo`` / ``all_sh_tensors`` in place to
        match the serial path's accumulation order. Atlases come back via temp
        ``.npz`` files (deleted after load) to keep them out of the pool's IPC.
        """
        # Split CPU cores across workers so each FAISS search doesn't oversubscribe.
        faiss_threads = max(1, (os.cpu_count() or 8) // n_workers)
        means_min_cpu = global_means_min.detach().cpu()
        means_max_cpu = global_means_max.detach().cpu()

        args_list = [
            ChunkWorkerArgs(
                chunk_idx=idx,
                frame_tensors=[frame_to_cpu_dict(f) for f in frames[start:end]],
                ranges=global_ranges,
                means_min=means_min_cpu,
                means_max=means_max_cpu,
                lo_snap_k=lo_snap_k,
                sh_bands=sh_bands,
                sh_max_centroids=self.chunk_config.sh_max_centroids,
                n_gaussians=n_gaussians,
                matcher_k_passes=self.chunk_config.matcher_k_passes,
                faiss_threads=faiss_threads,
                device=self.device,
                reuse_inactive=self.chunk_config.reuse_inactive,
                keyframe_snap=self.chunk_config.keyframe_snap,
            )
            for idx, (start, end) in enumerate(chunk_boundaries)
        ]

        logger.info(
            f"Encoding {len(args_list)} chunks across {n_workers} workers "
            f"({faiss_threads} FAISS threads each)..."
        )
        results = run_parallel_chunks(args_list, n_workers)

        for result in tqdm(results, desc="Loading chunk atlases"):
            with np.load(result.npz_path) as data:
                all_presence.extend(data["presence"])
                atlases = data["atlases"]  # [F, H, W, 3]
                means_lo = data["means_lo"]  # [F, N, 3]
                all_atlases.extend(atlases[i] for i in range(atlases.shape[0]))
                all_means_lo.extend(means_lo[i] for i in range(means_lo.shape[0]))
            os.unlink(result.npz_path)
            if result.sh_tensors is not None:
                all_sh_tensors.append(result.sh_tensors)

    def _load_ply_sequence(self, input_dir: Path) -> list[GSTensor]:
        """Load PLY/SPZ files as GSTensor list (parallelized I/O)."""
        files = sorted(
            (p for p in (*input_dir.glob("*.ply"), *input_dir.glob("*.spz"))),
            key=lambda x: int(re.findall(r"\d+", x.stem)[-1])
            if re.findall(r"\d+", x.stem)
            else x.stem,
        )

        if not files:
            raise ValueError(f"No PLY or SPZ files found in {input_dir}")

        suffixes = {p.suffix.lower() for p in files}
        logger.info(f"Found {len(files)} frame files ({'/'.join(sorted(s[1:] for s in suffixes))})")

        def _read_frame(path: Path) -> gsply.GSData:
            # gsply reads SPZ and any-property-order GS PLYs natively.
            if path.suffix.lower() == ".spz":
                return gsply.read_spz(path)
            return gsply.plyread(path)

        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 8, 16)) as pool:
            gsdata_list = list(pool.map(_read_frame, files))

        # Keep on CPU to avoid OOM. Tensors will be moved to GPU chunk-by-chunk.
        frames = []
        for gsdata in gsdata_list:
            opacities = gsdata.opacities
            if opacities.ndim == 1:
                opacities = opacities.reshape(-1, 1)

            shN = None
            if hasattr(gsdata, "shN") and gsdata.shN is not None and gsdata.shN.size > 0:
                shN = torch.from_numpy(gsdata.shN).float()

            # PLY logit -inf represents exactly zero opacity. Treat it as
            # inactive, including when PLY has no explicit mask property.
            presence = ~np.isneginf(opacities.reshape(-1))
            if gsdata.masks is not None:
                presence &= np.asarray(gsdata.masks).astype(bool).reshape(-1)

            gstensor = GSTensor(
                means=torch.from_numpy(gsdata.means).float(),
                scales=torch.from_numpy(gsdata.scales).float(),
                quats=torch.from_numpy(gsdata.quats).float(),
                opacities=torch.from_numpy(opacities).float(),
                sh0=torch.from_numpy(gsdata.sh0).float(),
                shN=shN,
                masks=torch.from_numpy(presence.copy()),
            )
            frames.append(gstensor)

        logger.info(f"Loaded {len(frames)} frames (CPU)")
        return frames

    # Odd K values safe for K-snap: ((255 + K//2) // K * K) < 256.
    # K=13,29,33,37,39,43,45 cause modular wrapping and data corruption.
    _SAFE_K_VALUES = [
        1,
        3,
        5,
        7,
        9,
        11,
        15,
        17,
        19,
        21,
        23,
        25,
        27,
        31,
        35,
        41,
        47,
        49,
        51,
    ]

    def _compute_auto_k(
        self,
        frames: list[GSTensor],
        means_min: torch.Tensor,
        means_max: torch.Tensor,
    ) -> int:
        """Compute optimal K from scene Gaussian scale distribution.

        Targets error/scale ratio of ~5% so K-snap displacement is small
        relative to Gaussian size. Selects from verified safe K values only.

        Returns:
            Optimal safe K value.
        """
        from gscodec.constants import SCALE_16BIT

        n_sample = min(5, len(frames))
        frames = active_frames(frames)
        n_sample = min(n_sample, len(frames))
        all_scales = torch.cat([f.scales[:, :3].to(self.device).exp() for f in frames[:n_sample]])
        median_scale = all_scales.median().item()

        extent = (means_max - means_min).max().item()

        # Target: K-snap error ≈ 10% of median Gaussian scale.
        # Factor 0.20 accounts for round() halving the average error vs floor().
        k_raw = 0.20 * SCALE_16BIT * median_scale / max(extent, 1e-8)
        k_raw = max(3, min(127, round(k_raw)))

        # Select nearest safe K value
        best_k = min(self._SAFE_K_VALUES, key=lambda k: abs(k - k_raw))
        logger.info(
            f"Auto-K: median_scale={median_scale:.4f}, extent={extent:.2f}, "
            f"raw_k={k_raw}, selected K={best_k}"
        )
        return best_k

    def _prune_invisible(
        self,
        frames: list,
        opacity_threshold: float = 0.10,
        scale_threshold: float = 1e-3,
    ) -> list:
        """Remove Gaussians that are invisible across ALL frames.

        A Gaussian is pruned only if its max opacity (sigmoid) across every
        frame is below threshold OR its max linear scale is sub-pixel across
        every frame. This is safe because the Gaussian never contributes to
        any rendered view in the entire sequence.

        Args:
            frames: List of GSTensor frames.
            opacity_threshold: Max sigmoid opacity below which a Gaussian is invisible.
            scale_threshold: Max linear scale below which a Gaussian is sub-pixel.

        Returns:
            List of pruned GSTensor frames (may have fewer Gaussians).
        """
        import torch

        if not frames:
            return frames

        N = frames[0].means.shape[0]
        device = self.device

        # Skip pruning for variable-count sequences (no 1:1 correspondence)
        if any(f.means.shape[0] != N for f in frames):
            return frames
        if self.chunk_config.identity_mode != "stable":
            return frames
        ever_active = torch.stack([f.masks.reshape(-1).bool() for f in frames]).any(dim=0)
        if not self.chunk_config.lossy_pruning:
            if not ever_active.any():
                ever_active[0] = True  # Keep a disabled slot for a valid empty scene stream.
            return [f[ever_active] for f in frames]

        # Track max opacity and max scale across ALL frames
        max_opacity = torch.full((N,), -float("inf"), device=device)
        max_scale = torch.full((N,), -float("inf"), device=device)
        for f in frames:
            max_opacity = torch.maximum(
                max_opacity, torch.where(f.masks.to(device).reshape(-1), torch.sigmoid(f.opacities.to(device).squeeze(-1)), 0)
            )
            max_scale = torch.maximum(max_scale, torch.where(f.masks.to(device).reshape(-1), f.scales.to(device).exp().max(dim=1).values, 0))

        # Volume-weighted importance: catches Gaussians that are small AND transparent
        max_volume = torch.full((N,), -float("inf"), device=device)
        for f in frames:
            vol = f.scales.to(device).exp().prod(dim=1)
            vol = torch.where(f.masks.to(device).reshape(-1), vol, 0)
            max_volume = torch.maximum(max_volume, vol)
        importance = max_opacity * max_volume
        importance_threshold = torch.quantile(importance, 0.005)

        # Prune: invisible OR negligible importance (bottom 0.5%)
        prune_mask = (
            (max_opacity < opacity_threshold)
            | (max_scale < scale_threshold)
            | (importance < importance_threshold)
        )
        keep_mask = (~prune_mask).cpu()
        if not keep_mask.any():
            keep_mask[0] = True
        n_pruned = prune_mask.sum().item()

        if n_pruned == 0:
            return frames

        pruned = [f[keep_mask] for f in frames]
        logger.info(
            f"Pruned {n_pruned}/{N} always-invisible Gaussians "
            f"({100 * n_pruned / N:.1f}%, op<{opacity_threshold} or sc<{scale_threshold})"
        )
        return pruned

    @staticmethod
    def _compress_means_lo(lo_frames: list[np.ndarray]) -> bytes:
        """Compress means_lo with per-frame zstd for random access.

        Each frame is independently zstd-compressed, enabling per-frame random
        access. Uses K-aware base-N packing when the number of unique lo values
        is small enough (n_vals^3 <= 256) to pack xyz triples as single bytes,
        otherwise falls back to column-major layout.

        Format: u32 (n_frames | flags) + N x (u32 compressed_size + zstd blob).
        Bit 31 (0x80000000): column-major layout.
        Bit 30 (0x40000000): base-N packed (1 byte per triple), base in first
        byte of each frame blob (before zstd).

        Args:
            lo_frames: List of [N, 3] uint8 lo arrays.

        Returns:
            Compressed bytes.
        """
        import struct

        _ZSTD_LEVEL = 13
        _N_WORKERS = min(os.cpu_count() or 4, 8)

        # Detect unique lo values across all frames for K-aware base-N packing
        all_vals = np.unique(np.concatenate([lo.ravel() for lo in lo_frames]))
        n_vals = len(all_vals)
        use_base_n = n_vals**3 <= 256

        # Build payloads first, then compress all in parallel
        payloads: list[bytes] = []
        if use_base_n:
            lut = np.zeros(256, dtype=np.uint8)
            lut[all_vals] = np.arange(n_vals, dtype=np.uint8)
            base = n_vals
            for lo in lo_frames:
                idx = lut[lo]  # [N, 3]
                packed = (
                    idx[:, 0].astype(np.uint16) * base * base
                    + idx[:, 1].astype(np.uint16) * base
                    + idx[:, 2]
                )
                payloads.append(
                    bytes([base]) + all_vals.tobytes() + packed.astype(np.uint8).tobytes()
                )
            logger.info(
                f"  Means Lo: base-{base} packing ({n_vals} unique vals, "
                f"{n_vals**3} triples fit in uint8)"
            )
        else:
            for lo in lo_frames:
                payloads.append(lo.T.copy().tobytes())  # [3, N] contiguous col-major

        def _zstd_compress(payload: bytes) -> bytes:
            return zstd.ZstdCompressor(level=_ZSTD_LEVEL).compress(payload)

        with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
            frame_blobs = list(pool.map(_zstd_compress, payloads))

        raw_total = sum(lo.nbytes for lo in lo_frames)
        compressed_total = sum(len(b) for b in frame_blobs)

        flags = 0x40000000 if use_base_n else 0x80000000
        n_frames_flagged = len(lo_frames) | flags
        parts: list[bytes] = [struct.pack("<I", n_frames_flagged)]
        for blob in frame_blobs:
            parts.append(struct.pack("<I", len(blob)) + blob)
        out = b"".join(parts)

        logger.info(
            f"  Means Lo: {raw_total:,} raw -> {len(out):,} compressed "
            f"({compressed_total/raw_total*100:.1f}%)"
        )
        return out

    @staticmethod
    def _compress_sh_payload(
        sh_data: SHChunkData,
        chunk_frame_counts: list[int],
    ) -> bytes:
        """Compress SH payload: global centroids + per-chunk frame labels.

        Layout:
            GLOBAL:
                u16  n_centroids
                u8   sh_bands
                256 x f32  codebook (1024 bytes)
                u32  compressed_centroids_size
                zstd blob  (centroids [K, coeffs*3] uint8)
            PER-CHUNK:
                u32  n_frames_in_chunk
                For each frame:
                    u32  compressed_label_size
                    zstd blob  (labels — XOR delta from frame 0 of chunk)

        Args:
            sh_data: Global SHChunkData with codebook, centroids, and flat per-frame labels.
            chunk_frame_counts: Number of frames per chunk.

        Returns:
            Serialized bytes.
        """
        import struct

        _ZSTD_LEVEL = 13
        _N_WORKERS = min(os.cpu_count() or 4, 8)
        parts: list[bytes] = []

        # Global header: n_centroids + sh_bands + codebook + centroids
        parts.append(struct.pack("<HB", sh_data.n_centroids, sh_data.sh_bands))
        parts.append(sh_data.codebook.astype(np.float32).tobytes())

        raw_centroids = sh_data.centroids.tobytes()
        compressed_centroids = zstd.ZstdCompressor(level=_ZSTD_LEVEL).compress(raw_centroids)
        parts.append(struct.pack("<I", len(compressed_centroids)))
        parts.append(compressed_centroids)

        # Pre-compute all label payloads (XOR delta against chunk keyframe),
        # then compress all frames in parallel via ThreadPoolExecutor.
        # Delta always refs frame 0, so all frames are independent.
        total_raw = 0
        frame_offset = 0
        chunk_payload_counts: list[int] = []
        all_label_payloads: list[bytes] = []

        for n_frames_chunk in chunk_frame_counts:
            chunk_labels = sh_data.labels[frame_offset : frame_offset + n_frames_chunk]
            keyframe_labels = chunk_labels[0]
            chunk_payload_counts.append(n_frames_chunk)
            for t, labels in enumerate(chunk_labels):
                total_raw += labels.nbytes
                if t == 0:
                    all_label_payloads.append(labels.tobytes())
                else:
                    all_label_payloads.append(np.bitwise_xor(labels, keyframe_labels).tobytes())
            frame_offset += n_frames_chunk

        def _zstd_compress(payload: bytes) -> bytes:
            return zstd.ZstdCompressor(level=_ZSTD_LEVEL).compress(payload)

        with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
            all_label_blobs = list(pool.map(_zstd_compress, all_label_payloads))

        # Reassemble per-chunk structure
        total_compressed = 0
        blob_idx = 0
        for n_frames_chunk in chunk_payload_counts:
            parts.append(struct.pack("<I", n_frames_chunk))
            for _ in range(n_frames_chunk):
                blob = all_label_blobs[blob_idx]
                blob_idx += 1
                total_compressed += len(blob)
                parts.append(struct.pack("<I", len(blob)))
                parts.append(blob)

        out = b"".join(parts)
        logger.info(
            f"  SH payload: {total_raw:,} raw labels -> {len(out):,} total "
            f"({len(chunk_frame_counts)} chunks, labels {total_compressed/max(total_raw,1)*100:.1f}%)"
        )
        return out

    def _compute_global_ranges(self, frames: list[GSTensor]) -> QuantRanges:
        """Compute global min/max ranges across all frames.

        Uses exact min/max for all attributes to avoid precision loss from
        percentile clipping. Scale ranges in particular need full coverage —
        percentile clipping (p1/p99) wastes 13-16% of quantization range and
        causes 15+ dB PSNR degradation on dense scenes.

        Processes frames in batches of _RANGE_BATCH to reduce Python loop
        overhead (3000 iters → ~94 iters for 3k frames).
        """
        _RANGE_BATCH = 32
        # Range reduction never reads SHN. Do not copy the largest field merely
        # to apply presence masks to means/scales/rotation/opacity/base color.
        frames = active_frames([
            GSTensor(means=f.means, scales=f.scales, quats=f.quats,
                     opacities=f.opacities, sh0=f.sh0, shN=None, masks=f.masks)
            for f in frames
        ])
        device = self.device
        def color_values(frame):
            values = frame.sh0.to(device).float()
            return values if self.chunk_config.identity_mode == "preserve" else values.clamp(-2, 4)

        _ycbcr_mat = torch.tensor(
            [
                [0.299, 0.587, 0.114],
                [-0.169, -0.331, 0.500],
                [0.500, -0.419, -0.081],
            ],
            device=device,
            dtype=torch.float32,
        )

        first = frames[0]
        means_log = log_transform(first.means.to(device))
        means_min = means_log.amin(dim=0)
        means_max = means_log.amax(dim=0)
        scales_min = first.scales.to(device).amin(dim=0)
        scales_max = first.scales.to(device).amax(dim=0)
        quats_asin = torch.asin(first.quats.to(device).clamp(-0.9999, 0.9999)) / (3.14159265 / 2)
        quats_min = quats_asin.amin(dim=0)
        quats_max = quats_asin.amax(dim=0)
        opacity_min = first.opacities.to(device).amin()
        opacity_max = first.opacities.to(device).amax()
        sh0_ycc = torch.einsum("nc,dc->nd", color_values(first), _ycbcr_mat)
        sh0_min = sh0_ycc.amin(dim=0)
        sh0_max = sh0_ycc.amax(dim=0)

        # Concatenate (not stack) along the point axis so frames with differing
        # Gaussian counts still reduce to a single per-channel global min/max.
        for i in range(1, len(frames), _RANGE_BATCH):
            batch = frames[i : i + _RANGE_BATCH]

            means_b = torch.cat(
                [log_transform(f.means.to(device)) for f in batch], dim=0
            )  # [Total_N, 3]
            means_min = torch.minimum(means_min, means_b.amin(dim=0))
            means_max = torch.maximum(means_max, means_b.amax(dim=0))
            del means_b

            scales_b = torch.cat([f.scales.to(device) for f in batch], dim=0)  # [Total_N, 3]
            scales_min = torch.minimum(scales_min, scales_b.amin(dim=0))
            scales_max = torch.maximum(scales_max, scales_b.amax(dim=0))
            del scales_b

            quats_b = torch.cat([f.quats.to(device).clamp(-0.9999, 0.9999) for f in batch], dim=0)
            qa_b = torch.asin(quats_b) / (3.14159265 / 2)
            quats_min = torch.minimum(quats_min, qa_b.amin(dim=0))
            quats_max = torch.maximum(quats_max, qa_b.amax(dim=0))
            del quats_b, qa_b

            op_b = torch.cat([f.opacities.to(device) for f in batch], dim=0)  # [Total_N, 1]
            opacity_min = torch.minimum(opacity_min, op_b.amin())
            opacity_max = torch.maximum(opacity_max, op_b.amax())
            del op_b

            sh0_b = torch.cat(
                [color_values(f) for f in batch], dim=0
            )  # [Total_N, 3]
            sh0_ycc_b = torch.einsum("nc,dc->nd", sh0_b, _ycbcr_mat)
            sh0_min = torch.minimum(sh0_min, sh0_ycc_b.amin(dim=0))
            sh0_max = torch.maximum(sh0_max, sh0_ycc_b.amax(dim=0))
            del sh0_b, sh0_ycc_b

        quats_min = quats_min.clamp(min=QUATS_CLIP[0])
        quats_max = quats_max.clamp(max=QUATS_CLIP[1])
        opacity_min = opacity_min.clamp(min=OPACITY_CLIP[0])
        opacity_max = opacity_max.clamp(max=OPACITY_CLIP[1])
        # SH0 clip in YCbCr space — wider range for Cb/Cr channels
        if self.chunk_config.identity_mode != "preserve":
            sh0_min = sh0_min.clamp(min=-4.0)
            sh0_max = sh0_max.clamp(max=4.0)

        return QuantRanges(
            means_min=means_min.cpu().numpy(),
            means_max=means_max.cpu().numpy(),
            scales_min=scales_min.cpu().numpy(),
            scales_max=scales_max.cpu().numpy(),
            quats_min=quats_min.cpu().numpy(),
            quats_max=quats_max.cpu().numpy(),
            opacity_min=float(opacity_min.cpu()),
            opacity_max=float(opacity_max.cpu()),
            sh0_min=sh0_min.cpu().numpy(),
            sh0_max=sh0_max.cpu().numpy(),
        )

    def _compute_chunk_boundaries(self, n_frames: int) -> list[tuple[int, int]]:
        """Compute chunk start/end indices.

        Uses GSFlow metadata if available, otherwise falls back to fixed chunk_size.
        """
        if self.chunk_config.gsflow_metadata:
            gsflow = GSFlowMetadata.load(self.chunk_config.gsflow_metadata)
            boundaries = [(c.start_frame, c.end_frame + 1) for c in gsflow.chunks]
            logger.info(f"Using GSFlow metadata: {len(boundaries)} chunks")
            return boundaries

        chunk_size = self.chunk_config.size
        n_chunks = math.ceil(n_frames / chunk_size)
        return [(i * chunk_size, min((i + 1) * chunk_size, n_frames)) for i in range(n_chunks)]
