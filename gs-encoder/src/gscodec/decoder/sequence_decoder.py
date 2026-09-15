"""
Sequence decoder for frame-by-frame GSAV codec.

Every frame is self-contained in the video atlas. No anchor binding needed.

Optimized for streaming:
- Single ffmpeg process for all frames (no temp files)
"""

import logging
from collections.abc import Iterator
from pathlib import Path

from gsply import GSData

from gscodec.common.types import HAS_MASK_FLAG, STATIC_ENCODING_NATIVE
from gscodec.common.v3 import static_encoding
from gscodec.decoder.chunk_decoder import ChunkDecoder
from gscodec.decoder.providers import GSAVFileProvider

logger = logging.getLogger(__name__)


class SequenceDecoder:
    """High-level decoder for GSAV files.

    Example:
        ```python
        decoder = SequenceDecoder.from_file("scene.gsav")

        for frame in decoder:
            render(frame)

        frames = decoder.decode_all()
        ```
    """

    def __init__(self, provider: GSAVFileProvider):
        """Initialize decoder with a provider.

        Args:
            provider: GSAVFileProvider for file access.
        """
        self._provider = provider
        self._static_decoded = False
        self._static_scene = None
        self._native = None
        self._decoder = ChunkDecoder(
            ranges=provider.ranges,
            atlas_width=provider.atlas_width,
            n_gaussians=provider.n_gaussians,
            n_atlas_cols=provider.n_atlas_cols,
            sh_bands=provider.sh_bands,
            has_mask=bool(provider.header["flags"] & HAS_MASK_FLAG),
        )

    def get_static_asset(self) -> bytes | None:
        """Return the embedded GSST track unchanged."""
        return self._provider.get_static_asset()

    def decode_static_asset(self) -> GSData | None:
        """Decode once and cache the static scene for persistent rendering."""
        if self._static_decoded:
            return self._static_scene
        encoding = static_encoding(self._provider.header)
        if encoding == STATIC_ENCODING_NATIVE:
            if self._native is not None:
                self._static_scene = self._native_frame(self._native.decode_static_asset())
            else:
                from gscodec.common.static_track import static_track_to_gsav

                decoder = SequenceDecoder(GSAVFileProvider(static_track_to_gsav(self.get_static_asset())))
                self._static_scene = decoder.decode_frame(0)
        self._static_decoded = True
        return self._static_scene

    def create_scene_buffer(self, device: str = "cpu"):
        """Allocate a persistent static-prefix/dynamic-suffix rendering buffer."""
        from gscodec.decoder.scene_buffer import SceneBuffer
        return SceneBuffer(self, device=device)

    @classmethod
    def from_file(cls, path: str | Path, *, backend: str = "python") -> "SequenceDecoder":
        """Create decoder from GSAV file path.

        Args:
            path: Path to .gsav file.

        Returns:
            SequenceDecoder instance.
        """
        if backend not in ("python", "native"):
            raise ValueError("backend must be python or native")
        result = cls(GSAVFileProvider(path))
        if backend == "native":
            import _gscodec_native
            result._native = _gscodec_native.SequenceDecoder(str(path))
        return result

    @staticmethod
    def _native_frame(frame: dict) -> GSData:
        import numpy as np
        return GSData(
            means=frame["means"], scales=frame["scales"], quats=frame["quats"],
            opacities=frame["opacities"], sh0=frame["sh0"],
            shN=frame["shN"] if frame["shN"] is not None else np.empty((len(frame["means"]), 0, 3), np.float32),
            masks=frame["presence"].astype(bool),
        )

    def __len__(self) -> int:
        """Return total number of frames."""
        return self._provider.n_frames

    def __iter__(self) -> Iterator[GSData]:
        """Iterate through all frames with streaming decode.

        Yields:
            GSData for each frame in order.
        """
        yield from self.decode_all()

    def decode_all(self) -> list[GSData]:
        """Decode all frames in one pass.

        Returns:
            List of all decoded GSData frames.
        """
        logger.info(f"Decoding {self._provider.n_frames} frames...")
        if self._native is not None:
            return [self.decode_frame(i) for i in range(len(self))]

        atlases = self._provider.decode_all_frames()
        logger.debug(f"Decoded {len(atlases)} atlas frames")

        # Cache SH chunk data per chunk to avoid re-reading
        sh_cache: dict[int, object] = {}

        frames = []
        for i, atlas in enumerate(atlases):
            means_lo = self._provider.get_means_lo(i)

            sh_labels = None
            sh_chunk_data = None
            if self._provider.sh_bands > 0:
                chunk_idx = self._provider.get_chunk_for_frame(i)
                if chunk_idx not in sh_cache:
                    sh_cache[chunk_idx] = self._provider.get_sh_chunk_data(chunk_idx)
                sh_chunk_data = sh_cache[chunk_idx]
                sh_labels = self._provider.get_sh_labels(i)

            frames.append(self._decoder.decode_frame(atlas, means_lo, sh_labels, sh_chunk_data))

        logger.info(f"Reconstructed {len(frames)} frames")
        return frames

    def decode_frame(self, frame_idx: int) -> GSData:
        """Decode a specific frame by index.

        Args:
            frame_idx: Global frame index.

        Returns:
            Decoded GSData frame.
        """
        if frame_idx < 0 or frame_idx >= len(self):
            raise IndexError(f"Frame index {frame_idx} out of range [0, {len(self)})")
        if self._native is not None:
            return self._native_frame(self._native.decode_frame(frame_idx))

        atlases = self._provider.decode_video_frames(frame_idx, 1)
        means_lo = self._provider.get_means_lo(frame_idx)

        sh_labels = None
        sh_chunk_data = None
        if self._provider.sh_bands > 0:
            chunk_idx = self._provider.get_chunk_for_frame(frame_idx)
            sh_chunk_data = self._provider.get_sh_chunk_data(chunk_idx)
            sh_labels = self._provider.get_sh_labels(frame_idx)

        return self._decoder.decode_frame(atlases[0], means_lo, sh_labels, sh_chunk_data)

    @property
    def n_gaussians(self) -> int:
        """Get number of Gaussians per frame."""
        return self._provider.n_gaussians

    @property
    def chunk_size(self) -> int:
        """Get frames per chunk (GOP size)."""
        return self._provider.chunk_size

    @property
    def fps(self) -> int:
        """Get frames per second."""
        return self._provider.fps

    @property
    def has_audio(self) -> bool:
        """Check if the GSAV file contains audio."""
        return self._provider.has_audio

    def get_audio(self) -> bytes | None:
        """Get the raw OGG/Opus audio bytes.

        Returns:
            Raw OGG/Opus bytes, or None if no audio is present.
        """
        return self._provider.get_audio()
