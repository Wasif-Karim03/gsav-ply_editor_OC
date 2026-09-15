"""
Data providers for the GSAV codec.

GSAVFileProvider reads GSAV container files and provides:
- Header and metadata access
- Video frame decoding via ffmpeg (piped, no temp files)

GSAV Layout:
    Header (128B) -> Ranges (112B) -> ChunkIndex (16B x nChunks) -> FrameIndex (8B x nFrames)
    -> MeansLoPayload -> VideoPayload -> AudioPayload (optional)
"""

import logging
import struct
import subprocess
import threading
from pathlib import Path
from typing import BinaryIO

import numpy as np

from gscodec.common.binary_format import (
    get_frame_size,
    is_keyframe,
    read_chunk_index,
    read_frame_index,
    read_header,
    read_ranges,
)
from gscodec.common.types import (
    ChunkEntry,
    FrameEntry,
    GSAVHeader,
    QuantRanges,
    SHChunkData,
)

logger = logging.getLogger(__name__)


def _ivf_fourcc(codec: str) -> int:
    """Derive IVF FourCC from WebCodecs codec string."""
    if codec.startswith("vp09"):
        return 0x30395056  # "VP90"
    return 0x31305641  # "AV01"


class GSAVFileProvider:
    """Provides access to GSAV file contents.

    Parses the container format and provides methods to:
    - Get header, global ranges, chunk index
    - Decode video frames via ffmpeg (piped)
    """

    def __init__(self, path: str | Path | bytes):
        """Initialize provider from GSAV file.

        Args:
            path: Path to .gsav file.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is invalid.
        """
        self._data = path if isinstance(path, bytes) else None
        self.path = None if self._data is not None else Path(path)
        if self.path is not None and not self.path.exists():
            raise FileNotFoundError(f"GSAV file not found: {self.path}")

        self._file: BinaryIO | None = None
        self._header: GSAVHeader | None = None
        self._ranges: QuantRanges | None = None
        self._chunk_index: list[ChunkEntry] | None = None
        self._frame_index: list[FrameEntry] | None = None

        # Cached video payload for streaming decode
        self._video_payload: bytes | None = None

        self._parse_container()

    def _open(self) -> BinaryIO:
        if self._data is not None:
            import io
            return io.BytesIO(self._data)
        return open(self.path, "rb")

    def _parse_container(self) -> None:
        """Parse container header, indices, and chunk info."""
        with self._open() as f:
            self._header = read_header(f)
            from gscodec.common.v3 import validate_extensions

            validate_extensions(self._header, len(self._data) if self._data is not None else self.path.stat().st_size)
            f.seek(self._header["ranges_offset"])
            self._ranges = read_ranges(f)

            # Read chunk index
            if self._header["n_chunks"] > 0:
                f.seek(self._header["chunk_index_offset"])
                self._chunk_index = read_chunk_index(f, self._header["n_chunks"])
            else:
                self._chunk_index = []

            # Read frame index
            f.seek(self._header["frame_index_offset"])
            self._frame_index = read_frame_index(f, self._header["n_frames"])

        logger.debug(
            "Parsed GSAV: %d frames, %d chunks, chunk_size=%d, %dx%d atlas (side=%d)",
            self._header["n_frames"],
            self._header["n_chunks"],
            self._header["chunk_size"],
            self.atlas_width,
            self.atlas_height,
            self._header["atlas_side"],
        )

    @property
    def header(self) -> GSAVHeader:
        """Get file header."""
        assert self._header is not None
        return self._header

    @property
    def ranges(self) -> QuantRanges:
        """Get global quantization ranges."""
        assert self._ranges is not None
        return self._ranges

    @property
    def n_frames(self) -> int:
        """Get total number of frames."""
        return self.header["n_frames"]

    @property
    def n_chunks(self) -> int:
        """Get number of chunks."""
        return self.header["n_chunks"]

    @property
    def chunk_size(self) -> int:
        """Get frames per chunk (GOP size)."""
        return self.header["chunk_size"]

    @property
    def n_gaussians(self) -> int:
        """Get number of Gaussians per frame."""
        return self.header["n_gaussians"]

    @property
    def atlas_side(self) -> int:
        """Get gaussian grid side (ceil(sqrt(n_gaussians)) rounded even)."""
        return self.header["atlas_side"]

    @property
    def n_atlas_cols(self) -> int:
        """Get number of atlas columns."""
        return self.header["n_atlas_cols"]

    @property
    def n_atlas_rows(self) -> int:
        """Get number of atlas rows."""
        return self.header["n_atlas_rows"]

    @property
    def atlas_width(self) -> int:
        """Get atlas width (derived from atlas_side * n_atlas_cols)."""
        return self.atlas_side * self.n_atlas_cols

    @property
    def atlas_height(self) -> int:
        """Get atlas height (derived from atlas_side * n_atlas_rows)."""
        return self.atlas_side * self.n_atlas_rows

    @property
    def fps(self) -> int:
        """Get frames per second."""
        return self.header["fps"]

    @property
    def chunk_index(self) -> list[ChunkEntry]:
        """Get chunk index entries."""
        assert self._chunk_index is not None
        return self._chunk_index

    def get_chunk_for_frame(self, frame_idx: int) -> int:
        """Get chunk index for a given global frame index.

        Args:
            frame_idx: Global frame index.

        Returns:
            Chunk index (0-based).
        """
        for i, entry in enumerate(self.chunk_index):
            if entry["start_frame"] <= frame_idx <= entry["end_frame"]:
                return i
        raise IndexError(f"Frame {frame_idx} not found in any chunk")

    def get_frame_entry(self, frame_idx: int) -> FrameEntry:
        """Get frame index entry."""
        assert self._frame_index is not None
        return self._frame_index[frame_idx]

    def is_keyframe(self, frame_idx: int) -> bool:
        """Check if a frame is a keyframe (I-frame)."""
        return is_keyframe(self.get_frame_entry(frame_idx))

    def decode_all_frames(self) -> list[np.ndarray]:
        """Decode ALL video frames in one ffmpeg pass (streaming, no temp files).

        Returns:
            List of grayscale atlas frames [H, W] for all frames.
        """
        ivf_data = self._build_full_ivf()
        return self._decode_ivf_piped(ivf_data, self.n_frames)

    def decode_video_frames(
        self,
        start_frame: int = 0,
        n_frames: int | None = None,
    ) -> list[np.ndarray]:
        """Decode video frames from the embedded AV1 stream.

        Args:
            start_frame: First frame to decode.
            n_frames: Number of frames to decode (None = all remaining).

        Returns:
            List of grayscale atlas frames [H, W].
        """
        if n_frames is None:
            n_frames = self.n_frames - start_frame

        if start_frame == 0 and n_frames == self.n_frames:
            return self.decode_all_frames()

        # Start from nearest keyframe for VP9 inter-frame decode
        chunk_i = self.get_chunk_for_frame(start_frame)
        keyframe_idx = self.chunk_index[chunk_i]["start_frame"]
        decode_count = (start_frame - keyframe_idx) + n_frames
        ivf_data = self._build_ivf_range(keyframe_idx, decode_count)
        all_frames = self._decode_ivf_piped(ivf_data, decode_count)
        skip = start_frame - keyframe_idx
        return all_frames[skip : skip + n_frames]

    def _load_video_payload(self) -> bytes:
        """Load entire video payload into memory (cached).

        Reads only the video section, excluding any trailing audio payload.
        """
        if self._video_payload is None:
            video_base = self.header["video_payload_offset"]
            with self._open() as f:
                f.seek(video_base)
                if self.header["audio_payload_size"] > 0:
                    video_size = self.header["audio_payload_offset"] - video_base
                    self._video_payload = f.read(video_size)
                else:
                    self._video_payload = f.read()
            logger.debug(f"Loaded video payload: {len(self._video_payload):,} bytes")
        return self._video_payload

    def _build_full_ivf(self) -> bytes:
        """Build IVF container for all frames using contiguous video payload."""
        assert self._frame_index is not None

        video_payload = self._load_video_payload()
        width = self.atlas_width
        height = self.atlas_height
        n_frames = self.n_frames
        fourcc = _ivf_fourcc(self.header.get("codec", "av01.0.04M.08"))

        header = struct.pack(
            "<4sHHIHHIIII",
            b"DKIF",
            0,
            32,
            fourcc,
            width,
            height,
            self.fps,
            1,
            n_frames,
            0,
        )

        total_size = 32 + n_frames * 12
        for entry in self._frame_index:
            total_size += get_frame_size(entry)

        ivf = bytearray(total_size)
        ivf[:32] = header
        offset = 32

        for i, entry in enumerate(self._frame_index):
            obu_size = get_frame_size(entry)
            obu_offset = entry["offset"]

            struct.pack_into("<IQ", ivf, offset, obu_size, i)
            offset += 12

            ivf[offset : offset + obu_size] = video_payload[obu_offset : obu_offset + obu_size]
            offset += obu_size

        return bytes(ivf)

    def _build_ivf_range(self, start_frame: int, n_frames: int) -> bytes:
        """Build IVF container for a range of frames."""
        assert self._frame_index is not None

        video_payload = self._load_video_payload()
        width = self.atlas_width
        height = self.atlas_height
        fourcc = _ivf_fourcc(self.header.get("codec", "av01.0.04M.08"))

        header = struct.pack(
            "<4sHHIHHIIII",
            b"DKIF",
            0,
            32,
            fourcc,
            width,
            height,
            self.fps,
            1,
            n_frames,
            0,
        )

        parts = [header]
        for i in range(start_frame, start_frame + n_frames):
            entry = self._frame_index[i]
            obu_size = get_frame_size(entry)
            obu_offset = entry["offset"]

            frame_header = struct.pack("<IQ", obu_size, i - start_frame)
            parts.append(frame_header)
            parts.append(video_payload[obu_offset : obu_offset + obu_size])

        return b"".join(parts)

    def _decode_ivf_piped(self, ivf_data: bytes, n_frames: int) -> list[np.ndarray]:
        """Decode IVF via piped ffmpeg (no temp files)."""
        width = self.atlas_width
        height = self.atlas_height
        frame_size = width * height

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-color_range",
            "pc",
            "-i",
            "pipe:0",
            "-vf",
            "extractplanes=y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-color_range",
            "pc",
            "pipe:1",
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        def write_stdin() -> None:
            assert process.stdin is not None
            try:
                process.stdin.write(ivf_data)
            finally:
                process.stdin.close()

        writer = threading.Thread(target=write_stdin)
        writer.start()

        frames: list[np.ndarray] = []
        assert process.stdout is not None

        for _ in range(n_frames):
            raw = process.stdout.read(frame_size)
            if len(raw) != frame_size:
                writer.join()
                stderr = process.stderr.read() if process.stderr else b""
                raise RuntimeError(
                    f"Expected {frame_size} bytes, got {len(raw)}. "
                    f"ffmpeg stderr: {stderr.decode('utf-8', errors='replace')}"
                )
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
            frames.append(frame.copy())

        writer.join()
        process.wait()

        return frames

    def get_means_lo(self, frame_idx: int) -> np.ndarray:
        """Read and decompress means_lo for a specific frame.

        Supports three formats:
        - Legacy raw: uncompressed [N, 3] row-major bytes
        - Per-frame zstd (row-major): n_frames header without high bit
        - Per-frame zstd (column-major): n_frames header with high bit (0x80000000)

        Args:
            frame_idx: Global frame index.

        Returns:
            [N, 3] uint8 raw lo-bytes.
        """
        import zstandard as zstd

        n = self.n_gaussians
        frame_size = n * 3
        lo_offset = self.header["means_lo_payload_offset"]

        # Build frame offset index on first access
        if not hasattr(self, "_lo_frame_offsets"):
            self._lo_frame_offsets, self._lo_col_major = self._build_lo_index()

        offsets = self._lo_frame_offsets
        if offsets is None:
            # Legacy raw format: direct seek
            offset = lo_offset + frame_idx * frame_size
            with self._open() as f:
                f.seek(offset)
                raw = f.read(frame_size)
            return np.frombuffer(raw, dtype=np.uint8).reshape(n, 3).copy()

        # Per-frame zstd: seek to frame blob and decompress
        file_offset, blob_size = offsets[frame_idx]
        with self._open() as f:
            f.seek(file_offset)
            blob = f.read(blob_size)

        dctx = zstd.ZstdDecompressor()

        if getattr(self, "_lo_base_n", False):
            max_size = 1 + 256 + n
            raw = dctx.decompress(blob, max_output_size=max_size)
            base = raw[0]
            lut = np.frombuffer(raw[1 : 1 + base], dtype=np.uint8)
            packed = np.frombuffer(raw[1 + base :], dtype=np.uint8)
            ix = packed // (base * base)
            rem = packed % (base * base)
            iy = rem // base
            iz = rem % base
            return np.stack([lut[ix], lut[iy], lut[iz]], axis=1)

        # Check if temporal XOR is active
        lo_xor = getattr(self, "_lo_xor", False)

        if not lo_xor:
            # No XOR: decompress directly
            raw = dctx.decompress(blob, max_output_size=frame_size)
            arr = np.frombuffer(raw, dtype=np.uint8)
            if self._lo_col_major:
                return arr.reshape(3, n).T.copy()
            return arr.reshape(n, 3).copy()

        # Temporal XOR: decode from chunk keyframe, apply XOR chain
        chunk_i = self.get_chunk_for_frame(frame_idx)
        chunk_start = self.chunk_index[chunk_i]["start_frame"]

        result = None
        for fi in range(chunk_start, frame_idx + 1):
            fo, bs = offsets[fi]
            with self._open() as f2:
                f2.seek(fo)
                b = f2.read(bs)
            raw = dctx.decompress(b, max_output_size=frame_size)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(3, n).T.copy()

            if result is None:
                result = arr  # keyframe: raw data
            else:
                result = np.bitwise_xor(result, arr)  # apply XOR delta

        return result

    def _build_lo_index(self) -> tuple[list[tuple[int, int]] | None, bool]:
        """Build per-frame offset index for compressed means_lo.

        Returns:
            Tuple of (offsets, col_major) where offsets is a list of
            (file_offset, blob_size) per frame (or None for legacy),
            and col_major indicates column-major storage layout.
        """
        import struct

        n = self.n_gaussians
        frame_size = n * 3
        lo_offset = self.header["means_lo_payload_offset"]
        video_offset = self.header["video_payload_offset"]
        payload_size = video_offset - lo_offset
        expected_raw = frame_size * self.n_frames

        if payload_size == expected_raw:
            return None, False  # Legacy raw format

        # Per-frame zstd format: u32 header + N x (u32 size + blob)
        # High bit of header signals column-major layout
        with self._open() as f:
            f.seek(lo_offset)
            raw_header = struct.unpack("<I", f.read(4))[0]
            col_major = bool(raw_header & 0x80000000)
            self._lo_base_n = bool(raw_header & 0x40000000)
            self._lo_xor = bool(raw_header & 0x20000000)
            n_frames_header = raw_header & 0x1FFFFFFF
            offsets: list[tuple[int, int]] = []
            pos = lo_offset + 4
            for _ in range(n_frames_header):
                f.seek(pos)
                blob_size = struct.unpack("<I", f.read(4))[0]
                offsets.append((pos + 4, blob_size))
                pos += 4 + blob_size

        return offsets, col_major

    def get_video_byte_range(
        self,
        start_frame: int,
        n_frames: int,
    ) -> tuple[int, int]:
        """Calculate byte range for video frames (for HTTP Range requests)."""
        assert self._frame_index is not None
        video_base = self.header["video_payload_offset"]

        start_entry = self._frame_index[start_frame]
        end_frame_idx = min(start_frame + n_frames - 1, self.n_frames - 1)
        end_entry = self._frame_index[end_frame_idx]

        start_byte = video_base + start_entry["offset"]
        end_byte = video_base + end_entry["offset"] + get_frame_size(end_entry)

        return start_byte, end_byte

    @property
    def sh_bands(self) -> int:
        """Get SH band level (0=none, 1-3)."""
        return self.header.get("sh_bands", 0)

    def get_sh_chunk_data(self, chunk_idx: int) -> SHChunkData | None:
        """Read global SH centroids + codebook (shared across all chunks).

        Args:
            chunk_idx: Chunk index (ignored — data is global).

        Returns:
            SHChunkData (without labels — those come from get_sh_labels), or None.
        """
        if self.sh_bands == 0 or self.header.get("sh_payload_offset", 0) == 0:
            return None

        if not hasattr(self, "_sh_global"):
            self._build_sh_index()

        return self._sh_global

    def get_sh_labels(self, frame_idx: int) -> np.ndarray | None:
        """Read and decompress SH labels for a specific frame.

        Labels are delta-encoded (XOR with keyframe) and zstd-compressed.

        Args:
            frame_idx: Global frame index.

        Returns:
            [N] uint16 labels, or None if no SH data.
        """
        import zstandard as zstd

        if self.sh_bands == 0 or self.header.get("sh_payload_offset", 0) == 0:
            return None

        if not hasattr(self, "_sh_global"):
            self._build_sh_index()

        chunk_idx = self.get_chunk_for_frame(frame_idx)
        chunk_start = self.chunk_index[chunk_idx]["start_frame"]
        frame_in_chunk = frame_idx - chunk_start

        label_offsets = self._sh_label_offsets[chunk_idx]
        n = self.n_gaussians
        dctx = zstd.ZstdDecompressor()

        # Read keyframe labels
        kf_offset, kf_size = label_offsets[0]
        with self._open() as f:
            f.seek(kf_offset)
            kf_blob = f.read(kf_size)
        keyframe_labels = np.frombuffer(
            dctx.decompress(kf_blob, max_output_size=n * 2),
            dtype=np.uint16,
        ).copy()

        if frame_in_chunk == 0:
            return keyframe_labels

        # Read delta frame and XOR with keyframe
        d_offset, d_size = label_offsets[frame_in_chunk]
        with self._open() as f:
            f.seek(d_offset)
            d_blob = f.read(d_size)
        delta = np.frombuffer(
            dctx.decompress(d_blob, max_output_size=n * 2),
            dtype=np.uint16,
        )
        return np.bitwise_xor(keyframe_labels, delta)

    def _build_sh_index(self) -> None:
        """Parse SH payload: read global header + index per-chunk label offsets."""
        from gscodec.common.binary_format import build_sh_label_offsets, read_sh_global_data

        sh_offset = self.header["sh_payload_offset"]

        with self._open() as f:
            self._sh_global = read_sh_global_data(f, sh_offset)
            _, self._sh_label_offsets = build_sh_label_offsets(
                f, sh_offset, self.n_chunks,
            )

    @property
    def has_audio(self) -> bool:
        """Check if GSAV file contains an audio payload."""
        return self.header["audio_payload_size"] > 0

    def get_static_asset(self) -> bytes | None:
        """Read the separately indexed embedded GSST track, if present."""
        size = self.header.get("static_asset_size", 0)
        if not size:
            return None
        with self._open() as f:
            f.seek(self.header["static_asset_offset"])
            data = f.read(size)
        if len(data) != size:
            raise ValueError("Truncated static asset")
        from gscodec.common.static_track import validate_static_track

        validate_static_track(data)
        return data

    def get_audio(self) -> bytes | None:
        """Extract the audio payload from the GSAV file.

        Returns:
            Raw OGG/Opus bytes, or None if no audio is present.
        """
        if not self.has_audio:
            return None

        with self._open() as f:
            f.seek(self.header["audio_payload_offset"])
            return f.read(self.header["audio_payload_size"])

    def __len__(self) -> int:
        """Return total number of frames."""
        return self.n_frames
