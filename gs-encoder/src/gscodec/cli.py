"""
Command-line interface for gscodec.

Provides two commands:
- compress: Encode Gaussian Splatting PLY sequences to GSAV format
- decompress: Decode GSAV files back to PLY sequences
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from gscodec.decoder import SequenceDecoder
from gscodec.encoder import ChunkConfig, SequenceEncoder, VideoConfig


@dataclass
class CompressConfig:
    """Configuration for the compress command."""

    input_dir: Path
    """Path to directory containing per-frame PLY or SPZ files."""

    output: Path
    """Output GSAV file path."""

    fps: int = 30
    """Frames per second."""

    chunk_size: int = 30
    """Frames per chunk (ignored if gsflow_metadata is provided)."""

    gsflow_metadata: Path | None = None
    """Path to GSFlow metadata.json for chunk boundaries."""

    device: str = "cuda:0"
    """Torch device for computation."""

    matching: bool = False
    """Enable temporal matching for unstructured point clouds with varying counts."""

    identity_mode: Literal["auto", "stable", "unstructured"] = "auto"
    """Use stable for persistent row identities; unstructured enables temporal matching."""

    lossy_pruning: bool = False
    """Allow opacity/size pruning of stable slots; disabled by default."""

    matcher_k_passes: int = 10
    """Number of iterative NN matching passes for conflict resolution."""

    lo_snap_k: int = 0
    """Round lo bytes to nearest K for compression (0=auto, 1=off). Auto-tunes from scene stats."""

    sh_bands: int = -1
    """SH band level (-1=auto-detect from source [default], 0=DC only, 1=SH1, 2=SH2, 3=SH3)."""

    parallel_chunks: int = 1
    """Worker processes for parallel chunk encoding (1=serial, 0=auto). Only used
    with --matching; the single GPU saturates around 4 workers (~2x speedup)."""

    audio: Path | None = None
    """Optional audio file to embed (any format, transcoded to Opus)."""

    static_asset: Path | None = None
    """Static PLY/SPZ to encode natively, or a precompressed GSST track."""

    verbose: bool = False
    """Enable verbose logging."""


@dataclass
class DecompressConfig:
    """Configuration for the decompress command."""

    input: Path
    """Input GSAV file path."""

    output_dir: Path
    """Output directory for PLY files."""

    no_audio: bool = False
    """Skip audio extraction."""

    verbose: bool = False
    """Enable verbose logging."""


def compress_main() -> None:
    """Entry point for the compress command."""

    config = tyro.cli(CompressConfig)

    log_level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Ensure output has .gsav extension
    output_path = config.output
    if output_path.suffix != ".gsav":
        output_path = output_path.with_suffix(".gsav")

    encoder = SequenceEncoder(
        video_config=VideoConfig(fps=config.fps),
        chunk_config=ChunkConfig(
            size=config.chunk_size,
            gsflow_metadata=config.gsflow_metadata,
            matching_enabled=config.matching,
            identity_mode=config.identity_mode,
            lossy_pruning=config.lossy_pruning,
            matcher_k_passes=config.matcher_k_passes,
            lo_snap_k=config.lo_snap_k,
            sh_bands=config.sh_bands,
            parallel_chunks=config.parallel_chunks,
        ),
        device=config.device,
    )
    encoder.compress(input_dir=config.input_dir, output=output_path, audio_path=config.audio,
                     static_asset_path=config.static_asset)


def decompress_main() -> None:
    """Entry point for the decompress command."""
    import gsply

    config = tyro.cli(DecompressConfig)

    log_level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Decode frames
    decoder = SequenceDecoder.from_file(config.input)
    total_frames = len(decoder)

    logging.info(f"Decompressing {total_frames} frames from {config.input}")

    for i, frame_data in enumerate(decoder):
        output_file = config.output_dir / f"frame_{i:06d}.ply"
        gsply.plywrite(output_file, frame_data)

        if (i + 1) % 10 == 0 or i == total_frames - 1:
            logging.info(f"Processed {i + 1}/{total_frames} frames")

    # Extract audio if present
    if not config.no_audio and decoder.has_audio:
        audio_path = config.output_dir / "audio.ogg"
        audio_data = decoder.get_audio()
        if audio_data:
            with open(audio_path, "wb") as f:
                f.write(audio_data)
            logging.info(f"Extracted audio to {audio_path}")

    static_data = decoder.get_static_asset()
    if static_data is not None:
        (config.output_dir / "static.gsst").write_bytes(static_data)
        gsply.plywrite(config.output_dir / "static.ply", decoder.decode_static_asset())

    logging.info(f"Decompression complete. Output saved to {config.output_dir}")


if __name__ == "__main__":
    compress_main()
