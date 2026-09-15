"""Audio transcoding for the GSAV container format.

Transcodes any input audio format to OGG/Opus via FFmpeg, trimmed or
padded to match video duration. The output is stored as a single
continuous blob in the GSAV audio payload section.
"""

import logging
import tempfile
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of an audio file in seconds via ffprobe.

    Args:
        audio_path: Path to audio file.

    Returns:
        Duration in seconds.

    Raises:
        FileNotFoundError: If audio file doesn't exist.
        RuntimeError: If ffprobe fails.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        probe = ffmpeg.probe(str(audio_path))
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"ffprobe failed for {audio_path}: {e.stderr.decode('utf-8', errors='replace') if e.stderr else 'unknown error'}"
        ) from e

    return float(probe["format"]["duration"])


def transcode_to_opus(input_path: Path, target_duration: float) -> bytes:
    """Transcode any audio file to OGG/Opus, matched to target duration.

    If the input is longer than target_duration, it is trimmed.
    If shorter, it is padded with silence.

    Args:
        input_path: Path to source audio file (MP3, WAV, AAC, OGG, etc.).
        target_duration: Target duration in seconds (typically n_frames / fps).

    Returns:
        Raw OGG/Opus bytes.

    Raises:
        FileNotFoundError: If input_path doesn't exist.
        RuntimeError: If FFmpeg transcoding fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    input_duration = get_audio_duration(input_path)
    needs_padding = input_duration < target_duration

    logger.info(
        f"Transcoding audio: {input_path.name} ({input_duration:.1f}s) -> OGG/Opus ({target_duration:.1f}s)"
    )
    if needs_padding:
        logger.warning(
            f"Audio ({input_duration:.1f}s) shorter than video ({target_duration:.1f}s), padding with silence"
        )
    elif input_duration > target_duration:
        logger.warning(
            f"Audio ({input_duration:.1f}s) longer than video ({target_duration:.1f}s), trimming"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        ogg_path = Path(tmpdir) / "audio.ogg"

        stream = ffmpeg.input(str(input_path))

        if needs_padding:
            stream = stream.filter("apad")

        output_kwargs = {
            "acodec": "libopus",
            "audio_bitrate": "128k",
            "ar": 48000,
            "ac": 2,
            "t": target_duration,
        }

        try:
            (
                stream
                .output(str(ogg_path), **output_kwargs)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(
                f"Audio transcode failed: {e.stderr.decode('utf-8', errors='replace') if e.stderr else 'unknown error'}"
            ) from e

        audio_bytes = ogg_path.read_bytes()

    logger.info(f"Transcoded audio: {len(audio_bytes):,} bytes")
    return audio_bytes
