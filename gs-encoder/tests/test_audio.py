"""Tests for audio transcoding module."""

import wave
from pathlib import Path

import numpy as np
import pytest

from gscodec.encoder.audio import get_audio_duration, transcode_to_opus


def _make_wav(path: Path, duration: float, sample_rate: int = 44100) -> None:
    """Create a WAV file with a sine tone."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


class TestGetAudioDuration:
    def test_wav_duration(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        _make_wav(wav_path, duration=2.0)

        duration = get_audio_duration(wav_path)
        assert abs(duration - 2.0) < 0.1

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_audio_duration(Path("/nonexistent/audio.wav"))


class TestTranscodeToOpus:
    def test_produces_ogg_output(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        _make_wav(wav_path, duration=1.0)

        result = transcode_to_opus(wav_path, target_duration=1.0)

        # OGG files start with "OggS" magic
        assert result[:4] == b"OggS"
        assert len(result) > 0

    def test_trim_long_audio(self, tmp_path: Path) -> None:
        """Audio longer than target should be trimmed."""
        wav_path = tmp_path / "long.wav"
        _make_wav(wav_path, duration=5.0)

        result = transcode_to_opus(wav_path, target_duration=2.0)

        assert result[:4] == b"OggS"
        # Write to file and probe duration
        ogg_path = tmp_path / "trimmed.ogg"
        ogg_path.write_bytes(result)
        duration = get_audio_duration(ogg_path)
        assert abs(duration - 2.0) < 0.2

    def test_pad_short_audio(self, tmp_path: Path) -> None:
        """Audio shorter than target should be padded with silence."""
        wav_path = tmp_path / "short.wav"
        _make_wav(wav_path, duration=1.0)

        result = transcode_to_opus(wav_path, target_duration=3.0)

        assert result[:4] == b"OggS"
        ogg_path = tmp_path / "padded.ogg"
        ogg_path.write_bytes(result)
        duration = get_audio_duration(ogg_path)
        assert abs(duration - 3.0) < 0.2

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            transcode_to_opus(Path("/nonexistent/audio.wav"), target_duration=1.0)


class TestHeaderAudioFields:
    """Test that header round-trips audio fields correctly."""

    def test_header_roundtrip_with_audio(self, tmp_path: Path) -> None:
        from gscodec.common.binary_format import read_header, write_header
        from gscodec.common.types import HAS_AUDIO_FLAG, GSAVHeader

        header = GSAVHeader(
            magic=b"GSAV",
            version=1,
            n_gaussians=1000,
            n_frames=30,
            n_chunks=1,
            chunk_size=30,
            atlas_side=32,
            n_atlas_cols=6,
            n_atlas_rows=3,
            fps=30,
            flags=HAS_AUDIO_FLAG,
            ranges_offset=128,
            chunk_index_offset=192,
            frame_index_offset=208,
            means_lo_payload_offset=448,
            video_payload_offset=538448,
            codec="av01.0.14M.08",
            audio_payload_offset=1000000,
            audio_payload_size=12345,
            means_hi_payload_offset=0,
            sh_bands=0,
            sh_payload_offset=0,
        )

        path = tmp_path / "test.gsav"
        with open(path, "wb") as f:
            write_header(f, header)

        with open(path, "rb") as f:
            result = read_header(f)

        assert result["audio_payload_offset"] == 1000000
        assert result["audio_payload_size"] == 12345
        assert result["flags"] & HAS_AUDIO_FLAG

    def test_header_roundtrip_without_audio(self, tmp_path: Path) -> None:
        from gscodec.common.binary_format import read_header, write_header
        from gscodec.common.types import HAS_AUDIO_FLAG, GSAVHeader

        header = GSAVHeader(
            magic=b"GSAV",
            version=1,
            n_gaussians=1000,
            n_frames=30,
            n_chunks=1,
            chunk_size=30,
            atlas_side=32,
            n_atlas_cols=6,
            n_atlas_rows=3,
            fps=30,
            flags=0,
            ranges_offset=128,
            chunk_index_offset=192,
            frame_index_offset=208,
            means_lo_payload_offset=448,
            video_payload_offset=538448,
            codec="av01.0.14M.08",
            audio_payload_offset=0,
            audio_payload_size=0,
            means_hi_payload_offset=0,
            sh_bands=0,
            sh_payload_offset=0,
        )

        path = tmp_path / "test.gsav"
        with open(path, "wb") as f:
            write_header(f, header)

        with open(path, "rb") as f:
            result = read_header(f)

        assert result["audio_payload_offset"] == 0
        assert result["audio_payload_size"] == 0
        assert not (result["flags"] & HAS_AUDIO_FLAG)
