"""Lossless video fallback must preserve atlas bytes and GOP seek points."""

import shutil
import subprocess

import numpy as np
import pytest

from gscodec.encoder.video_writer import _encode_ffmpeg, _parse_ivf


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_ffmpeg_exact_luma_and_fixed_gop(tmp_path):
    rng = np.random.default_rng(8)
    frames = [rng.integers(16, 236, (12, 20), dtype=np.uint8) for _ in range(5)]
    encoded = _encode_ffmpeg(frames, 24, 2)
    path = tmp_path / "atlas.ivf"
    path.write_bytes(encoded)
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    for i, expected in enumerate(frames):
        offset = i * 12 * 20 * 3 // 2
        actual = np.frombuffer(decoded[offset : offset + 12 * 20], np.uint8).reshape(12, 20)
        np.testing.assert_array_equal(actual, expected)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "frame=key_frame",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert probe.stdout.split() == [b"1", b"0", b"1", b"0", b"1"]
    _, entries = _parse_ivf(encoded)
    assert len(entries) == 5
