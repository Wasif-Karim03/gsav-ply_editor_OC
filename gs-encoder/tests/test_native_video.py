"""Native xllvp9 must preserve atlas bytes and GOP seek points without fallback."""

import shutil
import subprocess
import sys

import numpy as np
import pytest

from gscodec.encoder import video_writer
from gscodec.encoder.video_writer import _parse_ivf, require_native_encoder


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_native_exact_luma_and_fixed_gop(tmp_path, monkeypatch):
    rng = np.random.default_rng(8)
    frames = [rng.integers(16, 236, (12, 20), dtype=np.uint8) for _ in range(5)]
    import xllvp9.pipeline

    def reject_fallback(*args, **kwargs):
        raise AssertionError("FFmpeg encoding must not be used")

    monkeypatch.setattr(xllvp9.pipeline, "encode_ivf", reject_fallback)
    monkeypatch.setattr(video_writer, "_parse_ivf", lambda data: data)
    encoded = video_writer.encode_to_ivf(frames, 24, 2)
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


def test_missing_xllvp9_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "xllvp9", None)
    with pytest.raises(RuntimeError, match="Native xllvp9 is required"):
        require_native_encoder()


def test_python_package_without_native_module_cannot_fallback(monkeypatch):
    import xllvp9.native_backend

    monkeypatch.setattr(xllvp9.native_backend, "native_backend_available", lambda: False)
    with pytest.raises(RuntimeError, match="native _libvpx_ref module is unavailable"):
        video_writer.encode_to_ivf([np.zeros((12, 20), np.uint8)], 30, 1)
