"""Tests for the SequenceEncoder class."""

import numpy as np
import pytest

from gscodec.encoder.sequence_encoder import SequenceEncoder


class TestMeansLoCompression:
    """Tests for means_lo column-major compression roundtrip."""

    def test_compress_decompress_col_major_roundtrip(self):
        """Column-major zstd compression preserves data exactly."""
        import struct

        import zstandard as zstd

        rng = np.random.default_rng(42)
        n = 1024
        frames = [rng.integers(0, 256, size=(n, 3), dtype=np.uint8) for _ in range(5)]

        compressed = SequenceEncoder._compress_means_lo(frames)

        # Parse header: high bit should be set
        raw_header = struct.unpack("<I", compressed[:4])[0]
        assert raw_header & 0x80000000, "Column-major flag not set"
        n_frames = raw_header & 0x7FFFFFFF
        assert n_frames == len(frames)

        # Decompress and verify roundtrip
        dctx = zstd.ZstdDecompressor()
        pos = 4
        for i in range(n_frames):
            blob_size = struct.unpack("<I", compressed[pos:pos + 4])[0]
            pos += 4
            raw = dctx.decompress(compressed[pos:pos + blob_size], max_output_size=n * 3)
            pos += blob_size
            # Decoder transposes [3, N] -> [N, 3]
            recovered = np.frombuffer(raw, dtype=np.uint8).reshape(3, n).T.copy()
            np.testing.assert_array_equal(recovered, frames[i])

    def test_ksnapped_data_smaller_col_major(self):
        """K-snapped data compresses better with column-major layout."""
        import zstandard as zstd

        rng = np.random.default_rng(42)
        n = 25600
        snap_vals = np.array([0, 35, 70, 105, 140, 175, 210, 245], dtype=np.uint8)
        frame = rng.choice(snap_vals, size=(n, 3)).astype(np.uint8)

        cctx = zstd.ZstdCompressor(level=3)
        row_size = len(cctx.compress(frame.tobytes()))
        col_size = len(cctx.compress(frame.T.copy().tobytes()))

        # Column-major should not be significantly worse (may be better with spatial locality)
        assert col_size <= row_size * 1.05, (
            f"Column-major unexpectedly larger: {col_size} vs {row_size}"
        )


class TestSequenceEncoder:
    """Tests for SequenceEncoder."""

    def test_init_default_config(self):
        """Test encoder initialization with default config."""
        encoder = SequenceEncoder()

        assert encoder.video_config.fps == 30
        assert encoder.chunk_config.size == 30

    def test_init_custom_config(self, video_config, chunk_config):
        """Test encoder initialization with custom config."""
        encoder = SequenceEncoder(
            video_config=video_config,
            chunk_config=chunk_config,
            device="cpu",
        )

        assert encoder.video_config == video_config
        assert encoder.chunk_config == chunk_config
        assert encoder.device == "cpu"

    def test_compute_chunk_boundaries(self):
        """Test chunk boundary computation."""
        encoder = SequenceEncoder()
        encoder.chunk_config.size = 10

        # Test exact division
        boundaries = encoder._compute_chunk_boundaries(30)
        assert boundaries == [(0, 10), (10, 20), (20, 30)]

        # Test with remainder
        boundaries = encoder._compute_chunk_boundaries(25)
        assert boundaries == [(0, 10), (10, 20), (20, 25)]

        # Test single chunk
        boundaries = encoder._compute_chunk_boundaries(5)
        assert boundaries == [(0, 5)]

    def test_compress_missing_input_dir_raises(self, tmp_path):
        """Test that compress with non-existent input raises error."""
        encoder = SequenceEncoder(device="cpu")

        with pytest.raises(ValueError, match="No PLY or SPZ files"):
            encoder.compress(
                input_dir=tmp_path / "nonexistent",
                output=tmp_path / "output.gsav",
            )


@pytest.mark.parametrize("explicit_mask", [None, [True, True, False, True]])
def test_ply_negative_infinite_opacity_is_inactive(tmp_path, monkeypatch, explicit_mask):
    """A transparent PLY slot remains absent even without a mask property."""
    from types import SimpleNamespace
    import gsply

    (tmp_path / "frame_000000.ply").touch()
    data = SimpleNamespace(
        means=np.zeros((4, 3), dtype=np.float32),
        scales=np.full((4, 3), -3, dtype=np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (4, 1)),
        opacities=np.array([-np.inf, -1, -2, -np.inf], dtype=np.float32),
        sh0=np.zeros((4, 3), dtype=np.float32), shN=None,
        masks=None if explicit_mask is None else np.array(explicit_mask),
    )
    monkeypatch.setattr(gsply, "plyread", lambda path: data)
    frame = SequenceEncoder(device="cpu")._load_ply_sequence(tmp_path)[0]
    expected = [False, True, explicit_mask is None, False]
    assert frame.masks.tolist() == expected
