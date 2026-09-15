"""Verify lossless round-trip for VP9 and AV1 codecs using real Gaussian data.

Loads real PLY frames, quantizes them through the actual encoder pipeline,
encodes to IVF with both VP9 and AV1, decodes back, and compares pixel-by-pixel
against the pre-encoding quantized atlas.
"""

import subprocess
import struct
import tempfile
import threading
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Add the src directory to path so we can import gscodec
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gscodec.encoder.sequence_encoder import SequenceEncoder
from gscodec.encoder.chunk_encoder import ChunkEncoder
from gscodec.encoder.video_writer import encode_to_ivf, compute_atlas_dimensions
from gscodec.encoder.config import VideoConfig, ChunkConfig

logging.basicConfig(level=logging.WARNING)


def decode_ivf_to_y_planes(
    ivf_path: str,
    width: int,
    height: int,
    n_frames: int,
    color_range: str = "tv",
) -> list[np.ndarray]:
    """Decode IVF file back to Y-plane pixels via ffmpeg."""
    frame_size = width * height
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", ivf_path,
        "-vf", "extractplanes=y",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "pipe:1",
    ]

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    decoded = []
    for _ in range(n_frames):
        raw = proc.stdout.read(frame_size)
        if len(raw) != frame_size:
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Short read: got {len(raw)}/{frame_size}. stderr: {stderr}")
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        decoded.append(frame.copy())

    proc.wait()
    return decoded


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify codec round-trip with real data")
    parser.add_argument("--input-dir", required=True, help="Directory containing PLY files")
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames to test")
    parser.add_argument("--chunk-size", type=int, default=30, help="Chunk size")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Use the actual encoder pipeline to load and quantize
    encoder = SequenceEncoder(
        video_config=VideoConfig(fps=30, codec="libsvtav1"),
        chunk_config=ChunkConfig(size=args.chunk_size),
        device="cuda:0",
    )

    print(f"Loading PLY files from {input_dir}...")
    frames = encoder._load_ply_sequence(input_dir)
    n_frames = min(len(frames), args.max_frames)
    frames = frames[:n_frames]
    n_gaussians = frames[0].means.shape[0]
    print(f"Loaded {n_frames} frames with {n_gaussians} gaussians")

    # Compute global ranges
    print("Computing global ranges...")
    global_ranges = encoder._compute_global_ranges(frames)

    # Quantize through chunk encoder
    print("Quantizing frames...")
    chunk_enc = ChunkEncoder(device="cuda:0", skip_sorting=False)
    results = chunk_enc.encode(frames, global_ranges)

    # Collect pre-encoding atlases
    original_atlases = [atlas.copy() for atlas, _ in results]

    atlas_side, atlas_width, atlas_height = compute_atlas_dimensions(n_gaussians)
    print(f"Atlas: {atlas_width}x{atlas_height} (side={atlas_side})")

    # Test configurations
    configs = [
        ("libsvtav1", "tv", {"preset": "6", "svtav1-params": "lossless=1"}),
        ("libvpx-vp9", "tv", {"lossless": "1", "row-mt": "1", "cpu-used": "4"}),
        ("libvpx-vp9", "pc", {"lossless": "1", "row-mt": "1", "cpu-used": "4"}),
    ]

    for codec, cr, extra_opts in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {codec}  color_range={cr}")
        print(f"{'='*60}")

        # Encode via the same encode_to_ivf used by the real encoder,
        # but we also need the IVF file on disk for decoding.
        # Rebuild IVF manually using ffmpeg-python like encode_to_ivf does.
        import ffmpeg as ffmpeg_lib

        with tempfile.TemporaryDirectory() as tmpdir:
            ivf_path = Path(tmpdir) / "test.ivf"

            ffmpeg_opts = {
                "vcodec": codec,
                "r": "30",
                "g": str(n_frames),
                "loglevel": "error",
                "color_range": cr,
                "pix_fmt": "yuv420p",
                "f": "ivf",
            }
            ffmpeg_opts.update(extra_opts)

            process = (
                ffmpeg_lib.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="yuv420p",
                    s=f"{atlas_width}x{atlas_height}",
                    r="30",
                    color_range=cr,
                )
                .output(str(ivf_path), **ffmpeg_opts)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

            uv_w, uv_h = atlas_width // 2, atlas_height // 2
            u_plane = np.full((uv_h, uv_w), 128, dtype=np.uint8).tobytes()
            v_plane = np.full((uv_h, uv_w), 128, dtype=np.uint8).tobytes()

            for atlas in original_atlases:
                process.stdin.write(atlas.astype(np.uint8).tobytes())
                process.stdin.write(u_plane)
                process.stdin.write(v_plane)

            process.stdin.close()
            process.wait()

            ivf_size = os.path.getsize(ivf_path)
            print(f"IVF file size: {ivf_size:,} bytes ({ivf_size/1024/1024:.2f} MB)")

            # Decode back
            decoded = decode_ivf_to_y_planes(
                str(ivf_path), atlas_width, atlas_height, n_frames, cr
            )

        # Compare
        total_pixels = 0
        total_errors = 0
        max_error = 0
        error_hist = np.zeros(256, dtype=np.int64)

        for i, (orig, dec) in enumerate(zip(original_atlases, decoded)):
            diff = np.abs(orig.astype(np.int16) - dec.astype(np.int16))
            n_err = np.count_nonzero(diff)
            frame_max = int(diff.max())
            total_pixels += orig.size
            total_errors += n_err
            max_error = max(max_error, frame_max)

            for v in range(min(frame_max + 1, 256)):
                error_hist[v] += np.count_nonzero(diff == v)

            if n_err > 0 and i < 5:
                print(f"  Frame {i}: {n_err}/{orig.size} errors ({n_err/orig.size*100:.2f}%), max={frame_max}")

        pct = total_errors / total_pixels * 100
        print(f"\nTOTAL: {total_errors:,}/{total_pixels:,} errors ({pct:.4f}%), max_error={max_error}")

        if max_error > 0:
            print("Error histogram:")
            for v in range(1, min(max_error + 1, 20)):
                cnt = error_hist[v]
                if cnt > 0:
                    print(f"  error={v}: {cnt:,} pixels ({cnt/total_pixels*100:.3f}%)")


if __name__ == "__main__":
    main()
