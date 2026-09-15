"""Test different VP9 encoding modes to find one that preserves pixel values."""

import tempfile
import subprocess
import os
import sys
from pathlib import Path

import ffmpeg as ffmpeg_lib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gscodec.encoder.sequence_encoder import SequenceEncoder
from gscodec.encoder.chunk_encoder import ChunkEncoder
from gscodec.encoder.video_writer import compute_atlas_dimensions
from gscodec.encoder.config import VideoConfig, ChunkConfig


def main():
    # Load real data
    encoder = SequenceEncoder(
        video_config=VideoConfig(fps=30, codec="libsvtav1"),
        chunk_config=ChunkConfig(size=30),
        device="cuda:0",
    )
    frames = encoder._load_ply_sequence(Path("../data/gsflow/elly/plys/"))[:30]
    n_gaussians = frames[0].means.shape[0]
    global_ranges = encoder._compute_global_ranges(frames)
    chunk_enc = ChunkEncoder(device="cuda:0", skip_sorting=False)
    results = chunk_enc.encode(frames, global_ranges)
    original_atlases = [atlas.copy() for atlas, _ in results]
    n_frames = len(original_atlases)
    atlas_side, atlas_width, atlas_height = compute_atlas_dimensions(n_gaussians)

    uv_w, uv_h = atlas_width // 2, atlas_height // 2
    u420 = np.full((uv_h, uv_w), 128, dtype=np.uint8).tobytes()
    v420 = np.full((uv_h, uv_w), 128, dtype=np.uint8).tobytes()
    u444 = np.full((atlas_height, atlas_width), 128, dtype=np.uint8).tobytes()
    v444 = u444

    configs = [
        ("VP9 lossless yuv420p", "yuv420p", {"lossless": "1", "cpu-used": "4"}),
        ("VP9 lossless yuv444p", "yuv444p", {"lossless": "1", "cpu-used": "4"}),
        ("VP9 lossy crf=0 420p", "yuv420p", {"crf": "0", "b:v": "0", "cpu-used": "4"}),
        ("VP9 lossy crf=4 420p", "yuv420p", {"crf": "4", "b:v": "0", "cpu-used": "4"}),
        ("VP9 lossy qmin=0 420p", "yuv420p", {"qmin": "0", "qmax": "0", "cpu-used": "4"}),
    ]

    for name, pix_fmt, extra in configs:
        print(f"\n--- {name} ---")
        with tempfile.TemporaryDirectory() as tmpdir:
            ivf_path = Path(tmpdir) / "test.ivf"

            opts = {
                "vcodec": "libvpx-vp9",
                "r": "30",
                "g": str(n_frames),
                "loglevel": "error",
                "color_range": "tv",
                "pix_fmt": pix_fmt,
                "f": "ivf",
            }
            opts.update(extra)

            process = (
                ffmpeg_lib.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt=pix_fmt,
                    s=f"{atlas_width}x{atlas_height}",
                    r="30",
                    color_range="tv",
                )
                .output(str(ivf_path), **opts)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

            for atlas in original_atlases:
                process.stdin.write(atlas.astype(np.uint8).tobytes())
                if pix_fmt == "yuv420p":
                    process.stdin.write(u420)
                    process.stdin.write(v420)
                else:
                    process.stdin.write(u444)
                    process.stdin.write(v444)

            process.stdin.close()
            process.wait()

            ivf_size = os.path.getsize(ivf_path)

            # Decode
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(ivf_path),
                "-vf", "extractplanes=y",
                "-f", "rawvideo", "-pix_fmt", "gray",
                "pipe:1",
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            frame_size = atlas_width * atlas_height
            total_err = 0
            total_pix = 0
            max_err = 0
            for i in range(n_frames):
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    print(f"  Short read at frame {i}")
                    break
                dec = np.frombuffer(raw, dtype=np.uint8).reshape(atlas_height, atlas_width)
                diff = np.abs(original_atlases[i].astype(np.int16) - dec.astype(np.int16))
                total_err += np.count_nonzero(diff)
                total_pix += diff.size
                max_err = max(max_err, int(diff.max()))
            proc.wait()

            pct = total_err / total_pix * 100 if total_pix else 0
            print(f"  Size: {ivf_size / 1024 / 1024:.2f} MB  Errors: {pct:.4f}%  MaxErr: {max_err}")


if __name__ == "__main__":
    main()
