"""Measure means_lo compression ratios: per-frame vs per-chunk, raw vs delta."""

import zlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gscodec.encoder.sequence_encoder import SequenceEncoder
from gscodec.encoder.chunk_encoder import ChunkEncoder
from gscodec.encoder.config import VideoConfig, ChunkConfig


def compress_deflate(data: bytes, level: int = 6) -> bytes:
    return zlib.compress(data, level)[2:-4]  # strip zlib header/trailer = deflate-raw


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=30)
    args = parser.parse_args()

    encoder = SequenceEncoder(
        video_config=VideoConfig(fps=30, codec="libsvtav1"),
        chunk_config=ChunkConfig(size=args.chunk_size),
        device="cuda:0",
    )

    frames = encoder._load_ply_sequence(Path(args.input_dir))
    n_gaussians = frames[0].means.shape[0]
    global_ranges = encoder._compute_global_ranges(frames)

    chunk_size = args.chunk_size
    n_frames = len(frames)
    n_chunks = (n_frames + chunk_size - 1) // chunk_size

    print(f"{n_frames} frames, {n_gaussians} gaussians, {n_chunks} chunks of {chunk_size}")
    print(f"Per-frame raw size: {n_gaussians * 3:,} bytes")
    print(f"Total raw size: {n_gaussians * 3 * n_frames:,} bytes\n")

    # Quantize all frames
    chunk_enc = ChunkEncoder(device="cuda:0", skip_sorting=False)
    all_means_lo = []
    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, n_frames)
        results = chunk_enc.encode(frames[start:end], global_ranges)
        for _, means_lo in results:
            all_means_lo.append(means_lo)  # [N, 3] uint8

    frame_bytes = [lo.tobytes() for lo in all_means_lo]
    frame_size = len(frame_bytes[0])

    # ---- Per-frame compression ----
    pf_raw_sizes = []
    pf_deflate_sizes = []
    pf_delta_deflate_sizes = []

    prev = None
    for i, fb in enumerate(frame_bytes):
        arr = np.frombuffer(fb, dtype=np.uint8)
        pf_raw_sizes.append(len(fb))
        pf_deflate_sizes.append(len(compress_deflate(fb)))

        if prev is not None:
            delta = (arr.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)
            pf_delta_deflate_sizes.append(len(compress_deflate(delta.tobytes())))
        else:
            pf_delta_deflate_sizes.append(len(compress_deflate(fb)))  # first frame raw
        prev = arr

    # ---- Per-chunk compression ----
    pc_raw_sizes = []
    pc_deflate_sizes = []
    pc_delta_deflate_sizes = []

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, n_frames)
        chunk_frames = frame_bytes[start:end]

        raw = b"".join(chunk_frames)
        pc_raw_sizes.append(len(raw))
        pc_deflate_sizes.append(len(compress_deflate(raw)))

        # Delta within chunk
        delta_parts = [chunk_frames[0]]  # first frame raw
        for j in range(1, len(chunk_frames)):
            curr = np.frombuffer(chunk_frames[j], dtype=np.uint8)
            prev_f = np.frombuffer(chunk_frames[j - 1], dtype=np.uint8)
            delta = (curr.astype(np.int16) - prev_f.astype(np.int16)).astype(np.uint8)
            delta_parts.append(delta.tobytes())
        pc_delta_deflate_sizes.append(len(compress_deflate(b"".join(delta_parts))))

    # ---- Report ----
    total_raw = sum(pf_raw_sizes)

    print("=" * 60)
    print("PER-FRAME compression:")
    print(f"  Raw:              {total_raw:>12,} bytes")
    total_pf_deflate = sum(pf_deflate_sizes)
    print(f"  Deflate:          {total_pf_deflate:>12,} bytes  ({total_pf_deflate/total_raw*100:.1f}%)")
    total_pf_dd = sum(pf_delta_deflate_sizes)
    print(f"  Delta + Deflate:  {total_pf_dd:>12,} bytes  ({total_pf_dd/total_raw*100:.1f}%)")

    print()
    print("PER-CHUNK compression:")
    print(f"  Raw:              {total_raw:>12,} bytes")
    total_pc_deflate = sum(pc_deflate_sizes)
    print(f"  Deflate:          {total_pc_deflate:>12,} bytes  ({total_pc_deflate/total_raw*100:.1f}%)")
    total_pc_dd = sum(pc_delta_deflate_sizes)
    print(f"  Delta + Deflate:  {total_pc_dd:>12,} bytes  ({total_pc_dd/total_raw*100:.1f}%)")

    print()
    print("WHOLE-FILE compression (for reference):")
    whole_raw = b"".join(frame_bytes)
    whole_deflate = len(compress_deflate(whole_raw))
    print(f"  Deflate:          {whole_deflate:>12,} bytes  ({whole_deflate/total_raw*100:.1f}%)")

    # Delta whole
    delta_whole = [frame_bytes[0]]
    for i in range(1, len(frame_bytes)):
        curr = np.frombuffer(frame_bytes[i], dtype=np.uint8)
        prev_f = np.frombuffer(frame_bytes[i - 1], dtype=np.uint8)
        delta = (curr.astype(np.int16) - prev_f.astype(np.int16)).astype(np.uint8)
        delta_whole.append(delta.tobytes())
    whole_dd = len(compress_deflate(b"".join(delta_whole)))
    print(f"  Delta + Deflate:  {whole_dd:>12,} bytes  ({whole_dd/total_raw*100:.1f}%)")

    # Per-chunk detail
    print()
    print("Per-chunk detail (delta+deflate):")
    for ci in range(min(n_chunks, 10)):
        raw_sz = pc_raw_sizes[ci]
        dd_sz = pc_delta_deflate_sizes[ci]
        print(f"  Chunk {ci:2d}: {raw_sz:>10,} -> {dd_sz:>10,} ({dd_sz/raw_sz*100:.1f}%)")


if __name__ == "__main__":
    main()
