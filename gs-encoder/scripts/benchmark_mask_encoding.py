"""Repeatable DC benchmark for mask-aware encoding; stores encoded outputs and timings."""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import gsply
import numpy as np
import torch

from gscodec.common.binary_format import read_header
from gscodec.encoder.config import ChunkConfig
from gscodec.encoder.sequence_encoder import SequenceEncoder
from gscodec.encoder.static_encoder import tensor_from_gsdata


def render_comparisons(frames, paths):
    """Shared camera, active source reference, sampled frames, RGB PSNR."""
    from gsplat import rasterization

    from gscodec.decoder.sequence_decoder import SequenceDecoder

    points = torch.cat([f.means[f.masks] for f in frames])
    center = (points.amin(0) + points.amax(0)) / 2
    radius = max(float((points - center).norm(dim=1).max()), 0.1)
    view = torch.eye(4, device="cuda")[None]
    view[0, :3, 3] = -center.cuda()
    view[0, 2, 3] += 3 * radius
    K = torch.tensor([[[240.0, 0, 128], [0, 240.0, 128], [0, 0, 1]]], device="cuda")

    def render(frame):
        if not isinstance(frame.means, torch.Tensor):
            frame = tensor_from_gsdata(frame)
        frame = frame[frame.masks].to("cuda")
        return rasterization(
            means=frame.means,
            quats=frame.quats,
            scales=frame.scales.exp(),
            opacities=frame.opacities.sigmoid().reshape(-1),
            colors=(0.28209479177387814 * frame.sh0 + 0.5).clamp_min(0),
            viewmats=view,
            Ks=K,
            width=256,
            height=256,
        )[0]

    results = {}
    references = {t: render(frames[t]) for t in (0, 5, 11)}
    for path in paths:
        decoder = SequenceDecoder.from_file(path)
        samples = []
        for t, reference in references.items():
            reconstructed = render(decoder.decode_frame(t))
            mse = float((reference - reconstructed).square().mean())
            foreground = (reference.amax(-1) > 0.01) | (reconstructed.amax(-1) > 0.01)
            foreground_mse = float((reference - reconstructed)[foreground].square().mean())
            samples.append(
                {
                    "frame": t,
                    "mse": mse,
                    "psnr_db": float(-10 * np.log10(max(mse, 1e-20))),
                    "foreground_psnr_db": float(-10 * np.log10(max(foreground_mse, 1e-20))),
                    "reference_nonblack_fraction": float((reference.amax(-1) > 0.01).float().mean()),
                }
            )
        results[str(path)] = samples
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["python", "native"], default="python")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--inactive-values", choices=["stable", "noise"], default="noise")
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--render-compare", type=Path, nargs="*")
    args = parser.parse_args()
    rng = np.random.default_rng(731)
    n, count, gop = 2048, 12, 4
    if args.source:
        base = gsply.plyread(args.source)[:n]
        n = len(base.means)
    else:
        q = rng.normal(size=(n, 4)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        base = gsply.GSData(
            means=rng.normal(size=(n, 3)).astype(np.float32),
            scales=np.full((n, 3), -3, np.float32),
            quats=q,
            opacities=np.full(n, 3, np.float32),
            sh0=rng.normal(size=(n, 3)).astype(np.float32),
            shN=np.empty((n, 0, 3), np.float32),
        )
    frames = []
    for t in range(count):
        frame = tensor_from_gsdata(base)
        frame.shN = None
        frame = frame.clone()
        frame.means += t * 0.002
        active = (np.arange(n) + (t // 2) * 149) % 10 < 4
        frame.masks[:] = torch.from_numpy(active)
        if args.inactive_values == "noise":
            frame.means[~frame.masks] = torch.from_numpy(
                rng.normal(0, 4, size=((~active).sum(), 3)).astype(np.float32)
            )
        frames.append(frame)
    start = time.perf_counter()
    if args.backend == "python":
        config = ChunkConfig(size=gop, sh_bands=0, lo_snap_k=1)
        if hasattr(config, "identity_mode"):
            config.identity_mode = "stable"
            config.reuse_inactive = not args.no_reuse
        data = SequenceEncoder(device="cpu", chunk_config=config).encode_frames(frames, prune=False)
    else:
        built = Path(__file__).resolve().parents[1] / "cpp" / "built"
        sys.path.insert(0, str(built))
        _dll = os.add_dll_directory(str(built)) if os.name == "nt" else None
        import _gscodec_native as native

        arrays = [
            [np.ascontiguousarray(getattr(f, k).numpy()) for f in frames]
            for k in ("means", "scales", "quats", "opacities", "sh0")
        ]
        arrays[3] = [a.reshape(-1) for a in arrays[3]]
        data = native.encode_sequence_with_options(
            *arrays,
            None,
            gop,
            30,
            1,
            1,
            1,
            0,
            "vp09.00.51.08",
            prune=False,
            ordering_mode="stable",
            reuse_inactive=not args.no_reuse,
            presence_list=[f.masks.numpy().astype(np.uint8) for f in frames],
        )
    seconds = time.perf_counter() - start
    header = read_header(io.BytesIO(data))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.backend}.gsav").write_bytes(data)
    result = {
        "reuse_inactive": not args.no_reuse,
        "inactive_values": args.inactive_values,
        "backend": args.backend,
        "source": str(args.source),
        "gaussians": n,
        "frames": count,
        "gop": gop,
        "total_bytes": len(data),
        "video_bytes": len(data) - header["video_payload_offset"],
        "means_lo_bytes": header["video_payload_offset"] - header["means_lo_payload_offset"],
        "encode_seconds": seconds,
    }
    if args.render_compare is not None:
        result["render_comparisons"] = render_comparisons(
            frames, [*args.render_compare, args.output / f"{args.backend}.gsav"]
        )
    (args.output / f"{args.backend}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
