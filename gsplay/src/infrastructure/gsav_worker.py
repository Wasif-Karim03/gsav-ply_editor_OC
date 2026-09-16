"""Executed by the separate gscodec v3 interpreter, never imported by GSPlay."""

import json
import sys
from pathlib import Path


def stored_sh_degree(header: dict) -> int:
    """Legacy SH0 files can advertise degree 3 without storing any SH sidecar."""
    bands = header.get("sh_bands", 0)
    if (
        header.get("version") == 1
        and not header.get("flags", 0) & 2
        and not header.get("sh_payload_offset", 0)
    ):
        return 0
    return bands


def convert(operation: str, source: Path, destination: Path, options: dict) -> dict:
    import gsply
    from gscodec.decoder import SequenceDecoder

    if operation == "decode":
        decoder = SequenceDecoder.from_file(source)
        if not 0 < len(decoder) <= 10000 or not 0 < decoder.n_gaussians <= 5_000_000:
            raise ValueError("Scene exceeds supported frame/Gaussian limits.")
        # Decoder currently materializes the sequence; bound its estimated working set.
        if len(decoder) * decoder.n_gaussians * 256 > 8 * 1024**3:
            raise ValueError("Scene exceeds the 8 GiB decode budget; split it into smaller clips.")
        if decoder.get_static_asset():
            raise ValueError("Embedded static GSST tracks are not supported by PLY editing yet.")
        destination.mkdir(parents=True, exist_ok=False)
        bands = stored_sh_degree(decoder._provider.header)
        expected = {0: 0, 1: 3, 2: 8, 3: 15}[bands]
        for index, frame in enumerate(decoder):
            if expected and (frame.shN is None or frame.shN.shape[1:] != (expected, 3)):
                raise ValueError("SH payload did not reconstruct the declared SH degree.")
            gsply.plywrite(destination / f"frame_{index:06d}.ply", frame)
        audio = decoder.get_audio() if decoder.has_audio else None
        if audio:
            (destination / "audio.ogg").write_bytes(audio)
        return {
            "frames": len(decoder),
            "fps": decoder.fps,
            "sh_bands": bands,
            "audio": str(destination / "audio.ogg") if audio else None,
            "warning": (
                "This legacy file contains no SH sidecar; only SH0 is available."
                if bands != decoder._provider.sh_bands
                else None
            ),
        }

    if operation == "export-ply":
        files = sorted(source.glob("*.ply"))
        if not files:
            raise ValueError("No edited PLY frames to export.")
        destination.mkdir(parents=True, exist_ok=False)
        for path in files:
            # Same raw Gaussian representation / writer used by codec decompression.
            gsply.plywrite(destination / path.name, gsply.plyread(path))
        return {"frames": len(files), "format": "ply"}

    if operation != "encode":
        raise ValueError("Unknown conversion operation")
    from gscodec.encoder import ChunkConfig, SequenceEncoder, VideoConfig
    from gscodec.encoder.video_writer import require_native_encoder

    require_native_encoder()

    files = sorted(source.glob("*.ply"))
    if not files:
        raise ValueError("No edited PLY frames to encode.")
    from gscodec.encoder.sh_compress import compute_sh_bands

    bands = compute_sh_bands(gsply.plyread(files[0]).shN)
    if any(compute_sh_bands(gsply.plyread(path).shN) != bands for path in files[1:]):
        raise ValueError("All exported frames must have the same SH degree.")

    def progress(message):
        if options.get("progress_path"):
            # Windows readers can prevent atomic replacement. A progress update
            # is advisory; readers tolerate partial JSON and retry on next poll.
            try:
                Path(options["progress_path"]).write_text(
                    json.dumps({"message": message}), encoding="utf-8"
                )
            except OSError:
                pass

    preserve = bool(options.get("preserve_layout", False))
    chunk_size = 30
    if preserve:
        import numpy as np

        layout = json.loads((source / "source-layout.json").read_text(encoding="utf-8"))
        if (
            layout.get("version") != 1
            or layout.get("frames") != len(files)
            or not isinstance(layout.get("rows"), int)
            or layout["rows"] <= 0
            or not isinstance(layout.get("chunk_size"), int)
            or layout["chunk_size"] <= 0
        ):
            raise ValueError("Invalid source layout manifest")
        chunk_size = layout["chunk_size"]
        for index, path in enumerate(files):
            frame = gsply.plyread(path)
            presence = np.load(source / f"presence_{index:06d}.npy", allow_pickle=False)
            if (
                len(frame.means) != layout["rows"]
                or presence.dtype != np.bool_
                or presence.shape != (layout["rows"],)
                or not np.array_equal(presence, ~np.isneginf(frame.opacities.reshape(-1)))
            ):
                raise ValueError("Staged frame does not preserve source rows/presence")
        progress("Fast export: retaining original GSAV arrangement")
    else:
        progress("Standard export: temporal matching enabled")
    encoder = SequenceEncoder(
        video_config=VideoConfig(fps=options["fps"]),
        chunk_config=ChunkConfig(
            sh_bands=bands,
            identity_mode="preserve" if preserve else "unstructured",
            size=chunk_size,
            lo_snap_k=1,
            keyframe_snap=False,
            lossy_pruning=False,
        ),
        device=options["device"],
        progress=progress,
    )
    geometry_source = options.get("geometry_source")
    if geometry_source and not preserve:
        raise ValueError("Source geometry reuse requires verified preserved rows")
    encoder.compress(
        input_dir=source,
        output=destination,
        audio_path=options.get("audio"),
        geometry_source=geometry_source,
    )
    progress("Validating exported container")
    decoded = SequenceDecoder.from_file(destination)
    if len(decoded) != len(files) or decoded._provider.sh_bands != bands:
        raise ValueError("Encoded frame count or SH degree does not match the edited sequence.")
    return {"frames": len(files), "fps": options["fps"], "sh_bands": bands}


if __name__ == "__main__":
    mode, input_path, output_path, report_path, settings = sys.argv[1:]
    result = convert(mode, Path(input_path), Path(output_path), json.loads(settings))
    Path(report_path).write_text(json.dumps(result), encoding="utf-8")
