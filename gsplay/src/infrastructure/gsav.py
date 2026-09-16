"""Isolated gscodec v3 process adapter. No codec imports enter the viewer runtime."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


class GsavError(RuntimeError):
    """Actionable import/export error safe to display in the viewer."""


def codec_python() -> Path:
    configured = os.environ.get("GSPLAY_CODEC_PYTHON")
    repository = Path(__file__).resolve().parents[3] / "gs-encoder"
    executable = (
        Path(configured)
        if configured
        else repository / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    if not executable.is_file():
        raise GsavError(
            "GSAV codec is not configured. Set GSPLAY_CODEC_PYTHON to the Python executable in the gs-encoder v3 environment."
        )
    return executable.resolve()


def validate_gsav(path: Path) -> None:
    if path.suffix.lower() != ".gsav" or not path.is_file():
        raise GsavError("Choose an existing .gsav file.")
    with path.open("rb") as stream:
        header = stream.read(128)
    if len(header) != 128 or header[:4] != b"GSAV":
        raise GsavError("Invalid or truncated GSAV header.")


def _run(operation: str, source: Path, destination: Path, **options) -> dict:
    progress = options.pop("progress", None)
    executable = codec_python()
    worker = Path(__file__).with_name("gsav_worker.py")
    # File-backed logs prevent large codec progress output consuming viewer memory.
    with tempfile.TemporaryDirectory(prefix="gsav-job-") as directory:
        result_path = Path(directory) / "result.json"
        progress_path = Path(directory) / "progress.json"
        options["progress_path"] = str(progress_path)
        with (Path(directory) / "codec.log").open("w+b") as log:
            command = [
                str(executable),
                str(worker),
                operation,
                str(source.resolve()),
                str(destination.resolve()),
                str(result_path),
                json.dumps(options),
            ]
            started = time.monotonic()
            last_message = None
            last_reported = 0.0
            with subprocess.Popen(
                command,
                stdout=log,
                stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            ) as process:
                while process.poll() is None:
                    if time.monotonic() - started > 3600:
                        process.kill()
                        process.wait()
                        raise GsavError(
                            "GSAV conversion exceeded the one-hour limit; use a shorter sequence."
                        )
                    if progress is not None:
                        try:
                            message = json.loads(progress_path.read_text(encoding="utf-8"))[
                                "message"
                            ]
                        except (OSError, ValueError, KeyError):
                            message = None
                        now = time.monotonic()
                        if message and (message != last_message or now - last_reported >= 5):
                            progress(f"{message} (encoder elapsed {int(now - started)}s)")
                            last_message = message
                            last_reported = now
                    time.sleep(0.25)
                returncode = process.returncode
            if returncode:
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 4000))
                raise GsavError("GSAV conversion failed: " + log.read().decode(errors="replace"))
        if not result_path.exists():
            raise GsavError("Codec exited without reporting a result.")
        return json.loads(result_path.read_text(encoding="utf-8"))


def decode_gsav(source: Path, destination: Path) -> dict:
    validate_gsav(source)
    return _run("decode", source, destination)


def encode_gsav(
    source: Path,
    destination: Path,
    fps: int = 30,
    device: str = "cpu",
    audio: str | None = None,
    *,
    preserve_layout: bool = False,
    geometry_source: str | None = None,
    progress=None,
) -> dict:
    if not 1 <= fps <= 240:
        raise GsavError("GSAV FPS must be between 1 and 240.")
    if destination.suffix.lower() != ".gsav":
        raise GsavError("GSAV output must end in .gsav.")
    if destination.exists():
        raise GsavError(f"Output already exists; choose a new filename: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Encode and validate before publishing; never replace the user's original.
    with tempfile.TemporaryDirectory(prefix=".gsav-export-", dir=destination.parent) as directory:
        pending = Path(directory) / "scene.gsav"
        result = _run(
            "encode",
            source,
            pending,
            fps=fps,
            device=device,
            audio=audio,
            preserve_layout=preserve_layout,
            geometry_source=geometry_source,
            progress=progress,
        )
        validate_gsav(pending)
        # Exclusive publication also catches a destination created during encoding.
        with destination.open("xb") as output:
            try:
                import shutil

                with pending.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
            except BaseException:
                output.close()
                destination.unlink(missing_ok=True)
                raise
    return result


def export_ply(source: Path, destination: Path) -> dict:
    """Write edited raw Gaussian data with the codec's PLY serializer."""
    if destination.exists():
        raise GsavError(f"Output already exists; choose a new folder: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ply-export-", dir=destination.parent) as directory:
        pending = Path(directory) / "frames"
        result = _run("export-ply", source, pending)
        # Reserve the destination exclusively; a competing export must never replace it.
        destination.mkdir(exist_ok=False)
        published = []
        try:
            for frame in pending.glob("*.ply"):
                target = destination / frame.name
                with target.open("xb") as output, frame.open("rb") as input_file:
                    published.append(target)
                    import shutil

                    shutil.copyfileobj(input_file, output)
        except BaseException:
            for target in published:
                target.unlink(missing_ok=True)
            # Do not remove files added by another process.
            if not any(destination.iterdir()):
                destination.rmdir()
            raise
    return result
