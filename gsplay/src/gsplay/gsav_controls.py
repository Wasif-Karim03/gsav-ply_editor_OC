"""GSAV import/export orchestration around the existing editor and PLY writer."""

from __future__ import annotations

import logging
import math
import tempfile
import threading
from pathlib import Path

from src.infrastructure.gsav import GsavError, encode_gsav, export_ply


logger = logging.getLogger(__name__)


def _notify(app, title: str, body: str, color: str = "blue") -> None:
    for client in app.server.get_clients().values():
        client.add_notification(title=title, body=body, color=color)


def add_gsav_upload(server, app, path_input) -> None:
    app._scene_status = server.gui.add_markdown(
        "**No scene loaded.** Upload a GSAV file or choose a data path."
        if app.model is None
        else f"**Loaded:** {app.model.get_total_frames()} frames"
    )
    app._export_status = server.gui.add_markdown("**Export:** idle")
    upload = server.gui.add_upload_button(
        "Upload GSAV",
        mime_type=".gsav",
        hint="Open GSAV directly as editable Gaussian frames (up to 1 GiB).",
    )

    @upload.on_upload
    def _uploaded(event) -> None:
        upload.disabled = True
        directory = None
        try:
            value = event.target.value
            if not value.name.lower().endswith(".gsav") or value.content[:4] != b"GSAV":
                raise GsavError("Choose a valid .gsav file.")
            if len(value.content) > 1024**3:
                raise GsavError("Upload limit is 1 GiB. Use Data Path for larger local files.")
            directory = tempfile.TemporaryDirectory(prefix="gsplay-upload-")
            # Never interpret a browser-provided filename as a filesystem path.
            source = Path(directory.name) / "uploaded.gsav"
            source.write_bytes(value.content)
            _notify(app, "Opening GSAV", "Reading the timeline and decoding the first frame…")
            if app._handle_load_data(
                str(source), display_name=Path(value.name.replace("\\", "/")).name
            ):
                app.model_component._gsav_directories.append(directory)
                directory = None
                path_input.value = str(source)
                app._set_export_path(Path.cwd() / "export_with_edits")
                meta = app.model_component.gsav_metadata
                _notify(
                    app,
                    "GSAV ready",
                    f"Loaded {meta['frames']} frames at {meta['fps']} FPS with SH degree {meta['sh_bands']}.",
                    "green",
                )
                if meta.get("warning"):
                    _notify(app, "GSAV source detail", meta["warning"], "yellow")
        except Exception as exc:
            logger.exception("GSAV upload failed")
            _notify(app, "GSAV import failed", str(exc), "red")
        finally:
            if directory is not None:
                directory.cleanup()
            upload.disabled = False


def export_times(
    model,
    scope: str,
    current: float,
    start: float | None = None,
    end: float | None = None,
    step: float | None = None,
    snap: bool = False,
) -> list[float]:
    """Resolve a bounded, ordered selection in the model's native time units."""
    domain = model.time_domain
    if scope == "Snapshot at Current Time":
        return [current]
    if scope != "Custom Time Range":
        return [
            domain.from_normalized(model.get_frame_time(i)) for i in range(model.get_total_frames())
        ]
    if (
        start is None
        or end is None
        or step is None
        or not all(map(math.isfinite, (start, end, step)))
        or step <= 0
    ):
        raise GsavError("Choose a finite time range and positive time step.")
    start, end = sorted((start, end))
    if start < domain.min_time or end > domain.max_time:
        raise GsavError("Export range lies outside the scene timeline.")
    count = int(math.floor((end - start) / step + 1e-8)) + 1
    if count > 10000:
        raise GsavError("Export is limited to 10,000 frames.")
    times = [start + i * step for i in range(count)]
    if snap and domain.keyframe_times is not None:
        times = list(dict.fromkeys(domain.source_time_to_nearest_keyframe(t)[1] for t in times))
    return times


def write_edited_sequence(
    model,
    times,
    edit_applier,
    destination: Path,
    *,
    fps: int,
    device: str,
    audio: str | None = None,
    progress=None,
    output_format: str = "GSAV",
    status=None,
    fast_export: bool = True,
) -> dict:
    """Use the same PLY export normalization as ordinary PLY export, including SHN."""
    from src.domain.data import GaussianData
    from src.infrastructure.exporters.ply_exporter import PlyExporter

    if not times:
        raise GsavError("There are no frames to export.")
    if destination.exists():
        raise GsavError(f"Output already exists; choose a new filename: {destination}")
    import json

    import numpy as np

    from src.gsplay.gsav_export_layout import SourceLayout

    layout = SourceLayout(model, times, fps, enabled=fast_export and output_format == "GSAV")
    writer = PlyExporter()
    with tempfile.TemporaryDirectory(prefix="gsplay-edited-") as directory:
        for index, time in enumerate(times):
            data = model.get_frame_at_source_time(time)
            if data is None:
                raise GsavError(f"Frame at time {time} could not be loaded; export aborted.")
            if isinstance(data, GaussianData):
                data = data.to_gstensor(device)
                # gsply's packed backing buffer can become stale when gsmod replaces
                # arrays during edits. Make field arrays authoritative at this boundary.
                data._base = None
            reference = layout.before(data)
            edited = edit_applier(data)
            presence = layout.after(index, reference, edited)
            if presence is not None:
                np.save(Path(directory) / f"presence_{index:06d}.npy", presence, allow_pickle=False)
            writer.export_frame(edited, Path(directory) / f"frame_{index:06d}.ply")
            if progress:
                progress(index + 1, len(times))
            if status and (index == 0 or (index + 1) % 10 == 0 or index + 1 == len(times)):
                status(f"Preparing edited frames: {index + 1}/{len(times)}")
        if output_format == "PLY":
            return export_ply(Path(directory), destination)
        if output_format != "GSAV":
            raise GsavError(f"Unsupported export format: {output_format}")
        if layout.valid:
            (Path(directory) / "source-layout.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "frames": len(times),
                        "rows": layout.count,
                        "chunk_size": layout.chunk_size,
                    }
                ),
                encoding="utf-8",
            )
        if status:
            status(layout.reason)
        logger.info("GSAV export path: %s", layout.reason)
        return encode_gsav(
            Path(directory),
            destination,
            fps=fps,
            device=device,
            audio=audio,
            preserve_layout=layout.valid,
            progress=status,
        )


def write_edited_gsav(*args, **kwargs) -> dict:
    """Compatibility entry point for callers requesting GSAV."""
    return write_edited_sequence(*args, **kwargs)


def export_sequence(app) -> None:
    """Start one export with frozen edit settings; report completion or failure in UI."""
    from copy import deepcopy

    from src.gsplay.core.container import create_edit_manager

    if getattr(app, "_gsav_export_running", False):
        _notify(app, "Export in progress", "Wait for the current export to finish.")
        return
    if app.model is None:
        _notify(app, "Export failed", "Load a scene first.", "red")
        return
    if not app._scene_job_lock.acquire(blocking=False):
        _notify(app, "Scene operation in progress", "Wait for the current import or export.")
        return
    try:
        ui = app.ui
        output_format = ui.export_format.value
        model = app.model
        scope = ui.export_scope_dropdown.value if ui.export_scope_dropdown else "Original Frames"
        times = export_times(
            model,
            scope,
            app.playback_controller.current_source_time,
            ui.export_start_time_slider.value if ui.export_start_time_slider else None,
            ui.export_end_time_slider.value if ui.export_end_time_slider else None,
            ui.export_time_step_slider.value if ui.export_time_step_slider else None,
            bool(ui.export_snap_to_keyframe and ui.export_snap_to_keyframe.value),
        )
        path = ui.export_path.value.strip()
        if not path or "://" in path:
            raise GsavError("Choose a new local output folder (PLY) or .gsav filename (GSAV).")
        destination = Path(path).resolve()
        if output_format == "PLY" and destination.suffix.lower() == ".gsav":
            raise GsavError(
                "Choose a new output folder for the PLY sequence, not a .gsav filename."
            )
        if output_format == "GSAV" and destination.suffix.lower() != ".gsav":
            destination /= "scene.gsav"
        device = "cuda:0" if ui.export_device.value == "GPU" else "cpu"
        configured_fps = ui.source_fps_input.value if ui.source_fps_input else 0
        source_fps = model.time_domain.source_fps or app.config.animation.play_speed_fps
        fps_value = configured_fps or source_fps
        if output_format == "GSAV" and (
            not math.isfinite(fps_value) or fps_value != int(fps_value)
        ):
            raise GsavError("This GSAV codec requires an integer Source FPS.")
        fps = int(fps_value) if output_format == "GSAV" else 30
        app._update_edit_history()
        manager = create_edit_manager(deepcopy(app.config), device)
        bounds = deepcopy(app.scene_bounds_manager.get_bounds())
        metadata = dict(app.model_component.gsav_metadata)
        audio = metadata.get("audio")
        if (
            output_format == "GSAV"
            and audio
            and (scope != "Original Frames" or fps != metadata.get("fps"))
        ):
            raise GsavError(
                "To retain audio, export Original Frames at the imported FPS. Partial/audio-retimed GSAV export is not supported yet."
            )
    except Exception as exc:
        app._scene_job_lock.release()
        _notify(app, "Export failed", str(exc), "red")
        return

    app._gsav_export_running = True
    ui.export_ply_button.disabled = True
    app.playback_controller.pause()
    _notify(
        app,
        f"Exporting {output_format}",
        f"Applying edits to {len(times)} frames, then writing {output_format}…",
    )

    def report_status(message):
        if getattr(app, "_export_status", None) is not None:
            app._export_status.content = f"**Export:** {message}"

    def work():
        try:
            report_status("Preparing edited frames")
            result = write_edited_sequence(
                model,
                times,
                lambda data: manager.apply_edits(data, scene_bounds=bounds),
                destination,
                fps=fps,
                device=device,
                audio=audio,
                output_format=output_format,
                status=report_status,
            )
            _notify(
                app,
                f"{output_format} exported",
                f"Saved {result['frames']} frames: {destination}",
                "green",
            )
            report_status(f"Complete - saved {result['frames']} frames")
            logger.info("%s exported to %s", output_format, destination)
        except Exception as exc:
            report_status("Failed - see the error notification")
            logger.exception("Export failed")
            _notify(app, "Export failed", str(exc), "red")
        finally:
            app._gsav_export_running = False
            ui.export_ply_button.disabled = False
            app._scene_job_lock.release()

    threading.Thread(target=work, name="gsav-export", daemon=True).start()
