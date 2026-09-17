"""UI coordination for explicit, reusable crossing analysis."""

import threading
from copy import deepcopy

from src.gsplay.crossing_crop import POLICIES, active_plan, analyze, analyze_motion


def refresh_status(app):
    ui = app.ui
    if (
        not ui
        or not getattr(ui, "crossing_status", None)
        or getattr(app, "_crossing_running", False)
    ):
        return
    if app.config.crossing_policy == POLICIES[0]:
        ui.crossing_status.content = "**Crop:** show each row only while inside."
    elif active_plan(app.model, app.config) is not None:
        ui.crossing_status.content = "**Applied:** " + active_plan(app.model, app.config).summary
    else:
        ui.crossing_status.content = (
            "**Not applied:** click Analyze & Apply. Preview uses the ordinary boundary "
            "until analysis finishes; export is blocked for this choice."
        )


def bind_controls(app, ui):
    if ui.crossing_policy is None or ui.crossing_apply is None:
        return

    @ui.crossing_policy.on_update
    def changed(_):
        app.config.crossing_policy = ui.crossing_policy.value
        refresh_status(app)
        app.render_component.rerender()

    if ui.crossing_method is not None:

        @ui.crossing_method.on_update
        def method_changed(_):
            app.config.crossing_method = ui.crossing_method.value
            refresh_status(app)
            app.render_component.rerender()

    @ui.crossing_apply.on_click
    def apply(_):
        from src.gsplay.gsav_controls import _notify

        if app.config.crossing_policy == POLICIES[0]:
            app.config.crossing_plan = None
            refresh_status(app)
            app.render_component.rerender()
            return
        if not app._scene_job_lock.acquire(blocking=False):
            _notify(app, "Scene busy", "Wait for the current import, analysis or export.")
            return
        try:
            app._update_edit_history()
            model = app.model
            values = deepcopy(app.config.filter_values)
            policy = app.config.crossing_policy
            method = app.config.crossing_method
            app.playback_controller.pause()
        except Exception as exc:
            app._scene_job_lock.release()
            _notify(app, "Crossing analysis failed", str(exc), "red")
            return
        app._crossing_running = True
        ui.crossing_apply.disabled = True
        ui.crossing_policy.disabled = True
        if ui.crossing_method is not None:
            ui.crossing_method.disabled = True

        def work():
            try:

                def progress(message):
                    ui.crossing_status.content = f"**Crop:** {message}"

                if method == "Motion-aware (preview)":
                    import tempfile

                    if not hasattr(app, "_motion_cache_directory"):
                        app._motion_cache_directory = tempfile.TemporaryDirectory(
                            prefix="gsplay-motion-"
                        )
                    plan = analyze_motion(
                        model, values, policy, app._motion_cache_directory.name, progress
                    )
                else:
                    plan = analyze(model, values, policy, app.device, progress)
                # Controls can move during analysis; never apply stale settings.
                app._update_edit_history()
                if app.model is not model or not plan.matches(model, app.config):
                    raise ValueError(
                        "Boundary changed during analysis. Click Analyze & Apply again."
                    )
                app.config.crossing_plan = plan
                app.edit_manager.invalidate_cache()
                app.render_component.rerender()
                _notify(
                    app,
                    "Crossing crop applied",
                    "Scrub or play to inspect all chunks before export.",
                    "green",
                )
            except Exception as exc:
                _notify(app, "Crossing analysis failed", str(exc), "red")
            finally:
                app._crossing_running = False
                ui.crossing_apply.disabled = False
                ui.crossing_policy.disabled = False
                if ui.crossing_method is not None:
                    ui.crossing_method.disabled = False
                app._scene_job_lock.release()
                refresh_status(app)

        threading.Thread(target=work, name="crossing-analysis", daemon=True).start()
