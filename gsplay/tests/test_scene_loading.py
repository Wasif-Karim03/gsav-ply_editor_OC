"""Scene replacement contract in a clean interpreter (conftest stubs viser)."""

import subprocess
import sys
from pathlib import Path


def test_empty_start_and_scene_replacement():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import threading
from types import SimpleNamespace
from unittest.mock import Mock
from src.gsplay.core.main import ViewCmd
from src.gsplay.core.app import UniversalGSPlay
assert ViewCmd().config is None
app = UniversalGSPlay.__new__(UniversalGSPlay)
model = Mock()
model.get_total_frames.return_value = 30
app.model_component = Mock()
app.model_component.get_model.return_value = model
app.playback_controller = Mock()
app.scene_bounds_manager = Mock()
app.camera_controller = Mock()
app.ui = SimpleNamespace(time_slider=Mock())
app.config = Mock()
app._apply_edits_fn = Mock()
renderer = SimpleNamespace(_cached_frame=object())
viewer = SimpleNamespace(lock=threading.Lock(), _renderer=renderer, render_fn=object())
app.render_component = Mock()
app.render_component.get_viewer.return_value = viewer
app._on_model_loaded(None)
assert viewer.render_fn is app.render_component.create_render_function.return_value
assert renderer._cached_frame is None
assert viewer.total_frames == 30
app.camera_controller.focus_on_bounds.assert_called_once()
app.render_component.rerender.assert_called_once()
from unittest.mock import patch
from src.gsplay.rendering.renderer import create_render_function
with patch('src.gsplay.rendering.renderer.encode_and_cache') as publish:
    empty_render = create_render_function(None, None, 'cpu', lambda x: x)
    image = empty_render(None, SimpleNamespace(viewer_width=16, viewer_height=8, jpeg_quality=90))
    assert image.shape == (8, 16, 3) and not image.any()
    publish.assert_called_once()
""",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")
