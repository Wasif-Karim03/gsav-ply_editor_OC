from types import SimpleNamespace
from unittest.mock import Mock

from src.gsplay.core.app import UniversalGSPlay


def test_filter_reset_refreshes_application_visualizer():
    app = UniversalGSPlay.__new__(UniversalGSPlay)
    app.ui = None
    app.config = SimpleNamespace(volume_filter=None)
    viewer = SimpleNamespace(rerender=Mock())
    app.render_component = SimpleNamespace(get_viewer=lambda: viewer)
    app._update_filter_visualization = Mock()
    app._handle_filter_reset()
    app._update_filter_visualization.assert_called_once_with()
    app.viewer.rerender.assert_called_once_with(None)
    assert app.config.volume_filter is not None
