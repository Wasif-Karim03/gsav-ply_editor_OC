"""Apply keeps manual controls; only the explicit preset button replaces them."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.gsplay.config.settings import GSPlayConfig
from src.gsplay.config.ui_handles import UIHandles
from src.gsplay.core.app import UniversalGSPlay
from src.gsplay.initialization.ui_setup import UISetup


def manual_app(brightness):
    app = UniversalGSPlay.__new__(UniversalGSPlay)
    app.model_component = SimpleNamespace(get_model=lambda: object())
    app.config = GSPlayConfig()
    app.ui = UIHandles(
        brightness_slider=SimpleNamespace(value=brightness),
        color_adjustment_dropdown=SimpleNamespace(value="Auto Enhance"),
        apply_adjustment_button=Mock(),
        apply_preset_button=Mock(),
    )
    app._get_camera_state = lambda: (None, None)
    app.render_component = SimpleNamespace(rerender=Mock())
    app._apply_color_adjustment = Mock(side_effect=AssertionError("Preset called by Apply"))
    return app


@pytest.mark.parametrize("brightness", [0.7, 1.0, 5.0])
def test_apply_click_keeps_manual_brightness_and_shared_export_config(brightness):
    app = manual_app(brightness)
    UISetup(app).setup_auto_learn_color()
    click = app.ui.apply_adjustment_button.on_click.call_args.args[0]
    for _ in range(2):
        click(None)
        assert app.ui.brightness_slider.value == brightness
        assert app.config.color_values.brightness == brightness
        assert app.config.edits_active == (brightness != 1.0)
    app._apply_color_adjustment.assert_not_called()
    assert app.render_component.rerender.call_count == 2
    # Export synchronizes the controls once more before taking its config snapshot.
    app._update_edit_history()
    assert app.config.color_values.brightness == brightness


def test_preset_requires_its_own_button():
    app = manual_app(5.0)
    app._apply_color_adjustment = Mock()
    UISetup(app).setup_auto_learn_color()
    app.ui.apply_preset_button.on_click.call_args.args[0](None)
    app._apply_color_adjustment.assert_called_once_with()
    app.render_component.rerender.assert_not_called()
