from types import SimpleNamespace

import pytest

from src.infrastructure.folder_picker import export_destination


@pytest.mark.parametrize("format_name, name", [("GSAV", "scene.gsav"), ("PLY", "gsplay-export")])
def test_destination_is_fresh_and_inside_selected_folder(tmp_path, format_name, name):
    destination = export_destination(tmp_path, format_name)
    assert destination == tmp_path / name
    destination.write_bytes(b"existing")
    next_path = export_destination(tmp_path, format_name)
    assert next_path.parent == tmp_path and next_path != destination
    assert destination.read_bytes() == b"existing"
    assert not next_path.exists()


@pytest.mark.parametrize("cancelled", [True, False])
def test_picker_updates_path_only_when_selected(monkeypatch, tmp_path, cancelled):
    from src.gsplay.folder_picker_controls import add_folder_picker

    class Button:
        disabled = False

        def on_click(self, callback):
            self.callback = callback

    button = Button()
    server = SimpleNamespace(gui=SimpleNamespace(add_button=lambda *a, **k: button))
    controls = {
        "export_path": SimpleNamespace(value="original"),
        "export_format": SimpleNamespace(value="GSAV"),
    }
    monkeypatch.setattr(
        "src.gsplay.folder_picker_controls.choose_folder", lambda _: None if cancelled else tmp_path
    )
    add_folder_picker(server, controls)
    button.callback(SimpleNamespace(client=None))
    assert controls["export_path"].value == (
        "original" if cancelled else str(tmp_path / "scene.gsav")
    )
    assert not button.disabled
