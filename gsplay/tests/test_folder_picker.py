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

    buttons = []

    def add_button(*a, **k):
        button = Button()
        buttons.append(button)
        return button

    server = SimpleNamespace(
        gui=SimpleNamespace(
            add_button=add_button, add_markdown=lambda *a: SimpleNamespace(content="")
        )
    )
    controls = {
        "export_path": SimpleNamespace(value="original"),
        "export_format": SimpleNamespace(value="GSAV"),
    }
    monkeypatch.setattr(
        "src.gsplay.folder_picker_controls.choose_folder",
        lambda _, **kwargs: None if cancelled else tmp_path,
    )
    add_folder_picker(server, controls)
    button = buttons[0]
    button.callback(SimpleNamespace(client=None))
    assert controls["export_path"].value == (
        "original" if cancelled else str(tmp_path / "scene.gsav")
    )
    assert not button.disabled
    assert not buttons[1].visible


def test_native_picker_cancel_terminates_helper(monkeypatch, tmp_path):
    import threading

    from src.infrastructure import folder_picker

    if folder_picker.os.name != "nt":
        pytest.skip("Windows native picker")

    class Process:
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout):
            assert self.stopped

    process = Process()
    monkeypatch.setattr(folder_picker.subprocess, "Popen", lambda *a, **k: process)
    cancelled = threading.Event()
    cancelled.set()
    assert folder_picker.choose_folder(str(tmp_path), cancel=cancelled) is None
    assert process.stopped
