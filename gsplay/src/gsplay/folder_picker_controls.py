"""Export folder picker controls for the Windows desktop server."""

import logging
import threading

from src.infrastructure.folder_picker import choose_folder, export_destination


logger = logging.getLogger(__name__)


def add_folder_picker(server, controls) -> None:
    button = server.gui.add_button(
        "Save To Path…",
        hint="Open Windows File Explorer to choose where the export will be saved.",
    )
    cancel_button = server.gui.add_button("Cancel Folder Selection", visible=False)
    status = server.gui.add_markdown("")
    cancelled = threading.Event()

    @cancel_button.on_click
    def cancel_pick(_event):
        cancelled.set()

    @button.on_click
    def pick(event):
        if button.disabled:
            return
        button.disabled = True
        cancelled.clear()
        cancel_button.visible = True
        status.content = "Choose a folder in the Windows dialog, or cancel here to type a path."
        try:
            folder = choose_folder(controls["export_path"].value, cancel=cancelled)
            if folder is not None:
                destination = export_destination(folder, controls["export_format"].value)
                controls["export_path"].value = str(destination)
        except Exception as exc:
            logger.exception("Export folder picker failed")
            if event.client:
                event.client.add_notification(title="Folder picker", body=str(exc), color="red")
        finally:
            button.disabled = False
            cancel_button.visible = False
            status.content = ""
