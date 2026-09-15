"""Export folder picker controls for the Windows desktop server."""

import logging

from src.infrastructure.folder_picker import choose_folder, export_destination


logger = logging.getLogger(__name__)


def add_folder_picker(server, controls) -> None:
    button = server.gui.add_button(
        "Save To Path…",
        hint="Open Windows File Explorer to choose where the export will be saved.",
    )

    @button.on_click
    def pick(event):
        button.disabled = True
        try:
            folder = choose_folder(controls["export_path"].value)
            if folder is not None:
                destination = export_destination(folder, controls["export_format"].value)
                controls["export_path"].value = str(destination)
        except Exception as exc:
            logger.exception("Export folder picker failed")
            if event.client:
                event.client.add_notification(title="Folder picker", body=str(exc), color="red")
        finally:
            button.disabled = False
