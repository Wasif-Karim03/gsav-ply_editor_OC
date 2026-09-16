"""Windows native folder picker for the local desktop viewer."""

import base64
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path


def choose_folder(initial_path: str, cancel: threading.Event | None = None) -> Path | None:
    """Show the Windows folder dialog in an isolated STA process; None on cancel."""
    if os.name != "nt":
        raise RuntimeError("The native folder picker is currently available on Windows only.")
    initial = Path(initial_path)
    if not initial.is_dir():
        initial = initial.parent
    if not initial.is_dir():
        initial = Path.home()
    script = """
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Choose the folder for your GSPlay export'
$dialog.ShowNewFolderButton = $true
$dialog.SelectedPath = $env:GSPLAY_INITIAL_FOLDER
$owner = New-Object System.Windows.Forms.Form
$owner.Text = 'GSPlay export folder'
$owner.TopMost = $true
$owner.StartPosition = 'CenterScreen'
$owner.Width = 420
$owner.Height = 100
$owner.ShowInTaskbar = $true
try {
    $owner.Show()
    $owner.Activate()
    [System.Windows.Forms.Application]::DoEvents()
    $selected = $null
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
        $selected = $dialog.SelectedPath
    }
    @{folder=$selected} | ConvertTo-Json -Compress | Set-Content -LiteralPath $env:GSPLAY_PICKER_RESULT -Encoding UTF8
} finally {
    $dialog.Dispose()
    $owner.Dispose()
}
"""
    with tempfile.TemporaryDirectory(prefix="gsplay-picker-") as directory:
        result = Path(directory) / "result.json"
        environment = os.environ.copy()
        environment["GSPLAY_INITIAL_FOLDER"] = str(initial.resolve())
        environment["GSPLAY_PICKER_RESULT"] = str(result)
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-STA",
                "-EncodedCommand",
                base64.b64encode(script.encode("utf-16-le")).decode("ascii"),
            ],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            deadline = time.monotonic() + 300
            while process.poll() is None:
                if cancel is not None and cancel.wait(0.1):
                    return None
                if cancel is None:
                    time.sleep(0.1)
                if time.monotonic() >= deadline:
                    raise RuntimeError("Folder selection timed out. Try again or type a path.")
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
        if process.returncode or not result.is_file():
            raise RuntimeError(
                "Windows could not open the folder picker. You can still type a path."
            )
        folder = json.loads(result.read_text(encoding="utf-8-sig"))["folder"]
        return Path(folder) if folder else None


def export_destination(folder: Path, export_format: str) -> Path:
    """Select a fresh file/subfolder inside the chosen folder, without overwriting."""
    if not folder.is_dir():
        raise ValueError("The selected folder is no longer available.")
    suffix = ".gsav" if export_format == "GSAV" else ""
    stem = "scene" if suffix else "gsplay-export"
    index = 0
    while True:
        name = f"{stem}{'-' + str(index) if index else ''}{suffix}"
        candidate = folder / name
        if not candidate.exists():
            return candidate
        index += 1
