"""Persistent isolated codec process; no intermediate PLY files or pickle IPC."""

import atexit
import io
import json
import os
import queue
import struct
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

from src.infrastructure.gsav import GsavError, codec_python, validate_gsav


class GsavStream:
    def __init__(self, source: Path, timeout: float = 120):
        validate_gsav(source)
        executable = codec_python()
        self.timeout = timeout
        self._lock = threading.Lock()
        self._directory = tempfile.TemporaryDirectory(prefix="gsplay-gsav-stream-")
        self._log = (Path(self._directory.name) / "codec.log").open("w+b")
        self._closed = False
        self._responses = queue.Queue(maxsize=1)
        self._process = None
        try:
            self._process = subprocess.Popen(
                [
                    str(executable),
                    str(Path(__file__).with_name("gsav_stream_worker.py")),
                    str(source.resolve()),
                    self._directory.name,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            threading.Thread(target=self._read_loop, daemon=True, name="gsav-pipe").start()
            self.metadata = json.loads(self._receive())
            atexit.register(self.close)
        except BaseException:
            self.close()
            raise

    def _read_exact(self, size):
        result = bytearray()
        while len(result) < size:
            part = self._process.stdout.read(size - len(result))
            if not part:
                raise EOFError("Codec process closed its output.")
            result.extend(part)
        return bytes(result)

    def _read_loop(self):
        try:
            while not self._closed:
                size = struct.unpack("<Q", self._read_exact(8))[0]
                if size > 2 * 1024**3:
                    raise ValueError("Decoded frame exceeds the 2 GiB IPC limit.")
                self._responses.put(self._read_exact(size))
        except Exception as exc:
            if not self._closed:
                self._responses.put(exc)

    def _receive(self):
        try:
            response = self._responses.get(timeout=self.timeout)
        except queue.Empty as exc:
            self.close()
            raise GsavError("GSAV frame decoding timed out.") from exc
        if isinstance(response, Exception):
            self._log.seek(0, 2)
            self._log.seek(max(0, self._log.tell() - 3000))
            detail = self._log.read().decode(errors="replace")
            self.close()
            raise GsavError(f"GSAV frame decoding failed: {detail or response}")
        return response

    def frame(self, index: int) -> dict:
        with self._lock:
            if self._closed:
                raise GsavError("The GSAV source has been closed.")
            if not 0 <= index < self.metadata["frames"]:
                raise GsavError("Frame index outside the GSAV timeline.")
            try:
                self._process.stdin.write((json.dumps({"frame": index}) + "\n").encode())
                self._process.stdin.flush()
                payload = self._receive()
            except (BrokenPipeError, OSError) as exc:
                self.close()
                raise GsavError("GSAV decoder is unavailable; reload the scene.") from exc
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                return {name: data[name] for name in data.files}

    def close(self):
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
            self._process.wait(timeout=10)
            self._process.stdin.close()
            self._process.stdout.close()
        self._log.close()
        self._directory.cleanup()
