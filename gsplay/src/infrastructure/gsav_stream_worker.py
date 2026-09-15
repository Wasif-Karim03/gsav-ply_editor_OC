"""Private, length-prefixed raw-array IPC worker for the isolated v3 codec."""

import io
import json
import struct
import sys
from contextlib import redirect_stdout
from pathlib import Path

from gsav_worker import stored_sh_degree


def serve(source: Path, workspace: Path, output) -> None:
    import numpy as np
    from gscodec.decoder import SequenceDecoder

    def send(payload):
        output.write(struct.pack("<Q", len(payload)))
        output.write(payload)
        output.flush()

    decoder = SequenceDecoder.from_file(source)
    provider = decoder._provider
    if not 0 < len(decoder) <= 10000 or not 0 < decoder.n_gaussians <= 5_000_000:
        raise ValueError("Scene exceeds supported frame/Gaussian limits.")
    if decoder.get_static_asset():
        raise ValueError("Embedded static GSST tracks are not supported yet.")
    bands = stored_sh_degree(provider.header)
    audio = decoder.get_audio() if decoder.has_audio else None
    audio_path = workspace / "audio.ogg"
    if audio:
        audio_path.write_bytes(audio)
    send(
        json.dumps(
            {
                "frames": len(decoder),
                "fps": decoder.fps,
                "sh_bands": bands,
                "audio": str(audio_path) if audio else None,
                "warning": "This legacy file has no SH sidecar; only SH0 is available."
                if bands != provider.sh_bands
                else None,
            }
        ).encode()
    )
    # A bounded decoded video block amortizes FFmpeg startup while retaining seeking.
    block_size = max(1, min(32, (128 * 1024**2) // (provider.atlas_width * provider.atlas_height)))
    block_start, atlases = -1, []
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("close"):
            return
        index = request["frame"]
        if not isinstance(index, int) or not 0 <= index < len(decoder):
            raise ValueError("Frame index outside the GSAV timeline.")
        if not block_start <= index < block_start + len(atlases):
            block_start = (index // block_size) * block_size
            atlases = provider.decode_video_frames(
                block_start, min(block_size, len(decoder) - block_start)
            )
        chunk = provider.get_chunk_for_frame(index)
        frame = decoder._decoder.decode_frame(
            atlases[index - block_start],
            provider.get_means_lo(index),
            provider.get_sh_labels(index) if bands else None,
            provider.get_sh_chunk_data(chunk) if bands else None,
        )
        expected = {0: 0, 1: 3, 2: 8, 3: 15}[bands]
        if expected and (frame.shN is None or frame.shN.shape[1:] != (expected, 3)):
            raise ValueError("SH payload did not reconstruct the declared SH degree.")
        arrays = {
            name: getattr(frame, name)
            for name in ("means", "scales", "quats", "opacities", "sh0", "shN")
        }
        if arrays["shN"] is None:
            arrays["shN"] = np.empty((len(frame), 0, 3), dtype=np.float32)
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        send(buffer.getvalue())


if __name__ == "__main__":
    # Library progress/logging must never contaminate the binary protocol.
    pipe = sys.stdout.buffer
    with redirect_stdout(sys.stderr):
        serve(Path(sys.argv[1]), Path(sys.argv[2]), pipe)
