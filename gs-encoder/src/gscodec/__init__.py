"""
gscodec - Encode and decode dynamic 3D Gaussian Splatting sequences to/from GSAV format.

This library provides a complete codec for compressing and decompressing dynamic
Gaussian Splatting sequences using frame-by-frame video encoding.

Key Features:
- High-quality compression using video-safe quantization (16-235 range)
- Frame-by-frame encoding: every frame fully quantized into a 3x5 atlas
- 16-bit means (hi in atlas + lo in binary payload), 8-bit for all other attributes
- Memory-efficient chunk-based processing (GOP alignment only)
- Single-file GSAV container format

Typical Usage (Encoding):
    ```python
    from gscodec.encoder import SequenceEncoder

    encoder = SequenceEncoder()
    encoder.compress(input_dir="path/to/ply_files", output="scene.gsav")
    ```

Typical Usage (Decoding):
    ```python
    from gscodec.decoder import SequenceDecoder

    decoder = SequenceDecoder.from_file("scene.gsav")
    for frame in decoder:
        print(f"Decoded frame with {len(frame.means)} Gaussians.")
    ```
"""

from gscodec.constants import (
    MIN_VAL,
    N_ATLAS_COLS,
    N_ATLAS_ROWS,
    N_LEVELS,
    OPACITY_CLIP,
    QUATS_CLIP,
    SCALE_8BIT,
    SCALE_16BIT,
    SH0_CLIP,
)

__all__ = [
    # Constants
    "N_LEVELS",
    "MIN_VAL",
    "SCALE_8BIT",
    "SCALE_16BIT",
    "N_ATLAS_COLS",
    "N_ATLAS_ROWS",
    "QUATS_CLIP",
    "OPACITY_CLIP",
    "SH0_CLIP",
]
