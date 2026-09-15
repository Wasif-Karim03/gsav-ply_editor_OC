# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gscodec is a dynamic 3D Gaussian Splatting video codec. It compresses PLY sequences into `.gsav` container files optimized for web streaming via AV1/VP9 video encoding with WebCodecs API compatibility.

- Format Specification: [GSAV_FORMAT.md](GSAV_FORMAT.md)
- Coding Standards: [CONTRIBUTING.md](CONTRIBUTING.md)

## Commands

```bash
# Setup
uv pip install -e ".[dev]"
pre-commit install

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_chunk_encoder.py

# Run a specific test
uv run pytest tests/test_chunk_encoder.py::TestClassName::test_method_name -v

# Lint & type check
uv run ruff check .
uv run mypy src/

# Pre-commit (all hooks)
pre-commit run --all-files

# CLI usage
uv run compress --input-dir ./plys --output scene.gsav
uv run decompress --input scene.gsav --output-dir ./decoded_plys
```

**Always use `uv run` for Python execution** — never bare `python` or `pytest`.

## Agent Guidelines

- **Communication**: Be concise. Focus on solutions.
- **Refactoring**: Ask before removing unused code. Delete dead code rather than commenting it out.
- **Verification**: Run `uv run pytest` after changes. Any change to encoder/decoder logic must pass roundtrip tests.
- **Tensor shapes**: Document tensor shapes in comments (e.g., `# [N_Gaussians, 3]`).
- **Performance**: Never use Python `for` loops for array/tensor processing — use NumPy/PyTorch vectorized ops. Prefer in-place operations for large tensors.
- **Dependencies**: Use `uv add` / `uv remove` for dependency management.

## Architecture

### Encoder Flow

```
PLY Files → SequenceEncoder → .gsav
                ↓
    1. Load PLYs as GSTensor (via gsply)
    2. Compute global quantization ranges across all frames
    3. For each chunk:
       a. Per-frame Morton sort with global bbox (consistent quantization across frames)
       b. Per frame: log_transform + clip → quantize all attributes → build 3x5 atlas
    4. Keyframe-based temporal snapping (≤1 level → snap to frame 0, prevents drift)
    5. Encode atlases to VP9 video (xllvp9 lossless)
    6. Transcode audio to OGG/Opus (optional)
    7. Assemble GSAV container
```

Entry points: `SequenceEncoder.compress()` (Python API), `compress` CLI command (via `tyro`).

### Decoder Flow

```
.gsav → SequenceDecoder → GSData frames
              ↓
    1. Parse header and indices (GSAVFileProvider)
    2. For each frame:
       a. Decode video frame (3x5 atlas) via FFmpeg
       b. Extract 14 channels from grid via inverse 2D Morton curve (ChunkDecoder)
       c. Dequantize all attributes using global ranges
```

Entry points: `SequenceDecoder.from_file()` (Python API), `decompress` CLI command.

### Key Design Decisions

- **Frame-by-frame encoding**: Every frame is self-contained in the video atlas — no anchor/delta split, no drift accumulation.
- **Per-frame Morton sort with global bbox**: Each frame is independently sorted by 3D position using consistent coordinate quantization across all frames. This maximizes spatial smoothness (SOGS co-sorting effect) while maintaining temporal coherence.
- **Keyframe-based temporal snapping**: Per-attribute thresholds (scales≤2, quats≤2, opacity≤3, sh0≤3) snapped to chunk keyframe (frame 0). Prevents drift accumulation.
- **Arcsine quaternion quantization**: `asin(q)/(π/2)` transform before 8-bit quantization redistributes precision to match the quat distribution. Decoder applies `sin(q*π/2)` — single GPU instruction.
- **SH0 YCbCr decorrelation**: RGB→YCbCr before quantization concentrates color energy in Y channel. Cb/Cr channels compress better for grayscale-ish content.
- **2D Morton curve placement**: Gaussians are placed in atlas cells along a 2D Z-order curve instead of row-major, improving VP9's TM_PRED spatial prediction.
- **Auto-K with scene-adaptive heuristic**: K-snap value computed from median Gaussian scale vs scene extent (factor 0.20). Expanded safe K set includes all wrapping-safe odd values ≤51.
- **Volume-weighted auto-pruning**: Removes Gaussians invisible across ALL frames (max opacity < 0.10 or max scale < 1e-3) plus bottom 0.5% by opacity×volume importance.
- **Video-safe range [16, 235]**: All quantized values stay within TV-range to avoid TMU range expansion inconsistencies across devices.
- **Index-first binary layout**: Enables HTTP Range requests and random frame access for web streaming.
- **GOP = chunk_size**: I-frames align with chunk boundaries for seeking.

## Key Constants (`gscodec/constants.py`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `N_LEVELS` | 220 | Video-safe 8-bit range (16-235) |
| `MIN_VAL` | 16 | Video-safe range start |
| `SCALE_8BIT` | 219 | N_LEVELS - 1 |
| `SCALE_16BIT` | 56319 | 220 × 256 - 1 for 16-bit means |
| `N_ATLAS_COLS` | 5 | Atlas columns |
| `N_ATLAS_ROWS` | 3 | Atlas rows |

## Quantization (per-frame, using global ranges)

- **Means**: 16-bit base-256 split into hi/lo bytes. Hi: base-220 video-safe [16, 235]. Lo: raw [0, 255] stored in separate binary payload (not in video).
- **Scales, quats, opacity, sh0**: 8-bit linear in [16, 235]

## Atlas Layout (3 rows x 5 cols = 15 cells)

```
Row 0: [mean_hi.x][mean_hi.y][mean_hi.z][scale.x  ][scale.y  ]
Row 1: [scale.z  ][quat.w   ][quat.x   ][quat.y   ][quat.z   ]
Row 2: [opacity  ][sh0.r    ][sh0.g    ][sh0.b    ][padding  ]
```

14 active channels + 1 padding (filled with MIN_VAL=16). Gaussians placed along 2D Morton curve within each cell.

## GSAV Binary Layout

```
Header (128B) → Ranges (112B) → ChunkIndex → FrameIndex → MeansLoPayload → VideoPayload → AudioPayload (optional)
```

Header flags: bit 0 = `HAS_AUDIO` (0x0001). See [GSAV_FORMAT.md](GSAV_FORMAT.md) for full byte-level specification.

## Video Encoding

VP9 lossless via xllvp9 (native libvpx wrapper). Single optimal path — no codec selection needed.

Key params: `color_range=tv`, `pix_fmt=yuv420p`, `g=chunk_size`. Decoder uses FFmpeg for VP9 frame extraction.
