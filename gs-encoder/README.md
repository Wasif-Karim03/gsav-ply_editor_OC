<div align="center">

# gscodec

### Dynamic 3D Gaussian Splatting Codec

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Encode and decode dynamic Gaussian Splatting sequences to/from streamable `.gsav` containers.**

</div>

---

## Overview

`gscodec` compresses dynamic 3D Gaussian Splatting PLY sequences into a single `.gsav` container file optimized for web streaming. It uses VP9 lossless video encoding via [xllvp9](https://github.com/OpsiClear/xllvp9), enabling hardware-accelerated decoding in browsers via WebCodecs API.

**Key Features:**
- **Single File Output**: One `.gsav` file — video + binary means + optional audio
- **Frame-by-Frame Encoding**: Every frame is self-contained, no drift accumulation
- **Web-Ready**: VP9 video stream compatible with WebCodecs API for hardware decoding
- **Streaming-Optimized**: Index-first layout enables HTTP Range requests and random access
- **Auto-Tuned Compression**: Scene-adaptive K-snap, auto-pruning, temporal snapping, arcsine quat transform, SH0 YCbCr decorrelation

---

## Installation

```bash
uv pip install -e ".[dev]"
```

**Dependencies:** PyTorch, NumPy, gsply, xllvp9, zstandard, FFmpeg (system).

---

## Quick Start

### Compress (Encode)

**CLI:**
```bash
compress --input-dir ./plys --output scene.gsav
```

**With options:**
```bash
compress --input-dir ./plys --output scene.gsav \
    --chunk-size 30 \
    --fps 30 \
    --device cuda:0 \
    --lo-snap-k 0          # 0=auto (default), 1=off, 3-51=manual
```

**Python API:**
```python
from gscodec.encoder import SequenceEncoder, VideoConfig, ChunkConfig

encoder = SequenceEncoder(
    video_config=VideoConfig(fps=30),
    chunk_config=ChunkConfig(size=30, lo_snap_k=0),
    device="cuda:0",
)
encoder.compress(input_dir="./plys", output="scene.gsav")
```

### Decompress (Decode)

**CLI:**
```bash
decompress --input scene.gsav --output-dir ./decoded_plys
```

**Python API:**
```python
from gscodec.decoder import SequenceDecoder

decoder = SequenceDecoder.from_file("scene.gsav")

# Iterate all frames (streaming decode)
for frame in decoder:
    print(f"Frame: {frame.means.shape[0]} Gaussians")
    # frame.means  [N, 3] float32 — 3D positions
    # frame.scales [N, 3] float32 — log-space scales
    # frame.quats  [N, 4] float32 — unit quaternions (w, x, y, z)
    # frame.opacities [N] float32 — logit-space opacity
    # frame.sh0    [N, 3] float32 — SH DC color coefficients

# Random frame access (seeks to chunk keyframe internally)
frame_100 = decoder.decode_frame(100)

# Decode all at once
all_frames = decoder.decode_all()

# Properties
print(decoder.n_gaussians)  # Gaussians per frame
print(decoder.fps)          # Frames per second
print(decoder.chunk_size)   # Frames per chunk (GOP size)
print(len(decoder))         # Total frame count
print(decoder.has_audio)    # Audio track present?
```

### Quality Measurement

```python
from gscodec.metrics import rendering_psnr, rendering_psnr_batch

# Single frame PSNR (renders from orbit cameras via gsplat)
psnr = rendering_psnr(original_gstensor, decoded_gsdata)

# Batch PSNR with fixed cameras (consistent measurement)
avg, min_psnr, per_frame = rendering_psnr_batch(originals, decoded_frames)

# Custom cameras for consistent cross-frame comparison
from gscodec.metrics import _make_orbit_cameras
viewmats, Ks = _make_orbit_cameras(means, n_cameras=32, render_size=512, device="cuda")
psnr = rendering_psnr(orig, dec, viewmats=viewmats, Ks=Ks)
```

---

## How It Works

### Encoding Pipeline

```
PLY Files → SequenceEncoder.compress() → .gsav
                    ↓
    1. Load PLYs as GSTensor (parallel I/O via gsply)
    2. Auto-prune invisible Gaussians (opacity × volume importance)
    3. Compute global quantization ranges across all frames
    4. Auto-K: select K-snap value from scene scale distribution
    5. For each chunk (GOP-aligned):
       a. Morton sort with global bbox (stable correspondence detection)
       b. Arcsine quaternion transform (asin(q) / (π/2))
       c. SH0 RGB → YCbCr decorrelation
       d. 16-bit means quantization (hi/lo split) + 8-bit attribute quantization
       e. Keyframe temporal snapping (per-attribute thresholds)
       f. Build 3×5 grayscale atlas per frame
    6. VP9 lossless encode all atlases (xllvp9)
    7. Compress means_lo per-frame (base-N packing + zstd-13)
    8. Assemble GSAV container (index-first layout)
```

### Decoding Pipeline

```
.gsav → SequenceDecoder.decode_frame(idx) → GSData
                    ↓
    1. Seek to chunk keyframe via chunk index
    2. Decode VP9 frames (FFmpeg, keyframe → target)
    3. Extract 14 channels from 3×5 atlas via inverse Morton curve
    4. Dequantize attributes using global ranges
    5. Inverse arcsine on quaternions: sin(q × π/2)
    6. Inverse YCbCr on SH0: YCbCr → RGB
    7. Normalize quaternions to unit length
    8. Decompress means_lo from binary payload (zstd + base-N unpack)
    9. Reconstruct 16-bit means from hi + lo bytes
```

### Atlas Layout

Each video frame is a grayscale atlas (3 rows × 5 cols = 15 cells):

```
Row 0: [mean_hi.x][mean_hi.y][mean_hi.z][scale.x  ][scale.y  ]
Row 1: [scale.z  ][quat.w   ][quat.x   ][quat.y   ][quat.z   ]
Row 2: [opacity  ][sh0.Y    ][sh0.Cb   ][sh0.Cr   ][padding  ]
```

- **14 active channels + 1 padding** in video-safe range [16, 235] (220 levels)
- **Gaussians placed along 2D Morton (Z-order) curve** within each cell
- **Cell size**: `ceil(sqrt(N_gaussians))` rounded to even, per side

### Quantization

| Attribute | Precision | Storage | Transform | Range |
|-----------|-----------|---------|-----------|-------|
| Means | 16-bit (56,320 levels) | hi in atlas + lo in binary | log(1+\|x\|) | adaptive |
| Scales | 8-bit (220 levels) | Video atlas | none (already log-space) | adaptive |
| Quaternions | 8-bit (220 levels) | Video atlas | **asin(q)/(π/2)** | adaptive |
| Opacity | 8-bit (220 levels) | Video atlas | none (already logit) | [-6, 12] |
| SH0 (color) | 8-bit (220 levels) | Video atlas | **RGB→YCbCr** | adaptive |

**K-snap**: means_lo bytes are rounded to nearest multiple of K (scene-adaptive, odd, coprime with 256). Auto-K selects from `[1,3,5,7,...,51]` based on median Gaussian scale vs scene extent.

### Compression Techniques

| Technique | Target | Effect |
|-----------|--------|--------|
| **Arcsine quat transform** | Video (53% of bits) | Redistributes quantization precision to match quat distribution. Decoder: `sin(q×π/2)` — 1 GPU instruction. |
| **SH0 YCbCr decorrelation** | Video (16% of bits) | Concentrates color energy in Y channel. Cb/Cr compress better for grayscale content. |
| **Keyframe temporal snapping** | Video (all channels) | Per-attribute thresholds (sc≤2, qt≤2, op≤3, sh0≤3). Snaps to chunk frame 0 — VP9 encodes zero-diff for free. |
| **Volume-weighted pruning** | All payloads | Removes bottom 0.5% by `opacity × volume` importance, plus standard opacity/scale thresholds. |
| **Base-N packing** | Means_lo payload | When K-snap produces ≤6 unique values, packs xyz triples as single bytes (3:1 ratio). |
| **Morton spatial sort** | Video quality | 3D Morton code sorts Gaussians for spatial locality. Composite 4+4 bit (position + quat) for dynamic sequences. |

### GSAV Container Layout

```
Header (128B) → Ranges (112B) → ChunkIndex → FrameIndex → MeansLo → Video → Audio
```

| Section | Size | Purpose |
|---------|------|---------|
| Header | 128 B | Magic `"GSAV"`, dimensions, offsets, codec string, flags |
| Ranges | 112 B | Global min/max for dequantization (means, scales, quats, opacity, sh0) |
| Chunk Index | 16 B × N_chunks | Frame ranges per chunk (GOP alignment for seeking) |
| Frame Index | 8 B × N_frames | Per-frame VP9 byte offset + size + keyframe flag |
| MeansLo | variable | Per-frame zstd-compressed lo-bytes (random access via index) |
| Video | variable | VP9 lossless bitstream (3×5 grayscale atlas sequence) |
| Audio | variable | Optional OGG/Opus (synced to video timebase) |

See [GSAV_FORMAT.md](GSAV_FORMAT.md) for the full byte-level specification.

---

## Configuration

### ChunkConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `size` | 30 | Frames per chunk (= GOP size = seeking granularity) |
| `lo_snap_k` | 0 | K-snap value. 0=auto, 1=off, 3-51=manual (odd only) |
| `matching_enabled` | False | Temporal matching for variable Gaussian counts |
| `gsflow_metadata` | None | Path to GSFlow metadata.json for pre-sorted sequences |

### VideoConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fps` | 30 | Output video framerate |

### CLI Flags

| Flag | Description |
|------|-------------|
| `--input-dir` | Directory containing PLY files |
| `--output` | Output .gsav file path |
| `--fps` | Frames per second (default: 30) |
| `--chunk-size` | Frames per chunk (default: 30) |
| `--lo-snap-k` | K-snap value (default: 0 = auto) |
| `--device` | Torch device (default: cuda:0) |
| `--audio` | Optional audio file to embed |
| `--matching` | Enable temporal matching for varying Gaussian counts |
| `--verbose` | Enable debug logging |

---

## Benchmarks

Measured with 32 fixed orbit cameras at 512px, rendering PSNR via gsplat.

| Dataset | Frames | Gaussians | File Size | Avg PSNR | Min PSNR |
|---------|--------|-----------|-----------|----------|----------|
| Soccer | 351 | 25.5K | 18.95 MB | 40.88 dB | 39.97 dB |
| Elly | 220 | 86.8K | 70.03 MB | 41.05 dB | 39.00 dB |

Bits per Gaussian per frame: Soccer 4.23, Elly 36.9.

---

## Development

```bash
git clone https://github.com/OpsiClear/gscodec.git
cd gscodec
uv pip install -e ".[dev]"
pre-commit install

uv run pytest              # tests (29 tests)
uv run ruff check .        # lint
uv run mypy src/           # type check
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
