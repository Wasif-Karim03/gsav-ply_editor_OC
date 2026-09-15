# GSAV v1 File Format Specification

## Overview

GSAV (Gaussian Splatting Audio-Visual) is a binary container format for streaming dynamic 3D Gaussian Splatting sequences. Version 1 uses an **Index-First** layout optimized for HTTP Range requests and WebCodecs decoding.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              GSAV v1 File Layout                             │
├──────────────────────────────────────────────────────────────────────────────┤
│  Header (64 bytes)                                                           │
│  ├─ Magic: "GSAV"                                                            │
│  ├─ Dimensions, counts, fps                                                  │
│  └─ Section offsets                                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│  Quantization Ranges (112 bytes)                                             │
│  └─ Global min/max for each attribute                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│  Chunk Index (16 bytes × n_chunks)                                           │
│  └─ Frame ranges and anchor offsets per chunk                                │
├──────────────────────────────────────────────────────────────────────────────┤
│  Frame Index (8 bytes × n_frames)                                            │
│  └─ Video OBU offsets and sizes per frame                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│  Anchor Payload (variable)                                                   │
│  └─ Per-chunk canonical data + residual scaling                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  Video Payload (variable)                                                    │
│  └─ AV1 OBU bitstream (delta atlases)                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Header (64 bytes)

The header contains file identification, dimensions, and offsets to all sections.

### Structure

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | char[4] | `magic` | File signature: `"GSAV"` (0x47535632) |
| 4 | 4 | uint32 | `version` | Format version: `1` |
| 8 | 4 | uint32 | `n_gaussians` | Number of Gaussians per frame |
| 12 | 4 | uint32 | `n_frames` | Total number of frames |
| 16 | 4 | uint32 | `n_chunks` | Number of chunks |
| 20 | 4 | uint32 | `chunk_size` | Frames per chunk (= GOP size) |
| 24 | 2 | uint16 | `atlas_width` | Atlas width in pixels |
| 26 | 2 | uint16 | `atlas_height` | Atlas height in pixels |
| 28 | 2 | uint16 | `fps` | Frames per second |
| 30 | 2 | uint16 | `flags` | Reserved (0) |
| 32 | 4 | uint32 | `ranges_offset` | Byte offset to Quantization Ranges |
| 36 | 4 | uint32 | `chunk_index_offset` | Byte offset to Chunk Index |
| 40 | 4 | uint32 | `frame_index_offset` | Byte offset to Frame Index |
| 44 | 4 | uint32 | `anchor_payload_offset` | Byte offset to Anchor Payload |
| 48 | 4 | uint32 | `video_payload_offset` | Byte offset to Video Payload |
| 52 | 12 | - | `reserved` | Reserved for future use |

### Parsing (Python)

```python
import struct

def read_header(f):
    data = f.read(64)
    return {
        "magic": data[0:4],
        "version": struct.unpack("<I", data[4:8])[0],
        "n_gaussians": struct.unpack("<I", data[8:12])[0],
        "n_frames": struct.unpack("<I", data[12:16])[0],
        "n_chunks": struct.unpack("<I", data[16:20])[0],
        "chunk_size": struct.unpack("<I", data[20:24])[0],
        "atlas_width": struct.unpack("<H", data[24:26])[0],
        "atlas_height": struct.unpack("<H", data[26:28])[0],
        "fps": struct.unpack("<H", data[28:30])[0],
        "flags": struct.unpack("<H", data[30:32])[0],
        "ranges_offset": struct.unpack("<I", data[32:36])[0],
        "chunk_index_offset": struct.unpack("<I", data[36:40])[0],
        "frame_index_offset": struct.unpack("<I", data[40:44])[0],
        "anchor_payload_offset": struct.unpack("<I", data[44:48])[0],
        "video_payload_offset": struct.unpack("<I", data[48:52])[0],
    }
```

### Parsing (TypeScript)

```typescript
interface GSAVHeader {
  magic: string;
  version: number;
  nGaussians: number;
  nFrames: number;
  nChunks: number;
  chunkSize: number;
  atlasWidth: number;
  atlasHeight: number;
  fps: number;
  flags: number;
  rangesOffset: number;
  chunkIndexOffset: number;
  frameIndexOffset: number;
  anchorPayloadOffset: number;
  videoPayloadOffset: number;
}

function readHeader(buffer: ArrayBuffer): GSAVHeader {
  const view = new DataView(buffer);
  const decoder = new TextDecoder();

  return {
    magic: decoder.decode(new Uint8Array(buffer, 0, 4)),
    version: view.getUint32(4, true),
    nGaussians: view.getUint32(8, true),
    nFrames: view.getUint32(12, true),
    nChunks: view.getUint32(16, true),
    chunkSize: view.getUint32(20, true),
    atlasWidth: view.getUint16(24, true),
    atlasHeight: view.getUint16(26, true),
    fps: view.getUint16(28, true),
    flags: view.getUint16(30, true),
    rangesOffset: view.getUint32(32, true),
    chunkIndexOffset: view.getUint32(36, true),
    frameIndexOffset: view.getUint32(40, true),
    anchorPayloadOffset: view.getUint32(44, true),
    videoPayloadOffset: view.getUint32(48, true),
  };
}
```

---

## 2. Quantization Ranges (112 bytes)

Global min/max values used for anchor dequantization. All values are stored as little-endian float32.

### Structure

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 12 | float32[3] | `means_min` | XYZ minimums (log-space) |
| 12 | 12 | float32[3] | `means_max` | XYZ maximums (log-space) |
| 24 | 12 | float32[3] | `scales_min` | Log-scale minimums |
| 36 | 12 | float32[3] | `scales_max` | Log-scale maximums |
| 48 | 16 | float32[4] | `quats_min` | Quaternion (wxyz) minimums |
| 64 | 16 | float32[4] | `quats_max` | Quaternion (wxyz) maximums |
| 80 | 4 | float32 | `opacity_min` | Logit opacity minimum |
| 84 | 4 | float32 | `opacity_max` | Logit opacity maximum |
| 88 | 12 | float32[3] | `sh0_min` | SH DC coefficient minimums |
| 100 | 12 | float32[3] | `sh0_max` | SH DC coefficient maximums |

**Total: 112 bytes**

### Usage

These ranges are used to dequantize anchor values:

```python
# For 16-bit means: [0, 65535] → [min, max]
value = (quantized / 65535.0) * (max - min) + min

# For 8-bit attributes: [16, 235] → [min, max]  (video-safe range)
value = ((quantized - 16) / 219.0) * (max - min) + min
```

---

## 3. Chunk Index (16 bytes × n_chunks)

Maps chunks to their frame ranges and anchor data locations.

### Entry Structure

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | uint32 | `start_frame` | First frame index (inclusive) |
| 4 | 4 | uint32 | `end_frame` | Last frame index (inclusive) |
| 8 | 4 | uint32 | `offset` | Byte offset **relative to** `anchor_payload_offset` |
| 12 | 4 | uint32 | `size` | Anchor data size in bytes |

### Example

For a file with 220 frames in 1 chunk:
```
Chunk 0: start=0, end=219, offset=0, size=1757880
```

For a file with 90 frames in 3 chunks of 30:
```
Chunk 0: start=0,  end=29, offset=0,       size=585960
Chunk 1: start=30, end=59, offset=585960,  size=585960
Chunk 2: start=60, end=89, offset=1171920, size=585960
```

### Calculating Absolute Offset

```python
absolute_offset = header["anchor_payload_offset"] + chunk_entry["offset"]
```

---

## 4. Frame Index (8 bytes × n_frames)

Maps each frame to its video data (AV1 OBU) location.

### Entry Structure

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | uint32 | `offset` | Byte offset **relative to** `video_payload_offset` |
| 4 | 4 | uint32 | `size` | OBU size with keyframe flag |

### Keyframe Flag

The high bit of `size` indicates whether the frame is a keyframe (I-frame):

```python
KEYFRAME_FLAG = 0x80000000
SIZE_MASK = 0x7FFFFFFF

is_keyframe = (entry["size"] & KEYFRAME_FLAG) != 0
actual_size = entry["size"] & SIZE_MASK
```

Keyframes occur at chunk boundaries (first frame of each chunk).

### Calculating Absolute Offset

```python
absolute_offset = header["video_payload_offset"] + frame_entry["offset"]
```

---

## 5. Anchor Payload

Contains per-chunk canonical (average) frame data and residual scaling factors.

### Per-Chunk Layout

Each chunk's anchor data is stored contiguously:

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 56 | float32[14] | `residual_max` | Per-channel max delta values |
| 56 | N×6 | uint16[N,3] | `means` | Quantized canonical means |
| 56+N×6 | N×3 | uint8[N,3] | `scales` | Quantized canonical scales |
| ... | N×4 | uint8[N,4] | `quats` | Quantized canonical quaternions |
| ... | N×1 | uint8[N,1] | `opacities` | Quantized canonical opacities |
| ... | N×3 | uint8[N,3] | `sh0` | Quantized canonical SH DC |

Where **N = n_gaussians**.

### Residual Max Layout (14 channels)

```
Index 0-2:   means (x, y, z)
Index 3-5:   scales (x, y, z)
Index 6-9:   quats (w, x, y, z)
Index 10:    opacity
Index 11-13: sh0 (r, g, b)
```

### Anchor Size Calculation

```python
def compute_anchor_size(n_gaussians: int) -> int:
    residual_max_size = 14 * 4  # 56 bytes
    means_size = n_gaussians * 3 * 2  # uint16
    scales_size = n_gaussians * 3 * 1  # uint8
    quats_size = n_gaussians * 4 * 1   # uint8
    opacities_size = n_gaussians * 1   # uint8
    sh0_size = n_gaussians * 3 * 1     # uint8

    return residual_max_size + means_size + scales_size + quats_size + opacities_size + sh0_size
    # = 56 + n_gaussians * 18
```

### Dequantization

**Means (16-bit):**
```python
normalized = quantized.astype(np.float32) / 65535.0
dequantized = normalized * (ranges.means_max - ranges.means_min) + ranges.means_min
```

**Other attributes (8-bit, video-safe):**
```python
normalized = (quantized.astype(np.float32) - 16) / 219.0
dequantized = normalized * (max_val - min_val) + min_val
```

---

## 6. Video Payload

Raw AV1 OBU (Open Bitstream Unit) frames concatenated sequentially.

### Structure

The video payload is a concatenation of AV1 OBU frames without any container format. Each frame's location is defined by the Frame Index.

```
[OBU Frame 0][OBU Frame 1][OBU Frame 2]...[OBU Frame N-1]
```

### Atlas Layout

Each video frame is a grayscale atlas with **14 rows** of height `side` where:
- `side = ceil(sqrt(n_gaussians))` rounded up to even number
- Total height = `14 × side`
- Width = `side`

**Row assignment:**
```
Row 0:  means delta X
Row 1:  means delta Y
Row 2:  means delta Z
Row 3:  scales delta X
Row 4:  scales delta Y
Row 5:  scales delta Z
Row 6:  quats delta W
Row 7:  quats delta X
Row 8:  quats delta Y
Row 9:  quats delta Z
Row 10: opacity delta
Row 11: sh0 delta R
Row 12: sh0 delta G
Row 13: sh0 delta B
```

### Delta Encoding

Each pixel encodes a quantized delta from the canonical value:

```
quantized = ((delta / residual_max) + 1) / 2 × 219 + 16
```

Where:
- `delta = frame_value - canonical_value`
- `residual_max` = per-channel maximum from anchor
- Output range: [16, 235] (video-safe)
- Center value (delta=0): ~125

### Decoding to IVF

To decode with ffmpeg, wrap the OBU frames in an IVF container:

```python
def build_ivf(obu_frames: list[bytes], width: int, height: int, fps: int) -> bytes:
    n_frames = len(obu_frames)

    # IVF header (32 bytes)
    header = struct.pack(
        "<4sHHIHHIIII",
        b"DKIF",      # signature
        0,            # version
        32,           # header size
        0x31305641,   # FourCC "AV01"
        width,
        height,
        fps,          # timebase numerator
        1,            # timebase denominator
        n_frames,
        0,            # unused
    )

    parts = [header]
    for i, obu in enumerate(obu_frames):
        frame_header = struct.pack("<IQ", len(obu), i)  # size + timestamp
        parts.append(frame_header)
        parts.append(obu)

    return b"".join(parts)
```

---

## 7. Decoding Pipeline

### Step 1: Parse Metadata

```python
with open("scene.gsav", "rb") as f:
    header = read_header(f)

    f.seek(header["ranges_offset"])
    ranges = read_ranges(f)

    f.seek(header["chunk_index_offset"])
    chunks = read_chunk_index(f, header["n_chunks"])

    f.seek(header["frame_index_offset"])
    frames = read_frame_index(f, header["n_frames"])
```

### Step 2: Load Anchors

```python
anchors = []
for chunk in chunks:
    f.seek(header["anchor_payload_offset"] + chunk["offset"])
    anchor_bytes = f.read(chunk["size"])
    anchor = deserialize_anchor(anchor_bytes, header["n_gaussians"], ranges)
    anchors.append(anchor)
```

### Step 3: Decode Video

```python
# Read all video OBUs
f.seek(header["video_payload_offset"])
video_payload = f.read()

# Build IVF container
obu_frames = []
for entry in frames:
    size = entry["size"] & SIZE_MASK
    obu_frames.append(video_payload[entry["offset"]:entry["offset"] + size])

ivf_data = build_ivf(obu_frames, header["atlas_width"], header["atlas_height"], header["fps"])

# Decode via ffmpeg (pipe)
atlases = decode_ivf_with_ffmpeg(ivf_data)
```

### Step 4: Reconstruct Frames

```python
for frame_idx in range(header["n_frames"]):
    # Find chunk for this frame
    chunk_idx = get_chunk_for_frame(frame_idx, chunks)
    anchor = anchors[chunk_idx]
    atlas = atlases[frame_idx]

    # Extract delta channels from atlas
    deltas = extract_channels(atlas, header["n_gaussians"])

    # Dequantize deltas
    for ch in range(14):
        deltas[ch] = ((deltas[ch] - 16) / 219.0) * 2 * anchor.residual_max[ch] - anchor.residual_max[ch]

    # Reconstruct: frame = canonical + delta
    means = anchor.means + deltas[0:3]
    scales = anchor.scales + deltas[3:6]
    quats = anchor.quats + deltas[6:10]
    opacities = anchor.opacities + deltas[10]
    sh0 = anchor.sh0 + deltas[11:14]

    # Apply inverse transforms
    means = inverse_log_transform(means)  # exp(x) - 1
```

---

## 8. HTTP Range Request Access

The Index-First layout enables efficient streaming:

### Initial Fetch (~200 bytes + indices)

```
Byte Range: 0 to frame_index_offset + (n_frames × 8) - 1
```

This gives you:
- Header (64 bytes)
- Ranges (112 bytes)
- Chunk Index (16 × n_chunks bytes)
- Frame Index (8 × n_frames bytes)

### Fetch Specific Chunk Anchor

```python
start = anchor_payload_offset + chunk["offset"]
end = start + chunk["size"] - 1
# Range: bytes=start-end
```

### Fetch Specific Video Frame

```python
start = video_payload_offset + frame["offset"]
size = frame["size"] & SIZE_MASK
end = start + size - 1
# Range: bytes=start-end
```

### Fetch Frame Range for Seeking

To seek to frame N:
1. Find chunk containing frame N
2. Fetch chunk anchor
3. Fetch video from chunk start to frame N
4. Decode sequentially from keyframe

---

## 9. WebCodecs Integration

### Configure VideoDecoder

```typescript
const decoder = new VideoDecoder({
  output: (frame) => processFrame(frame),
  error: (e) => console.error(e),
});

await decoder.configure({
  codec: "av01.0.04M.08",  // AV1 Main Profile, Level 4.0, 8-bit
  codedWidth: header.atlasWidth,
  codedHeight: header.atlasHeight,
  hardwareAcceleration: "prefer-hardware",
});
```

### Decode Frame

```typescript
const frameEntry = frameIndex[frameIdx];
const isKey = (frameEntry.size & 0x80000000) !== 0;
const size = frameEntry.size & 0x7FFFFFFF;

const obuData = await fetchRange(
  url,
  header.videoPayloadOffset + frameEntry.offset,
  size
);

decoder.decode(new EncodedVideoChunk({
  type: isKey ? "key" : "delta",
  timestamp: frameIdx * (1_000_000 / header.fps),
  data: obuData,
}));
```

---

## 10. Example File Analysis

For a file with 220 frames, 97344 Gaussians, 1 chunk:

```
Section              Offset        Size
─────────────────────────────────────────
Header               0             64 bytes
Ranges               64            112 bytes
Chunk Index          176           16 bytes
Frame Index          192           1,760 bytes
Anchor Payload       1,952         1,757,880 bytes
Video Payload        1,759,832     ~40 MB

Total: ~42 MB
```

Breakdown:
- **Indices** (header through frame index): 1,952 bytes (0.004%)
- **Anchor**: 1.76 MB (4.1%)
- **Video**: ~40 MB (95.9%)
