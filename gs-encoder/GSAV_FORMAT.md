# GSAV: Gaussian Splatting Audio-Visual Container Format

The v3 extensions and current version/flag rules are specified in
[docs/gsav-v3.md](docs/gsav-v3.md). The layout below describes the v2 software's
legacy wire version 1 base.

## Overview

GSAV (`.gsav`) is a binary container format for streaming dynamic 3D Gaussian Splatting sequences over the web. It uses **frame-by-frame encoding** where every frame is fully quantized independently using global ranges and packed into a 3x6 atlas.

### Design Goals

1. **Single file delivery**: One URL per sequence, simplifies caching and CDN distribution
2. **WebCodecs compatible**: AV1 video stream for hardware-accelerated decoding in browsers
3. **Streaming-ready**: Index-first layout enables HTTP Range requests for random access
4. **Drift-free**: Each frame decoded independently, errors don't accumulate
5. **Simple**: No anchor/delta split — every frame is self-contained
6. **Audio-visual**: Optional embedded audio synced to video via shared timebase

---

## File Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ HEADER (128 bytes)                                               │
│   Magic "GSAV", dimensions, offsets, codec, audio metadata      │
├─────────────────────────────────────────────────────────────────┤
│ QUANTIZATION RANGES (112 bytes)                                 │
│   Global min/max for dequantization                             │
├─────────────────────────────────────────────────────────────────┤
│ CHUNK INDEX (16 bytes × n_chunks)                               │
│   Frame ranges (offset=0, size=0, GOP alignment only)           │
├─────────────────────────────────────────────────────────────────┤
│ FRAME INDEX (8 bytes × n_frames)                                │
│   Video OBU offsets and sizes, keyframe flags                   │
├─────────────────────────────────────────────────────────────────┤
│ MEANS LO PAYLOAD (n_gaussians × 3 bytes × n_frames)            │
│   Raw lo-bytes for 16-bit mean positions                        │
├─────────────────────────────────────────────────────────────────┤
│ VIDEO PAYLOAD (variable)                                        │
│   Raw AV1 OBU bitstream (frame atlases)                         │
├─────────────────────────────────────────────────────────────────┤
│ AUDIO PAYLOAD (variable, optional)                              │
│   OGG/Opus bitstream                                            │
└─────────────────────────────────────────────────────────────────┘
```

The "Index-First" design allows clients to download all metadata upfront (~100KB) and then stream video and audio data via HTTP Range requests.

---

## Header (128 bytes)

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | char[4] | magic | `"GSAV"` |
| 4 | 4 | u32 | version | Format version (1) |
| 8 | 4 | u32 | n_gaussians | Gaussians per frame |
| 12 | 4 | u32 | n_frames | Total frames |
| 16 | 4 | u32 | n_chunks | Number of chunks |
| 20 | 4 | u32 | chunk_size | Frames per chunk (= GOP size) |
| 24 | 2 | u16 | atlas_side | Gaussian grid side |
| 26 | 1 | u8 | n_atlas_cols | Atlas columns (6) |
| 27 | 1 | u8 | n_atlas_rows | Atlas rows (3) |
| 28 | 2 | u16 | fps | Frames per second |
| 30 | 2 | u16 | flags | Bit 0: `HAS_AUDIO` (0x0001) |
| 32 | 4 | u32 | ranges_offset | Offset to quantization ranges |
| 36 | 4 | u32 | chunk_index_offset | Offset to chunk index |
| 40 | 4 | u32 | frame_index_offset | Offset to frame index |
| 44 | 4 | u32 | means_lo_payload_offset | Offset to means lo data |
| 48 | 4 | u32 | video_payload_offset | Offset to video data |
| 52 | 16 | char[16] | codec | WebCodecs codec string (null-padded) |
| 68 | 4 | u32 | audio_payload_offset | Offset to audio data (0 if none) |
| 72 | 4 | u32 | audio_payload_size | Audio data size in bytes (0 if none) |
| 76 | 52 | - | reserved | Future use |

Codec values: `"av01.0.14M.08"` for AV1 (Level 5.2, Main tier), `"vp09.00.51.08"` for VP9 (Level 5.1).

### Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | `HAS_AUDIO` | Audio payload is present |
| 1-15 | - | Reserved (0) |

---

## Quantization Ranges (112 bytes)

Global min/max values for frame dequantization:

| Field | Size | Type | Description |
|-------|------|------|-------------|
| means_min | 12 | f32[3] | XYZ minimums (log-transformed) |
| means_max | 12 | f32[3] | XYZ maximums (log-transformed) |
| scales_min | 12 | f32[3] | Log-scale minimums |
| scales_max | 12 | f32[3] | Log-scale maximums |
| quats_min | 16 | f32[4] | Quaternion minimums |
| quats_max | 16 | f32[4] | Quaternion maximums |
| opacity_min | 4 | f32 | Logit opacity minimum |
| opacity_max | 4 | f32 | Logit opacity maximum |
| sh0_min | 12 | f32[3] | SH DC minimums |
| sh0_max | 12 | f32[3] | SH DC maximums |

---

## Chunk Index

Array of `n_chunks` entries (16 bytes each). Chunks exist only for GOP/I-frame alignment:

| Field | Size | Type | Description |
|-------|------|------|-------------|
| start_frame | 4 | u32 | First frame index (inclusive) |
| end_frame | 4 | u32 | Last frame index (inclusive) |
| offset | 4 | u32 | Always 0 (no anchor payload) |
| size | 4 | u32 | Always 0 (no anchor payload) |

---

## Frame Index

Array of `n_frames` entries (8 bytes each):

| Field | Size | Type | Description |
|-------|------|------|-------------|
| offset | 4 | u32 | Offset relative to video_payload_offset |
| size | 4 | u32 | OBU size (high bit = keyframe flag) |

The high bit of `size` (`0x80000000`) indicates a keyframe. Use `size & 0x7FFFFFFF` to get the actual size.

---

## Video Payload

Raw AV1 OBU (Open Bitstream Unit) stream. Frames are concatenated sequentially.

Each frame is a grayscale atlas containing all attributes for that frame:

### Atlas Layout (3 rows x 6 cols = 18 cells)

```
Row 0: [mean_hi.x][mean_hi.y][mean_hi.z][mean_lo.x][mean_lo.y][mean_lo.z]
Row 1: [scale.x  ][scale.y  ][scale.z  ][quat.w   ][quat.x   ][quat.y   ]
Row 2: [quat.z   ][opacity  ][sh0.r    ][sh0.g    ][sh0.b    ][padding  ]
```

- 17 active channels + 1 padding (filled with MIN_VAL=16)
- Each cell is `side x side` pixels where `side = ceil(sqrt(n_gaussians))` rounded to even

### Quantization

- **Means**: 16-bit base-220 split into hi/lo bytes, both in video-safe range [16, 235]
  - `quantized = normalize(value, min, max) * 48399`
  - `hi = quantized // 220 + 16`, `lo = quantized % 220 + 16`
- **All other attributes**: 8-bit linear in video-safe range [16, 235]
  - `quantized = normalize(value, min, max) * 219 + 16`

### Dequantization

- **Means**: `value = ((hi - 16) * 220 + (lo - 16)) / 48399 * (max - min) + min`
- **Others**: `value = (pixel - 16) / 219 * (max - min) + min`

### Video encoding

- Codec: AV1 (libaom-av1, libsvtav1, av1_nvenc) or VP9 (libvpx-vp9)
- GOP size: Matches chunk_size (I-frame at each chunk boundary)
- Quality: Lossless (or QP=1 for av1_nvenc)
- Color range: TV (limited, values in [16, 235])
- Pixel format: YUV420P with neutral chroma (128)

---

## Audio Payload (Optional)

A single continuous **OGG/Opus** bitstream. Present only when `flags & HAS_AUDIO` is set.

### Format

- **Container**: OGG
- **Codec**: Opus (libopus)
- **Sample rate**: 48 kHz (Opus native)
- **Channels**: Stereo (2)
- **Bitrate**: 128 kbps VBR (typical)

### Input handling

The encoder accepts any audio format supported by FFmpeg (MP3, WAV, AAC, FLAC, OGG, etc.) and transcodes to OGG/Opus. The audio is trimmed or padded with silence to match the video duration (`n_frames / fps` seconds).

### Synchronization

Audio and video share a common timebase starting at `t = 0`:

- Video frame `i` corresponds to time `t = i / fps`
- The OGG container provides granule positions for sample-accurate seeking
- No per-frame audio index is needed — the client seeks the Opus stream by timestamp

To seek to frame `i`:
1. Compute `t = i / fps`
2. Seek the OGG/Opus stream to time `t`
3. Decode the video frame at index `i`

### Client playback (browser)

```typescript
// Fetch audio blob via Range request
const audioBlob = await fetchRange(url, audioPayloadOffset, audioPayloadOffset + audioPayloadSize);

// Decode and play via Web Audio API or <audio> element
const audioContext = new AudioContext();
const audioBuffer = await audioContext.decodeAudioData(audioBlob);
const source = audioContext.createBufferSource();
source.buffer = audioBuffer;
source.connect(audioContext.destination);
source.start();

// Sync: when seeking to frame i
const seekTime = frameIndex / fps;
source.stop();
// Recreate source at new offset...
```

### Size impact

Audio adds negligible overhead to GSAV files:

| Content | Typical size |
|---------|-------------|
| Video payload (300 frames, 50K gaussians) | 5-50 MB |
| Audio (10 seconds, Opus 128kbps) | ~160 KB |
| Audio overhead | < 1% |

---

## Encoding Pipeline

### 1. Load PLY Sequence

```python
frames = [gsply.plyread(f) for f in sorted(ply_files)]
```

### 2. Compute Global Ranges

Find min/max across all frames for quantization.

### 3. Process Chunks

For each chunk:

1. **Morton sort** first frame, apply indices to all frames in chunk
2. **Per frame**: log_transform + clip -> quantize all attributes -> build 3x6 atlas

### 4. Encode Video

Encode all atlases to AV1 with GOP = chunk_size.

### 5. Transcode Audio (optional)

If audio input provided, transcode to OGG/Opus via FFmpeg, trimmed/padded to `n_frames / fps` seconds.

### 6. Assemble Container

Write sections in order: Header -> Ranges -> ChunkIndex -> FrameIndex -> MeansLo -> Video -> Audio

---

## Decoding Pipeline

### 1. Fetch Indices

```typescript
const header = await fetchRange(url, 0, 128);
const { rangesOffset, frameIndexOffset, videoPayloadOffset, audioPayloadOffset, audioPayloadSize, flags, ... } = parseHeader(header);

const indices = await fetchRange(url, rangesOffset, frameIndexOffset + n_frames * 8);
```

### 2. Decode Frame

```typescript
function decodeFrame(atlas: VideoFrame, ranges: QuantRanges): GSData {
    // Extract 18 channels from 3x6 atlas grid
    const channels = extractChannels(atlas);

    // Dequantize means from hi/lo base-220
    const means = dequantMeans16bit(channels[0..2], channels[3..5], ranges);

    // Dequantize other attributes with global ranges
    const scales = dequant8bit(channels[6..8], ranges.scales);
    const quats = dequant8bit(channels[9..12], ranges.quats);
    const opacity = dequant8bit(channels[13], ranges.opacity);
    const sh0 = dequant8bit(channels[14..16], ranges.sh0);

    return { means, scales, quats, opacities, sh0 };
}
```

### 3. Fetch Audio (optional)

```typescript
if (flags & HAS_AUDIO) {
    const audioData = await fetchRange(url, audioPayloadOffset, audioPayloadOffset + audioPayloadSize);
    // Pipe to AudioContext or <audio> element
}
```

---

## Performance Characteristics

### Decode Latency (per frame)

| Stage | Time |
|-------|------|
| Video decode (WebCodecs HW) | ~1 ms |
| Dequantization (all attributes) | ~1 ms |
| **Total** | **~2 ms** |

Audio decoding runs independently on the browser's audio thread and does not affect frame decode latency.
