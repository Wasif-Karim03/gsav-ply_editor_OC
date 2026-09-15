# GSAV SH Compression Specification

Higher-order spherical harmonics (SH1-SH3) compression for the GSAV format, using a codebook + palette + label approach adapted from the PlayCanvas SOG format. All SH data is stored as a binary payload with zstd compression — the video atlas (3x5) is unchanged.

## 1. Overview

SH compression uses three levels of indirection:

1. **Scalar codebook** (220 float32 values per chunk): Maps a uint8 index to a float SH coefficient value. 220 entries match the video-safe quantization level count.
2. **Palette centroids** (up to 65,535 entries per chunk): Each entry stores a complete SH profile as uint8 codebook indices. Built via K-means clustering of all gaussians across all frames in the chunk.
3. **Labels** (per gaussian, per frame): A uint16 index selecting a palette entry. Stored in binary payload with per-frame zstd compression and delta encoding (XOR with keyframe).

Decode path per gaussian:
```
label (uint16, from binary payload)
  -> palette entry (coeffs_per_channel * 3 uint8 codebook indices)
  -> codebook[index] for each
  -> float SH coefficient values
```

## 2. SH Band Levels

| sh_bands | Band | Coefficients per channel | Total values (x3 colors) | Centroid bytes |
|----------|------|--------------------------|--------------------------|----------------|
| 0 | DC only (SH0) | 0 | 0 | 0 |
| 1 | SH1 | 3 | 9 | 9 |
| 2 | SH2 | 8 | 24 | 24 |
| 3 | SH3 | 15 | 45 | 45 |

SH0 (DC) is always encoded separately in the atlas (channels 11-13, YCbCr-transformed, 8-bit). This spec covers SH1+ only.

Typical centroid count: `n_gaussians / 16` (matching SOG's ratio for million-scale scenes).

## 3. GSAV Header Changes

Two new fields carved from the 48-byte reserved area at offset 80:

| Offset | Size | Field | Type | Description |
|--------|------|-------|------|-------------|
| 80 | 1 | `sh_bands` | uint8 | 0=none, 1=SH1, 2=SH2, 3=SH3 |
| 81 | 4 | `sh_payload_offset` | uint32 | Absolute file offset (0 if sh_bands=0) |
| 85 | 43 | reserved | padding | Remaining reserved bytes |

Header struct format:
```
"<4s I I I I I H B B H H I I I I I 16s I I I B I 43x"
```

Header flags: bit 1 (`0x0002`) = `HAS_SH`.

**Backward compatibility:** Existing files have `0x00` in byte 80, so `sh_bands=0` and all SH logic is skipped.

**Atlas layout is unchanged:** 3x5 grid (15 cells), `n_atlas_cols=5`, `n_atlas_rows=3`.

## 4. File Layout

```
Header (128B) -> Ranges (112B) -> ChunkIndex -> FrameIndex
  -> MeansLoPayload -> SHPayload -> VideoPayload -> AudioPayload
```

The SH payload sits between means_lo and video, just like means_lo sits between frame index and video.

## 5. SH Binary Payload Format

The SH payload contains data for each chunk, concatenated sequentially.

### Per-chunk structure:

```
u16   n_centroids                              (2 bytes)
u8    sh_bands                                 (1 byte)
f32   codebook[220]                            (880 bytes)
u32   compressed_centroids_size                (4 bytes)
bytes zstd_compressed(centroids_raw)           (variable)
u32   n_frames_in_chunk                        (4 bytes)
For each frame:
  u32   compressed_label_size                  (4 bytes)
  bytes zstd_compressed(labels_or_delta)       (variable)
```

### Centroids

`centroids_raw` is `n_centroids * coeffs_per_channel * 3` bytes (uint8 codebook indices, row-major). Compressed with zstd level 13.

### Labels (delta-encoded)

- **Frame 0 (keyframe):** Raw uint16 labels, `N * 2` bytes, zstd compressed.
- **Frames 1+:** XOR delta against frame 0 labels, then zstd compressed. For gaussians whose SH doesn't change (most within a chunk), the delta is zero — zstd compresses this to nearly nothing.

### Decode:

```
keyframe_labels = zstd_decompress(frame_0_blob)  // [N] uint16
frame_labels = keyframe_labels XOR zstd_decompress(frame_t_blob)  // [N] uint16
```

## 6. Encoding Algorithm

### Per chunk (30 frames):

**Step 1: Collect all SH values**
```
all_sh = concatenate([frame.shN for frame in chunk_frames])
shape: [T * N, coeffs_per_channel * 3]
```

**Step 2: Build scalar codebook**
```
codebook = linspace(min(all_sh), max(all_sh), 220)  // [220] float32
```

**Step 3: Quantize to codebook indices**
```
indices = nearest_index(all_sh, codebook)  // [T * N, coeffs * 3] uint8
```

**Step 4: K-means clustering**
```
centroids, labels = MiniBatchKMeans(n_clusters, indices)
centroids: [K, coeffs * 3] uint8
labels: [T * N] uint16
```

**Step 5: Split labels per frame, delta-encode, zstd compress**

**Step 6: Serialize codebook + centroids + labels into binary payload**

## 7. Decoding Algorithm

### Per chunk (once per ~30 frames):

1. Read codebook (220 floats) + compressed centroids from SH payload
2. Decompress centroids: `zstd -> [K, coeffs * 3] uint8`
3. Expand via codebook: `expanded[i][j] = codebook[centroids[i][j]]` -> `[K, coeffs * 3] float32`
4. Upload expanded centroid table to GPU as texture/buffer

### Per frame (30 Hz):

5. Read compressed labels from SH payload
6. Decompress: zstd + XOR with keyframe -> `[N] uint16`
7. Upload labels to GPU
8. Shader: `sh_values = expanded_centroids[label]` -> 45 floats for SH3

## 8. Web Decoder Integration

### CPU Side (per chunk, once per ~1 second)

- Fetch SH payload section via HTTP Range request
- zstd decompress centroids in Web Worker (reuse means_lo infra)
- Expand codebook indices to floats on CPU
- Upload expanded centroids as GPU texture (up to ~8.7 MB for 48K entries at SH3)

### CPU Side (per frame, 30 Hz)

- zstd decompress labels in Web Worker
- XOR with cached keyframe labels
- Upload `[N]` uint16 labels as GPU texture

### GPU Side (per frame)

- Shader reads label per gaussian -> indexes into centroid texture -> gets SH floats
- No atlas changes needed

### Performance Budget (Quest 3)

| Operation | Frequency | Cost |
|-----------|-----------|------|
| zstd decompress centroids | Once/chunk (~1/sec) | ~2ms CPU |
| Codebook expand | Once/chunk | ~1ms CPU |
| Centroid GPU upload | Once/chunk | ~2ms |
| zstd decompress labels | Per frame | ~0.5ms CPU |
| Label GPU upload | Per frame | ~0.2ms |
| Centroid lookup in shader | Per frame | 12 texture reads/gaussian (RGBA32F) |

No CPU readbacks. No second VideoDecoder. No atlas layout changes.
