# Version 3.0.7 — compact cropped exports

Version 3.0.6 preserved source ordering by hiding excluded rows. Those rows
still occupied storage. This release physically removes rows that are never
visible within each encoded chunk, retaining the order of surviving rows.

## Correctness contract

The operation does not assume a row is the same physical Gaussian over time.
For each chunk it computes the union of visible storage slots and applies one
fixed selection to every frame in that chunk. Every visible sample keeps its
position, scale, rotation, opacity and colors. Per-frame masks still enforce
the exact crop boundary. Never-visible slots are removed from the atlas,
position-detail arrays and SH label arrays. Unreferenced SH palette entries
are pruned by remapping labels, without reclustering.

The existing format has one global row count. Chunks are padded invisibly to
the largest retained chunk, with four-byte row-buffer alignment and even atlas
dimensions for VP9. An entirely hidden sequence is represented by invisible
placeholder rows. This is not per-frame compaction or physical-object tracking.

The requested keep/remove-crossing choices are deferred until reliable identity
tracking exists. No chunk-local identity assumption is presented as whole-scene
tracking. The user delegated this choice; strict per-frame boundary behavior is
retained.

## Two export paths

- **Crop only:** Verify geometry and color fields were not edited, stage only
  boolean masks, then copy retained encoded geometry and SH data directly from
  the source. No PLY transport, temporal matching, color range fitting or SH
  clustering. Geometry ranges, scalar codebook values and audio are retained.
- **Crop plus color:** Use the existing verified color-edit path, then compact
  the encoded arrays before native VP9 and position/SH compression. Compaction
  adds no quantization loss beyond the existing color export.

Both paths require the existing source-layout eligibility: full original GSAV
timeline and FPS, neutral transforms and unchanged opacity scaling. Other
edits and PLY outputs retain their existing behavior. Mask shapes, source
presence, frame count and SH degree are validated. Output is fully encoded and
validated before exclusive publication; original files are not replaced.

Native xllvp9 is still the encoder. FFmpeg is used only for decoding. No new
dependencies or container schema changes were introduced. A mask-aware v3
viewer is required, as with the previous filtered export.

## Measured source scene

180 frames, SH3, 78,124 original slots. Sphere radius 1.1, center
(-0.26989809, 0.53084362, 0.24318331), maximum opacity 0.98. Color test uses
brightness 3.78. The compact output has 51,824 slots (the aligned largest
retained chunk), rather than 78,124.

| Output | Bytes | Decimal MB | Elapsed seconds |
| --- | ---: | ---: | ---: |
| Original source | 40,651,825 | 40.65 | — |
| v3.0.6 crop + brightness reference | 41,140,793 | 41.14 | 84.34 |
| Compact crop only | 27,290,228 | 27.29 | 37.64 |
| Compact crop + brightness | 27,278,988 | 27.28 | 73.53 |

The edited compact output is about 33.7% smaller than the previous output and
32.9% smaller than the source. These are separate local runs, not controlled
hardware benchmarks or promises for every crop. Crop-only and color-edited
timings measure different workloads. Light crops or crops that retain every
slot at some point in a chunk may save little space.

All 180 crop-only frames match original visible means, scales, quaternions,
opacity, SH0 and SHN exactly. All 180 crop-plus-color frames match the verified
v3.0.6 edited reference exactly on those fields. Visibility matches the previously
verified GPU preview. Random access passed at frames 0, 29, 30, 90, 150 and 179.

After restarting port 6034, a browser-triggered GPU crop-only export produced
27,290,228 bytes, SHA-256
`724033340c3411a62f7c4896b7cd540bcf4242b84ce347f5ecf6c8d6aa716a75`.
It is byte-identical to the fully verified crop-only benchmark. The original
sphere settings, brightness 3.78 and frame 49 were restored afterward.

## Verification and rollout

- Full codec suite: 133 passed, 1 skipped.
- Full editor suite: 124 passed, 25 existing warnings.
- New tests cover compact mapping across chunks, partial final chunks, SH0/SH3,
  fully hidden scenes, invalid masks, source-presence validation, exact retained
  fields, audio copying and random access. The crop-only editor integration
  forbids PLY writing to prove the transport shortcut is used.
- Changed-file Ruff checks pass with the existing PLR0917 exclusion.
- Parent workspace `npm run check` passes. Its broad `npm test` discovery retains
  the existing 49 passes and 78 failures in unrelated JavaScript snapshots.

Restart to activate. Previous tags remain available; `v3.0.6` restores the
visibility-only export. `GSPLAY_FAST_EXPORT=0` retains its existing role of
disabling source reuse, which also disables this compact source path.
