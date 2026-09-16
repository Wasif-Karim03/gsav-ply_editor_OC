# Version 3.0.5 — preserve geometry during color-only GSAV export

The previous export path retained Gaussian row identities but rebuilt all
quantized attributes. On the audited 180-frame SH3 scene, the source position
detail used 14 distinct low-byte values in the first frame; reconstruction
introduced many more values and enlarged that section. Scale and opacity
normalization also changed geometry despite a color-only edit.

## Behavior

For a full-timeline GSAV export at the source FPS, the editor verifies every
frame's positions, scales, rotations, opacity and row count before selecting
the new path. The codec then:

1. Keeps the original geometry, opacity and presence/padding atlas tiles.
2. Replaces only the three SH0 color tiles with edited values.
3. Preserves original geometry quantization ranges and copies the compressed
   low-position section byte-for-byte.
4. Encodes the resulting atlas with native lossless xllvp9 and rebuilds edited
   SH color tables through the existing codec.
5. Copies embedded audio without transcoding and writes updated container
   offsets and indexes.

Geometry changes, opacity changes, filtering, changed row counts, partial
exports and changed FPS continue through the existing full export path.
`GSPLAY_FAST_EXPORT=0` disables reuse. Unsupported source layouts fail clearly.
The internal `geometry_source` codec argument requires the caller to verify
color-only edits and stable source rows; it is not a general geometry override.

Native xllvp9 remains required. No dependency, container schema or PLY export
behavior changed. Legacy v1 and current v3 inputs are tested; output remains v3
and retains the source's explicit-mask or legacy-opacity visibility semantics.
Color quantization and SH clustering remain lossy. Unedited-file passthrough
and direct compressed SH-table transformations are not implemented here.

## Controlled 180-frame comparison

The test took colors from the user's previous edited export and geometry from
the original source, then exported through the normal editor bridge.

| File | Bytes | MiB |
| --- | ---: | ---: |
| Original | 40,651,825 | 38.77 |
| Previous edited export | 47,465,440 | 45.27 |
| Geometry-preserving test export | 40,762,242 | 38.87 |

The new file is **14.1% smaller than the previous export** and **0.27% above
the source size**. This controlled check reads two models and took 116 seconds;
it is not a like-for-like UI speed benchmark.

All 180 frames have exactly equal decoded positions, scales, rotations,
opacity and visibility to the original. The compressed position-detail bytes
also match exactly. All frames retain color edits; seeks at 0, 29, 30, 90, 150
and 179 pass. Against the prior edited export, maximum per-frame mean absolute
SH0 coefficient error is 0.002087 and SHN error is 0.000559; worst individual
coefficient errors are 0.029733 and 0.102275 respectively. These are additional
color recompression differences, not exact color preservation or a perceptual
quality guarantee.

## Live browser check

After restarting port 6034, the original scene and observed user settings were
restored: brightness 2.4870500564575195, contrast 0.5, temperature slider 0.625,
tint slider 0.5650818943977356, with remaining color controls neutral. Apply
was clicked and all 180 frames were exported from the browser with CPU selected.
The result is **40,772,397 bytes (38.88 MiB)**, about **0.30% above the original**
and **14.10% below the previous export**. All decoded geometry fields and masks
match the original exactly across all frames; compressed low-position bytes
match exactly; all frames have edited colors; the six seek checks pass.
SHN coefficients match the previous user export exactly. SH0 maximum absolute
coefficient difference is 0.051372 and maximum per-frame mean absolute
difference is 0.012686 (RGB errors are smaller by the SH0 constant, 0.282095).
This does not assert bit-exact color compression or perceptually identical
rendering. The user scene and restored settings remain open in the editor.

## Tests and rollout

- Full codec suite: 122 passed, 1 skipped. Subsequently added audio-copy
  regression passed together with all six source-geometry tests.
- Full editor suite: 120 passed, 25 warnings; real codec integration now
  requires exact geometry equality and exercises the geometry-edit fallback.
- New codec cases cover SH0/SH3, legacy/v3 masks, snapped position detail,
  unsafe mode rejection, timeline mismatch, and byte-exact audio copying.
- Changed-file Ruff lint passes with the pre-existing `PLR0917` exclusion.
- Parent workspace syntax checks pass; unrelated JavaScript snapshot/audit
  discovery retains 49 passes and 78 failures.

Restart the editor to load the new export orchestration. Previous tags remain
available; use `v3.0.4` for rollback or disable the fast path temporarily.
