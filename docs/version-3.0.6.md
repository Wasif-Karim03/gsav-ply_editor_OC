# Version 3.0.6 — filtered GSAV export without temporal rematching

Filtering previously compacted each frame independently. This discarded source
row identities and selected the full temporal matching path, even when GPU
export was selected. Matching uses CPU nearest-neighbor searches; native xllvp9
also runs on CPU. Reordered geometry and SH labels compressed poorly.

## Behavior

Full-timeline GSAV exports at the original FPS now retain source rows when
edits consist of colors and filters. The export-only filter records visibility
and applies colors to the original arrays. GPU filtering uses the exact native
preview filter with explicit source indices in a separate scratch tensor.
CPU filtering uses the existing CPU mask computation. Source arrays are not
modified by this selection step.

The encoder keeps original geometry tiles, geometry ranges, compressed position
detail, and audio. It updates color tiles and the version-3 visibility tile.
Visibility must be boolean, match every original row, and never revive a row
absent from the source. An unexpected geometry change aborts this path rather
than exporting an unfiltered scene. Removed Gaussians remain stored but hidden;
this is visibility filtering, not physical removal of their data.

Verification also exposed clipping of bright SH0 coefficients to [-2, 4].
Preserved-row exports now derive color ranges from the actual edited values.
When the exact SH palette contains at most 256 distinct scalar values, its
codebook now stores those values directly instead of fitting a lossy histogram.
Larger palettes continue through the existing quantizer.

Transforms, opacity scaling, partial timelines, changed FPS, PLY sources and
PLY export retain their previous paths. Native xllvp9 remains mandatory. No
dependencies or container schema changed; filtered output requires a viewer
that supports the existing version-3 visibility mask.

## Measured 180-frame SH3 scene

The original upload was confirmed byte-identical to the benchmark source.
Observed settings: brightness 3.78, maximum opacity 0.98, sphere radius 1.1,
center (-0.26989809, 0.53084362, 0.24318331); other color controls and transforms
neutral. Export used GPU and the normal editor staging/codec bridge.

| File | Bytes | MB (decimal) | MiB |
| --- | ---: | ---: | ---: |
| Imported source | 40,651,825 | 40.65 | 38.77 |
| Previous filtered export | 71,017,786 | 71.02 | 67.73 |
| Updated filtered export | 41,140,793 | 41.14 | 39.23 |

The updated export is 42.1% smaller than the previous result and 1.2% above
the source. Its position section is byte-identical to the source (17,659,911
bytes); SH is 3,545,474 bytes versus 3,568,435; video is 19,933,632 bytes versus
19,421,703. The previous output's video alone occupied 41,189,138 bytes.

The full updated export took 84.34 seconds, including 34 seconds preparing
frames. The previous logged codec stage took approximately 7 minutes 14 seconds.
These timings come from separate runs with different background activity,
not a controlled hardware benchmark or a guarantee for other scenes.

Across all 180 frames, decoded positions, scales and rotations match the source
exactly. Visible opacity is exact. Visibility matches the actual GPU preview,
including filter boundary decisions. Edited SHN coefficients are exact; maximum
SH0-derived RGB error is 0.002100 and maximum per-frame mean error is 0.000794.
This is a numerical check, not a claim of lossless base-color compression or
a perceptual quality certification. Seek checks passed at frames 0, 29, 30, 90,
150 and 179. File-size savings depend on source layout and edit type.

## Validation and rollout

The restarted editor at port 6034 also completed a browser-triggered GPU export:
41,142,665 bytes for 180 frames. Its captured settings had sphere center Z=0
because the automated restoration had not committed that field; all other
settings matched the benchmark. All 180 frames passed geometry, visibility and
color checks against those captured settings (SHN exact; RGB maximum error
0.002100). Center Z was subsequently restored to 0.24318331 and confirmed in
the exported configuration. This browser check verifies the UI export wiring;
the original-setting comparison is the separate benchmark above.

- Full editor suite: 124 passed, 25 existing warnings.
- Full codec suite: 128 passed, 1 skipped.
- Tests cover exact filter row selection on CPU/CUDA, source immutability,
  filtered round trips, fully hidden frames, visibility validation, bright
  SH0/SH3, legacy/v3 sources, audio copying, and geometry-edit fallback.
- Changed-file Ruff lint uses the existing PLR0917 exclusion.
- Parent workspace `npm run check` passes. Its broad `npm test` discovery has
  the existing 49 passes / 78 failures in unrelated JavaScript snapshots.

Restart the editor to activate this release. `GSPLAY_FAST_EXPORT=0` disables
source reuse. Tag `v3.0.5` remains available for rollback; previous tags and
the main branch are preserved.
