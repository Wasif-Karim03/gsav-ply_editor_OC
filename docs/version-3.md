# Version 3.0.0 - guarded fast GSAV export

## What changed

Full-sequence GSAV exports can retain the imported Gaussian arrangement instead
of rebuilding temporal correspondence. This eliminates matching and sorting in
the new codec `preserve` mode. Original frame/chunk structure and explicit masks
are retained; no permanent physical identity is inferred from a row number.

The exporter checks every frame before choosing this path. It requires:

- A directly loaded GSAV with regular source chunks and explicit presence masks.
- All original frames in order, at the imported frame rate.
- Unchanged row counts, positions, scales, rotations and opacity after editing.
- Presence masks consistent with the prepared PLY data.

Color-only and no-edit exports can qualify. Geometry/opacity edits, filtering
that changes rows, partial/retimed exports, PLY sources or missing metadata use
the ordinary matching path. A codec-side contract check rejects corrupted layout
metadata rather than silently producing an incorrect export.

No compression precision, SH degree, pruning or snapping quality settings were
relaxed. The codec still recompresses SH and quantizes attributes; this is not a
lossless source-payload preservation feature. Temporary PLY transport remains.

The UI now reports preparation, original-arrangement or matching mode, chunk
encoding, SH compression, VP9 encoding, validation and completion. Encoder elapsed
time refreshes during long stages. SH compression has stage status, not an
invented percentage. Advisory progress writes tolerate Windows file-sharing races.

## Measurements on the existing hardware

Windows / RTX 2000 Ada laptop GPU (8 GiB VRAM) / approximately 64 GiB system RAM.
Single runs; these are measured cases, not a general performance guarantee.

| Case | Result |
| --- | --- |
| Full 180-frame SH3 source, approximately 78,124 points/frame, brightness 0.7 | 347.30 s end-to-end; 47,253,245 output bytes |
| Full preparation | About 37 s including source startup; codec quantization ranges started at 49.91 s |
| Six chunks, original arrangement | About 3.1 s inside the codec |
| SH compression | Approximately 285 s; now the dominant stage |
| Historical same-source matching run | Chunk log: 24 min 15 s; worker launch to completion: about 28 min 43 s |
| Controlled two-frame SH3 sample, ordinary matching | 10.234 s; 1,070,930 bytes |
| Same two-frame source/edit/device, fast path | 8.688 s; 1,117,943 bytes (about 4.4% larger) |

The historical large run used different edit settings and its launch timing
excludes earlier preparation. It is not a controlled end-to-end speedup ratio.
The small A/B shows why skipping matching can trade some compression efficiency
for speed, and why small-file speedups should not be extrapolated.

During the large SH stage, sampled GPU memory reached about 7,922 MiB. One process
working-set sample reached about 44.7 GB. These are samples, not instrumented peak
measurements. SH clustering remains memory intensive; this release does not
establish support for the same workload on lower-memory machines.

## Quality and regression evidence

- Full source and output: 180 frames, 30 FPS, SH3, matching source chunk boundaries.
- Explicit visibility masks matched exactly for every frame.
- All 180 frames rendered against the brightness-edited reference at 512 x 512.
  Six sample/boundary frames also used two additional camera views (192 pairs).
- Mean foreground RGB absolute error: 0.001287 on a 0..1 scale. Maximum per-view
  foreground mean error: 0.001566, below the pre-check threshold of 0.02.
- Per-frame geometry and SH coefficient errors were recorded; synthetic tests
  assert row-aligned geometry tolerances and mask lifetimes, including hidden
  frames and reused slots. Representative paired screenshots were inspected.
- Browser: empty startup, SH3 upload, brightness edit, GSAV fast export and live
  status all passed without JavaScript page errors.
- Independent OpsiClear viewer (WebGL2): the full 180-frame export loaded,
  sought to frames 0, 29, 30, 90 and 179, and played to the final frame without
  JavaScript errors. The local verification server supplied HTTP byte ranges.
- GSPlay: 105 reported passes, with the same 25 inherited boolean-return warnings.
  Those 25 are not reliable assertion-based checks and should not be counted as
  such. Codec: 107 passes, one optional renderer skip.
- Ruff passes on changed GSPlay files and new codec tests. The codec sequence
  file retains one pre-existing PLR0917 complaint about a 13-argument method.
- Parent workspace `npm run check` passes; `npm test` still reports 49 passes and
  76 failures from unrelated TypeScript viewer snapshots. It is not green.

This release does not claim bit-exact color, original encoded-geometry reuse,
coverage of every viewer, or exhaustive performance coverage. Arbitrary GSAV
files and color operations can have different clustering time and error.

## Reproduce and opt out

From a configured `gsplay/` environment, use a NEW output filename for each run:

```powershell
uv run --no-sync python -m scripts.benchmark_gsav_export INPUT.gsav fast.gsav
uv run --no-sync python -m scripts.benchmark_gsav_export INPUT.gsav standard.gsav --standard
```

Each run writes a sibling `.benchmark.json` with stage observations, total time,
settings and output size. Use `--device cpu` or `--brightness VALUE` as needed.
The codec's SH implementation can still use CUDA internally even for CPU export.

Set `GSPLAY_FAST_EXPORT=0` before launching GSPlay to force ordinary exports.
Neither export path overwrites an existing output. No dependency versions changed.
Raw source scenes, test exports and screenshots stay outside Git. Summarized
measurements are included in `version-3-benchmark.json`.

## Versioning and rollback

- `version-3` branch and `v3.0.0` tag: this release.
- `pre-v3` tag: original combined baseline, commit `58855ad`.
- `main` remains on that baseline; it was not force-pushed or replaced.

To use the old code, stop the server after any export finishes, switch a clean
checkout to `pre-v3`, and restart using environments installed for that checkout.
Do not expect a browser refresh or a Git checkout alone to replace code already
loaded in a running Python process. Keep the viewer and codec sources on the
same release; editable installations must point to the intended checkout.
