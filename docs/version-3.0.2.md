# Version 3.0.2 - reliable full-sequence export

## Changes

Original Frames now supplies exact integer frame identities for GSAV sources.
The previous normalized-time round trip produced 61 fractional values in the
600-frame test (for example, 49.00000000000001), incorrectly disabling the
preserved-layout path. Custom time ranges retain their existing behavior.
Eligibility appears before preparation, with a reason when standard encoding
is required. Per-frame geometry, presence, timeline and FPS checks still decide
whether preserving the source arrangement is safe.

For preserved GSAV sequences, the encoder first collects exact repeated SH
vectors on the CPU. Float32 byte patterns are compared without rounding or
sampling. If every distinct vector fits the existing centroid limit, the
encoder stores that palette directly instead of clustering billions of repeated
values. Weighted scalar statistics retain each value's original frequency and
use the existing scalar quantizer. No SH degree or Gaussian-count reduction is
introduced. This does not make the lossy codec bit-exact.

The encoder avoids redundant full-sequence SH copies during range calculation,
constructs the palette before allocating GPU chunks, and releases processed
input references. When an exact palette does not fit, the ordinary clustering
path remains; the preserved path rejects an estimated GPU allocation larger
than its available-memory allowance with an actionable error. This is not a
general streaming-memory solution for arbitrary large PLY sequences.

Scene replacement now installs the new render closure under the existing viewer
lock before closing the old decoder. Decoder shutdown uses its close protocol;
on Windows a timed-out worker is terminated with its owned process tree, since
killing only a Python launcher can leave a child holding the output pipe open.
These address a closed-source render race and a shutdown hang risk found during
live switching. Subsequent tested GSAV/PLY switches completed without that error.

## Full-size local result

Windows, Python 3.12, RTX 2000 Ada 8 GB; existing dependencies unchanged.
The 600-frame source contains 339,536 Gaussians per frame, 30 FPS, SH3, and
20 chunks of 30 frames. The saved brightness, contrast, temperature and tint
edits were applied through the actual editor export functions.

| Measured stage | Elapsed from start |
| --- | ---: |
| Edited-frame preparation complete and source arrangement verified | 303.88 s |
| Repeated SH check starts after reading/validating prepared frames | 499.44 s |
| Exact 65,535-vector palette collected | 725.33 s |
| Final chunk encoded | 790.31 s |
| VP9 stage begins | 794.09 s |
| Container writing begins | 850.81 s |
| Export complete and validated | **869.88 s (14 min 30 s)** |

Input: 608,582,759 bytes. Output: 648,568,815 bytes (about 6.6% larger).
This is one observed local run with other diagnostic work on the machine,
not a controlled speedup comparison. The previous standard-path attempt was
cancelled, so there is no completed same-file baseline ratio. Preparation,
reading temporary PLYs and scanning SH still dominate; export is not instant.

## Verification

- All 600 decoded output frames retained frame count, FPS, SH3, chunk boundaries,
  Gaussian count and exact presence masks.
- 612 paired 512x512 renders compared edited decoded input against decoded output:
  one view per frame, plus two extra views at frames 0, 29, 30, 299, 300 and 599.
  Foreground RGB mean absolute error averaged 0.00042175 and was at most
  0.00050839 on a 0-1 scale (about 0.042% average and 0.051% maximum).
  The preselected acceptance threshold was 0.02. Paired renders at the first
  and last frames were also visually inspected. This is sampled camera coverage,
  not proof that every possible view is identical.
- The independent local GSAV.js/PlayCanvas viewer loaded the 20-second output,
  played across its timeline and looped, and accepted a midpoint seek; no browser
  console errors were reported during the check.
- Live editor: empty startup, GSAV upload, brightness edit, GSAV export,
  standard PLY export, PLY reimport and standard-path GSAV export succeeded.
  Folder-picker cancellation restored Save To Path immediately.
- Editor suite: **116 passed**, with 25 inherited boolean-return warnings.
  Codec suite: **115 passed, 1 skipped**. These include exact frame-selection,
  weighted-quantizer, palette, mask, edit, real codec round-trip, graceful worker
  exit and model replacement ordering regressions.
- Changed modules pass Ruff (the two pre-existing codec modules retain their
  existing PLR0917 exemption). No dependency versions changed.
- Parent workspace `npm run check`: passed (10 modules / 35 catalog rules).
  Its broad `npm test`: 49 passed, 78 failed in unrelated viewer snapshots and
  local browser audit scripts. That unrelated workspace suite is not green.

Private scene files, output images and detailed local logs are excluded from Git.
The measured full export and quality report are kept locally in the audit folder.

## Compatibility and rollback

The container remains GSAV v3. The application patch is v3.0.2 on `version-3`.
Use `v3.0.1` to return to the previous patch, `v3.0.0` for the original fast-export
release, or `pre-v3` for the older combined baseline. `main` is unchanged.
Stop the editor after saving edits before switching tags; run editor and codec
from the same checkout. No data migration or dependency upgrade is required.
