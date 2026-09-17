# Version 3.0.8 — chunk-based crossing controls

## User workflow

In **Edit → Filter**, position a spatial boundary, then choose **Crossing rows
(per chunk)**:

- **Show only inside** (default): the existing per-frame crop.
- **Keep crossing rows**: retain a source storage row throughout each chunk in
  which it is inside at least once, only in frames where it originally exists.
- **Remove crossing rows**: retain a row only if it never goes outside while
  present in that chunk.

Click **Analyze & Apply** for either new choice. Progress is shown while a separate
decoder scans the timeline. Scrub/play to inspect the result, then export PLY or
GSAV. Analysis is session-only and is not serialized with viewer settings.
Changing the source, boundary, opacity/scale limits or selected policy makes a
nonmatching plan inactive. The panel says **Not applied**, the preview returns to
ordinary cropping, and export rejects the unapplied choice. Reanalyze before
exporting. Returning to identical settings can reuse the matching plan.

## Scope and correctness

This is explicitly a **storage-row policy within each encoded chunk**, not
physical-object tracking across the animation. Decisions may change at chunk
boundaries; rows themselves are not guaranteed physical identities. This does
not implement the original whole-animation tracking proposal.

Spatial tests use Gaussian centers and the existing filter implementation.
Absent frames do not count as outside; absent samples cannot be revived.
Opacity and scale restrictions still apply per frame after the crossing decision.
Inverted filters and non-GSAV sources are rejected for the new choices.
Compressed PLY export is rejected while a crossing choice is selected; regular
PLY and GSAV are supported.

Packed boolean masks are shared by preview and export. The edit pipeline accepts
an explicit per-frame mask before transforms/colors. Eligible GSAV exports
preserve source rows during staging and use the 3.0.7 compaction path; crop-only
exports still skip PLY staging and SH fitting. Edited arrays are never used as
identity tags. Native xllvp9 remains the encoder.

Analysis holds one chunk of masks and retains packed bits for the timeline,
with explicit memory limits. The source is decoded through an independent
bounded-cache model, closed on success or failure. Scene operations are mutually
exclusive. Settings are checked again before a completed plan is published.
Export freezes the plan and settings so later UI changes cannot alter an export.

Keeping crossings can retain more samples than ordinary cropping, so it may
produce a larger output than that crop. No unconditional size or speed guarantee
is made. Playback and exports reuse the masks without repeating analysis.

## Verification

- Full editor suite: `python -m pytest gsplay/tests` — **132 passed**, 25 existing
  warnings. CPU/GPU source selection and native-codec integration included.
- Final targeted crossing and Apply suite verifies CPU slicing, empty masks,
  absent rows, partial chunks, stale settings, PLY and GSAV preview/export parity.
- Ruff changed Python files with `--ignore PLR0917` and `git diff --check` pass.
- Root `npm run check`: 10 modules and 35 catalog rules pass. Root `npm test`:
  **49 passed, 78 failed** in the existing unrelated archived JavaScript snapshot
  discovery; not a passing global gate.
- Live 180-frame source: restored the user's sphere at
  (-0.2698980868, 1.1451966763, -0.1076771021), radius 1.7, neutral color edits.
  Frame 154 showed roughly 55.1K rows with ordinary crop, 56.4K with Keep, and
  50.4K with Remove. Both analysis choices completed in the UI.
- Live Keep GSAV output: **29,613,326 bytes**, 56,496 storage rows, versus
  **40,651,825 bytes** in the source. An independent decoder check recomputed the
  chunk union directly from source positions and presence. All 180 frames matched
  exactly for visible positions, scales, rotations, opacity, SH0 and SHN. Frame
  154 contained exactly 56,436 visible rows.
- Moving radius to 1.8 changed status to Not applied and correctly blocked export;
  restoring 1.7 made the existing matching plan available again.

No dependencies or GSAV container layout changed. Older defaults retain ordinary
cropping. Roll back to tag `v3.0.7` for the previous UI; earlier tags and `main`
remain preserved.
