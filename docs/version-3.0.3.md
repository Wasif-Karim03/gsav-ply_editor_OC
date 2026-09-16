# Version 3.0.3 — native xllvp9 exports

GSAV exports now require the user's xllvp9 native encoder. The application
checks `_libvpx_ref` availability before the codec worker loads staged PLYs,
and again at the video boundary. Missing native support produces an actionable
error; neither FFmpeg encoding fallback is allowed. Progress names the native
encoder. FFmpeg is still used for decoding and audio.

The pinned xllvp9 commit is `fc0ad99972e96af08fffd62489dd1fa926323c21`.
Its implementation calls its bundled libvpx directly, without an FFmpeg encoder
process. See [build fixes and installation](native-xllvp9.md). The tested wheel
SHA-256 is `fcbc5d555f79d42a1c0136384622ad8ce4374559eef7c00e2d31b03926aa27f3`.
Private sources, the compiled wheel and customer scenes are not committed.

## Measurements (September 16, 2026)

Same 220-frame, 33,056-Gaussian scene, 910 × 546 atlases, 30 FPS and GOP 30.
Native xllvp9 uses its default `fast` lossless preset. Timings are individual
local measurements, not a cross-machine performance guarantee.

| Case | Prior FFmpeg video bytes | Native video bytes | Native video seconds |
| --- | ---: | ---: | ---: |
| Original atlas data | 29,819,447 | 25,859,354 | 5.79 |
| Rebuilt neutral atlas data | 30,159,361 | 26,212,392 | 5.81 |
| Edited atlas data | 30,200,205 | 26,238,404 | 5.91 |

Original source file: **45,939,286 bytes (43.81 MiB)**.
Previous edited export: **50,277,611 bytes (47.95 MiB)**.
Replacing only its video with native output: **46,315,810 bytes (44.17 MiB)**,
7.88% smaller and 0.82% above the original file. Neutral rebuilt file with native
video: 46,289,778 bytes (44.15 MiB). This sample has no stored higher-order SH.

All 220 decoded atlas frames are byte-exact against the same pre-encoding
atlases in each case; fixed keyframes occur every 30 frames. Native-container
repacks preserve all 220 low-position sections and ranges and pass seeks at
frames 0, 29, 30, 90, 210 and 219. The test prohibits FFmpeg encoder fallback.
This isolates compression from upstream Gaussian quantization and SH clustering.

A separate complete editor-bridge import/export using the preserved edited
file finished in **31.41 seconds**, produced **46,315,123 bytes**, decoded all
220 frames and passed those six seeks. It performed another full round trip;
its bytes are not expected to equal the video-only repack. The original
temporary upload had already been removed, so the complete bridge check used
the preserved edited file; original-atlas comparison used the earlier captured
raw atlas data. No new browser-driven export was claimed by this check.

## Verification

- Codec: `python -m pytest gsav-ply_editor_OC/gs-encoder/tests -q`:
  **117 passed, 1 skipped**, including SH3, presence masks, row preservation,
  native lossless luma, GOP boundaries and missing-native error paths.
- Editor: `python -m pytest gsav-ply_editor_OC/gsplay/tests -q`:
  **116 passed**, 25 warnings. Both use their documented separate environments.
- Ruff formatting and changed-file lint passed with the pre-existing
  `PLR0917` positional-argument warning excluded.
- Parent resume workspace `npm run check` passed. Its broad `npm test` also
  discovers unrelated copied JavaScript viewer snapshots and audit scripts;
  that check remains failing, as before this change. It is not a codec test.

The native path changes compression, not GSAV quantization policy. Exact atlas
preservation does not make the entire Gaussian round trip lossless. Unedited
source passthrough and reuse of unchanged compressed sections remain separate
future work. Large SH3 scenes have not been re-benchmarked with native xllvp9.

## Compatibility and rollback

Container layout, SH sidecars and decoded viewer interfaces are unchanged.
GSAV export now needs the installed native wheel; PLY export and GSAV import
do not. Install it into the interpreter selected by `GSPLAY_CODEC_PYTHON`.
New codec subprocesses see the installed module without discarding the current
editor scene. No server restart was required for this dependency installation.
Earlier tags, including `v3.0.2`, preserve the FFmpeg-era implementation for
rollback. The prior `main` baseline is unchanged.
