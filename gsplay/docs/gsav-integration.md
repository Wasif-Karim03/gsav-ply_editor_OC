# GSAV editing with gs-encoder v3

## Use

Start an empty editor with `uv run --no-sync gsplay view`. A file is loaded on
startup only when explicitly supplied: `uv run --no-sync gsplay view path/to/clip.gsav`.
The local server on port 6019 is started without a file. Browser refreshes share
the current server session; they do not automatically clear an active edit.
Successful loads show the filename, replace the cached render, and fit the camera
to the new scene. An empty scene publishes a blank canvas rather than retaining
the browser's previous image.
Use **Upload GSAV** in the data loader, or enter a local `.gsav` in Data Path.
In Convert, select the export format and click **Save To Path…** to open the
native Windows folder picker. **Selected Export Path** shows the full destination
and remains editable. GSAV uses a fresh `scene.gsav` filename; PLY uses a fresh
`gsplay-export` subfolder inside the selected folder. Numbered names avoid
overwriting prior outputs. Cancel leaves the destination unchanged. The picker
opens on the Windows computer running GSPlay; remote-device folder selection
is not implemented by this local desktop dialog.
Decoded frames enter the editor directly, without intermediate PLY files. In **Convert**, choose **PLY** or
**GSAV**, set the output path and scope, and export. GSAV accepts a new `.gsav`
filename or a folder (writes `scene.gsav`). Existing GSAV files are never replaced.
Use standard PLY when exporting SH coefficients; the existing Compressed PLY
format is a separate compact format.

GSAV export applies a snapshot of the editor settings to the selected frames.
The source data remains unchanged. Loading another scene is blocked during GSAV
export because the existing loader shuts down the previous model.

## Architecture and compatibility

The conversion path is:

```text
GSAV input -> persistent v3 decoder -> raw arrays through a local binary pipe
           -> GSPlay direct GSAV source -> editing
           -> snapshot current edits -> edited Gaussian arrays
              -> temporary PLY transport to the isolated codec environment
                 -> PLY: codec reads raw GSData and writes standard PLY
                 -> GSAV: v3 SequenceEncoder compresses the edited sequence
```

Existing PLY input enters the editor directly. It does not need an extra lossy
GSAV encode/decode before editing. The raw data stage is in-memory Gaussian
arrays, not a separate undocumented binary format. Both standard PLY and GSAV
exports use the same edit/staging routine and isolated codec bridge. PLY output
requires a new folder, and GSAV requires a new filename; neither replaces an
existing destination. The older Compressed PLY option remains available through
its existing exporter.

The codec checkout is **gs-encoder v3**, tested at `906de3f`.
`src/infrastructure/gsav.py` runs `gsav_worker.py` using a separate Python
environment. GSPlay retains gsply 0.2.13; the codec uses gsply 0.4.2.
Do not install the codec dependencies into the viewer environment.

Direct import uses `gsav_stream.py`, `gsav_stream_worker.py` and `GsavModel`.
The worker decodes video blocks on demand (up to 32 frames, targeting 128 MiB
retained atlas data). The editor keeps up to four raw frames, targeting 128 MiB;
a single larger frame is allowed. NumPy array messages use a length-prefixed
pipe with pickle disabled. No frame files are written during import. Temporary
storage is limited to logs, an uploaded source file, and optional audio.
FFmpeg may briefly decode extra frames back to the nearest codec keyframe, so
cache budgets are not strict peak-memory limits. A worker is closed on model
replacement, unload or normal process exit. The original PLY decode command
remains available for explicit conversion and regression comparisons.

The worker reconstructs SH coefficients through v3's SH sidecar decoder and
passes those arrays to the viewer. Export uses the existing edit manager and PLY writer,
detects the edited SH degree, then explicitly encodes that degree with v3.
SH3 has 15 RGB higher-order coefficients per Gaussian. Degree is preserved;
Color quantization and SH clustering mean a round trip is **not bit-exact**.
For verified color-only full-timeline GSAV exports, version 3.0.5 preserves
geometry tiles/ranges and compressed position detail exactly; see
[geometry preservation](../../docs/version-3.0.5.md).
Unstructured identity mode supports changing Gaussian counts after filtering.

GSAV export requires native xllvp9 at commit
`fc0ad99972e96af08fffd62489dd1fa926323c21`, including its compiled
`_libvpx_ref` extension. Both the application's and xllvp9's FFmpeg encoding
fallbacks are disallowed. FFmpeg still decodes video and handles audio.
See [native build and installation](../../docs/native-xllvp9.md).

## Tested Windows setup

Python 3.12, NVIDIA RTX 2000 Ada, CUDA 12.8, VS 2022 C++ tools, FFmpeg with
VP9 decoding and ffprobe on PATH. Keep the two repositories beside one another.
From the patched `gs-encoder` v3 checkout:

```powershell
uv venv --python 3.12
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy==2.3.5 gsply==0.4.2 faiss-cpu==1.15.0 ffmpeg-python==0.2.0 tyro==0.9.35 tqdm==4.70.1 zstandard==0.25.0 numba==0.62.1 triton-windows==3.5.1.post23
uv pip install --no-deps flash-kmeans==0.2.0 -e .
uv pip install pytest==9.1.1 ruff==0.16.7
```

These pin the directly installed dependencies of the tested local environment;
they are not a cross-platform lockfile. `--no-deps` deliberately avoids the
unused optional codec backends. Install the native xllvp9 wheel separately
using the linked guide. Use `uv run --no-sync`
to retain this environment. The default adapter finds the sibling codec's
`.venv`; alternatively set `GSPLAY_CODEC_PYTHON` to its Python executable.
CPU codec operation is tested; the GSPlay viewer here still uses CUDA.

## Limits

- Local editor workflow, not a hardened public upload service. Browser uploads
  are checked against 1 GiB after receipt; this is not a transport memory cap.
- Direct import checks: 10,000 frames, 5 million Gaussians, 2 GiB per IPC message.
  Explicit whole-clip PLY decoding retains its estimated 8 GiB limit.
- Direct decoder requests time out after 120 seconds. Export retains a one-hour
  codec timeout and uses temporary PLY staging; large exports need disk space.
- Embedded GSST static tracks are rejected rather than silently discarded.
- Audio is retained for Original Frames at the imported FPS. Partial or
  retimed exports with audio are rejected. Integer FPS from 1 to 240 is required.
- Uploaded source files remain temporary for the viewer lifetime.
- Export writes to the server's local filesystem, as existing PLY export does.

## Verification (2026-09-15)

- Codec suite: 96 passed, 1 skipped, including exact FFmpeg luma/GOP test.
- GSPlay integration: 9 passed. Tests cover direct decode equivalence against PLY, random
  seeking, frame-cache isolation and eviction, direct-source edits and both
  exports, no PLY frame files on import, and decoder process cleanup.
- Elly direct GPU check: 220 frames at 30 FPS, first frame ready in approximately
  1.5 seconds in the measured local run; final-frame seek succeeded. This is a
  smoke measurement, not a cross-machine performance guarantee.
- Actual edit manager translation, GPU-backed
  GaussianData conversion, SH3 export/decode/re-encode, loader/FPS, invalid
  headers, codec configuration errors, existing-file protection and failed
  export cleanup are covered.
- Shared codec export regression verifies standard PLY positions match the
  edits and SH3 coefficients match exactly; GSAV is compared within compression
  tolerance. PLY codec failure and existing-folder protection are also tested.
- Existing collectable GSPlay tests: 35 passed, 1 skipped. Full collection is
  blocked by two upstream tests importing the absent `src.viewer` package.
- Parent workspace `npm run check` passed. Parent `npm test` fails on unrelated
  TypeScript snapshot tests beneath `outputs/gsavjs-audit-2026-09-10`.
- A two-frame subset of the supplied soccer PLYs encoded with SH3 and imported
  through the browser upload button. Original source files were not changed.
- Browser export completed in both GSAV (SH3) and standard PLY formats.
- Legacy `elly (1).gsav` decoded: 220 frames at 30 FPS. Its header advertises
  SH3 but its SH flag and payload offset are zero. Such legacy files import as
  SH0 with a notice; no missing coefficients are fabricated. Files declaring
  a stored SH payload remain subject to strict SH reconstruction validation.

Run focused checks from GSPlay:

```powershell
uv run --no-sync pytest -q tests/test_gsav_integration.py
uv run --no-sync ruff check src/infrastructure/gsav.py src/infrastructure/gsav_worker.py src/gsplay/gsav_controls.py tests/test_gsav_integration.py
```

Run codec checks from gs-encoder: `uv run --no-sync pytest -q`.
To undo the integration, revert the GSPlay changes and the codec video-writer
patch; generated data and the isolated environments are ignored local files.

## Color export audit (2026-09-15)

Two confirmed defects were fixed:

1. `GaussianData` compared gsply `DataFormat` enums against legacy strings and
   wrote those strings back. RGB SH0 became mislabeled as SH coefficients, so
   PLY staging skipped `(RGB - 0.5) / SH_C0`. It now reads native public flags
   and restores native enums while retaining SH ordering metadata. Scale and
   opacity activation flags are also preserved.
2. gsply cloning can produce interleaved color tensors. gsmod RGB Triton kernels
   assume contiguous color storage and read incorrect values after the first
   Gaussian. The GPU color processor now clones once, makes color arrays
   contiguous, invalidates the packed buffer, then applies edits in place to
   that private copy.

The path remains GSAV -> v3 raw arrays -> activated editor data -> edits ->
standard PLY staging -> v3 GSAV encoding (or standard PLY output). RGB-only
rendering uses RGB; SH3 rendering and PLY/GSAV storage use SH coefficients.
No dependency versions or codec quantization settings were changed in this fix.

Validation:

- 38 focused tests passed: 24 color/format CPU+CUDA tests plus existing GSAV,
  scene-loading and folder-picker tests. Color tests cover PLY/linear flags,
  RGB/SH flags, cached tensor conversion, no-edit and brightness exports, and SH3.
- Current uploaded 30-frame clip, first frame (33,056 Gaussians): old export
  mean RGB absolute error 0.29807; corrected export 0.001643. RGB scale is 0..1.
- Brightness 0.7, actual first-frame GSAV encode/decode: RGB clip mean color
  error 0.001332; SH3 soccer clip 0.001039; mean higher SH coefficient error 0.00856.
- Real-file comparisons use nearest positions after codec reordering. Matches
  are not all unique, so these are diagnostic estimates, not exact per-identity
  or rendered-image quality scores. Exact PLY coefficient tests complement them.
- Audit outputs and JSON reports are under ignored `data/color-audit/`.
- Ruff and parent `npm run check` pass. Parent `npm test` still fails on the
  unrelated TypeScript snapshots under `outputs/gsavjs-audit-2026-09-10`.

Independent visual verification subsequently passed in the available OpsiClear
PlayCanvas/WebGL2 viewer build: original, old and corrected exports were compared
at frames 0, 15 and 29; the corrected 30-frame clip played to completion without
JavaScript errors. Foreground screenshot RGB error dropped from 0.175�0.197
to 0.00303�0.00332. Brightness-edited RGB and SH3 samples also rendered. Evidence
is in the parent workspace outputs/gsavjs-audit-2026-09-10/visual-evidence.
The user�s exact viewer remains unidentified; other backends were not verified.
Existing bad exports must be regenerated from the source.
