# GSAV / PLY Editor - Version 3

Application release **3.0.0** adds guarded fast GSAV export and live export progress.
See [release notes and benchmark evidence](docs/version-3.md). The `pre-v3` tag
and unchanged `main` branch preserve the previous baseline; this release is on
`version-3` with tag `v3.0.0`. The codec format remains GSAV v3.

Combined GSPlay editor and gs-encoder **v3**, including the local integration,
color/export fixes, regression tests, and September 15, 2026 audit.

## What it does

- Opens GSAV as decoded Gaussian arrays directly, without converting input to PLY.
- Loads existing PLY sequences through GSPlay.
- Provides scene, color, opacity and transform editing.
- Exports edited sequences as standard PLY or GSAV, retaining stored SH degree.
- Opens a native Windows folder picker for local export destinations.
- Starts empty unless a scene is explicitly supplied on the command line.

```text
GSAV -> v3 decoder -> raw Gaussian arrays -> GSPlay editor
PLY -------------------------------^           |
                                      edited arrays
                                            |
                                  temporary PLY transport
                                      /             \
                               standard PLY       v3 GSAV
```

GSAV compression is lossy; retaining SH3 does not mean a bit-exact round trip.
Legacy files without stored higher-order SH cannot recover missing coefficients.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `gsplay/` | Viewer, editor, direct GSAV source, export bridge, UI and regression tests |
| `gs-encoder/` | Patched v3 codec, SH sidecar support and FFmpeg VP9 fallback |
| `docs/source-manifest.json` | Upstream commits and SHA-256 hashes of the imported working files |
| `docs/*-environment.json` | Package/version inventory from the tested environments |

Both source trees are normal tracked directories, not submodules. Optional
native third-party dependencies retain their upstream submodule references.
The two projects need **separate Python environments**: the tested editor uses
gsply 0.2.13, while the codec uses gsply 0.4.2.

## Setup and launch

The verified machine used Windows, Python 3.12, an NVIDIA RTX 2000 Ada GPU,
CUDA 12.8, Visual Studio 2022 C++ build tools, and FFmpeg/ffprobe on PATH.
This repository is a source baseline, not a bundled standalone executable.

1. Clone normally; do not require recursive submodule initialization for the
   tested FFmpeg fallback. The upstream private xllvp9/xllav1 URLs may be
   inaccessible. They are not required by that fallback.
2. Set up GSPlay using its [installation documentation](gsplay/README.md).
   Its installer manages CUDA/MSVC and gsplat. Review its actions before use.
3. Set up the separate codec `.venv` using the pinned commands in
   [the integration guide](gsplay/docs/gsav-integration.md#tested-windows-setup).
   Avoid a default codec `uv sync`: upstream optional native dependencies are
   unavailable in the tested setup. Keep the documented `--no-deps` procedure.
4. Consult the environment inventories when reproducing the audited package
   versions. They are records, not portable lockfiles. A fresh installation of
   this combined checkout has not been verified; the existing environments were
   used for the audit.

Once the two environments are configured, run from `gsplay/`:

```powershell
uv run --no-sync gsplay view --host 127.0.0.1 --port 6030
```

Open **http://127.0.0.1:6030/**. Upload GSAV, or enter a PLY folder in Data Path.
Use **Convert -> PLY / GSAV -> Save To Path** to choose an export destination.
The picker opens on the machine running the server, not a remote browser device.

The bridge automatically finds `gs-encoder/.venv`. For a different environment,
set `GSPLAY_CODEC_PYTHON` to its Python executable. Do not copy `.venv` directories
between checkouts; editable installs and launchers can contain absolute paths.

## Verification and known limits

See [the audit report](gsplay/docs/audit-2026-09-15.md) and
[integration details](gsplay/docs/gsav-integration.md).
Tests live within each project and must run with that project's environment:

```powershell
cd gsplay
uv run --no-sync python -m pytest -q
cd ../gs-encoder
uv run --no-sync python -m pytest -q
```

Version 3 reports 105 GSPlay passes (25 older tests have unreliable boolean
return conventions), 107 encoder passes and one skip. Browser upload, color
editing, fast export and progress were checked. All 180 frames of the large
benchmark were compared with the edited reference; see release evidence for
exact coverage and quality limits. GPU/large-file,
audio/static-track, and native-dialog coverage limits are recorded in the report.
Local datasets, exported scenes, screenshots, environments and raw audit logs
are intentionally excluded from this source repository.

## Provenance and licensing

- GSPlay: https://github.com/OpsiClear-4DGS/gsplay.git at
  `37997a4dae8df4f2dfa9d646af4ea20c8deafb9a`, plus the imported working changes.
- gs-encoder v3: https://github.com/OpsiClear/gs-encoder.git at
  `906de3f7cee2a514e53179feb76d54439c445877`, plus the imported working changes.

Original source notices are retained. GSPlay includes its
[AGPL-3.0 license](gsplay/LICENSE). This combined repository does not relicense
the encoder or third-party components. The initial commit captures the working
source baseline; upstream histories remain in the repositories linked above.
