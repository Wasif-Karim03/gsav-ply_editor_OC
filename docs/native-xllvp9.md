# Native xllvp9 setup (Windows / Python 3.12)

GSAV exports require the compiled `_libvpx_ref` extension from
`https://github.com/Opsiclear-internal/xllvp9.git`, pinned at
`fc0ad99972e96af08fffd62489dd1fa926323c21`. Repository access is required.
Installing only its Python source is insufficient. The editor rejects that
configuration instead of silently encoding through FFmpeg. FFmpeg remains a
decoder and audio dependency. xllvp9 itself wraps its bundled libvpx directly.

## Reproduce the tested build

Use a short checkout path without spaces, for example `C:/build/xllvp9`.
Use Python 3.12 x64, Visual Studio 2022 C++ tools, CMake 3.31.6, Git Bash/Perl,
GNU Make 4.4.1 and NASM 3.02. Build with nanobind 2.12.0 and
scikit-build-core 0.12.2 from the repository's locked development environment.
Run `uv sync --locked --no-install-project --group dev --python 3.12` there.

Two source packaging issues were found in this pinned commit:

1. `reference/libvpx_lossless/build` is absent. Clone the official
   `https://github.com/webmproject/libvpx.git` at tag `v1.16.0`, verify commit
   `1024874c5919305883187e2953de8fcb4c3d7fa6`, and copy **only its `build`
   directory** into that missing location. This matches `reference/README.md`.
2. In `src/cpp/libvpx_ref.cpp`, replace `VP9E_SET_CQ_LEVEL` with
   `VP8E_SET_CQ_LEVEL` (including the diagnostic string). The bundled header
   exposes the latter control for both VP8 and VP9. This fixes compilation of
   the lossy branch; GSAV continues to use lossless encoding.

Neither fix has been pushed to the upstream private repository. Keep them with
the build until upstream incorporates them. No private source is vendored here.

In a VS developer shell, configure and build from the checkout:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" `
  -DPython_EXECUTABLE=C:/build/xllvp9/.venv/Scripts/python.exe `
  -DVP9_LOSSLESS_LIBVPX_BASH_EXECUTABLE=C:/path/to/git/bin/bash.exe `
  -DVP9_LOSSLESS_LIBVPX_MAKE_EXECUTABLE=C:/path/to/make.exe `
  -DVP9_LOSSLESS_LIBVPX_NASM_EXECUTABLE=C:/path/to/nasm.exe
cmake --build build --config Release --parallel 4
```

Use real absolute tool paths. Make needs its MSYS runtime on PATH; NASM needs
its DLL dependencies accessible even when the build initializes the VS shell.
The tested MSYS2 NASM needed `zlib1.dll` next to its executable. Ensure CMake is
on PATH for the wheel step. Package with native support **enabled**, reusing
the library just built:

```powershell
$env:CMAKE_GENERATOR = 'Visual Studio 17 2022'
uv build --wheel --no-build-isolation --python .venv/Scripts/python.exe `
  --config-setting cmake.define.VP9_LOSSLESS_ENABLE_NATIVE=ON `
  --config-setting cmake.define.VP9_LOSSLESS_LIBVPX_LIBRARY=C:/build/xllvp9/build/native/libvpx_optimized/x64/Release/vpxmd.lib `
  --config-setting cmake.define.VP9_LOSSLESS_LIBVPX_INCLUDE_DIR=C:/build/xllvp9/reference/libvpx_lossless
```

The wheel must contain `xllvp9/_libvpx_ref.cp312-win_amd64.pyd`. A wheel built
with native support disabled may still have a platform tag but omit the
extension: inspect its contents, then verify the installed module.

```powershell
uv pip install --python C:/path/to/codec/.venv/Scripts/python.exe --no-deps C:/path/to/xllvp9-0.1.0-cp312-cp312-win_amd64.whl
C:/path/to/codec/.venv/Scripts/python.exe -c "from xllvp9.native_backend import native_backend_available; assert native_backend_available()"
```

Keep the editor and codec environments separate. `--no-deps` here preserves
the existing tested codec dependencies; grayscale export uses its NumPy.
The standalone xllvp9 video CLI also declares OpenCV and needs its own full
dependency environment. The native wheel is ABI/platform specific and is not
committed to this repository. Other platforms have not been tested here.
