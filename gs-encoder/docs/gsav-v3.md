# GSAV v3

`v3` contains the Python implementation. `v3-cpp` adds the C++ implementation
and cross-reader parity tests. Both share the same Python code and wire format.
Static PLY/SPZ assets use the native GSST codec. Precompressed GSST tracks can
also be embedded directly.

The `v3` branch starts at `v2`. Software v2 wrote wire version **1**;
v3 readers accept wire versions 1 and 3. Wire version 2 is not defined by
this repository and is rejected rather than guessed. New encodes write 3.

The existing 128-byte little-endian header, 112-byte ranges, chunk/frame indices,
means-lo, SH, VP9, and audio encodings are retained. The v3 layout is:

```
Header -> Ranges -> ChunkIndex -> FrameIndex -> StaticAsset (optional)
       -> MeansLo -> SH (optional) -> Video -> Audio (optional)
```

## Metadata

| Header field | Offset | Type | Meaning |
| --- | ---: | --- | --- |
| version | 4 | u32 | 3 for new files |
| flags | 30 | u16 | Feature flags below |
| static_asset_offset | 85 | u32 | Absolute byte offset, or zero |
| static_asset_size | 89 | u32 | Byte length, or zero |
| static_asset_encoding | 93 | u8 | 0 absent, 1 native GSST |
| reserved | 94 | 34 bytes | Zero in v3 |

Existing HAS_AUDIO (`0x0001`) and HAS_SH (`0x0002`) flags retain their meaning.
Bits `0x0004` and `0x0008` remain reserved in this prototype based on v2.

| Flag | Value | Meaning |
| --- | --- | --- |
| HAS_MASK | 0x0010 | Binary Gaussian presence mapped to video cell 14 |
| HAS_STATIC_ASSET | 0x0020 | Independent static scene is embedded |

HAS_STATIC_ASSET requires a nonempty GSST payload and encoding 1.
Unknown flags and encoding values are rejected.
The static section starts immediately after the frame index and ends at
means_lo_payload_offset. Its Gaussian count, ranges and SH level are independent
of the dynamic track. Both tracks use the same world coordinate system.

## Native static track

Native static PLY/SPZ inputs reuse the existing single-frame Gaussian encoder.
The resulting static section is flat: a 64-byte descriptor, 112-byte ranges,
the standard single-frame means-lo section, optional standard SH section, and
one lossless VP9 keyframe. No nested GSAV file is stored and no WebM wrapper is needed.

| Descriptor offset | Type | Meaning |
| ---: | --- | --- |
| 0 | char[4] | GSST magic |
| 4 | u32 | Static descriptor version 1 |
| 8 | u32 | Static Gaussian count |
| 12 | u32 | Atlas cell side (5x3 layout) |
| 16 | u32 | SH bands, 0 through 3 |
| 20 | u32 | HAS_MASK and HAS_SH only |
| 24 | char[16] | Null-padded vp09.00.51.08 codec |
| 40 | u32 | Means-lo section size |
| 44 | u32 | SH section size, zero when absent |
| 48 | u32 | VP9 keyframe size |
| 52 | bytes[12] | Reserved zero |

The sizes must cover the static section exactly. All fields are little-endian.
The in-memory decoder adapter reuses ordinary GSAV parsing and dequantization;
static attributes are decoded once and cached.

## Combined rendering

Rendering uses one persistent set of attribute buffers:
`[static primitives][dynamic slots]`. Static data initializes the prefix once;
only the dynamic suffix is decoded/uploaded on frame changes. Static and dynamic
SH levels may differ; missing coefficients are padded with zero. The mask sets
inactive alpha to exactly zero.

The entire buffer is submitted to one rasterization call, so visibility and
depth ordering include both tracks together. Static data still participates in
camera-dependent projection and sorting. This is not image compositing of
separately rendered tracks. Scene buffers are mutable and must not be updated
concurrently with rendering.

```python
decoder = SequenceDecoder.from_file("scene.gsav", backend="native")
scene = decoder.create_scene_buffer(device="cuda")
scene.update(frame_index)
image, alpha, info = scene.render(viewmats, Ks, width, height)
```

The native module must be importable (`cpp/built` on PYTHONPATH after building).
Omit `backend` to use the Python codec. Install the optional `render` extra for
gsplat rasterization. CUDA renderer compatibility depends on the local PyTorch
and CUDA toolchain. `render` returns the usual gsplat output tuple.

On Windows, gsplat 1.5.3 required two local build fixes: undefining the Windows
`small` macro before PyTorch's caching allocator header, and using MSVC flags
for C++ sources. The reproducible [dependency patch](gsplat-1.5.3-windows.patch)
is provided separately; gs-encoder does not modify dependencies at runtime.
For a fresh gsplat 1.5.3 install in this repository's Windows virtual environment:

```powershell
git apply --directory=.venv/Lib/site-packages/gsplat docs/gsplat-1.5.3-windows.patch
```

Reinstalling gsplat removes this local patch. Newer versions may already fix
these issues and should be checked before applying it.

Native file reads are demand-paged through a read-only mapping. Video packets
are passed directly from mapped storage to libvpx. The native decoder retains
at most two decoded atlases; older backward seeks restart at their keyframe.

## Mask video mapping

The 5-column, 3-row atlas retains all fourteen attribute mappings. Cell 14
(row 2, column 4) follows exactly the same Gaussian order and Morton placement.
Its lossless byte samples are **16 = active**, **235 = inactive**. These encode
a Boolean per primitive per frame; they are not bit-packed. Masks bypass
temporal attribute snapping. Sorting and matching carry source masks; padded
and retired slots are inactive. Masked decoded opacity is negative infinity
in logit space (effective alpha exactly zero), and GSData also exposes `masks`.

When HAS_MASK is absent, all primitives are active and cell 14 is ignored.
This preserves v2's padding semantics. A flagged mask requires the 5x3 atlas;
non-binary decoded samples are rejected. Current lossless VP9 preserves endpoints.

## Static asset workflow

```
compress --input-dir frames --output scene.gsav --static-asset background.ply
decompress --input scene.gsav --output-dir decoded
```

PLY/SPZ input is compressed into a native static track. Decompression extracts
`static.gsst` and reconstructs `static.ply`. Existing `.gsst` inputs are embedded
unchanged.

```python
decoder = SequenceDecoder.from_file("scene.gsav")
bundle = decoder.get_static_asset()       # bytes or None
static_scene = decoder.decode_static_asset()  # GSData or None
```

## Verification

`uv run python -m pytest tests` covers existing v2 tests plus v3 container,
mask, padding, random-frame, static GSST decode, and invalid metadata cases.
`tests/test_static_track.py` also checks persistent buffer addresses, mixed SH
degrees, and joint rasterizer submission. With CUDA and the `render` extra,
its GPU test moves a dynamic Gaussian from behind to in front of a static
Gaussian and verifies the resulting occlusion and static buffer residency.
These are correctness checks; no representative throughput benchmark is claimed.

On `v3-cpp`, `cpp/README.md` describes the native CPU build,
embedding/extraction APIs, and cross-reader parity tests. This branch extends
the Python prototype with the native codec; it uses the same version and flags.
