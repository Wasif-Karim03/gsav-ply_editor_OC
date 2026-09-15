# Mask-aware encoding

The presence plane controls visibility. Inactive source attributes are undefined
and may contain nonfinite placeholders. Encoders validate active values and
sanitize inactive rows on a copy before sorting or quantization.

Python `ChunkConfig(identity_mode="stable")` declares persistent source-row
identities. A GOP uses one permutation based on each row's first active sample.
`auto` conservatively detects correspondence; `unstructured` enables matching
using only active candidates. Stable identity cannot be combined with matching.

Pruning requires explicit stable identity. By default it removes only rows that
are never active. `lossy_pruning=True` additionally enables opacity, scale, and
importance thresholds. Entirely inactive streams retain one disabled row.

Ranges, automatic position precision, and SH training use active samples only.
After quantization, inactive rows repeat their previous attributes and SH labels.
An inactive GOP keyframe uses its first active sample within that GOP. No previous
GOP is required. Masks remain exact and active values are never replaced by this
reuse step. Keyframe snapping excludes activation transitions. Set
`reuse_inactive=False` for ablation and `keyframe_snap=False` for an unsnapped
attribute comparison.

Unstructured matching first matches active source candidates to previously active
slots. Unmatched candidates rank up to eight vacant spatial neighbors using
position, DC color, and log scale, then resolve collisions and fill remaining
capacity deterministically. Retirement clears only the mask.

The GSAV v3 wire format and embedded GSST static asset are unchanged. Static and
active dynamic Gaussians continue to share rendering and depth sorting.

## Native interface

C++ uses `NativeOrderingMode::StableIdentity`, exposed as `ordering_mode="stable"`
in `encode_sequence_with_options` and `--ordering-mode stable` in the CLI.
`matching_enabled=True` selects unstructured matching. Stable counts may differ
between GOPs, but must remain constant within a GOP; lifetime pruning additionally
requires a constant count across the sequence.

The options binding exposes `reuse_inactive`, `keyframe_snap`, and `lossy_pruning`.
CLI controls are `--no-reuse-inactive`, `--no-keyframe-snap`, and `--lossy-pruning`.
Equivalent native environment controls are `GSCODEC_REUSE_INACTIVE`,
`GSCODEC_KEYFRAME_SNAP`, and `GSCODEC_LOSSY_PRUNING` (0 or 1).

Native masked SH training uses bounded active samples and one frame of float
scratch. Its existing quantized palette strategy differs from Python clustering:
parity means compatible decoded behavior, not identical compressed bytes.

## Initial measurements

Deterministic SH0 stress fixture: 2,048 source Gaussians, 12 frames, GOP 4, about
40% active; geometry comes from a subset of one Dymensium frame. This is a
synthetic lifecycle test, not a representative compression or quality benchmark.

| Inactive positions | Baseline Python bytes | Mask-aware Python bytes |
| --- | ---: | ---: |
| Coherently moving with source | 175,159 | 92,282 |
| Random noisy placeholders | 422,603 | 92,282 |

These measurements combine range, sorting, and reuse changes. They do not isolate
the gain from reuse. The Python prototype passed
86 tests, including activation, all-inactive streams, random access, parallel
worker masks/SH, and existing joint static/dynamic GPU rendering coverage.

The native noisy-placeholder fixture fell from 422,132 to 92,118 bytes. Under
identical mask-aware ranges and sorting, disabling reuse produced 139,568 bytes
in Python and 138,683 in C++. Reuse therefore saved 33.9% and 33.6% respectively
in this fixture. Timings are single-run diagnostics, not throughput claims.

GPU renders used a shared camera at frames 0, 5, and 11. Native foreground PSNR
against active source Gaussians was 57.63, 45.01, and 56.34 dB, with identical
results when reuse was disabled. Against the stable-placeholder baseline, the
combined changes improve two samples and slightly worsen the third (56.67 to
56.34 dB). Only about 3.7% of pixels contain visible geometry; this small SH0
fixture does not establish quality on full scenes or higher-order SH.

Reproduction: run `scripts/benchmark_mask_encoding.py --output <directory>
--source <ply> --backend python|native`, optionally with `--no-reuse`,
`--inactive-values stable`, or `--render-compare <saved.gsav> ...` (CUDA required).
Raw measurements are in [benchmarks/mask-aware-encoding.json](benchmarks/mask-aware-encoding.json).

Final validation: 125 tests passed on `v3-cpp`, including native cross-reader
lifecycles, mixed source SH degrees, optional pruning, exact active-attribute
preservation, and GPU mask transitions with an embedded static asset. The Release
native build and Python lint checks passed.

Stable sorting constructs first-active positions only; it does not copy SH
coefficients between source frames. This also supports a requested SH band when
the source frames contain different, higher SH degrees.

Python selects the requested SH layout before serial or parallel matching and
rejects frames missing the requested coefficients. SH0-only encoding excludes
unused higher-order coefficients from matching. Source tensors retain their
original SH degrees. Every frame is padded to the sequence population, including
GOPs whose first frame is already the largest.
