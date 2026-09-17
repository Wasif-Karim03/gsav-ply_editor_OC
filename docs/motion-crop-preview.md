# Motion-aware crop preview: 3.0.9-preview.1

## Use

Load a GSAV, choose a spatial boundary in Filter, select **Motion-aware (preview)**
under Crossing method, choose **Keep crossing rows** or **Remove crossing rows**,
then click **Analyze & Apply**. Inspect the animation before export. The existing
opacity, scale, sphere, box, ellipsoid and frustum controls remain available.
Chunk-based remains the default method.

Keep retains an eligible estimated track throughout its observed fragment if it
enters the boundary at least once. Remove retains it only if all its observations
are inside. Tracks shorter than three observations use the ordinary per-frame
boundary. Opacity and scale limits still apply per frame.

## What changed

The original spatial/appearance matcher could confuse dense similarly colored
Gaussians. The revised matcher adds local distance-shape descriptors, unique
quantized proposals and agreement with at least three nearby correspondences.
Position-based fallback requires locally distinctive appearance. Conflicts and
ambiguous matches break tracks instead of forcing an association. Source row
numbers are never treated as physical identities.

The first analysis decodes the source and builds a disk-backed cache. Changing
the crop or keep/remove choice reuses that cache in the same editor session.
Source SHA-256 and algorithm-version checks prevent reuse for different content
or algorithms. Incomplete caches are never reused. A 1 GiB cache limit rejects
larger analyses with an error; the user can still use the existing crop method.
The cache is temporary analysis data and is not included in exported GSAV files.

Preview and export consume the same frozen selection masks. A changed boundary,
source, method or policy invalidates the plan and requires Analyze & Apply again.
Native xllvp9 and the existing compact crop export path remain in use.

## Limits

These are estimated track fragments, not semantic hands, heads or other objects.
Missed observations, occlusion, very fast motion, deformation, repeated shapes
and changed sampling can break or misassign tracks. Coverage shown in the panel
is the fraction using long-enough estimated fragments, **not tracking accuracy**.
Ordinary-crop fallback can still cut a moving part outside the boundary.
No universal hand-preservation or quality guarantee is made. The offline benchmark
does not certify real-object tracking and keeps its general-release gate closed.

## Rollback

The stable branch `version-3` and tag `v3.0.8` remain unchanged. This preview is on
`motion-crop-prototype`. Choose Chunk-based in the preview for the earlier method,
or run a separate checkout of v3.0.8. No GSAV format migration is required.

## Measured validation

- Nine deterministic synthetic cases: zero wrong accepted links, including dense
  similar-appearance translation, rotation and independently moving groups.
  Independent dense-group recall was 97.4%; indistinguishable duplicates were
  rejected. These controlled cases do not establish real-scene accuracy.
- Saved 180-frame, 78,124-row scene: 391.0 seconds to build a 464,072,477-byte
  temporary cache; subsequent keep/remove/boundary evaluations 4.63–4.75 seconds.
  Other checks shared the machine during measurement, so this is not an isolated
  performance comparison. 82.3% of samples used fragments of at least three
  observations; the remainder fell back to ordinary cropping.
- Editor pytest suite: 139 passed, 25 warnings. Targeted integration tests verify
  both crop methods through preview, PLY export and native GSAV export, comparing
  all selected Gaussian fields after decoding, including SH coefficients.
- Full 180-frame crop-only native export: 42.1 seconds, 29,616,041 bytes versus
  40,651,825 source bytes (27.1% smaller). Decoding and comparing every selected
  position, scale, rotation, opacity, SH0 and SHN value passed exactly. This
  verifies export fidelity to the masks, not semantic correctness of the masks.
- Root `npm run check` passes. Root `npm test`: 49 passed, 78 failed because it
  also discovers unrelated archived JavaScript audit fixtures under `outputs/`.
  This is not a passing workspace-wide Node test run.
- Separate live viewer: loaded the saved scene, applied Motion-aware Keep through
  the UI, observed the shared-mask status and rendered frame 154. Visual inspection
  still shows the raised part cut at this boundary; this release is not evidence
  that the dancer's entire hand has been recovered. The conservative fallback
  limitation is visible in practice, not only theoretical.

Benchmark source: `python -m src.gsplay.motion_crop.benchmark`; historical report
schema called its default-release gate `ready_for_editor`. The current schema
uses `ready_for_default` to distinguish opt-in preview availability from approval
to make estimated tracking the default.
