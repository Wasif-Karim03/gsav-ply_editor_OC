# Motion-aware crop prototype 1 — not enabled in the editor

## Saved working version and rollback

The working editor is preserved at **v3.0.8**, commit
`4b81eddd2f37223ace1a8d663cd4da20bf713d88`. The remote annotated tag was verified
to resolve to that commit before this work. `version-3` remains on that release.
Prototype work is isolated on **motion-crop-prototype**. The live viewer was not
restarted or changed by this prototype, so no rollback is currently necessary.

To restore this exact application version later, stop the editor, preserve any
uncommitted work, run `git switch --detach v3.0.8`, and restart with the existing
editor/codec environments. Do not use a hard reset. To resume prototype work,
switch back to `motion-crop-prototype`. Input files, exports and cache directories
are not managed by Git and must be retained separately.

## Implemented offline slice

- A local-candidate tracker using the existing SciPy dependency. It compares
  predicted position, base color and log-scale features, checks ambiguity and
  requires agreement in both matching directions. It ignores row numbering and
  chunk boundaries. Matches are **estimated**, not authoritative identities.
- Consecutive observations form track fragments. A missing or rejected link
  breaks the track; the implementation does not invent occluded geometry or
  claim to recover the same identity after occlusion.
- A disk-backed geometry/track cache, keyed by full source SHA-256 and an
  algorithm version. It stores no SHN arrays. A complete marker is written only
  after successful construction and a second source hash check.
- A 1 GiB default cache limit and metadata/type/shape validation. Incomplete,
  changed-source and invalid-track caches fail closed. Existing directories and
  report files are not overwritten.
- Boundary and Keep/Remove evaluation reuses the cache. Tracks with fewer than
  three observations fall back to per-frame cropping. Policies apply to observed
  fragments, **not** proven whole-animation physical identities. Opacity/scale
  thresholds remain per frame. Results are packed masks; no scene is exported.
- Synthetic ground-truth benchmarks, a real-source timing benchmark, and an
  explicit promotion gate. There are no new dependencies, UI registrations,
  codec modifications or server changes.

## Results and decision

See [machine-readable measurements](benchmarks/motion-crop-prototype-1.json).
Measurements are one local run while other checks were active, not portable
performance promises.

On the saved 180-frame, 78,124-row source:

| Work | Measurement |
| --- | --- |
| Initial decode + tracking + cache | 129.48 seconds |
| Cache size | 464,072,478 bytes (464.07 MB) |
| Keep evaluation from existing cache | 4.67 seconds |
| Remove evaluation from existing cache | 4.49 seconds |
| Changed-radius Keep evaluation | 4.07 seconds |

82.85% of observed samples belonged to estimated fragments at least three frames
long. **This is coverage, not accuracy.** There were 2,657,069 fragments and a
median fragment length of one frame. The long-lived tracks may disproportionately
represent static parts; these figures do not establish hand-tracking quality.

Synthetic shuffled static scenes, ordinary translation, crossing groups and
appearance/disappearance cases achieved 100% adjacent-link recall with no wrong
links. A sudden jump caused a track break. Identical duplicate features were
rejected rather than arbitrarily assigned.

The dense, similar-appearance case **failed badly**: 639 of 661 accepted links
were wrong (3.33% precision). Nearby samples can appear to be good mutual matches
while actually belonging to different trajectories. Forward/backward agreement
and distance-ratio thresholds are insufficient confidence measures on their own.
**The prototype is not approved for editor integration.** The cache approach is
useful; this matcher must not replace v3.0.8 selection behavior.

## Next implementation gate

1. Estimate coherent local/group motion before matching individual Gaussians.
   Test several independently moving regions, not just global translation.
2. Calibrate ambiguity rejection against dense same-appearance sequences,
   acceleration, rotations, crossings, births/deaths and partial occlusion.
   Candidate-radius parameters currently depend on the first active frame's
   robust bounds; they are not calibrated for arbitrary sources or FPS.
3. Measure both incorrect links and missing useful tracks. Rejecting every
   difficult match is not sufficient to deliver the user's keep-hand behavior.
4. Review fixed-camera clips and selected source-frame annotations for the
   dancer, especially hand/head overlap and chunk boundaries. No real-scene
   visual correctness review or ground-truth hand annotations are claimed here.
5. Only after those gates pass, add an optional mode sharing frozen masks between
   preview and export, benchmark export size/time and verify every visible field.
   Keep ordinary and chunk-based crop available. No universal tracking guarantee.

## Reproduce

Validation: the complete editor/prototype suite passed **138 tests** (25 existing
warnings). Ruff on new Python files and Git whitespace checks passed. Root
`npm run check` passed 10 modules/35 catalog rules; root `npm test` retained the
existing **49 pass / 78 fail** result from unrelated archived JavaScript snapshot
discovery. Passing implementation tests does not override the failed tracking
quality benchmark.

Run from `gsplay/` using the existing editor environment. Configure
`GSPLAY_CODEC_PYTHON` for the existing codec environment as for the application.
The cache/report paths must be new for the first run; later runs reuse a complete
matching cache and need a new report filename.

```powershell
python -m src.gsplay.motion_crop.benchmark --report path/to/synthetic.json
python -m src.gsplay.motion_crop.benchmark --source path/to/scene.gsav --cache path/to/cache --report path/to/report.json --center 0 0 0 --radius 1
python -m pytest tests/test_motion_crop.py
```

Source data and caches remain local in ignored output directories. The committed
benchmark contains aggregate measurements only. No production migration is
required. The application release number remains 3.0.8 because no new editor
feature has been released.
