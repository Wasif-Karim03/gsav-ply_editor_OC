# Version 3.0.1 - playback and folder selection

## Reported problems and fixes

Large GSAV scenes stalled on frame loading. Each frame was serialized as an
uncompressed NumPy archive and copied through a subprocess pipe. On Windows,
the editor and isolated decoder now share a reusable named memory mapping.
Only array descriptors cross the pipe. The consumer copies the arrays before
the next request can overwrite them. Capacity and descriptor bounds are checked;
the OS releases the mapping when both processes close. Other operating systems
retain the existing pipe transport.

GPU upload now transfers the decoder's fields directly, avoiding an intermediate
CPU packing/transposition step. One pristine GPU frame, at most 128 MiB, is cached
for repeated camera/color interactions. Each caller receives a clone, so edits
cannot accumulate in the cached source. SH ordering and format metadata remain
intact. These changes do not reduce Gaussian count, SH degree or render resolution.

The folder picker could wait on a dialog without a visible owner while leaving
the browser button disabled. Its Windows owner is now shown and activated, with
a taskbar entry. The browser displays a waiting message and a **Cancel Folder
Selection** button that terminates the helper and restores the controls. The
existing five-minute timeout remains. Typed export paths still work.

A separate live error from Filter Reset called the visualization method on the
renderer instead of the application. Reset now refreshes the correct component.

## Measurements and limits

A local 600-frame SH3 scene with 339,536 Gaussians per frame was used for a
20-frame sequential loading comparison, after loading frame zero:

| Path | Time for 20 frames |
| --- | ---: |
| Original archive/pipe and CPU packing | 11.43 s |
| Shared-memory transfer alone | 6.89 s |
| Shared memory and direct GPU transfer | 4.76 s |

This is a single local loading benchmark, not end-to-end browser FPS or a
guarantee of real-time playback. Decode and GPU upload costs remain, especially
on large scenes. Decoded raw-array SHA-256 hashes matched exactly at frames 0,
30 and 599. The cache regression checks mutation isolation and SH ordering.
Real codec round trips continue to test edited SH3 exports and mask lifetimes.

The editor suite reports 109 passes and 25 inherited boolean-return warnings;
those inherited cases are not reliable assertion-based checks. Changed helper
modules and new tests pass Ruff. The parent workspace's `npm run check` passes;
its broad `npm test` discovery reports 49 passes and 78 failures in unrelated
viewer snapshots and local browser audit scripts, so that workspace suite is
not green.

Live verification on a separate editor instance loaded the reported 600-frame
scene and exercised playback and Filter Reset without the previous reset error.
The user confirmed the Windows folder dialog was visible, selected a folder,
and started a GSAV export. The Save To Path button recovered after selection.
Cancellation is covered by regression tests; the live cancel check was not run
because the user had already completed selection and started exporting.

## Rollback

Use tag `v3.0.0` for the previous version-3 build, or `pre-v3` for the older
combined baseline. Stop the editor after saving edits before changing versions;
the viewer and codec sources must belong to the same checkout. `main` remains
the original baseline. No dependency versions or GSAV container format changed.
