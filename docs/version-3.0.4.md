# Version 3.0.4 — Apply preserves manual edits

Previously the Color tab's **Apply** button executed the selected preset
(Auto Enhance by default), replacing manually chosen slider values. Export
then correctly read those replaced values, losing the user's intended look.

**Apply** now synchronizes the current color, opacity, transform and filter
controls into the shared preview/export configuration and refreshes rendering.
It does not overwrite sliders or bake changes repeatedly into source frames.
Edits remain live across playback and seeking; export takes a snapshot of the
same settings and applies them to the selected frame range.

Automatic corrections and style presets have a separate **Apply Preset**
button whose hint explicitly says it replaces the color sliders. **Reset**
retains its existing explicit reset behavior. Slider previews remain live.

## Verification

- Full editor suite: **120 passed**, 25 warnings. New regressions exercise the
  real button callbacks at brightness 0.7, 1 and 5, repeated Apply, export
  configuration synchronization, and independent preset wiring.
- Changed Python files pass Ruff lint and formatting checks.
- Browser: loaded the preserved 220-frame user scene, set brightness to 5,
  clicked Apply, sought to frame 110, and clicked Apply again. Brightness
  stayed at 5; the rendered preview showed the brighter scene. Exported all
  220 frames through the browser to a separate verification file.
- Decoded all 220 exported frames and compared against the CPU brightness-5
  operation (including its RGB clamp to 0–1). Every frame changed; maximum
  absolute RGB error was 0.006622, below the 0.03 quantization tolerance.
  This verifies edited data export, not pixel-identical CPU/GPU rendering.
- Parent workspace `npm run check` passes; its broad JavaScript test discovery
  retains the existing 49 passes / 78 failures in unrelated copied viewer
  snapshots and audit scripts.

The server must restart to register the new UI callbacks. The local editor at
6034 was restarted and the preserved source reloaded for verification.
GSAV native compression and container layout are unchanged. Earlier release
tags remain available for rollback.
