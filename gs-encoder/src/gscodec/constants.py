"""
Shared constants for gscodec encoding/decoding.

Frame-by-frame codec:
- All attributes quantized per-frame using global ranges.
- Means: 16-bit split into hi (base-220, video-safe [16,235]) and lo (base-256, raw [0,255]).
- means_lo stored as raw binary in a separate GSAV section (not video-encoded).
- Other attributes: 8-bit video-safe [16, 235].

Video atlas: 3 rows x 5 cols = 15 cells (14 channels + 1 padding).
"""

# 8-bit quantization levels (video-safe range 16-235)
N_LEVELS = 220

# Video-safe range start (offset added to quantized values)
MIN_VAL = 16

# Scaling factor for 8-bit quantization (N_LEVELS - 1)
SCALE_8BIT = 219

# Base for lo-byte of 16-bit means (full byte range, raw binary)
LO_BASE = 256

# Scaling factor for 16-bit means (220 * 256 - 1)
SCALE_16BIT = 56319

# Atlas layout (3 rows x 5 cols = 15 cells)
# Row 0: mean_hi.x | mean_hi.y | mean_hi.z | scale.x   | scale.y
# Row 1: scale.z   | quat.w    | quat.x    | quat.y    | quat.z
# Row 2: opacity   | sh0.r     | sh0.g     | sh0.b     | padding
N_ATLAS_COLS = 5
N_ATLAS_ROWS = 3

# Attribute clipping bounds
QUATS_CLIP = (-1.0, 1.0)
OPACITY_CLIP = (-6.0, 12.0)
SH0_CLIP = (-2.0, 4.0)

# SH compression (codebook + palette + labels)
SH_COEFFS: dict[int, int] = {0: 0, 1: 3, 2: 8, 3: 15}  # coefficients per channel per band
SH_CODEBOOK_SIZE = 256  # scalar codebook entries (full uint8 range)
MAX_SH_CENTROIDS = 65535  # max palette entries (uint16 max)
