---
paths:
  - "**/*.py"
---

# Performance & Optimization Rules

## Vectorization
- **Strictly Avoid Loops**: Never use Python `for` loops for processing pixel data, gaussian attributes, or large arrays.
- **Use NumPy/PyTorch**: Always use vectorized operations (broadcasting, masking, indexing).
  - *Bad*: `[x * 2 for x in data]`
  - *Good*: `data * 2`

## Memory Management
- **Generators**: Use generators (`yield`) for frame processing pipelines to keep memory footprint low.
- **In-place Operations**: Prefer in-place operations (`+=`, `*=`) for large tensors to avoid unnecessary allocations.
- **Data Types**: Be explicit about dtypes. Use `uint8` for image data and minimal precision necessary for intermediate calculations.

## I/O
- **Batching**: Process I/O in chunks (as defined in `ChunkConfig`) rather than single frames where possible.
- **FFmpeg**: Ensure FFmpeg pipes are properly buffered.
