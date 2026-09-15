---
paths:
  - "tests/**/*.py"
  - "src/**/*.py"
---

# Testing Guidelines

## Requirements
- **Roundtrip Integrity**: Any change to encoder/decoder logic MUST pass `test_video_roundtrip.py`.
- **Unit Tests**: Every new feature requires a dedicated test file in `tests/`.

## Best Practices
- **Mocking**: Mock external dependencies (like `ffmpeg` subprocesses) for unit tests to ensure speed.
- **Fixtures**: Use `conftest.py` for shared setup (e.g., generating dummy Gaussian data).
- **Parametrization**: Use `@pytest.mark.parametrize` to test edge cases (empty inputs, single frame, max values).

## Coverage
- Aim for high branch coverage, especially in `quantization/` and `sorting/` modules.
