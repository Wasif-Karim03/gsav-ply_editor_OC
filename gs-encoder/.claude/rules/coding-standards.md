---
paths:
  - "**/*.py"
---

# Code Style & Architecture Rules

## Core Principles
- **SOLID**: Adhere to SOLID principles.
  - *Single Responsibility*: Classes/functions should do one thing well.
  - *Open/Closed*: Open for extension, closed for modification.
- **DRY**: Don't Repeat Yourself. Extract common logic into `utils/`.
- **Type Safety**: strict type hints are **mandatory**.
  - Use `list[str] | None` (Python 3.10+ union syntax).
  - Avoid `Any` unless absolutely necessary.

## Tooling & Environment
- **Command Runner**: ALWAYS use `uv run` for executing Python commands, scripts, or tests.
  - *Correct*: `uv run pytest`, `uv run mypy src/`
  - *Incorrect*: `python -m pytest`, `pytest`
- **Dependencies**: Use `uv add` and `uv remove` for dependency management.

## Comments & Documentation
- **"Why" not "What"**: Comments should explain the *intent* or *reasoning*, not describe the syntax.
  - *Bad*: `i += 1 # increment i`
  - *Good*: `i += 1 # skip header row`
- **Docstrings**: Google Style is mandatory for all public APIs.
- **Tensor Shapes**: ALWAYS document the shape of tensors in docstrings or inline comments.
  - Example: `# [Batch, N_Gaussians, 3]`
- **Minimal comments**: Use coments only when necessary

## Refactoring
- **Bloat Prevention**: When refactoring, explicitly identify and ask to remove deprecated functions/classes.
- **Legacy Code**: Do not comment out old code; delete it. Git history preserves it.
