# Contributing to gscodec

We welcome contributions to `gscodec`! Please read this guide to understand how to contribute effectively.

## How to Contribute

There are several ways to contribute to the `gscodec` project:

-   **Report Bugs**: If you find a bug, please open an issue on our [GitHub Issues page](https://github.com/OpsiClear/gscodec/issues).
-   **Suggest Features**: Have an idea for a new feature? Open an issue to discuss it.
-   **Submit Pull Requests**: We love pull requests for bug fixes, new features, or improvements.

## Development Setup

To get your development environment ready, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/OpsiClear/gscodec.git
    cd gscodec
    ```

2.  **Install Dependencies**: We recommend using `uv` for dependency management.
    ```bash
    pip install -e ".[dev]"
    # or if you have uv installed
    uv pip install -e ".[dev]"
    ```

3.  **Install Pre-commit Hooks**: Our project uses pre-commit to maintain code quality.
    ```bash
    uv pip install pre-commit # if not already installed
    pre-commit install
    ```
    This will set up hooks that automatically run checks like linting and formatting before you commit your changes.

## Code Style and Conventions

To ensure consistency and readability across the codebase, please adhere to the following guidelines:

### Type Hints
-   **Required**: Use Python 3.10+ type hint syntax.
-   **Example**: `list[int]` not `List[int]`, `dict[str, Any]` not `Dict[str, Any]`.

### CLI Argument Parsing
-   **Use `tyro`** for command-line argument parsing.

### Logging
-   **Use the `logging` module** for informational messages instead of `print()` statements.
-   Example: `logging.info("Something happened.")`

### Code Formatting
-   **Linter**: `Ruff` (run with `ruff check .` or `pre-commit run ruff --all-files`)
-   **Type Checker**: `MyPy` (run manually with `mypy src/` or `pre-commit run --hook-stage manual mypy --all-files`)
-   **Line Length**: Maximum of 100 characters.
-   Follow **PEP 8** conventions.

### Naming Conventions
-   **Variables**: `snake_case`
-   **Classes**: `PascalCase`
-   **Constants**: `UPPER_SNAKE_CASE`
-   **Private Members**: Prefix with `_` (e.g., `_private_method`).

### Documentation Style
-   **Docstrings**: Use Google style with type hints for all functions, classes, and methods.
-   **Example format**:
    ```python
    def example_function(arg1: str, arg2: int) -> bool:
        """Brief description of what the function does.

        Args:
            arg1: Description of the first argument.
            arg2: Description of the second argument.

        Returns:
            Description of what the function returns.
        """
    ```

## Pull Request Guidelines

Before submitting a pull request, please ensure the following:

1.  **Run Pre-commit Hooks**:
    ```bash
    pre-commit run --all-files
    ```
2.  **Run Full Test Suite**:
    ```bash
    uv run pytest
    ```
3.  **Type Check**:
    ```bash
    uv run mypy src/
    # or
    pre-commit run --hook-stage manual mypy --all-files
    ```
4.  **Update `CHANGELOG.md`**: If your changes impact users (new features, bug fixes, breaking changes), please update the `CHANGELOG.md` file to reflect your contribution.

### Pull Request Title Format
Please follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for your PR titles:
-   `feat: Add new feature`
-   `fix: Fix bug in sequence encoder`
-   `perf: Optimize video stream writing`
-   `docs: Update API reference`
-   `test: Add tests for chunk processor`
-   `refactor: Simplify conformer logic`
-   `ci: Update CI configuration`
-   `build: Update build process`

## Testing

All new features and bug fixes should be accompanied by appropriate unit tests. Tests are located in the `tests/` directory.

-   **Run all tests**: `uv run pytest`
-   **Test Structure**:
    -   `tests/conftest.py`: Shared fixtures (mocks, config objects).
    -   `tests/test_*.py`: Unit tests for specific modules.

## Release Process

The release process is handled by the project maintainers.

## Contact

If you have any questions, feel free to reach out:
-   **Maintainer**: Aly El Hakie (aly@opsiclear.com)
-   **Repository**: https://github.com/OpsiClear/gscodec
-   **Issues**: Use [GitHub Issues](https://github.com/OpsiClear/gscodec/issues) for bug reports and feature requests.
