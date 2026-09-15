---
paths:
  - "**/*"
---

# Git & Commit Guidelines

## Conventional Commits
Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

### Format
`<type>(<scope>): <description>`

### Types
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `docs`: Documentation changes
- `test`: Adding missing tests or correcting existing tests
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `ci`: CI/CD changes
- `build`: Update build process or dependencies

### Examples
- `feat(encoder): implement pingpong quantization`
- `fix(decoder): resolve frame index offset error`
- `perf(io): optimize ffmpeg buffer size`
- `docs: update installation instructions`

### Rules
- **Imperative Mood**: Use "add" not "added", "fix" not "fixed".
- **Lowercase**: Keep the description lowercase.
- **Scope**: Optional but recommended for monorepos or distinct modules (e.g., `encoder`, `decoder`, `cli`).
