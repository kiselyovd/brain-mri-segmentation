# Contributing to brain-mri-segmentation

Thanks for your interest in improving the project. This document covers the developer workflow, conventions, and guard-rails the repository enforces.

## Development setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management and Python 3.13+.

```bash
# 1. Clone
git clone https://github.com/kiselyovd/brain-mri-segmentation.git
cd brain-mri-segmentation

# 2. Install all dependency groups (dev + serving + tracking + docs)
uv sync --all-groups

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Run the full quality-gate stack locally
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts
uv run deptry .
uv run bandit -r src scripts -c pyproject.toml
uv run interrogate -c pyproject.toml src scripts
uv run pytest
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the full data/training reproducibility story (seeds, Docker pins, DVC).

## Branch naming

Use short, kebab-case branch names prefixed by the change kind:

- `feat/<short-topic>` — new feature (e.g. `feat/dice-loss-focal-variant`)
- `fix/<short-topic>` — bug fix (e.g. `fix/segformer-resize-off-by-one`)
- `docs/<short-topic>` — documentation-only change
- `ci/<short-topic>` — CI, workflows, pre-commit, quality gates
- `chore/<short-topic>` — repo hygiene, dependency bumps, refactors

## Commit messages — Conventional Commits

All commits MUST follow [Conventional Commits](https://www.conventionalcommits.org/). Types used in this repo:

| Type | Purpose |
|---|---|
| `feat` | New user-visible capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither adds features nor fixes bugs |
| `perf` | Performance improvement |
| `test` | Tests only |
| `ci` | CI / workflows / pre-commit |
| `build` | Build system, packaging, Docker |
| `style` | Formatting (ruff, whitespace) — no logic change |
| `chore` | Everything else (deps, gitignore, metadata) |

Example:

```
feat(training): add SegFormer-B2 mixed-precision training path

- Enables bf16 on CUDA, fp16 fallback on older GPUs.
- Checkpoint size unchanged; throughput +18% on RTX 3080.
```

## No AI co-author trailers

Do **not** add `Co-Authored-By: Claude …`, `Co-Authored-By: GitHub Copilot …`, or any other AI-tool trailer to commits. This applies to both human contributors and any AI assistants they use. A commit should have exactly one author — the human pushing the change.

If an AI assistant helped you write the patch, that is completely fine. Just omit the trailer.

## Pull requests

- Keep PRs focused — one logical change per PR.
- The CI must be green (lint, type, test, Docker build, actionlint) before merge.
- Update `CHANGELOG.md` under `[Unreleased]` for any user-visible change.
- For breaking changes, call it out in the PR description and the changelog entry.

## Code style

- `ruff` and `ruff format` are the source of truth for formatting and lint.
- Type hints on all public functions; `mypy` runs in CI.
- Docstrings on public modules / classes / functions (`interrogate` enforces ≥35% coverage).
- Keep imports first-party-qualified (`from brain_mri_segmentation.training import …`) — relative imports beyond the immediate parent are discouraged.

## Reporting issues

Open a [GitHub issue](https://github.com/kiselyovd/brain-mri-segmentation/issues) with:

- What you expected.
- What happened.
- Minimal repro (command, config, OS, GPU).
- Full traceback if applicable.

Security issues: email `daniil.kiselev@umbrellait.com` instead of opening a public issue.
