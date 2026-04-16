# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-15

### Added

- Binary brain-tumor segmentation pipeline on MRI slices (LGG / TCGA, Buda et al.).
- Main model: fine-tuned **SegFormer-B2** (`nvidia/segformer-b2-finetuned-ade-512-512`, ~27M params) with a binary segmentation head.
- Baseline model: hand-rolled **U-Net** (4 levels, 32→256 channels, ~1.9M params).
- Patient-level 80/10/10 split (88/11/11 patients) to prevent cross-partition leakage.
- Training entrypoint with **Hydra + PyTorch Lightning** and MLflow tracking; deterministic seeding (`seed=42`).
- Evaluation reports Dice, IoU, and pixel accuracy on the held-out test split.
- Test-set metrics: SegFormer-B2 Dice=65.5% / IoU=66.2% / px-acc=99.73%; U-Net baseline Dice=51.9% / IoU=57.7% / px-acc=99.66%.
- FastAPI serving app (`POST /segment`, `GET /health`, `GET /metrics`) with request-ID propagation.
- Multi-stage Docker images (`training` + `serving`) and `docker compose` wiring.
- Hugging Face Hub publishing via `scripts/publish_to_hf.py` — native `safetensors` export, rich frontmatter, and widget PNG samples.
- DVC pipeline (`dvc.yaml`) for `prepare → train → evaluate` reproducibility.
- MkDocs Material documentation site with bilingual (EN/RU) README and model card.
- Quality gates: ruff, mypy, deptry, bandit, interrogate, codespell, actionlint, pre-commit.
- GitHub Actions workflows: CI (lint/type/test + Docker build), Docs (GitHub Pages), Release.

[Unreleased]: https://github.com/kiselyovd/brain-mri-segmentation/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kiselyovd/brain-mri-segmentation/releases/tag/v0.1.0
