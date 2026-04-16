# Reproducibility

This project is designed to produce identical results across re-runs on the same hardware. Here's how each moving part is pinned.

## Random seeds

- `seed: 42` in `configs/config.yaml` is threaded through `seed_everything()` at the start of `src/brain_mri_segmentation/training/train.py`.
- Lightning's `deterministic="warn"` mode avoids the `nll_loss2d_forward_out_cuda_template` crash while still catching non-deterministic kernels elsewhere.
- **Patient-level split is deterministic**: `prepare.py` shuffles TCGA patient dirs with `random.Random(seed=42)` before the 88/11/11 partition — re-running always produces the same split.

## Dependencies

- **`pyproject.toml`** declares direct deps; **`uv.lock`** is committed and pins every transitive package version + hash.
- **`.python-version`** = `3.13` — CI + local dev must match.
- CUDA torch pinned via `[[tool.uv.index]] pytorch-cu124` on Win/Linux.
- CI runs the suite against Python 3.12 + 3.13 via a matrix strategy.

## Data

- **`data/sample/`** (8 paired image+mask TIFs) is in git — CI + smoke tests work without any external download.
- **`data/raw/`** and **`data/processed/`** are DVC-tracked. DVC remote is local-only by default.
- Kaggle source: `kaggle_3m/` (Buda et al., TCGA-LGG). 110 patient subdirs with paired `*.tif` + `*_mask.tif`.

## Docker

- `Dockerfile` is multi-stage (`base`, `training`, `serving`); `.dockerignore` keeps context minimal.
- Published images live at `ghcr.io/kiselyovd/brain-mri-segmentation:<tag>`.
- Base image: `python:3.13-slim-bookworm`.

## Model weights

- Published to HF Hub at `kiselyovd/brain-mri-segmentation`. Each release tag maps to an HF commit SHA.
- Weights ship as `model.safetensors`.

## One-command reproduction

```bash
git clone https://github.com/kiselyovd/brain-mri-segmentation
cd brain-mri-segmentation
uv sync --all-groups
bash scripts/sync_data.sh "/path/to/Brain MRI segmentation/kaggle_3m"
uv run python -m brain_mri_segmentation.data.prepare --raw data/raw --out data/processed
uv run python -m brain_mri_segmentation.training.train +experiment=sota
uv run python -m brain_mri_segmentation.evaluation.evaluate --checkpoint artifacts/checkpoints/best.ckpt --out reports/metrics.json
```

Expected: the numbers in [BENCHMARKS.md](BENCHMARKS.md) ± 1% (small test-set variance + floating-point noise).
