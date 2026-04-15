# brain-mri-segmentation

[![CI](https://github.com/kiselyovd/brain-mri-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/kiselyovd/brain-mri-segmentation/actions/workflows/ci.yml)
[![Docs](https://github.com/kiselyovd/brain-mri-segmentation/actions/workflows/docs.yml/badge.svg)](https://kiselyovd.github.io/brain-mri-segmentation/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/kiselyovd/brain-mri-segmentation)

Binary brain-tumor segmentation on MRI slices — fine-tuned **SegFormer-B2** as the main model and a hand-rolled **U-Net** as a reproducible baseline, both trained on the Mateusz Buda LGG (TCGA) dataset with a strict patient-level split to prevent data leakage.

**Russian:** [README.ru.md](README.ru.md) · **Docs:** [kiselyovd.github.io/brain-mri-segmentation](https://kiselyovd.github.io/brain-mri-segmentation/) · **Model:** [kiselyovd/brain-mri-segmentation](https://huggingface.co/kiselyovd/brain-mri-segmentation)

## Dataset

Mateusz Buda's [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) on Kaggle — 110 patients, 3,929 paired FLAIR slices with binary tumor masks from The Cancer Genome Atlas (TCGA). `src/brain_mri_segmentation/data/prepare.py` performs a **patient-level 80/10/10 split** (88/11/11 patients), so no patient appears in more than one partition.

Resulting slice counts: **3,133 train / 409 val / 387 test**.

## Results

Test-set metrics after full training (fill in with real numbers from `reports/metrics.json`):

| Model | Dice | IoU | Pixel Accuracy |
|---|---|---|---|
| **SegFormer-B2** (main) | **65.5%** | **66.2%** | **99.73%** |
| U-Net 4-level baseline | 51.9% | 57.7% | 99.66% |

Full per-slice report lives in `reports/metrics.json` after running evaluation.

## Quick Start

```bash
# 1. Install
uv sync --all-groups

# 2. Sync Kaggle dataset into data/raw/ (once)
bash scripts/sync_data.sh /path/to/lgg-mri-segmentation

# 3. Build processed splits
uv run python -m brain_mri_segmentation.data.prepare --raw data/raw --out data/processed

# 4. Train (main model on GPU)
make train

# 5. Evaluate on test split
make evaluate

# 6. Serve the model locally
make serve
# or
docker compose up api
```

## Full Training Commands

**Main — SegFormer-B2:**

```bash
uv run python -m brain_mri_segmentation.training.train experiment=sota
```

**Baseline — U-Net (4 levels, 32→256 ch):**

```bash
uv run python -m brain_mri_segmentation.training.train \
  model=baseline \
  trainer.max_epochs=30 \
  trainer.output_dir=artifacts/baseline
```

Every run is tracked with MLflow under `./mlruns/`; launch `mlflow ui --backend-store-uri ./mlruns` to inspect.

## Inference

```python
from huggingface_hub import snapshot_download

from brain_mri_segmentation.inference.predict import load_model, predict

weights_dir = snapshot_download("kiselyovd/brain-mri-segmentation")
model = load_model(f"{weights_dir}/best.ckpt")
result = predict(model, "slice.tif")
print(f"Mask: {len(result['mask'])}×{len(result['mask'][0])}")
```

`result["mask"]` is a 2-D binary array (H × W) aligned to the input slice.

## Serving

```bash
docker compose up api
curl -X POST -F "file=@slice.tif" http://localhost:8000/segment
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/segment` | Multipart TIFF/PNG → JSON mask |
| `GET` | `/metrics` | Prometheus metrics |

Every response carries an `X-Request-ID` header for log correlation.

## Project Structure

```
src/brain_mri_segmentation/
├── data/           # MRIDataModule, MRIDataset, prepare.py (patient-level split)
├── models/         # factory.py, lightning_module.py, unet.py
├── training/       # Hydra entrypoint
├── evaluation/     # Dice / IoU / pixel-accuracy report
├── inference/      # load_model + predict
├── serving/        # FastAPI app
└── utils/          # logging, seeding, HF Hub helpers
configs/            # Hydra configs (data / model / trainer / experiment)
data/
├── raw/            # original Kaggle download (images + masks per patient)
└── processed/      # train / val / test splits
docs/               # MkDocs site sources
tests/              # pytest suite
```

## Intended Use

Research and educational only. **Not a medical device.** Predictions must not be used for clinical decisions.

## License

MIT — see [LICENSE](LICENSE).
