# Training

## Prerequisites

```bash
uv sync --all-groups
bash scripts/sync_data.sh /path/to/lgg-mri-segmentation
uv run python -m brain_mri_segmentation.data.prepare --raw data/raw --out data/processed
```

`prepare.py` performs a patient-level 80/10/10 split (110 patients → ~88/11/11) so no patient's slices span partitions.

## Main — SegFormer-B2

```bash
uv run python -m brain_mri_segmentation.training.train experiment=sota
```

Expected wall time: ~60 min on an RTX 3080 (batch 16, 256², ~50 epochs). Checkpoint written to `artifacts/checkpoints/best.ckpt`.

**GPU memory note:** SegFormer-B2 at batch size 16 and 256² input fits comfortably in ~8 GB VRAM on an RTX 3080. Reduce `data.batch_size` if you see OOM on smaller cards.

## Baseline — U-Net

```bash
uv run python -m brain_mri_segmentation.training.train \
  model=baseline \
  trainer.max_epochs=50 \
  trainer.output_dir=artifacts/baseline
```

## MLflow tracking

```bash
mlflow ui --backend-store-uri ./mlruns
```

Browse at http://localhost:5000 — every Hydra run is one MLflow run with the full resolved config logged as params and `train/loss`, `val/loss`, `val/dice`, `val/iou`, `val/pixel_acc` as metrics.

## Hydra overrides (common)

| Override | Effect |
|---|---|
| `trainer.max_epochs=100` | Longer training |
| `trainer.accelerator=gpu` | Force GPU |
| `data.batch_size=8` | Smaller batches for low-VRAM cards |
| `model.lr=1e-4` | Different learning rate |
| `seed=7` | Reproducibility |

Multi-run sweep example:

```bash
uv run python -m brain_mri_segmentation.training.train -m \
  model.lr=1e-5,3e-5,1e-4 \
  data.batch_size=8,16
```
