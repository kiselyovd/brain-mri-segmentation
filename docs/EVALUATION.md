# Evaluation contract and source recovery

## Original cohort

The original [Mateusz Buda LGG MRI dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) was retrieved on 2026-10-15, Kaggle dataset version `407317`.

- Archive: 748,584,920 bytes; SHA256 `cac105de01272f0ca28dc56178f8d164a7df85d118a5b4baca388d6bcb4e854e`.
- This archive contains two identical cohort trees. Their corresponding files were compared byte-for-byte; only `kaggle_3m/` was used. The clinical/genomic table is not needed for evaluation.
- Running the preparation recipe with seed 42 recovers **3133/409/387 slices** and **88/11/11 patients** for train/validation/test.
- The canonical records SHA256 is `8ea724f07ee8667c29411ce6758063c715bb6d0b6a67cd4e3cbebf38774d5391`.
- [Source receipt](https://github.com/kiselyovd/brain-mri-segmentation/blob/main/reports/provenance/lgg-source.json) and [complete split manifest](https://github.com/kiselyovd/brain-mri-segmentation/blob/main/reports/provenance/lgg-split-manifest.json) record the source, patients, image/mask filenames and SHA256 values. Dataset attribution and CC-BY-NC-SA-4.0 apply to the source-derived manifest.

This reconstructs the current code's split from the original data. It does **not** establish historical training membership for the published checkpoint: the old run did not retain that receipt.

## Foreground overlap

The previous implementation passed integer `B×H×W` masks into torchmetrics segmentation metrics with their default one-hot input contract. Those historical aggregates remain unchanged in the legacy reports; they are not interchangeable with the corrected measures.

`binary-foreground-global-v1` defines foreground as class 1 and pools integer TP, FP, FN, TN over all evaluated pixels before division:

- Dice = `2TP / (2TP + FP + FN)`.
- IoU = `TP / (TP + FP + FN)`.
- Pixel accuracy = `(TP + TN) / (TP + FP + FN + TN)`; background-heavy slices can make this misleadingly high.
- Empty prediction and target: Dice/IoU 1. Empty target with false positives: 0. No smoothing constant is added.
- Logits are upsampled to 256² using bilinear interpolation with `align_corners=False`, then argmax selects the class. Masks are resized with nearest-neighbor interpolation.
- Patient-macro scores average independently pooled patient scores. A seeded patient-cluster bootstrap reports percentile 95% intervals for cohort-pooled Dice/IoU. This is conditional on these 11 patients, not an external validation interval or a paired architecture comparison.

Validation and evaluation share the same count-based implementation. Tests cover hand-computed overlap, empty masks, invalid labels, batch invariance and changed cohort files.

## Replaying the native published artifact

Prepare into an empty destination (`.gitkeep` is allowed); stale files cause an error. Repeated runs should use a fresh output directory rather than combine old and new splits.

```bash
uv run python -m brain_mri_segmentation.data.prepare \
  --raw /path/to/kaggle_3m --out data/processed
uv run python -m brain_mri_segmentation.evaluation.evaluate \
  --hf-model kiselyovd/brain-mri-segmentation \
  --revision 838da0dff1f96aa3b87d56d2099f2b86dec483bc \
  --weights-sha256 b8e0816e76ee1f849ad37b119fc42647beba5072e842db958b848b0673dedfa4 \
  --data data/processed --out reports/replay.json
```

The evaluator verifies every test image and mask before inference. It writes the model identity, split hash, preprocessing and package versions, per-slice and per-patient counts, bootstrap intervals, and compressed prediction/target masks with a SHA256. The status explicitly says `training_membership_verified: false` for a reconstructed cohort.

For a recovered local baseline checkpoint use the same command with `--checkpoint artifacts/baseline/checkpoints/best.ckpt`, replacing the three HF arguments. The original U-Net checkpoint is absent from the current clone, public GitHub release and HF files. The dataset author's separately published U-Net weights are a different artifact and cannot replace this baseline in a historical replay.

## Native processor parity

The corrected exporter writes a 256² SegFormer processor with rescaling 1/255, no ImageNet normalization and no label reduction. The production dataset and inference CLI share `data.preprocessing.image_tensor`.

For an exact tensor-resize contract on input sizes other than the original 256² TIFs, pass an RGB **float32 array in [0,255]** to the native processor and select `backend="torchvision"`. This avoids uint8 rounding during resizing. Focused tests compare actual pixel tensors with a maximum absolute tolerance of 3e-7, including nonsquare inputs. The published processor revision is `48c79adfde8b063e74737357ab20a6f7f67a0b1b`; all other Hub files, including the weights, were unchanged. Actual model predictions matched on eight committed TIF samples and one seeded nonsquare input (zero disagreeing mask pixels); maximum tensor error was 1.79e-7 and maximum logits error was 5.25e-6. [The numerical receipt](https://github.com/kiselyovd/brain-mri-segmentation/blob/main/reports/provenance/processor-parity.json) records every case. This compares computation paths, not model quality. Replay intentionally pins the pre-fix revision and uses the repository transform.

See [component licenses](LICENSES.md) before reusing code, data or weights.
