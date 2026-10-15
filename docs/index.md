# Brain MRI segmentation

A fine-tuned SegFormer-B2 for FLAIR-abnormality masks on 2D LGG MRI slices. The project provides native weights, training and serving code, and a replay with identified source data and explicit foreground measures.

[Run the model](https://github.com/kiselyovd/brain-mri-segmentation#run-the-published-model) · [Model & weights](https://huggingface.co/kiselyovd/brain-mri-segmentation) · [Source](https://github.com/kiselyovd/brain-mri-segmentation)

<!-- ARTIFACT-MANIFEST:START -->
### Current artifact and evaluation

**Replayed on the reconstructed original patient split · 387 slices / 11 patients**

| Metric | Value |
|---|---:|
| Foreground Dice | 88.08% |
| Foreground IoU | 78.70% |

Foreground scores pool confusion counts across all test pixels. Historical checkpoint training membership is unverified; this is not an independently established held-out benchmark or a replay of the missing U-Net baseline. Patient-macro Dice: see the full report.

[Evaluation record](https://github.com/kiselyovd/brain-mri-segmentation/blob/main/reports/replay-segformer-2026-10-15.json) · [Pinned model](https://huggingface.co/kiselyovd/brain-mri-segmentation/tree/48c79adfde8b063e74737357ab20a6f7f67a0b1b)
<!-- ARTIFACT-MANIFEST:END -->

## What changed in evaluation

The original Kaggle cohort reproduces the expected 3133/409/387 slices and 88/11/11 patients. Preparation records file identities and rejects stale destinations. Validation and evaluation share foreground confusion-count metrics; the report includes per-patient results, bootstrap intervals and compressed prediction/target masks.

The native processor is corrected to 256², rescaling 1/255 and no ImageNet normalization. Tests compare production preprocessing, native tensors and actual model predictions.

[Evaluation contract](EVALUATION.md) · [Reproducibility](REPRODUCIBILITY.md) · [Legacy records](BENCHMARKS.md)

## Scope and reuse

The original U-Net baseline checkpoint is missing. The published main checkpoint's historical training membership is unverified. These limits prevent an independently established held-out benchmark or a historical architecture comparison. The limited LGG cohort and 2D input do not establish clinical validity.

[Limitations](LIMITATIONS.md) · [Component licenses](LICENSES.md) · [Training](training.md) · [Serving](serving.md)
