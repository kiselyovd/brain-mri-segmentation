# Evaluation records

The corrected replay is recorded in [replay-segformer-2026-10-15.json](https://github.com/kiselyovd/brain-mri-segmentation/blob/main/reports/replay-segformer-2026-10-15.json). Its result block is generated from the artifact manifest in the [README](https://github.com/kiselyovd/brain-mri-segmentation#current-artifact-and-evaluation) and documentation home page. It evaluates the published main weights on the reconstructed original 387-slice/11-patient split; historical training membership is unverified.

See the [explicit foreground metric contract, source receipt and processor parity](EVALUATION.md). The original baseline checkpoint is still absent, so a corrected historical comparison is unavailable.

## Legacy benchmark records

**Legacy reported aggregates — metric-definition audit pending.** Stored values are retained for traceability, not as verified performance claims. The Dice/IoU definitions, checkpoint/config lineage and per-patient evaluation manifest need reconciliation before comparisons.

The repository reports an evaluation size of 387 slices. The stored JSON does not include the per-slice/patient arrays or the checkpoint/config/data hashes needed to establish the complete run lineage.

| Historical record | Dice | IoU | Pixel accuracy |
|---|---|---|---|
| SegFormer-B2 main | 65.5% | 66.2% | 99.73% |
| U-Net baseline | 51.9% | 57.7% | 99.66% |

Values remain in their original fields; they have not been swapped or recomputed. This table cannot establish a reliable architecture advantage while definitions and aggregation remain unresolved. Published plots visualize the same legacy values.

## What the audit requires

- Exact checkpoint and model/config hash for each report.
- Original patient/slice evaluation manifest and mask/prediction arrays.
- Shared foreground labels, threshold, empty-mask handling and aggregation definitions.
- Patient-level uncertainty estimates computed from those arrays, rather than guessed confidence intervals.
- Recorded preprocessing: the repository uses 256×256 and [0, 1]; the corrected native processor is pinned and tested against that tensor transform. The pre-fix revisions retain their original different configuration.

[Reproduction commands](REPRODUCIBILITY.md) document the workflow, without promising recovery of the old numbers. [Limitations](LIMITATIONS.md) separates known source limits from unmeasured failure rates.
