# Limitations and provenance

Research and education only; the repository does not provide clinical validation.

**Legacy reported aggregates — metric-definition audit pending.** Stored values are retained for traceability, not as verified performance claims. The Dice/IoU definitions, checkpoint/config lineage and per-patient evaluation manifest need reconciliation before comparisons.

## Known scope

- The source is the Buda LGG/TCGA dataset: a limited low-grade-glioma cohort, not a representative clinical population.
- The model processes individual 2D slices, without a 3D volumetric or longitudinal context.
- Preparation code uses a patient-level split. The stored metric files do not record the complete patient/slice manifest or checkpoint lineage of the legacy runs.
- Input processing matters: repository inference scales RGB to [0, 1] and resizes the tensor to 256×256. The corrected native HF processor is tested against this path; use the pinned loading recipe and float32 RGB arrays described in the README.
- The API returns a binary mask. No calibrated uncertainty or externally validated operating point is published.

## Qualitative examples

The published panel contains selected tumor-bearing samples. `make_plots.py` prefers `data/sample`; `build_sample_data.py` defaults to a training source. A recorded split/patient manifest for that panel is absent, so it must not be presented as held-out test proof.

Small lesions, ambiguous boundaries, image/sequence shift and slices without tumor are relevant cases to inspect. The new replay includes per-slice/patient counts and patient-cluster bootstrap intervals conditional on the reconstructed cohort. It does not measure external population shift or establish architecture gains. The legacy aggregate reports do not provide those uncertainty estimates.

## Before interpreting quality

Reconcile the metric definitions and export per-patient predictions/masks with their provenance. Do not infer clinical suitability or comparative architecture capacity from the current legacy table.
