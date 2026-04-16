# Limitations & Failure Modes

This page is a deliberate call-out of where the published model falls short. Read it before using the weights for anything beyond demo / research.

## Dataset scope

- **Small cohort**: 110 patients (~3929 2D slices) from TCGA-LGG — the Buda et al. (2019) Kaggle release. Low-grade glioma only; high-grade glioma (HGG), metastases, meningiomas, and non-brain pathology are out of distribution.
- **Scanner bias**: The kaggle_3m set comes from ~4 scanner models in TCGA archives. Field-strength and sequence variations across real-world scanners will shift performance.
- **2D slice-level model**: The architecture processes each axial slice independently. Volumetric context (through-plane continuity of tumor boundary) is not exploited — a 3D U-Net or volumetric transformer would likely do better.
- **Small held-out test set**: 387 slices from 11 patients. Reported Dice=65.5% has ~±3pp confidence interval.

## Segmentation-specific failure modes

- **Thin tumor edges get under-segmented**: Dice of ~65% reflects edge-voxel misses. The model is confident in the tumor core but less so at the periphery.
- **Hyperintense non-tumor regions get over-segmented**: flair-hyperintense but non-tumor tissue (edema, gliosis) sometimes gets predicted as tumor.
- **Cross-midline tumors underperform**: tumors extending across the brain midline miss the far-side extension more often than expected.
- **Empty-mask slices**: ~40% of slices in the dataset are tumor-free. The model handles these correctly most of the time but occasionally predicts tiny spurious blobs.

## Not a medical device

- Not FDA-cleared, not CE-marked, not clinically validated.
- Any clinical use requires independent validation by qualified neuroradiologists.
- Do not use for treatment planning or patient-facing diagnostics.

## Adversarial & reliability

- No adversarial or corruption robustness testing.
- No uncertainty estimation — single mask output, no per-voxel confidence. Monte Carlo dropout or a Bayesian head would help.

## What this project *is* good for

- A production ML pipeline template: Lightning + Hydra + MLflow + DVC + FastAPI + Docker + HF Hub + CI/CD.
- A reproducible baseline for brain-tumor segmentation research.
- Comparing SegFormer vs a small U-Net on medical imaging with a modest compute budget.
