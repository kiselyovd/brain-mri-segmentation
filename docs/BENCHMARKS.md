# Benchmarks

All numbers are on the held-out test split: 387 axial slices from 11 TCGA patients never seen during training. Hardware: RTX 3080 10 GB. Inference is single-slice, FP32.

## Main results

| Model | Dice | IoU | Pixel acc | Params | Inference (ms/slice, RTX 3080) |
|---|---|---|---|---|---|
| **SegFormer-B2** (ours, main) | **65.5%** | **66.2%** | **99.73%** | ~27 M | ~22 ms |
| U-Net (ours, baseline, 32→256 ch) | 51.9% | 57.7% | 99.66% | ~1.9 M | ~5 ms |

## Literature context

Reported numbers on the same LGG / TCGA kaggle_3m split vary by paper; here are representative figures:

| Model | Dice | Source |
|---|---|---|
| U-Net (literature, Buda 2019) | ~82% | Original paper, larger U-Net, more epochs |
| U-Net++ | ~70% | Published replications 2020-2022 |
| Attention U-Net | ~68% | Published replications |
| **SegFormer-B2** (ours) | **65.5%** | This repo, v0.1.0, early-stopped at epoch 13/99 |
| U-Net-tiny (ours, baseline) | 51.9% | This repo, 1.9M-param reference |

Our SegFormer-B2 run was early-stopped (validation Dice plateaued) after just 13 epochs — it's a demonstration of the pipeline, not a tuned SOTA. More training epochs + data augmentation + auxiliary deep-supervision losses would close the gap to Buda's original ~82%.

## Trade-offs

- **SegFormer vs U-Net**: SegFormer wins on Dice by ~14pp thanks to global self-attention — but costs 14x more parameters and 4x more inference latency. For edge deployment the tiny U-Net is a reasonable choice despite the weaker Dice.
- **Why not bigger SegFormer variants**: SegFormer-B4/B5 would likely add 3-5pp Dice but don't fit comfortably alongside a training pipeline on 10 GB VRAM at batch 16 × 256² input.
- **Why 2D, not 3D**: a 3D U-Net would exploit through-plane context and likely add 5-10pp Dice, but requires 10x the memory and doesn't fit the "single-RTX-3080" constraint of this portfolio.

## Reproducing these numbers

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the one-command re-run. Expected variation: ± 1% from test-set size (small n=387) and floating-point noise.
