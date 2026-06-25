"""Generate the model-card visuals from real SegFormer inference.

Produces two figures under ``reports/``:

1. ``segmentation_examples.png`` -- the hero qualitative panel. For a handful of
   test slices that actually contain a tumor, each row shows the MRI slice, the
   ground-truth mask overlay, and the model's predicted mask overlay, with the
   per-image Dice score (real prediction vs ground truth) in the title.
2. ``metrics_bar.png`` -- a grouped bar chart comparing the main SegFormer-B2
   model against the U-Net baseline on Dice / IoU / pixel accuracy.

Inference deliberately mirrors the evaluation pipeline in
``brain_mri_segmentation.evaluation.evaluate`` (resize to ``image_size`` with a
0..1 rescale, no ImageNet normalization, bilinear upsample of the logits, then
argmax) so the per-image Dice is consistent with ``reports/metrics.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from PIL import Image
from torchvision.transforms import v2
from transformers import SegformerForSemanticSegmentation

IMAGE_SIZE = 256
MIN_TUMOR_PIXELS = 500


def load_pair(img_path: Path, mask_path: Path, size: int = IMAGE_SIZE):
    """Return (rgb_uint8 HxWx3, image_tensor 3xHxW, mask_tensor HxW) at ``size``."""
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    img_arr = np.asarray(img, dtype=np.float32) / 255.0
    mask_arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.int64)

    img_t = torch.from_numpy(img_arr).permute(2, 0, 1)
    mask_t = torch.from_numpy(mask_arr)

    img_t = v2.Resize((size, size), antialias=True)(img_t)
    mask_t = v2.Resize(
        (size, size),
        interpolation=v2.InterpolationMode.NEAREST,
    )(mask_t.unsqueeze(0)).squeeze(0)

    rgb = (img_t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    gray = rgb.mean(axis=2).astype(np.uint8)
    return rgb, gray, img_t, mask_t


@torch.no_grad()
def predict_mask(model: SegformerForSemanticSegmentation, img_t: torch.Tensor) -> np.ndarray:
    """Run the SegFormer model and return a HxW binary prediction at input size."""
    out = model(img_t.unsqueeze(0))
    logits = out.logits
    logits = torch.nn.functional.interpolate(
        logits,
        size=img_t.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice for the foreground (tumor) class."""
    pred_b = pred > 0
    gt_b = gt > 0
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred_b, gt_b).sum() / denom)


def find_tumor_cases(
    img_dir: Path,
    mask_dir: Path,
    n: int,
    min_pixels: int,
) -> list[tuple[Path, Path]]:
    """Pick the first ``n`` slices whose ground-truth mask holds a real tumor."""
    pairs: list[tuple[Path, Path]] = []
    candidates = sorted(img_dir.glob("*.tif")) + sorted(img_dir.glob("*.png"))
    for img_path in candidates:
        mask_path = mask_dir / f"{img_path.stem}_mask.tif"
        if not mask_path.exists():
            mask_path = mask_dir / f"{img_path.stem}_mask.png"
        if not mask_path.exists():
            continue
        mask_arr = np.asarray(Image.open(mask_path).convert("L"))
        if int((mask_arr > 0).sum()) < min_pixels:
            continue
        pairs.append((img_path, mask_path))
        if len(pairs) >= n:
            break
    return pairs


def overlay(ax, gray: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> None:
    """Draw a grayscale MRI slice, a translucent mask fill, and a bright contour.

    The base is rendered in gray so the colored mask stays legible on the
    green-tinted FLAIR slices, and a solid contour line marks the boundary.
    """
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    cmap = ListedColormap([(0, 0, 0, 0), (*color, 1.0)])
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=cmap, alpha=0.5, interpolation="nearest")
    if mask.any():
        ax.contour(mask, levels=[0.5], colors=[color], linewidths=1.4)


def make_examples_plot(
    model_dir: Path, cases: list[tuple[Path, Path]], out_path: Path
) -> list[dict]:
    """Render the qualitative results panel and return per-case Dice records."""
    model = SegformerForSemanticSegmentation.from_pretrained(str(model_dir))
    model.train(False)

    gt_color = (0.95, 0.15, 0.15)  # red ground truth
    pred_color = (0.10, 0.85, 0.90)  # cyan prediction

    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(9.0, 3.0 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    records: list[dict] = []
    for row, (img_path, mask_path) in enumerate(cases):
        rgb, gray, img_t, gt = load_pair(img_path, mask_path)
        pred = predict_mask(model, img_t)
        gt_np = gt.cpu().numpy()
        d = dice_score(pred, gt_np)
        records.append({"slice": img_path.stem, "dice": d})

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"MRI slice\n{img_path.stem}", fontsize=9)

        overlay(axes[row, 1], gray, gt_np, gt_color)
        axes[row, 1].set_title("Ground truth (red)", fontsize=9)

        overlay(axes[row, 2], gray, pred, pred_color)
        axes[row, 2].set_title(f"Prediction (cyan) - Dice {d:.3f}", fontsize=9)

        for col in range(3):
            axes[row, col].axis("off")

    fig.suptitle(
        "SegFormer-B2 brain-tumor segmentation - qualitative results",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return records


def make_metrics_bar(main_path: Path, baseline_path: Path, out_path: Path) -> None:
    """Render a grouped bar chart of Dice / IoU / pixel accuracy (percentages)."""
    main = json.loads(main_path.read_text(encoding="utf-8"))
    base = json.loads(baseline_path.read_text(encoding="utf-8"))

    labels = ["Dice", "IoU", "Pixel acc."]
    keys = ["dice", "iou", "pixel_accuracy"]
    main_vals = [main[k] * 100.0 for k in keys]
    base_vals = [base[k] * 100.0 for k in keys]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars_main = ax.bar(
        x - width / 2, main_vals, width, label="SegFormer-B2 (main)", color="#4C6FFF"
    )
    bars_base = ax.bar(x + width / 2, base_vals, width, label="U-Net (baseline)", color="#9AA7C7")

    ax.set_ylabel("Score (%)")
    ax.set_title("Test-set metrics: SegFormer-B2 vs U-Net baseline", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 109)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for bars in (bars_main, bars_base):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="artifacts/hf_export")
    parser.add_argument("--images", default="data/sample/images")
    parser.add_argument("--masks", default="data/sample/masks")
    parser.add_argument("--fallback-images", default="data/processed/test/images")
    parser.add_argument("--fallback-masks", default="data/processed/test/masks")
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--out-examples", default="reports/segmentation_examples.png")
    parser.add_argument("--out-bar", default="reports/metrics_bar.png")
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--metrics-baseline", default="reports/metrics_baseline.json")
    args = parser.parse_args()

    cases = find_tumor_cases(
        Path(args.images),
        Path(args.masks),
        args.num_cases,
        MIN_TUMOR_PIXELS,
    )
    if len(cases) < args.num_cases:
        extra = find_tumor_cases(
            Path(args.fallback_images),
            Path(args.fallback_masks),
            args.num_cases,
            MIN_TUMOR_PIXELS,
        )
        seen = {p.stem for p, _ in cases}
        for pair in extra:
            if pair[0].stem not in seen:
                cases.append(pair)
            if len(cases) >= args.num_cases:
                break
    if not cases:
        raise SystemExit("No tumor-bearing slices found for the qualitative panel.")

    records = make_examples_plot(Path(args.model_dir), cases, Path(args.out_examples))
    dices = [r["dice"] for r in records]
    mean_dice = float(np.mean(dices))
    print(f"Wrote {args.out_examples}")
    for r in records:
        print(f"  {r['slice']}: Dice={r['dice']:.4f}")
    print(f"  mean tumor-case Dice={mean_dice:.4f}")

    make_metrics_bar(
        Path(args.metrics),
        Path(args.metrics_baseline),
        Path(args.out_bar),
    )
    print(f"Wrote {args.out_bar}")


if __name__ == "__main__":
    main()
