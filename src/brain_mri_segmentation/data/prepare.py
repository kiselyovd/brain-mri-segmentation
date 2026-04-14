"""Patient-level train/val/test split for the LGG Brain MRI dataset."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

from ..utils import get_logger

log = get_logger(__name__)

SPLITS = ("train", "val", "test")


def _collect_patients(raw: Path) -> list[Path]:
    return sorted(p for p in raw.iterdir() if p.is_dir() and p.name.startswith("TCGA_"))


def prepare_data(
    raw_dir: Path | str,
    processed_dir: Path | str,
    *,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> None:
    raw = Path(raw_dir)
    out = Path(processed_dir)
    log.info("prepare_data.start", raw=str(raw), out=str(out))

    patients = _collect_patients(raw)
    if not patients:
        raise SystemExit(f"No TCGA_* patient dirs under {raw}")
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_total = len(patients)
    n_val = max(1, int(round(n_total * val_frac)))
    n_test = max(1, int(round(n_total * test_frac)))
    n_train = n_total - n_val - n_test

    splits: dict[str, list[Path]] = {
        "train": patients[:n_train],
        "val": patients[n_train : n_train + n_val],
        "test": patients[n_train + n_val :],
    }

    for split, split_patients in splits.items():
        images_dir = out / split / "images"
        masks_dir = out / split / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        for patient in split_patients:
            for src in patient.iterdir():
                if not src.is_file() or src.suffix.lower() != ".tif":
                    continue
                dst = masks_dir / src.name if src.stem.endswith("_mask") else images_dir / src.name
                shutil.copy2(src, dst)

    counts = {s: len(list((out / s / "images").glob("*.tif"))) for s in SPLITS}
    log.info("prepare_data.done", patients={s: len(v) for s, v in splits.items()}, images=counts)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    prepare_data(args.raw, args.out, seed=args.seed)
