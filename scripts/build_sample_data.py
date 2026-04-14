"""Populate data/sample/{images,masks}/ with a tiny paired subset for CI."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def build_sample(src: Path, dst: Path, n: int = 8, seed: int = 42) -> None:
    rng = random.Random(seed)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "masks").mkdir(parents=True, exist_ok=True)
    candidates = sorted((src / "images").glob("*.tif"))
    if len(candidates) < n:
        raise SystemExit(f"Not enough in {src / 'images'}: need {n}, have {len(candidates)}")
    chosen = rng.sample(candidates, n)
    for img in chosen:
        mask = src / "masks" / f"{img.stem}_mask.tif"
        if not mask.exists():
            raise SystemExit(f"Missing mask for {img}")
        shutil.copy2(img, dst / "images" / img.name)
        shutil.copy2(mask, dst / "masks" / mask.name)
        print(f"wrote {img.name} (+mask)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/processed/train")
    p.add_argument("--dst", default="data/sample")
    p.add_argument("-n", type=int, default=8)
    args = p.parse_args()
    build_sample(Path(args.src), Path(args.dst), n=args.n)
