"""Convert 3 test slices with visible tumors to PNG for HF widget examples."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/processed/test")
    parser.add_argument("--dst", default="data/sample/widget")
    parser.add_argument("-n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tumor-pixels", type=int, default=500)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    candidates = sorted((src / "images").glob("*.tif"))
    rng.shuffle(candidates)

    picked = []
    for img_path in candidates:
        mask_path = src / "masks" / f"{img_path.stem}_mask.tif"
        if not mask_path.exists():
            continue
        mask_arr = np.asarray(Image.open(mask_path).convert("L"))
        if (mask_arr > 0).sum() < args.min_tumor_pixels:
            continue
        img = Image.open(img_path).convert("RGB")
        out_png = dst / f"{img_path.stem}.png"
        img.save(out_png, format="PNG")
        picked.append(out_png.name)
        if len(picked) >= args.n:
            break

    if len(picked) < args.n:
        raise SystemExit(f"Found only {len(picked)} slices with tumor >={args.min_tumor_pixels} px")
    print(f"Wrote {len(picked)} widget samples to {dst}: {picked}")


if __name__ == "__main__":
    main()
