"""CLI entrypoint: python -m brain_mri_segmentation"""
from __future__ import annotations

import sys


def main() -> int:
    print("brain-mri-segmentation — use make train / make evaluate / make serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
