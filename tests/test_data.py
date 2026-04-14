"""Data layer smoke tests."""

from __future__ import annotations

from pathlib import Path

from brain_mri_segmentation.data.prepare import prepare_data


def test_prepare_splits_by_patient(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    for patient in ("TCGA_A", "TCGA_B", "TCGA_C", "TCGA_D", "TCGA_E"):
        (raw / patient).mkdir(parents=True)
        for i in range(3):
            (raw / patient / f"{patient}_{i}.tif").write_bytes(b"IMG")
            (raw / patient / f"{patient}_{i}_mask.tif").write_bytes(b"MASK")

    prepare_data(raw, out, seed=0, val_frac=0.2, test_frac=0.2)

    all_images = list(out.rglob("*.tif"))
    assert len(all_images) == 30

    for patient in ("TCGA_A", "TCGA_B", "TCGA_C", "TCGA_D", "TCGA_E"):
        splits_with_patient = {
            split
            for split in ("train", "val", "test")
            if list((out / split / "images").glob(f"{patient}_*.tif"))
        }
        assert len(splits_with_patient) == 1, (
            f"{patient} leaked across splits: {splits_with_patient}"
        )

    for split in ("train", "val", "test"):
        imgs = sorted(p.stem for p in (out / split / "images").glob("*.tif"))
        msks = sorted(
            p.stem.removesuffix("_mask") for p in (out / split / "masks").glob("*_mask.tif")
        )
        assert imgs == msks
