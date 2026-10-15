"""Source split identity and rejection of unsafe preparation inputs."""

import json
from pathlib import Path

import pytest

from brain_mri_segmentation.data.prepare import prepare_data


def cohort(root: Path) -> None:
    for patient in ("TCGA_A", "TCGA_B", "TCGA_C", "TCGA_D", "TCGA_E"):
        folder = root / patient
        folder.mkdir(parents=True)
        (folder / f"{patient}_1.tif").write_bytes(b"image")
        (folder / f"{patient}_1_mask.tif").write_bytes(b"mask")


def test_manifest_is_reproducible_and_records_bytes(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    cohort(raw)
    a, b = tmp_path / "a", tmp_path / "b"
    prepare_data(raw, a)
    prepare_data(raw, b)
    assert (a / "split-manifest.json").read_bytes() == (b / "split-manifest.json").read_bytes()
    manifest = json.loads((a / "split-manifest.json").read_text())
    assert manifest["image_counts"] == {"train": 3, "val": 1, "test": 1}
    assert len({r["patient"] for r in manifest["records"]}) == 5
    assert all(len(r["image_sha256"]) == 64 for r in manifest["records"])
    (raw / "TCGA_A/TCGA_A_1.tif").write_bytes(b"changed")
    c = tmp_path / "c"
    prepare_data(raw, c)
    changed = json.loads((c / "split-manifest.json").read_text())
    assert manifest["records_sha256"] != changed["records_sha256"]


def test_nonempty_destination_is_preserved(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    cohort(raw)
    out.mkdir()
    sentinel = out / "unrelated.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="empty"):
        prepare_data(raw, out)
    assert sentinel.read_text() == "preserve"


def test_tracked_gitkeep_does_not_block_preparation(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    cohort(raw)
    out.mkdir()
    (out / ".gitkeep").touch()
    prepare_data(raw, out)
    assert (out / "split-manifest.json").is_file()


def test_invalid_pairs_fail_before_copy(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    cohort(raw)
    (raw / "TCGA_A/TCGA_A_1_mask.tif").unlink()
    with pytest.raises(ValueError, match="pairs"):
        prepare_data(raw, out)
    assert not out.exists()


@pytest.mark.parametrize("val,test", [(0, 0.1), (0.8, 0.2), (-0.1, 0.1)])
def test_invalid_fractions_fail(tmp_path: Path, val: float, test: float) -> None:
    raw = tmp_path / "raw"
    cohort(raw)
    with pytest.raises(ValueError, match="fractions"):
        prepare_data(raw, tmp_path / "out", val_frac=val, test_frac=test)
