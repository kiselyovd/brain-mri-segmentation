"""Replay refuses changed cohorts and records patient-level uncertainty."""

import json
from pathlib import Path

import pytest

from brain_mri_segmentation.data.prepare import prepare_data
from brain_mri_segmentation.evaluation.replay import patient_bootstrap, verified_test_records


def prepared_cohort(tmp_path: Path) -> Path:
    raw, out = tmp_path / "raw", tmp_path / "processed"
    for patient in ("TCGA_A", "TCGA_B", "TCGA_C", "TCGA_D", "TCGA_E"):
        directory = raw / patient
        directory.mkdir(parents=True)
        (directory / f"{patient}_1.tif").write_bytes(b"image")
        (directory / f"{patient}_1_mask.tif").write_bytes(b"mask")
    prepare_data(raw, out)
    return out


def test_replay_verifies_exact_content_and_membership(tmp_path: Path) -> None:
    out = prepared_cohort(tmp_path)
    _, records = verified_test_records(out)
    assert len(records) == 1
    image = out / "test/images" / Path(records[0]["image"]).name
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="content hash"):
        verified_test_records(out)
    image.write_bytes(b"image")
    (out / "test/images/stale.tif").write_bytes(b"stale")
    with pytest.raises(ValueError, match="files differ"):
        verified_test_records(out)


def test_changed_manifest_rejected(tmp_path: Path) -> None:
    out = prepared_cohort(tmp_path)
    path = out / "split-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["records"][0]["patient"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest content hash"):
        verified_test_records(out)


def test_patient_cluster_bootstrap_is_reproducible() -> None:
    patients = [{"counts": [1, 0, 0, 3]}, {"counts": [0, 1, 1, 2]}]
    result = patient_bootstrap(patients, repeats=100)
    assert result == patient_bootstrap(patients, repeats=100)
    assert result["intervals"]["dice"] == [0.0, 1.0]
