"""A changed report, fabricated score or mutable revision must fail the drift gate."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "artifact_manifest", ROOT / "scripts/artifact_manifest.py"
)
assert SPEC and SPEC.loader
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def fixture(root: Path) -> dict:
    manifest = json.loads((ROOT / "artifact-manifest.json").read_text())
    report = root / manifest["evaluation"]["report"]
    report.parent.mkdir(parents=True)
    report.write_bytes((ROOT / manifest["evaluation"]["report"]).read_bytes())
    numerical = json.loads(report.read_text(encoding="utf-8"))
    arrays = Path("reports") / numerical["prediction_arrays"]["file"]
    (root / arrays).write_bytes((ROOT / arrays).read_bytes())
    (root / "artifact-manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_committed_contract_and_prediction_arrays() -> None:
    manifest, values = contract.validate(ROOT)
    report_path = ROOT / manifest["evaluation"]["report"]
    assert b"\r" not in report_path.read_bytes()
    report = json.loads(report_path.read_text())
    arrays = ROOT / "reports" / report["prediction_arrays"]["file"]
    assert contract.digest(arrays) == report["prediction_arrays"]["sha256"]
    with np.load(arrays, allow_pickle=False) as archive:
        prediction, target = archive["predictions"].astype(bool), archive["targets"].astype(bool)
    counts = [
        int((prediction & target).sum()),
        int((prediction & ~target).sum()),
        int((~prediction & target).sum()),
        int((~prediction & ~target).sum()),
    ]
    assert counts == report["counts"]
    tp, fp, fn, _ = counts
    assert values["Foreground Dice"] == 2 * tp / (2 * tp + fp + fn)
    assert values["Foreground IoU"] == tp / (tp + fp + fn)
    assert prediction.shape == (387, 256, 256)


def test_changed_report_fails(tmp_path: Path) -> None:
    manifest = fixture(tmp_path)
    (tmp_path / manifest["evaluation"]["report"]).write_text("{}")
    with pytest.raises(ValueError, match="report SHA256"):
        contract.validate(tmp_path)


@pytest.mark.parametrize(
    "change", ["mutable_revision", "different_weights", "false_heldout", "missing_metric"]
)
def test_invalid_contract_fails(tmp_path: Path, change: str) -> None:
    manifest = fixture(tmp_path)
    if change == "mutable_revision":
        manifest["hub"]["revision"] = "main"
    elif change == "different_weights":
        manifest["hub"]["files"][0]["sha256"] = "0" * 64
    elif change == "false_heldout":
        manifest["evaluation"]["status"] = "verified_held_out"
    else:
        manifest["evaluation"]["metrics"][0]["pointer"] = "/missing"
    (tmp_path / "artifact-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, KeyError)):
        contract.validate(tmp_path)


def test_readme_drift_is_detected_and_repaired(tmp_path: Path, monkeypatch) -> None:
    manifest = fixture(tmp_path)
    _, values = contract.validate(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(contract.render(manifest, values), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["artifact_manifest", "--root", str(tmp_path), "--check"])
    contract.main()
    readme.write_text(
        readme.read_text(encoding="utf-8").replace("88.08%", "99.99%"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="drift"):
        contract.main()
    monkeypatch.setattr("sys.argv", ["artifact_manifest", "--root", str(tmp_path), "--write"])
    contract.main()
    assert "88.08%" in readme.read_text(encoding="utf-8")


def test_unavailable_results_render_without_fake_zero() -> None:
    text = contract.render(
        {
            "evaluation": {
                "status": "unavailable",
                "label": "Report missing",
                "scope": "No numeric claim available",
            }
        },
        {},
    )
    assert "Report missing" in text
    assert "0.00%" not in text


def test_path_cannot_escape_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        contract.local_path(tmp_path, "../outside.json")


def test_changed_prediction_arrays_fail(tmp_path: Path) -> None:
    manifest = fixture(tmp_path)
    report = json.loads((tmp_path / manifest["evaluation"]["report"]).read_text(encoding="utf-8"))
    path = tmp_path / "reports" / report["prediction_arrays"]["file"]
    path.write_bytes(b"altered numerical evidence")
    with pytest.raises(ValueError, match="Prediction array SHA256"):
        contract.validate(tmp_path)
