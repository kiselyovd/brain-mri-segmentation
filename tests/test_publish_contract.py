"""Publication is owner-guarded and stages only explicitly reviewed artifacts."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def environment(tmp_path: Path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    publisher = importlib.import_module("publish_to_hf")
    manifest = json.loads((ROOT / "artifact-manifest.json").read_text(encoding="utf-8"))
    export = tmp_path / "artifacts/hf_export"
    export.mkdir(parents=True)
    for item in manifest["hub"]["files"]:
        data = item["path"].encode()
        (export / item["path"]).write_bytes(data)
        item["sha256"] = hashlib.sha256(data).hexdigest()
    paths = [
        manifest["evaluation"]["report"],
        manifest["dataset"]["source_receipt"],
        manifest["dataset"]["split_manifest"],
        "reports/provenance/processor-parity.json",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    report = json.loads((tmp_path / paths[0]).read_text())
    arrays = Path("reports") / report["prediction_arrays"]["file"]
    (tmp_path / arrays).write_bytes((ROOT / arrays).read_bytes())
    (tmp_path / "artifact-manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "artifacts/private.env").write_text("dummy unrelated artifact")
    (export / "unreviewed.txt").write_text("not in reviewed files")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(publisher, "validate", lambda _: (manifest, {"Foreground Dice": 0.8808}))
    monkeypatch.setattr(
        "sys.argv", ["publish_to_hf", "--template", str(ROOT / "docs/model_card.md.j2")]
    )
    return publisher


def test_prepare_has_no_authentication_or_unreviewed_files(tmp_path: Path, monkeypatch) -> None:
    publisher = environment(tmp_path, monkeypatch)

    def forbidden_api(**kwargs):
        raise AssertionError("Prepare must not authenticate")

    monkeypatch.setattr(publisher, "HfApi", forbidden_api)
    output = tmp_path / "prepared"
    monkeypatch.setattr("sys.argv", [*sys.argv, "--prepare-only", str(output)])
    publisher.main()
    assert not (output / "private.env").exists()
    assert not (output / "unreviewed.txt").exists()
    card = (output / "README.md").read_text(encoding="utf-8")
    assert "88.08%" in card
    assert "license: other" in card
    assert "model-index:" not in card
    assert "48c79adfde8b063e74737357ab20a6f7f67a0b1b" in card


def test_wrong_hf_account_cannot_publish(tmp_path: Path, monkeypatch) -> None:
    publisher = environment(tmp_path, monkeypatch)

    class WrongAccount:
        def whoami(self):
            return {"name": "umbrellaitdaniilkiselev"}

        def upload_folder(self, **kwargs):
            raise AssertionError("Wrong owner must not publish")

    monkeypatch.setattr(publisher, "HfApi", lambda **_: WrongAccount())
    with pytest.raises(SystemExit, match="owner guard"):
        publisher.main()


def test_changed_export_cannot_publish(tmp_path: Path, monkeypatch) -> None:
    publisher = environment(tmp_path, monkeypatch)
    (tmp_path / "artifacts/hf_export/model.safetensors").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="differs from reviewed"):
        publisher.main()


def test_wrong_target_cannot_publish(tmp_path: Path, monkeypatch) -> None:
    publisher = environment(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", [*sys.argv, "--repo-id", "other/model"])
    with pytest.raises(SystemExit, match="Target differs"):
        publisher.main()
