from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from uapd import cli
from uapd.pipelines import PIPELINES


@pytest.fixture
def frame_folder(tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    for i, value in enumerate([0, 200, 0]):
        cv2.imwrite(str(folder / f"{i:03d}.png"), np.full((8, 8, 3), value, np.uint8))
    return folder


@pytest.fixture
def fake_registered(monkeypatch, fake_pipeline):
    monkeypatch.setitem(PIPELINES, "fake", lambda **options: fake_pipeline)
    return fake_pipeline


def test_config_prints_defaults(capsys):
    assert cli.main(["config"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["candidates"]["min_contrast"] == 23
    assert len(data["fingerprint"]) == 12


def test_config_reports_bad_profile(tmp_path, capsys):
    profile = tmp_path / "bad.toml"
    profile.write_text("[candidates]\nmin_contrst = 1\n")
    assert cli.main(["config", "--profile", str(profile)]) == 2
    assert "unknown key" in capsys.readouterr().err


def test_eval_writes_report_and_closes_pipeline(frame_folder, fake_registered, tmp_path, capsys):
    out = tmp_path / "report.json"
    code = cli.main(["eval", str(frame_folder), "--pipeline", "fake", "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_text())
    assert report["summary"]["frames"] == 3
    assert report["summary"]["detections"] == 1
    assert fake_registered.closed
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "3 frames, 1 detections" in captured.err


def test_eval_prints_report_to_stdout(frame_folder, fake_registered, capsys):
    assert cli.main(["eval", str(frame_folder), "--pipeline", "fake"]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["frames"] == 3


def test_eval_missing_source(tmp_path, capsys):
    assert cli.main(["eval", str(tmp_path / "missing.mp4")]) == 2
    assert "source not found" in capsys.readouterr().err


def test_eval_legacy_missing_weights(frame_folder, tmp_path, capsys):
    code = cli.main(["eval", str(frame_folder), "--weights", str(tmp_path / "none.pth")])
    assert code == 2
    assert "model weights not found" in capsys.readouterr().err
