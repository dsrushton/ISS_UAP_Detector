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


@pytest.mark.parametrize("pipeline", ["core", "legacy-v1"])
def test_eval_missing_weights(pipeline, frame_folder, tmp_path, capsys):
    weights = str(tmp_path / "none.pth")
    code = cli.main(["eval", str(frame_folder), "--pipeline", pipeline, "--weights", weights])
    assert code == 2
    assert "model weights not found" in capsys.readouterr().err


def test_eval_defaults_to_the_core_pipeline():
    args = cli.build_parser().parse_args(["eval", "clip.mp4"])
    assert args.pipeline == "core"


def test_eval_rejects_options_of_the_other_pipeline(frame_folder, tmp_path, capsys):
    profile = tmp_path / "p.toml"
    profile.write_text("[candidates]\nmin_contrast = 15\n")
    source = str(frame_folder)
    assert cli.main(["eval", source, "--rcnn-cycle", "5"]) == 2
    assert "--rcnn-cycle applies to legacy-v1" in capsys.readouterr().err
    code = cli.main(["eval", source, "--pipeline", "legacy-v1", "--profile", str(profile)])
    assert code == 2
    assert "--profile applies to the core pipeline" in capsys.readouterr().err
