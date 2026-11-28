from __future__ import annotations

import sys
import time

import pytest

from uapd.app.jobs import Job, Settings, detect_command, parse_line, train_command


def test_settings_survive_a_save_and_a_damaged_file(tmp_path):
    path = tmp_path / "nested" / "app.json"
    Settings(source="clip.mp4", data=["a", "b"], epochs="3").save(path)
    loaded = Settings.load(path)
    assert (loaded.source, loaded.data, loaded.epochs) == ("clip.mp4", ["a", "b"], "3")
    path.write_text("{not json")
    assert Settings.load(path) == Settings()


def test_detect_command():
    assert detect_command(Settings())[1:] == [
        "-m",
        "uapd",
        "run",
        "--weights",
        "best_map_model.pth",
    ]
    command = detect_command(Settings(source=" clip.mp4 ", weights="m/best.pth", profile="p.toml"))
    assert command[4:] == ["clip.mp4", "--weights", "m/best.pth", "--profile", "p.toml"]
    both = detect_command(Settings(weights="seg/best.pt", second_weights="best_map_model.pth"))
    assert both[4:] == ["--weights", "seg/best.pt", "--second-weights", "best_map_model.pth"]


def test_train_command(tmp_path):
    with pytest.raises(ValueError, match="Add at least one folder"):
        train_command(Settings())
    with pytest.raises(ValueError, match="epochs is not a number"):
        train_command(Settings(data=["d"], epochs="ten"))
    weights = tmp_path / "current.pth"
    weights.write_bytes(b"")
    settings = Settings(data=["d1", "d2"], weights=str(weights), epochs="2")
    command = train_command(settings)
    assert command[3:10] == ["train", "--progress-json", "--data", "d1", "--data", "d2", "--epochs"]
    assert command[-2:] == ["--baseline", str(weights)]
    # No comparison when the current weights are not there; no output folder for a check.
    assert "--baseline" not in train_command(Settings(data=["d"], weights="missing.pth"))
    assert train_command(settings, check=True)[-1] == "--check"
    assert "--out" not in train_command(settings, check=True)


def test_progress_lines_are_told_from_text():
    assert parse_line('@uapd {"event": "step", "loss": 1.5}\n') == (
        "event",
        {"event": "step", "loss": 1.5},
    )
    assert parse_line("Epoch 1/2: loss 0.5\n") == ("text", "Epoch 1/2: loss 0.5")
    assert parse_line("@uapd not json") == ("text", "@uapd not json")


def test_job_collects_output_and_exit_code(tmp_path):
    job = Job([sys.executable, "-c", "print('one'); print('two'); raise SystemExit(3)"], tmp_path)
    lines, code, deadline = [], None, time.time() + 30
    while code is None and time.time() < deadline:
        new, code = job.poll()
        lines += new
        time.sleep(0.02)
    assert [line.strip() for line in lines] == ["one", "two"]
    assert code == 3


def test_train_command_for_the_outline_trainer_and_unchanged_for_boxes():
    boxes = train_command(Settings(data=["frames"]))
    assert "--kind" not in boxes and "--config" not in boxes
    outlines = train_command(
        Settings(data=["frames"], train_kind="yolo-seg", train_config="models/seg.toml")
    )
    assert outlines[outlines.index("--kind") + 1] == "yolo-seg"
    assert outlines[outlines.index("--config") + 1] == "models/seg.toml"
    assert "--baseline" not in outlines
