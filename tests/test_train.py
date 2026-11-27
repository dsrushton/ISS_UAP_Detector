from __future__ import annotations

import json

import numpy as np
import pytest

from tests.test_labels import make_csv_folder
from uapd.cli import main
from uapd.config import ConfigError, RegionConfig
from uapd.train.config import TrainConfig
from uapd.train.data import DataError, plan_data
from uapd.train.metrics import ImageBoxes, score


def test_train_config_rejects_unknown_and_invalid_settings(tmp_path):
    with pytest.raises(ConfigError, match="unknown training setting"):
        TrainConfig.from_dict({"epoch": 3})
    with pytest.raises(ConfigError, match="background is added automatically"):
        TrainConfig.from_dict({"classes": ["background", "space"]})
    profile = tmp_path / "train.toml"
    profile.write_text('epochs = 3\nclasses = ["space", "earth"]\n[aliases]\nsky = "space"\n')
    config = TrainConfig.load(profile)
    assert (config.epochs, config.classes, config.aliases) == (
        3,
        ("space", "earth"),
        {"sky": "space"},
    )
    assert config.fingerprint() != TrainConfig().fingerprint()


def test_plan_maps_notebook_names_and_splits_by_position(tmp_path):
    folder = make_csv_folder(tmp_path / "frames_1", frames=10)
    plan = plan_data([tmp_path], TrainConfig())
    assert [it.image.name for it in plan.val] == ["frame_8.jpg", "frame_9.jpg"]
    assert len(plan.train) == 8
    # Space -> space (class 1) by case; Earth_broad -> earth (class 2) by alias.
    assert plan.train[0].labels == (1, 2)
    assert plan.report["classes"]["earth"] == {"train_boxes": 8, "val_boxes": 2}
    assert plan.report["folders"][0]["image_size"] == [160, 120]
    assert plan.report["folders"][0]["folder"] == str(folder)
    assert any("'sun' has no training boxes" in w for w in plan.report["warnings"])


def test_unknown_labels_and_bad_boxes_are_counted_not_trained(tmp_path):
    rows = [[0, 0, 50, 50, "Space"], [0, 0, 9, 9, "e_t"], [5, 5, 0, 10, "ISS"]]
    make_csv_folder(tmp_path / "f", frames=5, rows=rows)
    plan = plan_data([tmp_path / "f"], TrainConfig())
    assert plan.train[0].labels == (1,)
    assert plan.report["dropped_unknown_labels"] == {"e_t": 5}
    assert plan.report["dropped_bad_boxes"] == 5
    with pytest.raises(DataError, match=r"e_t \(5\)"):
        plan_data([tmp_path / "f"], TrainConfig(unknown_labels="error"))


def test_whole_folders_can_be_held_out(tmp_path):
    make_csv_folder(tmp_path / "a", frames=4)
    make_csv_folder(tmp_path / "b", frames=3)
    plan = plan_data([tmp_path], TrainConfig(val_folders=("b",)))
    assert (len(plan.train), len(plan.val)) == (4, 3)
    assert {it.image.parent.name for it in plan.val} == {"b"}


def test_a_path_without_labels_is_an_error(tmp_path):
    with pytest.raises(DataError, match="no labelled frames"):
        plan_data([tmp_path], TrainConfig())


def test_map_scores_matches_by_class_and_overlap():
    truth = [ImageBoxes(np.array([[0, 0, 10, 10], [20, 20, 40, 40]]), ("space", "earth"))]
    perfect = [ImageBoxes(truth[0].boxes, ("space", "earth"), (0.9, 0.8))]
    assert score(perfect, truth, ["space", "earth", "sun"])["ap"] == {
        "space": 1.0,
        "earth": 1.0,
        "sun": None,
    }
    assert score(perfect, truth, ["space", "earth", "sun"])["map"] == 1.0
    # Right place, wrong class; and right class, too little overlap.
    wrong = [
        ImageBoxes(np.array([[0, 0, 10, 10], [30, 30, 50, 50]]), ("earth", "earth"), (0.9, 0.8))
    ]
    assert score(wrong, truth, ["space", "earth"])["map"] == 0.0
    # A second box on an already matched object is a false alarm ranked below the hit.
    double = [ImageBoxes(np.array([[0, 0, 10, 10], [0, 0, 10, 9]]), ("space", "space"), (0.9, 0.8))]
    assert score(double, truth, ["space"])["ap"]["space"] == 1.0
    empty = [ImageBoxes(np.zeros((0, 4)), (), ())]
    assert score(empty, truth, ["space"])["map"] == 0.0


def test_train_check_reports_without_training(tmp_path, capsys):
    make_csv_folder(tmp_path / "frames_1", frames=5)
    assert main(["train", "--data", str(tmp_path), "--check", "--progress-json"]) == 0
    out = capsys.readouterr().out
    assert "4 training images, 1 validation images" in out
    event = json.loads(next(line for line in out.splitlines() if line.startswith("@uapd "))[6:])
    assert event["event"] == "check" and event["train_images"] == 4
    assert main(["train", "--check"]) == 2


def test_model_input_modes():
    torch = pytest.importorskip("torch")
    from uapd.regions.rcnn import IMAGENET_MEAN, IMAGENET_STD, model_input

    bgr = torch.zeros(3, 2, 2)
    bgr[0] = 1.0  # a pure blue frame as OpenCV gives it
    rgb = model_input(bgr, "rgb")
    assert rgb[2].min() == 1.0 and rgb[:2].max() == 0.0
    legacy = model_input(bgr, "legacy-v1")
    # V1's input: blue stays in the first channel and is normalised with the red statistics.
    assert legacy[0, 0, 0] == pytest.approx((1 - IMAGENET_MEAN[0]) / IMAGENET_STD[0])


def test_training_run_produces_weights_the_detector_loads(tmp_path):
    pytest.importorskip("torchvision")
    from uapd.regions import load_region_model
    from uapd.train import Reporter, run_training

    make_csv_folder(tmp_path / "data" / "frames_1", frames=6, size=(96, 64))
    config = TrainConfig(
        init="none",
        epochs=1,
        batch_size=2,
        min_size=64,
        max_size=96,
        learning_rate=0.0001,
        warmup_steps=1,
        val_fraction=0.34,
    )
    quiet = Reporter(stream=open(tmp_path / "log.txt", "w"), machine=True)  # noqa: SIM115
    run = run_training(config, [tmp_path / "data"], tmp_path / "models", quiet, device="cpu")
    assert run["status"] == "finished" and len(run["epochs"]) == 1
    best = tmp_path / "models" / next((tmp_path / "models").iterdir()).name / "best.pth"
    assert run["result"]["best_weights"] == str(best)
    assert json.loads((best.parent / "run.json").read_text())["data"]["train_images"] == 4

    model = load_region_model(best, RegionConfig())
    info = model.describe()
    assert info["input"] == "rgb" and info["manifest"] is True
    assert info["classes"][:2] == ["background", "space"]
    assert isinstance(model.detect(np.zeros((64, 96, 3), np.uint8)), list)
    # The manifest's input can be overridden from the config.
    assert load_region_model(best, RegionConfig(input="legacy-v1")).describe()["input"] == (
        "legacy-v1"
    )

    # Continue from those weights, scoring them as the baseline on the same images.
    again = run_training(
        config.replace(init=str(best)),
        [tmp_path / "data"],
        tmp_path / "models2",
        quiet,
        device="cpu",
        baseline=best,
    )
    assert again["baseline"]["input"] == "rgb"
    assert again["baseline"]["val"]["images"] == 2
    assert "@uapd " in (tmp_path / "log.txt").read_text()
