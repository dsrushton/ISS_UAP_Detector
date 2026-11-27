"""The Faster R-CNN box trainer: the script that made ``best_map_model.pth``, rebuilt.

That script is ``Pytorch_Training_Jan25.py`` (January 2025; see docs/V2_PLAN.md,
section 10, for where it was found and how it was matched to the weights).

What is the same: torchvision's Faster R-CNN (ResNet-50 FPN) with a new box
head, started from the COCO-pretrained detector, trained on RGB frames with
Adam at 0.0001, batch 8, mixed precision, gradient clipping and random flips.

What that script got wrong and this does not:

* The split was random over consecutive video frames, so validation frames were
  near copies of training frames. The split is by position in each folder.
* Validation frames were randomly flipped too. Only training frames are.
* The learning rate was cut when the score stopped *falling*, that is, while
  the model was improving. Here it drops once, late in the run.
* The class weighting scaled the whole loss by one number, so it did nothing
  per class. It is left out.
* Frames were normalised twice (once in augmentation, once inside the model),
  and the V1 app then fed BGR where training had fed RGB. Frames are normalised
  once, and each weights file gets a manifest (classes, input, data, versions)
  that the detector reads, so detection prepares frames exactly as training did.
"""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from uapd.regions.manifest import (
    INPUT_LEGACY_V1,
    INPUT_RGB,
    INPUT_RGB_NORMALIZED,
    KIND_RCNN,
    LEGACY_MANIFEST,
    ModelManifest,
    read_manifest,
    write_manifest,
)
from uapd.regions.rcnn import build_model, model_input, sha256_file
from uapd.train import TrainError, TrainJob, now
from uapd.train.config import TrainConfig
from uapd.train.data import DataError, Item
from uapd.train.metrics import ImageBoxes, score

PROGRESS_SECONDS = 0.5


class Frames:
    """Map-style dataset: (model input, target) per labelled frame."""

    def __init__(
        self,
        items: list[Item],
        input_mode: str = INPUT_RGB,
        hflip: bool = False,
        vflip: bool = False,
    ):
        self.items = items
        self.input_mode = input_mode
        self.hflip = hflip
        self.vflip = vflip

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        import cv2
        import torch

        item = self.items[index]
        frame = cv2.imread(str(item.image), cv2.IMREAD_COLOR)  # BGR, as the detector gets it
        if frame is None:
            raise DataError(f"could not read image {item.image}")
        height, width = frame.shape[:2]
        image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(item.boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(item.labels, dtype=torch.int64)
        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
        boxes, labels = boxes[keep], labels[keep]
        if self.hflip and torch.rand(()) < 0.5:
            image = image.flip(2)
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        if self.vflip and torch.rand(()) < 0.5:
            image = image.flip(1)
            boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
        return model_input(image, self.input_mode), {"boxes": boxes, "labels": labels}


def collate(batch):
    return tuple(zip(*batch, strict=True))


def train_rcnn(job: TrainJob) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise TrainError(
            f"training needs PyTorch and torchvision: {exc} "
            "(install with: pip install -e '.[torch]')"
        ) from exc

    config, plan, report = job.config, job.plan, job.reporter
    torch.manual_seed(config.seed)
    device = torch.device(job.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classes = ("background", *config.classes)
    model_args = {"min_size": config.min_size, "max_size": config.max_size}

    model, init = _initial_model(config, classes, model_args, torch, report)
    model.to(device)

    train_loader = DataLoader(
        Frames(plan.train, hflip=config.hflip, vflip=config.vflip),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        collate_fn=collate,
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.epochs
    optimizer, scheduler = _optimizer(model, config, total_steps, torch)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    truths = [
        ImageBoxes(
            np.array(item.boxes, np.float64).reshape(-1, 4),
            tuple(classes[i] for i in item.labels),
        )
        for item in plan.val
    ]

    def evaluate(network: Any, names: tuple[str, ...], input_mode: str) -> dict[str, Any]:
        loader = DataLoader(
            Frames(plan.val, input_mode),
            batch_size=config.batch_size,
            num_workers=config.workers,
            collate_fn=collate,
        )
        predictions = _predict(network, loader, device, names, torch)
        return score(predictions, truths, config.classes, config.iou_threshold)

    report.log(
        f"Training {KIND_RCNN} on {device}: {len(plan.train)} images, "
        f"{config.epochs} epochs of {steps_per_epoch} steps"
    )
    if device.type == "cpu":
        report.log("No GPU found: training on the CPU is very slow.")
    report.event(
        "start",
        device=str(device),
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        train_images=len(plan.train),
        val_images=len(plan.val),
        run_dir=str(job.run_dir),
    )

    if job.baseline is not None and plan.val:
        job.run["baseline"] = _score_baseline(job.baseline, evaluate, device, torch)
        job.save_run()
        for line in _baseline_lines(job.baseline.name, job.run["baseline"]):
            report.log(line)
        report.event("baseline", **job.run["baseline"])

    def record(epoch: int, val: dict[str, Any] | None) -> dict[str, Any]:
        data_keys = (
            "train_images",
            "val_images",
            "classes",
            "dropped_unknown_labels",
            "labels_sha256",
        )
        return {
            "trained": {
                "run": job.run_dir.name,
                "epoch": epoch,
                "saved": now(),
                "uapd_version": job.run["uapd_version"],
                "commit": job.run["commit"],
                "train_config": config.to_dict(),
                "train_config_fingerprint": config.fingerprint(),
                "init": init,
                "data": {key: plan.report[key] for key in data_keys},
                "val": val,
            }
        }

    last, best = job.run_dir / "last.pth", job.run_dir / "best.pth"
    best_map, best_epoch = -1.0, 0
    # Early stopping: epochs since the score last rose by more than min_improvement.
    reference, stale, stopped_early = -1.0, 0, False
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        loss = _train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch, config, report, torch
        )
        val = evaluate(model, classes, INPUT_RGB) if plan.val else None

        torch.save(model.state_dict(), last)
        saved = ModelManifest(
            KIND_RCNN,
            classes,
            INPUT_RGB,
            model_args=model_args,
            record={**record(epoch, val), "weights_sha256": sha256_file(last)},
        )
        write_manifest(last, saved)

        value = val["map"] if val and val["map"] is not None else None
        is_best = value is None or value >= best_map
        if is_best:
            best_map, best_epoch = (value if value is not None else best_map), epoch
            shutil.copyfile(last, best)
            write_manifest(best, saved)

        seconds = round(time.perf_counter() - started, 1)
        entry = {
            "epoch": epoch,
            "loss": round(loss, 4),
            "val": val,
            "seconds": seconds,
            "best": is_best,
        }
        job.run["epochs"].append(entry)
        job.save_run()
        shown = "-" if value is None else f"{value:.4f}"
        report.log(
            f"Epoch {epoch}/{config.epochs}: loss {loss:.4f}, mAP {shown}"
            f"{' (best so far)' if is_best and value is not None else ''}, {seconds:.0f} s"
        )
        report.event("epoch", epochs=config.epochs, **entry)

        if value is not None and config.patience > 0:
            if reference < 0 or value > reference + config.min_improvement:
                reference, stale = value, 0
            else:
                stale += 1
            if stale >= config.patience and epoch < config.epochs:
                stopped_early = True
                report.log(
                    f"Stopping early: the validation score has not risen by more than "
                    f"{config.min_improvement} for {stale} epochs."
                )
                break

    result = {
        "best_weights": str(best),
        "best_epoch": best_epoch,
        "best_map": best_map if best_map >= 0 else None,
        "stopped_early": stopped_early,
        "baseline_map": job.run.get("baseline", {}).get("val", {}).get("map"),
    }
    report.log(f"\nBest model: {best} (epoch {best_epoch}, mAP {result['best_map']})")
    if "baseline" in job.run:
        for line in _baseline_lines(job.baseline.name, job.run["baseline"]):
            report.log(line)
    return result


def _initial_model(
    config: TrainConfig, classes: tuple[str, ...], model_args: dict, torch: Any, report: Any
) -> tuple[Any, dict[str, Any]]:
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    layers = config.trainable_backbone_layers

    def new_head(model: Any) -> None:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))

    if config.init == "none":
        report.log("Starting from random weights (init = none): only useful for testing.")
        return build_model(len(classes), layers, **model_args), {"from": "none"}

    if config.init == "coco":
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights,
            fasterrcnn_resnet50_fpn,
        )

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        try:
            model = fasterrcnn_resnet50_fpn(
                weights=weights, trainable_backbone_layers=layers, **model_args
            )
        except Exception as exc:
            raise TrainError(
                f"could not load the COCO-pretrained detector: {exc!r}. It is downloaded "
                "from download.pytorch.org the first time; or start from a weights file."
            ) from exc
        new_head(model)
        return model, {"from": "coco", "weights": str(weights)}

    path = Path(config.init)
    if not path.is_file():
        raise TrainError(f"init: {config.init!r} is not coco, none or a weights file")
    source = read_manifest(path) or LEGACY_MANIFEST
    if source.kind != KIND_RCNN:
        raise TrainError(f"{path} is a {source.kind!r} model; cannot continue from it")
    model = build_model(len(source.classes), layers, **model_args)
    try:
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    except Exception as exc:
        raise TrainError(f"could not load weights from {path}: {exc}") from exc
    kept_head = source.classes == classes
    if not kept_head:
        report.log(f"{path.name} has different classes: its box head is replaced.")
        new_head(model)
    return model, {
        "from": str(path),
        "weights_sha256": sha256_file(path),
        "input_it_was_run_with": source.input,
        "kept_box_head": kept_head,
    }


def _optimizer(model: Any, config: TrainConfig, total_steps: int, torch: Any):
    params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            params, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    warmup = min(config.warmup_steps, max(total_steps // 2, 1))
    drop_step = config.lr_drop_at * total_steps

    def factor(step: int) -> float:
        scale = 0.1 if config.lr_drop_at < 1.0 and step >= drop_step else 1.0
        if step < warmup:
            scale *= 0.001 + (1 - 0.001) * step / warmup
        return scale

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _train_epoch(
    model, loader, optimizer, scheduler, scaler, device, epoch, config, report, torch
) -> float:
    model.train()
    total, steps = 0.0, len(loader)
    last_report = 0.0
    for step, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=scaler is not None):
            loss = sum(model(images, targets).values())
        value = float(loss.detach())
        if not math.isfinite(value):
            raise TrainError(
                f"the loss became {value} at epoch {epoch}, step {step}; try a lower learning rate"
            )
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        total += value
        if time.perf_counter() - last_report >= PROGRESS_SECONDS or step == steps:
            last_report = time.perf_counter()
            report.event(
                "step", epoch=epoch, epochs=config.epochs, step=step, steps=steps, loss=value
            )
    return total / max(steps, 1)


def _predict(model, loader, device, names: tuple[str, ...], torch) -> list[ImageBoxes]:
    model.eval()
    out = []
    with torch.no_grad():
        for images, _ in loader:
            for prediction in model([image.to(device) for image in images]):
                out.append(
                    ImageBoxes(
                        prediction["boxes"].cpu().numpy().astype(np.float64).reshape(-1, 4),
                        tuple(names[i] for i in prediction["labels"].cpu().tolist()),
                        tuple(prediction["scores"].cpu().tolist()),
                    )
                )
    return out


def _score_baseline(path: Path, evaluate, device, torch) -> dict[str, Any]:
    """Score existing weights on this run's validation images, with their own input."""
    source = read_manifest(path) or LEGACY_MANIFEST
    if source.kind != KIND_RCNN:
        raise TrainError(f"baseline {path} is a {source.kind!r} model")
    model = build_model(len(source.classes), **source.model_args)
    try:
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    except Exception as exc:
        raise TrainError(f"could not load baseline weights from {path}: {exc}") from exc
    model.to(device)
    result = {
        "weights": str(path),
        "weights_sha256": sha256_file(path),
        "input": source.input,
        "val": evaluate(model, source.classes, source.input),
        # The baseline may have been trained on these very images.
        "note": "not a held-out comparison unless these images are new to the baseline",
    }
    if source.input == INPUT_LEGACY_V1:
        # The V1 weights were trained on RGB; score them that way too.
        result["val_as_trained"] = {
            "input": INPUT_RGB_NORMALIZED,
            **evaluate(model, source.classes, INPUT_RGB_NORMALIZED),
        }
    del model
    return result


def _baseline_lines(name: str, baseline: dict[str, Any]) -> list[str]:
    lines = [f"Current model {name} with {baseline['input']} input: mAP {baseline['val']['map']}"]
    if "val_as_trained" in baseline:
        trained = baseline["val_as_trained"]
        lines.append(f"Current model {name} with {trained['input']} input: mAP {trained['map']}")
    return lines
