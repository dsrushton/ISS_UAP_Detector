"""The YOLO segmentation trainer (ultralytics).

Learns the outline of each region instead of a box around it. Labels with an
outline (``uapd masks`` makes them from boxes) are trained as that shape; labels
that are only a box are trained as the rectangle.

The run folder gets ``dataset/`` (links to the images, and the labels in YOLO's
text format), ``yolo/`` (what ultralytics writes: curves, sample pictures) and
the usual ``best.pt`` and ``last.pt`` with their manifests. Colour order,
resizing and normalising are ultralytics' business at both ends, so the manifest
input is ``bgr-frame``.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from uapd.regions.manifest import INPUT_BGR_FRAME, KIND_YOLO_SEG, ModelManifest, write_manifest
from uapd.regions.rcnn import sha256_file
from uapd.train import TrainError, TrainJob, now
from uapd.train.data import Item

# The score shown per epoch: mask mAP at IoU 0.5 on the validation frames.
SCORE = "metrics/mAP50(M)"


def label_lines(item: Item, width: int, height: int) -> list[str]:
    """YOLO segmentation label lines for one image: class then x y pairs, 0 to 1."""
    lines = []
    for index, (box, label) in enumerate(zip(item.boxes, item.labels, strict=True)):
        points = item.outlines[index] if item.outlines else ()
        if len(points) < 3:
            x1, y1, x2, y2 = box
            points = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        values = []
        for x, y in points:
            values.append(f"{min(max(x / width, 0.0), 1.0):.5f}")
            values.append(f"{min(max(y / height, 0.0), 1.0):.5f}")
        lines.append(f"{label - 1} {' '.join(values)}")
    return lines


def export_dataset(job: TrainJob, root: Path) -> Path:
    """Write the data plan as a YOLO dataset below ``root``; returns its data file."""
    import cv2

    sizes: dict[Path, tuple[int, int]] = {}
    counter = 0
    for part, items in (("train", job.plan.train), ("val", job.plan.val)):
        images, labels = root / "images" / part, root / "labels" / part
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for item in items:
            folder = item.image.parent
            if folder not in sizes:
                # Frames of one folder come from one clip and share a size.
                frame = cv2.imread(str(item.image))
                if frame is None:
                    raise TrainError(f"could not read image {item.image}")
                sizes[folder] = (frame.shape[1], frame.shape[0])
            width, height = sizes[folder]
            counter += 1
            stem = f"{counter:06d}_{item.image.stem}"
            os.symlink(item.image.resolve(), images / f"{stem}{item.image.suffix.lower()}")
            (labels / f"{stem}.txt").write_text(
                "".join(line + "\n" for line in label_lines(item, width, height))
            )
    names = "".join(f"  {i}: {name}\n" for i, name in enumerate(job.config.classes))
    data = root / "data.yaml"
    val = "images/val" if job.plan.val else "images/train"
    data.write_text(f"path: {root.resolve()}\ntrain: images/train\nval: {val}\nnames:\n{names}")
    return data


def _total_loss(trainer: Any) -> float | None:
    """The epoch's training loss as one number, or None if ultralytics changed its shape."""
    try:
        parts = trainer.label_loss_items(trainer.tloss, prefix="train")
        return round(sum(float(v) for v in parts.values()), 4)
    except (AttributeError, TypeError, ValueError):
        return None


def train_yolo_seg(job: TrainJob) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise TrainError(
            f"the yolo-seg trainer needs the ultralytics package: {exc} "
            "(install with: pip install -e '.[yolo]')"
        ) from exc

    config, report, run_dir = job.config, job.reporter, job.run_dir.resolve()
    data = export_dataset(job, run_dir / "dataset")
    if config.init == "coco":
        start = config.yolo_model
    elif config.init == "none":
        start = config.yolo_model.replace(".pt", ".yaml")
    else:
        start = config.init
    model = YOLO(start)
    outlined = sum(1 for item in job.plan.train for points in item.outlines if points)
    total = sum(len(item.boxes) for item in job.plan.train)
    report.log(
        f"Training {start} on {len(job.plan.train)} images, up to {config.epochs} epochs; "
        f"{outlined} of {total} training labels have an outline, the rest are rectangles"
    )
    report.event(
        "start",
        device=job.device or "auto",
        epochs=config.epochs,
        train_images=len(job.plan.train),
        val_images=len(job.plan.val),
        run_dir=str(job.run_dir),
    )

    started = time.perf_counter()
    best = {"score": -1.0, "epoch": 0}
    fresh = {"epoch": False}

    def on_epoch_start(trainer: Any) -> None:
        fresh["epoch"] = True

    def on_epoch(trainer: Any) -> None:
        nonlocal started
        # ultralytics also calls this once after training, when it re-scores best.pt.
        if not fresh["epoch"]:
            return
        fresh["epoch"] = False
        epoch = len(job.run["epochs"]) + 1
        metrics = {k: round(float(v), 4) for k, v in (trainer.metrics or {}).items()}
        value = metrics.get(SCORE)
        # best.pt is the epoch ultralytics rates highest (its "fitness", mostly the
        # stricter mAP 50-95 of boxes and masks), which need not be the highest SCORE.
        fitness = getattr(trainer, "fitness", None)
        is_best = fitness is not None and fitness == getattr(trainer, "best_fitness", None)
        if is_best:
            best.update(score=value if value is not None else -1.0, epoch=epoch)
        seconds = round(time.perf_counter() - started, 1)
        started = time.perf_counter()
        entry = {
            "epoch": epoch,
            "loss": _total_loss(trainer),
            "val": {"map": value, "iou_threshold": 0.5, "metrics": metrics},
            "seconds": seconds,
            "best": is_best,
        }
        job.run["epochs"].append(entry)
        job.save_run()
        shown = "-" if value is None else f"{value:.4f}"
        report.log(
            f"Epoch {epoch}/{config.epochs}: mask mAP {shown}"
            f"{' (best so far)' if is_best else ''}, {seconds:.0f} s"
        )
        report.event("epoch", epochs=config.epochs, **entry)

    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_fit_epoch_end", on_epoch)
    model.train(
        data=str(data),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        # ultralytics counts epochs without improvement itself; 0 here means never stop.
        patience=config.patience or config.epochs,
        fliplr=0.5 if config.hflip else 0.0,
        flipud=0.5 if config.vflip else 0.0,
        **({} if config.color_jitter else {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0}),
        save_period=1 if config.keep_epochs else -1,
        workers=config.workers,
        seed=config.seed,
        device=job.device,
        project=str(run_dir),
        name="yolo",
        exist_ok=True,
        plots=True,
        verbose=False,
    )

    weights = run_dir / "yolo" / "weights"
    result: dict[str, Any] = {
        "best_epoch": best["epoch"],
        "best_map": best["score"] if best["score"] >= 0 else None,
        "stopped_early": len(job.run["epochs"]) < config.epochs,
    }
    # ultralytics numbers its per-epoch files from 0.
    kept = {
        f"epoch-{int(p.stem[5:]) + 1:02d}": (p, int(p.stem[5:]) + 1)
        for p in sorted(weights.glob("epoch*.pt"))
        if p.stem[5:].isdigit()
    }
    files = {
        "last": (weights / "last.pt", len(job.run["epochs"])),
        "best": (weights / "best.pt", best["epoch"]),
        **kept,
    }
    scores = {entry["epoch"]: entry["val"]["map"] for entry in job.run["epochs"]}
    for name, (source, epoch) in files.items():
        if not source.is_file():
            continue
        target = run_dir / f"{name}.pt"
        shutil.copyfile(source, target)
        write_manifest(
            target,
            ModelManifest(
                KIND_YOLO_SEG,
                ("background", *config.classes),
                INPUT_BGR_FRAME,
                task="masks",
                model_args={"image_size": config.image_size},
                record={
                    "trained": {
                        "run": job.run_dir.name,
                        "epoch": epoch,
                        "saved": now(),
                        "uapd_version": job.run["uapd_version"],
                        "commit": job.run["commit"],
                        "train_config": config.to_dict(),
                        "train_config_fingerprint": config.fingerprint(),
                        "init": start,
                        "val": {"map": scores.get(epoch)},
                    },
                    "weights_sha256": sha256_file(target),
                },
            ),
        )
        if name in ("last", "best"):
            result[f"{name}_weights"] = str(job.run_dir / f"{name}.pt")
    if "best_weights" not in result:
        raise TrainError(f"ultralytics wrote no weights in {weights}")
    report.log(
        f"\nBest model: {result['best_weights']} "
        f"(epoch {result['best_epoch']}, mask mAP {result['best_map']})"
    )
    return result
