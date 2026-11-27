"""What the app runs: command lines, saved settings and background processes.

No Tk here, so this is tested without a display. The app starts each branch as
its own ``python -m uapd ...`` process and only reads its output, so the window
never imports the detector or the trainer and one cannot break the other.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = Path.home() / ".uapd" / "app.json"
EVENT_PREFIX = "@uapd "  # the same marker as uapd.train.EVENT_PREFIX


@dataclass
class Settings:
    """What the forms held last time."""

    # Detect / broadcast
    source: str = ""  # empty = the live feed in the config
    weights: str = "best_map_model.pth"
    profile: str = ""
    # Retrain
    data: list[str] = field(default_factory=list)
    train_out: str = "models"
    init: str = "coco"
    epochs: str = "12"
    batch_size: str = "8"
    learning_rate: str = "0.0001"
    compare: bool = True

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> Settings:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        try:
            return cls(**known)
        except TypeError:
            return cls()

    def save(self, path: Path = SETTINGS_FILE) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(asdict(self), fh, indent=2)
        except OSError:
            pass  # remembering the form is a convenience


def _base() -> list[str]:
    return [sys.executable, "-m", "uapd"]


def detect_command(settings: Settings) -> list[str]:
    command = [*_base(), "run"]
    if settings.source.strip():
        command.append(settings.source.strip())
    command += ["--weights", settings.weights.strip()]
    if settings.profile.strip():
        command += ["--profile", settings.profile.strip()]
    return command


def train_command(settings: Settings, check: bool = False) -> list[str]:
    """Raises ValueError with a message for the user when the form is not usable."""
    if not settings.data:
        raise ValueError("Add at least one folder of labelled frames.")
    command = [*_base(), "train", "--progress-json"]
    for folder in settings.data:
        command += ["--data", folder]
    for flag, text, kind in (
        ("--epochs", settings.epochs, int),
        ("--batch-size", settings.batch_size, int),
        ("--lr", settings.learning_rate, float),
    ):
        try:
            kind(text)
        except ValueError:
            raise ValueError(f"{flag[2:]} is not a number: {text!r}") from None
        command += [flag, text.strip()]
    command += ["--init", settings.init.strip() or "coco"]
    if check:
        return [*command, "--check"]
    command += ["--out", settings.train_out.strip() or "models"]
    if settings.compare and settings.weights.strip():
        if (REPO_ROOT / settings.weights.strip()).is_file():
            command += ["--baseline", settings.weights.strip()]
    return command


def parse_line(line: str) -> tuple[str, Any]:
    """("event", dict) for a progress line, otherwise ("text", the line)."""
    line = line.rstrip("\r\n")
    if line.startswith(EVENT_PREFIX):
        try:
            return "event", json.loads(line[len(EVENT_PREFIX) :])
        except ValueError:
            pass
    return "text", line


class Job:
    """A child process whose output lines are collected for the window to poll."""

    def __init__(self, command: list[str], cwd: Path = REPO_ROOT):
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        self.command = command
        self.lines: queue.Queue[str] = queue.Queue()
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._exit_code: int | None = None
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.process.stdout:
            self.lines.put(line)
        self._exit_code = self.process.wait()  # set only after every line is queued

    def poll(self) -> tuple[list[str], int | None]:
        """Lines printed since the last call, and the exit code once it has ended."""
        code = self._exit_code  # read before draining so no line is missed
        lines = []
        while True:
            try:
                lines.append(self.lines.get_nowait())
            except queue.Empty:
                break
        return lines, code

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
