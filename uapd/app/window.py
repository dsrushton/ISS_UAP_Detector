"""The app window: a home page with three branches, each on its own page.

Detect and Retrain only draw forms and show output; their work is done by child
processes (see ``uapd.app.jobs``). Relabel shows pictures and writes small files
itself (see ``uapd.app.relabel``).
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from uapd.app.jobs import REPO_ROOT, Job, Settings, detect_command, parse_line, train_command

POLL_MS = 150
# Trainers, as the Retrain page names them.
TRAIN_KINDS = {
    "rcnn-boxes": "Boxes (Faster R-CNN)",
    "yolo-seg": "Outlines (YOLO segmentation)",
}
PAD = 12

BRANCHES = (
    (
        "detect",
        "Detect / Broadcast",
        "Watch the live feed or a clip, record what is found, restream to YouTube.",
    ),
    (
        "relabel",
        "Relabel captures",
        "Look through labelled frames and outlines before training, and give saved "
        "captures a verdict.",
    ),
    (
        "retrain",
        "Retrain",
        "Train a new region model on labelled frames and compare it with the current one.",
    ),
)


class App(ttk.Frame):
    def __init__(self, root: tk.Tk, settings: Settings | None = None):
        super().__init__(root, padding=PAD)
        self.root = root
        self.settings = settings or Settings.load()
        self.pack(fill="both", expand=True)

        style = ttk.Style(root)
        style.configure("Title.TLabel", font=("TkDefaultFont", 16, "bold"))
        style.configure("Branch.TButton", font=("TkDefaultFont", 12, "bold"), padding=14)
        style.configure("Hint.TLabel", foreground="#555555")

        self.pages: dict[str, ttk.Frame] = {
            "home": self._home(),
            "detect": DetectPage(self),
            "relabel": RelabelPage(self),
            "retrain": RetrainPage(self),
        }
        self.show("home")
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _home(self) -> ttk.Frame:
        page = ttk.Frame(self)
        ttk.Label(page, text="ISS UAP Detector", style="Title.TLabel").pack(anchor="w")
        ttk.Label(page, text="Choose what to do.", style="Hint.TLabel").pack(
            anchor="w", pady=(0, PAD)
        )
        for key, title, text in BRANCHES:
            row = ttk.Frame(page)
            row.pack(fill="x", pady=6)
            ttk.Button(
                row,
                text=title,
                style="Branch.TButton",
                width=20,
                command=lambda key=key: self.show(key),
            ).pack(side="left")
            ttk.Label(row, text=text, wraplength=420, justify="left").pack(
                side="left", padx=PAD, fill="x"
            )
        return page

    def show(self, key: str) -> None:
        for page in self.pages.values():
            page.pack_forget()
        self.pages[key].pack(fill="both", expand=True)

    def close(self) -> None:
        running = [p for p in self.pages.values() if getattr(p, "job", None) is not None]
        if running and not messagebox.askokcancel(
            "Quit", "Something is still running. Stop it and quit?"
        ):
            return
        for page in running:
            page.job.stop()
        for page in self.pages.values():
            if hasattr(page, "collect"):
                page.collect()
        self.settings.save()
        self.root.destroy()


class Page(ttk.Frame):
    """A branch page: title, back button, a form, and an output box for a job."""

    title = ""

    def __init__(self, app: App):
        super().__init__(app)
        self.app = app
        self.settings = app.settings
        self.job: Job | None = None

        head = ttk.Frame(self)
        head.pack(fill="x", pady=(0, PAD))
        ttk.Button(head, text="< Back", command=lambda: app.show("home")).pack(side="left")
        ttk.Label(head, text=self.title, style="Title.TLabel").pack(side="left", padx=PAD)

        self.form = ttk.Frame(self)
        self.form.pack(fill="x")
        self.form.columnconfigure(1, weight=1)

    # --- form helpers -------------------------------------------------------------------

    def field(self, row: int, label: str, value: str, browse: str | None = None) -> tk.StringVar:
        """A labelled entry. ``browse`` is "file" or "folder" for a Browse button."""
        var = tk.StringVar(value=value)
        ttk.Label(self.form, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(self.form, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6)
        if browse:
            ttk.Button(self.form, text="Browse...", command=lambda: self._browse(var, browse)).grid(
                row=row, column=2
            )
        return var

    def _browse(self, var: tk.StringVar, kind: str) -> None:
        start = str(REPO_ROOT)
        if kind == "folder":
            chosen = filedialog.askdirectory(initialdir=start)
        else:
            chosen = filedialog.askopenfilename(initialdir=start)
        if chosen:
            var.set(chosen)

    def output_box(self) -> None:
        box = ttk.Frame(self)
        box.pack(fill="both", expand=True, pady=(PAD, 0))
        self.output = tk.Text(box, height=14, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(box, command=self.output.yview)
        self.output.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.output.pack(side="left", fill="both", expand=True)

    def write(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.insert("end", text + "\n")
        self.output.see("end")
        self.output.configure(state="disabled")

    # --- jobs ---------------------------------------------------------------------------

    def start(self, command: list[str]) -> bool:
        if self.job is not None:
            return False
        self.collect()
        self.settings.save()
        self.write("> " + " ".join(command[2:]))
        try:
            self.job = Job(command)
        except OSError as exc:
            self.write(f"Could not start: {exc}")
            return False
        self.running(True)
        self.after(POLL_MS, self._poll)
        return True

    def _poll(self) -> None:
        if self.job is None:
            return
        lines, code = self.job.poll()
        for line in lines:
            kind, value = parse_line(line)
            if kind == "event":
                self.on_event(value)
            else:
                self.write(value)
        if code is None:
            self.after(POLL_MS, self._poll)
            return
        self.job = None
        self.running(False)
        self.finished(code)

    def stop(self) -> None:
        if self.job is not None:
            self.job.stop()

    # --- for pages to fill in -----------------------------------------------------------

    def collect(self) -> None:
        """Copy the form into ``self.settings``."""

    def running(self, active: bool) -> None:
        """Enable or disable buttons."""

    def on_event(self, event: dict[str, Any]) -> None:
        """A progress event from the job."""

    def finished(self, code: int) -> None:
        self.write(f"Finished (exit code {code}).")


class DetectPage(Page):
    title = "Detect / Broadcast"

    def __init__(self, app: App):
        super().__init__(app)
        s = self.settings
        self.source = self.field(0, "Clip or YouTube URL", s.source, browse="file")
        ttk.Label(self.form, text="Leave empty to watch the live feed.", style="Hint.TLabel").grid(
            row=1, column=1, sticky="w", padx=6
        )
        self.weights = self.field(2, "Model weights", s.weights, browse="file")
        self.second_weights = self.field(
            3, "Second model (optional)", s.second_weights, browse="file"
        )
        ttk.Label(
            self.form,
            text="With a second model, it finds lens flares, sun, panels, total darkness and "
            "no feed, and the first finds space, Earth and the station. "
            "For example: an outline model first, best_map_model.pth second.",
            style="Hint.TLabel",
            wraplength=560,
            justify="left",
        ).grid(row=4, column=1, sticky="w", padx=6)
        self.profile = self.field(5, "Settings profile (optional)", s.profile, browse="file")

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=(PAD, 0))
        self.start_button = ttk.Button(buttons, text="Start", command=self.begin)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(
            buttons, text="Force stop", command=self.stop, state="disabled"
        )
        self.stop_button.pack(side="left", padx=6)
        ttk.Label(
            self,
            text="In the video window: q quits and saves, s starts or stops the restream, "
            "t plants test frames. Force stop ends it without saving a recording in progress.",
            style="Hint.TLabel",
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))
        self.output_box()

    def collect(self) -> None:
        self.settings.source = self.source.get()
        self.settings.weights = self.weights.get()
        self.settings.second_weights = self.second_weights.get()
        self.settings.profile = self.profile.get()

    def begin(self) -> None:
        self.collect()
        self.start(detect_command(self.settings))

    def running(self, active: bool) -> None:
        self.start_button.configure(state="disabled" if active else "normal")
        self.stop_button.configure(state="normal" if active else "disabled")


class RelabelPage(Page):
    title = "Relabel captures"

    def __init__(self, app: App):
        super().__init__(app)
        from uapd.app import relabel

        self.form.pack_forget()
        relabel.build(self)

    def collect(self) -> None:
        pass  # the tabs write their folders into the settings as they are loaded


class RetrainPage(Page):
    title = "Retrain"

    def __init__(self, app: App):
        super().__init__(app)
        s = self.settings
        ttk.Label(self.form, text="Labelled frames").grid(row=0, column=0, sticky="nw", pady=3)
        self.folders = tk.Listbox(self.form, height=5, selectmode="extended")
        self.folders.grid(row=0, column=1, sticky="ew", padx=6)
        for folder in s.data:
            self.folders.insert("end", folder)
        side = ttk.Frame(self.form)
        side.grid(row=0, column=2, sticky="n")
        ttk.Button(side, text="Add folder...", command=self.add_folder).pack(fill="x")
        ttk.Button(side, text="Remove", command=self.remove_folders).pack(fill="x", pady=3)
        ttk.Label(
            self.form,
            text="A folder of frames with labels, or a folder that contains such folders.",
            style="Hint.TLabel",
        ).grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(self.form, text="Model type").grid(row=2, column=0, sticky="w", pady=3)
        self.kind = tk.StringVar(value=TRAIN_KINDS.get(s.train_kind, TRAIN_KINDS["rcnn-boxes"]))
        ttk.Combobox(
            self.form, textvariable=self.kind, values=list(TRAIN_KINDS.values()), state="readonly"
        ).grid(row=2, column=1, sticky="ew", padx=6)
        self.train_config = self.field(3, "Settings file (optional)", s.train_config, browse="file")
        ttk.Label(
            self.form,
            text="A .toml file with held-out folders, patience and the like. "
            "The fields below override it.",
            style="Hint.TLabel",
        ).grid(row=4, column=1, sticky="w", padx=6)
        self.init = self.field(5, "Start from", s.init, browse="file")
        ttk.Label(
            self.form,
            text="coco = the pretrained model (downloaded once), or a weights file.",
            style="Hint.TLabel",
        ).grid(row=6, column=1, sticky="w", padx=6)
        self.epochs = self.field(7, "Epochs", s.epochs)
        self.batch_size = self.field(8, "Batch size", s.batch_size)
        self.learning_rate = self.field(9, "Learning rate", s.learning_rate)
        self.out = self.field(10, "Save runs in", s.train_out, browse="folder")
        self.compare = tk.BooleanVar(value=s.compare)
        ttk.Checkbutton(
            self.form,
            text="Score the current model (Detect page) on the same validation frames "
            "(box models only)",
            variable=self.compare,
        ).grid(row=11, column=1, sticky="w", padx=6, pady=3)

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=(PAD, 0))
        self.check_button = ttk.Button(buttons, text="Check data", command=self.check)
        self.check_button.pack(side="left")
        self.start_button = ttk.Button(buttons, text="Start training", command=self.begin)
        self.start_button.pack(side="left", padx=6)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left")
        self.use_button = ttk.Button(
            buttons, text="Use best model in Detect", command=self.use_best, state="disabled"
        )
        self.use_button.pack(side="right")

        status = ttk.Frame(self)
        status.pack(fill="x", pady=(PAD, 0))
        self.progress = ttk.Progressbar(status, maximum=1.0)
        self.progress.pack(fill="x")
        self.status = tk.StringVar(value="Not running.")
        ttk.Label(status, textvariable=self.status).pack(anchor="w")
        self.output_box()
        self.best_weights: str | None = None

    def add_folder(self) -> None:
        chosen = filedialog.askdirectory(initialdir=str(REPO_ROOT))
        if chosen and chosen not in self.folders.get(0, "end"):
            self.folders.insert("end", chosen)

    def remove_folders(self) -> None:
        for index in reversed(self.folders.curselection()):
            self.folders.delete(index)

    def collect(self) -> None:
        s = self.settings
        s.data = list(self.folders.get(0, "end"))
        s.init = self.init.get()
        s.train_kind = next(
            (key for key, shown in TRAIN_KINDS.items() if shown == self.kind.get()), "rcnn-boxes"
        )
        s.train_config = self.train_config.get()
        s.epochs = self.epochs.get()
        s.batch_size = self.batch_size.get()
        s.learning_rate = self.learning_rate.get()
        s.train_out = self.out.get()
        s.compare = bool(self.compare.get())

    def _run(self, check: bool) -> None:
        self.collect()
        try:
            command = train_command(self.settings, check=check)
        except ValueError as exc:
            messagebox.showerror("Retrain", str(exc))
            return
        if self.start(command):
            self.status.set("Checking the data..." if check else "Starting...")
            self.progress.configure(value=0.0)

    def check(self) -> None:
        self._run(check=True)

    def begin(self) -> None:
        self._run(check=False)

    def running(self, active: bool) -> None:
        state = "disabled" if active else "normal"
        self.check_button.configure(state=state)
        self.start_button.configure(state=state)
        self.stop_button.configure(state="normal" if active else "disabled")

    def on_event(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "start":
            self.status.set(f"Training on {event['device']}: {event['train_images']} images")
        elif kind == "step":
            done = (event["epoch"] - 1 + event["step"] / event["steps"]) / event["epochs"]
            self.progress.configure(value=done)
            self.status.set(
                f"Epoch {event['epoch']} of {event['epochs']}, step {event['step']} of "
                f"{event['steps']}, loss {event['loss']:.3f}"
            )
        elif kind == "epoch":
            # The outline trainer reports once an epoch, not every step.
            self.progress.configure(value=event["epoch"] / event["epochs"])
            score = (event.get("val") or {}).get("map")
            shown = "-" if score is None else f"{score:.3f}"
            self.status.set(f"Epoch {event['epoch']} of {event['epochs']} done, score {shown}")
        elif kind == "done":
            self.progress.configure(value=1.0)
            self.best_weights = event.get("best_weights")
            score = event.get("best_map")
            text = f"Done. Best model: epoch {event.get('best_epoch')}, mAP {score}"
            if event.get("baseline_map") is not None:
                text += f" (current model: {event['baseline_map']})"
            self.status.set(text)
            self.use_button.configure(state="normal")

    def finished(self, code: int) -> None:
        if code != 0:
            self.status.set("Stopped." if code < 0 or code == 130 else "Failed; see the output.")
        elif self.status.get().startswith(("Checking", "Starting")):
            self.status.set("Check finished.")

    def use_best(self) -> None:
        if not self.best_weights:
            return
        if not messagebox.askokcancel(
            "Use this model",
            "The Detect page will use this model from its next start.\n\n"
            "It has only been scored on validation frames. The evaluation gate "
            "(planted objects, labelled junk) is not built yet, so nothing has checked "
            "that detection is no worse with it.",
        ):
            return
        try:
            shown = str(Path(self.best_weights).relative_to(REPO_ROOT))
        except ValueError:
            shown = self.best_weights
        self.app.pages["detect"].weights.set(shown)
        self.settings.weights = shown
        self.settings.save()


def main() -> int:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"uapd: error: could not open a window: {exc}")
        return 2
    root.title("ISS UAP Detector")
    root.minsize(760, 560)
    App(root)
    root.mainloop()
    return 0
