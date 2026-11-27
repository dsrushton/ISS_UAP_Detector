"""The Relabel page: look through labelled frames and saved captures.

Two tabs:

* Labelled frames - step through the frames of a label folder with its boxes and
  outlines drawn, mark the folder ok or rejected and flag single bad frames.
  The marks go to ``review.json`` in the folder (``uapd.labels.review``) and
  Retrain leaves rejected folders and flagged frames out. In an outlined folder
  (made by ``uapd masks``) a click outlines what is under the pointer and a
  right-click removes an outline, on one frame or the whole folder
  (``uapd.labels.edit``). The mask model for that is loaded on the first click.
* Captures - play each saved capture and give the event one verdict, written
  beside the clip (``uapd.app.captures``).

Unlike the other pages this one works in the window's own process: it only reads
images and writes small JSON files. Pictures are handed to Tk as PPM data, so it
needs nothing beyond OpenCV.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

import cv2
import numpy as np

from uapd.app.captures import VERDICT_KEYS, VERDICTS, Capture, list_captures, write_verdict
from uapd.app.jobs import REPO_ROOT
from uapd.labels import (
    IMAGE_SUFFIXES,
    FolderLabels,
    LabelError,
    Sample,
    edit,
    find_label_folders,
    load_folder,
)
from uapd.labels.automask import REVIEW_FOLDER, draw_annotations, sam_point_masker
from uapd.labels.review import (
    VERDICT_OK,
    VERDICT_REJECT,
    Review,
    read_review,
    write_review,
)

PAD = 8
MARK = {VERDICT_OK: "ok ", VERDICT_REJECT: "REJ", None: "   "}
PLAY_MS = 33
EDIT_POLL_MS = 100
CLICK_CLASSES = ("earth", "space", "iss", "lf", "sun", "panel")
# The mask model for click edits; downloaded here the first time if it is missing.
SAM_WEIGHTS = REPO_ROOT / "data" / "sam2.1_s.pt"


def resolve(path: str) -> Path:
    """A path from a form: relative paths are relative to the repository."""
    chosen = Path(path.strip()).expanduser()
    return chosen if chosen.is_absolute() else REPO_ROOT / chosen


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """``image`` scaled to fit inside ``width`` x ``height``, keeping its shape."""
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (max(int(image.shape[1] * scale), 1), max(int(image.shape[0] * scale), 1))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def to_ppm(image: np.ndarray) -> bytes:
    """A BGR image as PPM bytes, which ``tk.PhotoImage`` reads without Pillow."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return b"P6 %d %d 255\n" % (rgb.shape[1], rgb.shape[0]) + rgb.tobytes()


def render_sample(sample: Any, outlines: bool, boxes: bool) -> np.ndarray:
    """A labelled frame as the viewer shows it."""
    image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR)
    if image is None:
        image = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(image, "could not read image", (20, 180), 0, 0.8, (0, 0, 255), 2)
        return image
    shown = [
        a
        for a in sample.annotations
        if (outlines and a.kind == "polygon") or (boxes and a.kind != "polygon")
    ]
    return draw_annotations(image, shown) if shown else image


def label_folders(root: Path) -> list[Path]:
    """Label folders in or below ``root``, without the contact-sheet folder."""
    return [f for f in find_label_folders(root) if REVIEW_FOLDER not in f.parts]


PICTURES = "pictures only"


def has_pictures(folder: Path) -> bool:
    return folder.is_dir() and any(
        p.suffix.lower() in IMAGE_SUFFIXES for p in folder.iterdir() if p.is_file()
    )


def picture_folders(root: Path, labelled: list[Path]) -> list[Path]:
    """Folders of plain pictures worth listing: ``root`` itself when it holds
    pictures and no labels, and the contact-sheet folders directly below it."""
    found = []
    if root not in labelled and has_pictures(root):
        found.append(root)
    if root.is_dir():
        found += sorted(
            d
            for d in root.iterdir()
            if d.is_dir() and d.name.startswith(REVIEW_FOLDER) and has_pictures(d)
        )
    return found


def load_pictures(folder: Path) -> FolderLabels:
    """A folder of pictures as samples with no labels, so the viewer can show them."""
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return FolderLabels([Sample(p, (), group=str(folder)) for p in images], format=PICTURES)


class Picture(ttk.Frame):
    """A canvas that shows one image scaled to its own size."""

    def __init__(self, parent: tk.Misc):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, background="#111111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._photo: tk.PhotoImage | None = None
        self._image: np.ndarray | None = None
        self._placed: tuple[float, int, int] | None = None  # scale, left, top
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

    def image_point(self, x: int, y: int) -> tuple[float, float] | None:
        """A canvas position as a position in the image shown, or None outside it."""
        if self._image is None or self._placed is None:
            return None
        scale, left, top = self._placed
        px, py = (x - left) / scale, (y - top) / scale
        if not (0 <= px < self._image.shape[1] and 0 <= py < self._image.shape[0]):
            return None
        return px, py

    def show(self, image: np.ndarray | None) -> None:
        self._image = image
        self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self._image is None or width < 8 or height < 8:
            return
        shown = fit(self._image, width, height)
        left, top = (width - shown.shape[1]) // 2, (height - shown.shape[0]) // 2
        self._placed = (shown.shape[1] / self._image.shape[1], left, top)
        self._photo = tk.PhotoImage(data=to_ppm(shown), format="PPM")
        self.canvas.create_image(left, top, image=self._photo, anchor="nw")


class FramesTab(ttk.Frame):
    def __init__(self, parent: tk.Misc, page: Any):
        super().__init__(parent, padding=PAD)
        self.page = page
        self.folders: list[Path] = []
        self.reviews: dict[Path, Review] = {}
        self.labels: FolderLabels | None = None
        self.folder: Path | None = None
        self.index = 0
        self.busy = False
        self.plain: set[Path] = set()
        self._masker: Any = None
        self._work: dict[str, Any] = {}

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Frames folder").pack(side="left")
        self.root_var = tk.StringVar(value=page.settings.review_root)
        entry = ttk.Entry(top, textvariable=self.root_var)
        entry.pack(side="left", fill="x", expand=True, padx=6)
        entry.bind("<Return>", lambda _event: self.load_root())
        ttk.Button(top, text="Browse...", command=self.browse).pack(side="left")
        ttk.Button(top, text="Load", command=self.load_root).pack(side="left", padx=(6, 0))

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(PAD, 0))
        left = ttk.Frame(body)
        self.listbox = tk.Listbox(left, width=44, exportselection=False, font="TkFixedFont")
        scroll = ttk.Scrollbar(left, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", lambda _event: self.select_folder())
        body.add(left, weight=0)

        right = ttk.Frame(body)
        body.add(right, weight=1)
        self.picture = Picture(right)
        self.picture.pack(fill="both", expand=True)
        self.picture.canvas.bind("<Button-1>", lambda event: self.click(event, add=True))
        self.picture.canvas.bind("<Button-3>", lambda event: self.click(event, add=False))
        self.info = tk.StringVar(value="Load a folder of labelled frames.")
        ttk.Label(right, textvariable=self.info, wraplength=900, justify="left").pack(
            anchor="w", pady=(4, 0)
        )
        self.slider = ttk.Scale(right, from_=0, to=0, command=self._slide)
        self.slider.pack(fill="x", pady=4)

        controls = ttk.Frame(right)
        controls.pack(fill="x")
        ttk.Button(controls, text="< Prev", command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(controls, text="Next >", command=lambda: self.step(1)).pack(side="left", padx=4)
        self.outlines = tk.BooleanVar(value=True)
        self.boxes = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls, text="Outlines", variable=self.outlines, command=self.show_frame
        ).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(controls, text="Boxes", variable=self.boxes, command=self.show_frame).pack(
            side="left", padx=4
        )
        ttk.Button(
            controls, text="Reject folder (x)", command=lambda: self.mark(VERDICT_REJECT)
        ).pack(side="right")
        ttk.Button(controls, text="Folder OK (o)", command=lambda: self.mark(VERDICT_OK)).pack(
            side="right", padx=4
        )
        ttk.Button(controls, text="Flag frame (f)", command=self.flag_frame).pack(side="right")

        editing = ttk.Frame(right)
        editing.pack(fill="x", pady=(6, 0))
        ttk.Label(editing, text="Click on the picture to outline what is there as").pack(
            side="left"
        )
        self.click_class = tk.StringVar(value=CLICK_CLASSES[0])
        ttk.Combobox(
            editing, textvariable=self.click_class, values=CLICK_CLASSES, width=7, state="readonly"
        ).pack(side="left", padx=4)
        self.whole_folder = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            editing, text="on every frame of this folder", variable=self.whole_folder
        ).pack(side="left", padx=4)
        self.undo_button = ttk.Button(editing, text="Undo last edit (u)", command=self.undo_edit)
        self.undo_button.pack(side="right")
        ttk.Label(
            right,
            text="Right-click removes the outline under the pointer. The station (green) is "
            "whatever is left of its box after space and Earth, so fix a wrong station by "
            "adding the Earth or space it covers. Number keys 1 to 6 choose the class.",
            style="Hint.TLabel",
            wraplength=900,
            justify="left",
        ).pack(anchor="w")
        self.summary = tk.StringVar()
        ttk.Label(self, textvariable=self.summary, style="Hint.TLabel").pack(
            anchor="w", pady=(4, 0)
        )
        ttk.Label(
            self,
            text="Keys: left/right = frame, shift+left/right = 10 frames, up/down = folder, "
            "space = hide or show the labels, o = folder ok, x = reject folder, f = flag frame, "
            "u = undo the last edit.",
            style="Hint.TLabel",
            wraplength=1000,
            justify="left",
        ).pack(anchor="w")

    # --- folders ------------------------------------------------------------------------

    def browse(self) -> None:
        chosen = filedialog.askdirectory(initialdir=str(resolve(self.root_var.get())))
        if chosen:
            self.root_var.set(chosen)
            self.load_root()

    def load_root(self) -> None:
        root = resolve(self.root_var.get())
        note = ""
        # A contact-sheet folder was chosen: the frames themselves, which are what
        # can be marked and edited, are in the folder above it.
        if root.name.startswith(REVIEW_FOLDER) and root.parent.is_dir():
            try:
                above = label_folders(root.parent)
            except LabelError:
                above = []
            if above:
                note = (
                    f"{root.name} only holds contact sheets (12 sample frames of each clip "
                    "side by side). Showing the frames themselves instead; the sheets are at "
                    "the bottom of the list."
                )
                root = root.parent
                self.root_var.set(str(root))
        self.page.settings.review_root = self.root_var.get()
        try:
            self.folders = label_folders(root)
        except LabelError as exc:
            self.folders = []
            self.info.set(str(exc))
        # Folders of plain pictures (contact sheets, or a folder with no labels at
        # all) are listed too, so they can be looked through here.
        self.plain = set(picture_folders(root, self.folders))
        self.folders += sorted(self.plain)
        self.root = root
        self.reviews = {folder: read_review(folder) for folder in self.folders}
        self.listbox.delete(0, "end")
        for folder in self.folders:
            self.listbox.insert("end", self._row(folder))
        self.labels, self.folder = None, None
        self.picture.show(None)
        if self.folders:
            self.listbox.selection_set(0)
            self.select_folder()
        elif root.is_dir():
            self.info.set(f"No labelled frames or pictures in {root}.")
        else:
            self.info.set(f"There is no folder {root}.")
        if note:
            self.summary.set(note)
            return
        self._summarise()

    def _row(self, folder: Path) -> str:
        review = self.reviews[folder]
        try:
            name = str(folder.relative_to(self.root))
        except ValueError:
            name = folder.name
        if folder in self.plain:
            return f"    {name if name != '.' else folder.name}  (pictures)"
        flagged = f" ({len(review.bad_frames)} flagged)" if review.bad_frames else ""
        return f"{MARK[review.verdict]} {name}{flagged}"

    def _refresh_row(self, folder: Path) -> None:
        row = self.folders.index(folder)
        self.listbox.delete(row)
        self.listbox.insert(row, self._row(folder))
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(row)
        self._summarise()

    def _summarise(self) -> None:
        verdicts = [r.verdict for f, r in self.reviews.items() if f not in self.plain]
        self.summary.set(
            f"{len(verdicts)} folders: {verdicts.count(VERDICT_OK)} ok, "
            f"{verdicts.count(VERDICT_REJECT)} rejected, {verdicts.count(None)} not looked at. "
            "Retrain leaves out rejected folders and flagged frames."
        )

    def select_folder(self) -> None:
        chosen = self.listbox.curselection()
        if not chosen:
            return
        folder = self.folders[chosen[0]]
        if folder == self.folder:
            return
        if self.busy:  # an edit is running on the folder shown; stay on it
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(self.folders.index(self.folder))
            return
        try:
            self.labels = load_pictures(folder) if folder in self.plain else load_folder(folder)
        except (LabelError, OSError) as exc:
            self.labels = None
            self.info.set(str(exc))
            self.picture.show(None)
            return
        self.folder, self.index = folder, 0
        self.slider.configure(to=max(len(self.labels.samples) - 1, 0))
        self.show_frame()

    def move_folder(self, delta: int) -> None:
        if not self.folders:
            return
        chosen = self.listbox.curselection()
        row = min(max((chosen[0] if chosen else 0) + delta, 0), len(self.folders) - 1)
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(row)
        self.listbox.see(row)
        self.select_folder()

    # --- frames -------------------------------------------------------------------------

    def step(self, delta: int) -> None:
        if self.labels is None or not self.labels.samples:
            return
        self.index = min(max(self.index + delta, 0), len(self.labels.samples) - 1)
        self.show_frame()

    def _slide(self, value: str) -> None:
        if self.labels is None:
            return
        index = round(float(value))
        if index != self.index:
            self.index = index
            self.show_frame(move_slider=False)

    def toggle_labels(self) -> None:
        on = not (self.outlines.get() or self.boxes.get())
        self.outlines.set(on)
        self.boxes.set(on)
        self.show_frame()

    def show_frame(self, move_slider: bool = True) -> None:
        if self.labels is None or self.folder is None or not self.labels.samples:
            return
        sample = self.labels.samples[self.index]
        self.picture.show(render_sample(sample, self.outlines.get(), self.boxes.get()))
        if move_slider:
            self.slider.set(self.index)
        counts: dict[str, int] = {}
        for a in sample.annotations:
            key = f"{a.label} {'outline' if a.kind == 'polygon' else 'box'}"
            counts[key] = counts.get(key, 0) + 1
        listed = ", ".join(f"{n} {name}" for name, n in sorted(counts.items())) or "no labels"
        flagged = (
            "  FLAGGED AS BAD" if sample.image.name in self.reviews[self.folder].bad_frames else ""
        )
        self.info.set(
            f"{sample.image.name}   frame {self.index + 1} of {len(self.labels.samples)}   "
            f"({self.labels.format}){flagged}\n{listed}"
        )

    # --- click edits --------------------------------------------------------------------

    def click(self, event: Any, add: bool) -> None:
        if self.busy or self.labels is None or self.folder is None or not self.labels.samples:
            return
        point = self.picture.image_point(event.x, event.y)
        if point is None:
            return
        if self.labels.format != "labels-json":
            self.info.set(
                "This folder holds pictures only (contact sheets), so there is nothing to "
                "edit here. Choose one of the clip folders higher in the list."
                if self.labels.format == PICTURES
                else "These labels are the originals and are not edited here. "
                "Make an outlined copy with uapd masks, and edit that."
            )
            return
        sample = self.labels.samples[self.index]
        label = self.click_class.get()
        frames = None if self.whole_folder.get() else [sample.image.name]
        where = "every frame of this folder" if frames is None else sample.image.name
        folder = self.folder

        if add:
            self.info.set(f"Outlining {label} at {point[0]:.0f}, {point[1]:.0f} on {where}...")

            def work(progress: Any) -> str:
                if self._masker is None:
                    SAM_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
                    self._masker = sam_point_masker(str(SAM_WEIGHTS))
                done = edit.add_outline(folder, label, point, self._masker, frames, progress)
                return f"Added {label} on {done} frame(s)." if done else "Nothing was found there."

        else:
            found = edit.outline_at(sample, point)
            if found is None:
                self.info.set(
                    "No outline to remove there. The station is what is left over: "
                    "add the Earth or space it wrongly covers instead."
                )
                return
            self.info.set(f"Removing the {found.label} outline on {where}...")

            def work(progress: Any) -> str:
                done = edit.remove_outline(folder, point, found.label, frames, progress)
                return f"Removed {found.label} on {done} frame(s)."

        self.run_edit(work)

    def run_edit(self, work: Any) -> None:
        """Run an edit in a thread, so the window stays alive while the mask model works."""
        self.busy = True
        self._work = {"done": 0, "total": 0, "result": None, "error": None}

        def progress(done: int, total: int) -> None:
            self._work.update(done=done, total=total)

        def target() -> None:
            try:
                self._work["result"] = work(progress)
            except Exception as exc:  # shown to the person; the window must survive
                self._work["error"] = str(exc) or repr(exc)

        threading.Thread(target=target, name="uapd-edit", daemon=True).start()
        self.after(EDIT_POLL_MS, self._poll_edit)

    def _poll_edit(self) -> None:
        state = self._work
        if state["result"] is None and state["error"] is None:
            if state["total"] > 1:
                self.info.set(f"Working: frame {state['done']} of {state['total']}...")
            self.after(EDIT_POLL_MS, self._poll_edit)
            return
        self.busy = False
        self.reload_labels()
        self.info.set(
            self.info.get().split("\n")[0]
            + "\n"
            + (f"Could not do that: {state['error']}" if state["error"] else state["result"])
        )

    def reload_labels(self) -> None:
        if self.folder is None:
            return
        try:
            self.labels = (
                load_pictures(self.folder)
                if self.folder in self.plain
                else load_folder(self.folder)
            )
        except (LabelError, OSError) as exc:
            self.info.set(str(exc))
            return
        self.index = min(self.index, max(len(self.labels.samples) - 1, 0))
        self.show_frame()

    def undo_edit(self) -> None:
        if self.busy or self.folder is None:
            return
        undone = edit.undo(self.folder)
        self.reload_labels()
        note = "Last edit undone." if undone else "Nothing to undo in this folder."
        self.info.set(self.info.get().split("\n")[0] + "\n" + note)

    # --- marks --------------------------------------------------------------------------

    def mark(self, verdict: str) -> None:
        if self.folder is None or self.folder in self.plain:
            return
        review = self.reviews[self.folder]
        review.verdict = None if review.verdict == verdict else verdict
        write_review(self.folder, review)
        self._refresh_row(self.folder)

    def flag_frame(self) -> None:
        if self.folder is None or self.labels is None or not self.labels.samples:
            return
        if self.folder in self.plain:
            return
        review = self.reviews[self.folder]
        review.bad_frames ^= {self.labels.samples[self.index].image.name}
        write_review(self.folder, review)
        self._refresh_row(self.folder)
        self.show_frame()

    def key(self, name: str, shift: bool) -> None:
        actions = {
            "Left": lambda: self.step(-10 if shift else -1),
            "Right": lambda: self.step(10 if shift else 1),
            "Up": lambda: self.move_folder(-1),
            "Down": lambda: self.move_folder(1),
            "space": self.toggle_labels,
            "o": lambda: self.mark(VERDICT_OK),
            "x": lambda: self.mark(VERDICT_REJECT),
            "f": self.flag_frame,
            "u": self.undo_edit,
        }
        if self.busy:
            return
        if name in actions:
            actions[name]()
        elif name in ("1", "2", "3", "4", "5", "6"):
            self.click_class.set(CLICK_CLASSES[int(name) - 1])


class CapturesTab(ttk.Frame):
    def __init__(self, parent: tk.Misc, page: Any):
        super().__init__(parent, padding=PAD)
        self.page = page
        self.captures: list[Capture] = []
        self.shown: list[Capture] = []
        self.capture: Capture | None = None
        self.video: cv2.VideoCapture | None = None
        self.frames = 0
        self.position = 0
        self.playing = False
        self._timer: str | None = None

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Captures folder").pack(side="left")
        self.dir_var = tk.StringVar(value=page.settings.captures_dir)
        entry = ttk.Entry(top, textvariable=self.dir_var)
        entry.pack(side="left", fill="x", expand=True, padx=6)
        entry.bind("<Return>", lambda _event: self.load_dir())
        ttk.Button(top, text="Browse...", command=self.browse).pack(side="left")
        ttk.Button(top, text="Load", command=self.load_dir).pack(side="left", padx=(6, 0))
        self.only_open = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top, text="Only without a verdict", variable=self.only_open, command=self.fill
        ).pack(side="left", padx=(12, 0))

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(PAD, 0))
        left = ttk.Frame(body)
        self.listbox = tk.Listbox(left, width=34, exportselection=False, font="TkFixedFont")
        scroll = ttk.Scrollbar(left, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", lambda _event: self.select())
        body.add(left, weight=0)

        right = ttk.Frame(body)
        body.add(right, weight=1)
        self.picture = Picture(right)
        self.picture.pack(fill="both", expand=True)
        self.info = tk.StringVar(value="Load a folder of captures.")
        ttk.Label(right, textvariable=self.info, wraplength=900, justify="left").pack(
            anchor="w", pady=(4, 0)
        )
        self.slider = ttk.Scale(right, from_=0, to=0, command=self._slide)
        self.slider.pack(fill="x", pady=4)
        controls = ttk.Frame(right)
        controls.pack(fill="x")
        self.play_button = ttk.Button(controls, text="Play", command=self.toggle_play, width=8)
        self.play_button.pack(side="left")
        ttk.Button(controls, text="Restart", command=lambda: self.seek(0)).pack(side="left", padx=4)
        ttk.Button(controls, text="Clear verdict", command=lambda: self.decide(None)).pack(
            side="right"
        )

        verdicts = ttk.Frame(right)
        verdicts.pack(fill="x", pady=(6, 0))
        for number, name in enumerate(VERDICTS):
            key = VERDICT_KEYS[number].replace("minus", "-")
            ttk.Button(
                verdicts, text=f"{key}  {name}", command=lambda name=name: self.decide(name)
            ).grid(row=number // 4, column=number % 4, sticky="ew", padx=2, pady=2)
        for column in range(4):
            verdicts.columnconfigure(column, weight=1)
        self.summary = tk.StringVar()
        ttk.Label(self, textvariable=self.summary, style="Hint.TLabel").pack(
            anchor="w", pady=(4, 0)
        )
        ttk.Label(
            self,
            text="Keys: space = play or pause, left/right = step a frame, up/down = capture, "
            "1 to 9, 0 and - = verdict (moves on to the next capture). "
            "The clip is what was recorded, with the detector's boxes already drawn in it.",
            style="Hint.TLabel",
            wraplength=1000,
            justify="left",
        ).pack(anchor="w")

    def browse(self) -> None:
        chosen = filedialog.askdirectory(initialdir=str(resolve(self.dir_var.get())))
        if chosen:
            self.dir_var.set(chosen)
            self.load_dir()

    def load_dir(self) -> None:
        self.page.settings.captures_dir = self.dir_var.get()
        self.captures = list_captures(resolve(self.dir_var.get()))
        self.fill()

    def fill(self) -> None:
        self.shown = [c for c in self.captures if not (self.only_open.get() and c.verdict)]
        self.listbox.delete(0, "end")
        for capture in self.shown:
            self.listbox.insert("end", self._row(capture))
        self._summarise()
        if self.shown:
            self.listbox.selection_set(0)
            self.select()
        else:
            self.close_video()
            self.picture.show(None)
            self.info.set("No captures to show.")

    def _row(self, capture: Capture) -> str:
        return f"{capture.event_id:<8} {capture.verdict or '-'}"

    def _summarise(self) -> None:
        done = sum(1 for c in self.captures if c.verdict)
        self.summary.set(f"{len(self.captures)} captures, {done} with a verdict.")

    def select(self) -> None:
        chosen = self.listbox.curselection()
        if not chosen:
            return
        capture = self.shown[chosen[0]]
        if capture is self.capture and self.video is not None:
            return
        self.close_video()
        self.capture = capture
        self.video = cv2.VideoCapture(str(capture.clip))
        self.frames = max(int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        self.slider.configure(to=self.frames - 1)
        self._describe()
        self.seek(0)
        self.set_playing(True)

    def _describe(self) -> None:
        if self.capture is None:
            return
        note = (
            f"   verdict: {self.capture.verdict}" if self.capture.verdict else "   no verdict yet"
        )
        self.info.set(f"{self.capture.clip.name}: {self.capture.describe()}{note}")

    def move(self, delta: int) -> None:
        if not self.shown:
            return
        chosen = self.listbox.curselection()
        row = min(max((chosen[0] if chosen else 0) + delta, 0), len(self.shown) - 1)
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(row)
        self.listbox.see(row)
        self.select()

    # --- playback -----------------------------------------------------------------------

    def close_video(self) -> None:
        self.set_playing(False)
        if self.video is not None:
            self.video.release()
        self.video, self.capture = None, None

    def read(self) -> bool:
        if self.video is None:
            return False
        ok, frame = self.video.read()
        if not ok:
            return False
        self.position = int(self.video.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.picture.show(frame)
        self.slider.set(self.position)
        return True

    def seek(self, frame: int) -> None:
        if self.video is None:
            return
        self.video.set(cv2.CAP_PROP_POS_FRAMES, min(max(frame, 0), self.frames - 1))
        self.read()

    def _slide(self, value: str) -> None:
        if self.video is not None and abs(float(value) - self.position) >= 1:
            self.seek(round(float(value)))

    def set_playing(self, playing: bool) -> None:
        self.playing = playing
        self.play_button.configure(text="Pause" if playing else "Play")
        if self._timer is not None:
            self.after_cancel(self._timer)
            self._timer = None
        if playing:
            self._timer = self.after(PLAY_MS, self._tick)

    def toggle_play(self) -> None:
        if self.video is None:
            return
        if not self.playing and self.position >= self.frames - 1:
            self.seek(0)
        self.set_playing(not self.playing)

    def _tick(self) -> None:
        self._timer = None
        if not self.playing or not self.winfo_viewable():
            self.set_playing(False)
            return
        if not self.read():
            self.set_playing(False)
            return
        self._timer = self.after(PLAY_MS, self._tick)

    # --- verdicts -----------------------------------------------------------------------

    def decide(self, verdict: str | None) -> None:
        if self.capture is None:
            return
        capture = self.capture
        write_verdict(capture, verdict)
        row = self.shown.index(capture)
        self.listbox.delete(row)
        self.listbox.insert(row, self._row(capture))
        self.listbox.selection_set(row)
        self._summarise()
        self._describe()
        if verdict is not None and row + 1 < len(self.shown):
            self.move(1)

    def key(self, name: str, shift: bool) -> None:
        if name in VERDICT_KEYS:
            self.decide(VERDICTS[VERDICT_KEYS.index(name)])
        elif name == "space":
            self.toggle_play()
        elif name in ("Left", "Right"):
            self.set_playing(False)
            self.seek(self.position + (1 if name == "Right" else -1) * (10 if shift else 1))
        elif name in ("Up", "Down"):
            self.move(-1 if name == "Up" else 1)


def build(page: Any) -> ttk.Notebook:
    """Fill a ``Page`` with the two tabs and wire up its keys."""
    notebook = ttk.Notebook(page)
    notebook.pack(fill="both", expand=True)
    frames, captures = FramesTab(notebook, page), CapturesTab(notebook, page)
    notebook.add(frames, text="Labelled frames")
    notebook.add(captures, text="Captures")
    page.tabs = (frames, captures)
    loaded: set[int] = set()

    def on_tab(_event: Any = None) -> None:
        current = notebook.index("current")
        if current != 1:
            captures.set_playing(False)
        if current not in loaded and page.winfo_viewable():
            loaded.add(current)
            (frames.load_root if current == 0 else captures.load_dir)()

    def on_key(event: Any) -> str | None:
        if not page.winfo_viewable():
            return None
        if isinstance(event.widget, (tk.Entry, ttk.Entry)) and not isinstance(
            event.widget, ttk.Combobox
        ):
            return None
        page.tabs[notebook.index("current")].key(event.keysym, bool(event.state & 0x1))
        return "break"

    notebook.bind("<<NotebookTabChanged>>", on_tab)
    page.bind("<Map>", on_tab)
    page.winfo_toplevel().bind("<Key>", on_key)
    # The list boxes move their own selection on arrow keys; the page handles those.
    for tab in page.tabs:
        for key in ("<Up>", "<Down>", "<Left>", "<Right>", "<space>"):
            tab.listbox.bind(key, on_key)
    return notebook
