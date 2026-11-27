"""The point-and-click app: ``uapd app``.

    uapd/app/window.py   the Tk window (home page and one page per branch)
    uapd/app/jobs.py     command lines, saved form values, child processes

Each branch runs as its own ``python -m uapd ...`` process, so the window only
depends on the command line of the detector and the trainer, not on their code.
"""

from __future__ import annotations


def main() -> int:
    try:
        from uapd.app.window import main as window_main
    except ImportError as exc:  # Python built without Tk
        print(f"uapd: error: the app needs Tk (tkinter): {exc}")
        return 2
    return window_main()
