from __future__ import annotations

import cv2
import numpy as np
import pytest

from uapd.sources import FrameFolderSource, SourceError, VideoFileSource, open_source


def _frame(value: int) -> np.ndarray:
    return np.full((48, 64, 3), value, np.uint8)


def test_folder_source_reads_images_in_name_order(tmp_path):
    for name, value in [("b.png", 20), ("a.png", 10), ("c.png", 30)]:
        cv2.imwrite(str(tmp_path / name), _frame(value))
    (tmp_path / "notes.txt").write_text("ignored")

    source = open_source(tmp_path)
    assert isinstance(source, FrameFolderSource)
    frames = list(source.frames())
    assert [index for index, _ in frames] == [0, 1, 2]
    assert [int(frame[0, 0, 0]) for _, frame in frames] == [10, 20, 30]
    assert source.describe()["files"] == 3


def test_video_source_reads_every_frame(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    assert writer.isOpened()
    for value in (0, 100, 200, 100):
        writer.write(_frame(value))
    writer.release()

    source = open_source(path)
    assert isinstance(source, VideoFileSource)
    info = source.describe()
    assert (info["width"], info["height"]) == (64, 48)
    frames = list(source.frames())
    assert len(frames) == 4
    assert abs(int(frames[2][1][0, 0, 0]) - 200) < 10


def test_missing_source_is_an_error(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        open_source(tmp_path / "nope.mp4")


def test_empty_folder_is_an_error(tmp_path):
    with pytest.raises(SourceError, match="no image files"):
        open_source(tmp_path)


def test_unreadable_video_is_an_error(tmp_path):
    path = tmp_path / "broken.avi"
    path.write_bytes(b"not a video")
    with pytest.raises(SourceError, match="could not open"):
        open_source(path)
