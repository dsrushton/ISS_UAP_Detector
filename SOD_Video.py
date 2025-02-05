"""
Video management and processing module.
Handles video capture, writing, and frame buffering.
"""

import cv2
import queue
import threading
import time
from typing import Optional, Tuple
import numpy as np
from collections import deque
import os

from SOD_Constants import (
    BUFFER_SECONDS,
    VIDEO_FPS,
    VIDEO_SAVE_DIR,
    JPG_SAVE_DIR,
    POST_DETECTION_SECONDS
)

class ThreadedVideoWriter:
    """Handles video writing in a separate thread."""
    
    def __init__(self, max_queue_size: int = 300):
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.writer: Optional[cv2.VideoWriter] = None
        self.is_running = False
        self.thread = None
        self.current_filename = None
        
    def start(self, filename: str, fps: float, frame_size: Tuple[int, int]) -> bool:
        """Start the video writer thread."""
        if self.is_running:
            return False
            
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        if not self.writer.isOpened():
            return False
            
        self.current_filename = filename
        self.is_running = True
        self.thread = threading.Thread(target=self._write_frames)
        self.thread.daemon = True
        self.thread.start()
        return True
        
    def stop(self) -> None:
        """Stop the video writer thread."""
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.writer:
            self.writer.release()
            self.writer = None
        self.current_filename = None
        
    def write(self, frame: np.ndarray) -> bool:
        """Add a frame to the queue for writing."""
        try:
            self.frame_queue.put(frame.copy(), timeout=1)
            return True
        except queue.Full:
            print("Warning: Video write queue is full")
            return False
            
    def _write_frames(self) -> None:
        """Worker thread that processes the frame queue."""
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1)
                if self.writer and frame is not None:
                    self.writer.write(frame)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error writing video frame: {str(e)}")
                
class VideoManager:
    """Manages video capture and writing operations."""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_buffer = deque(maxlen=BUFFER_SECONDS * VIDEO_FPS)
        self.annotated_buffer = deque(maxlen=BUFFER_SECONDS * VIDEO_FPS)
        self.debug_buffer = deque(maxlen=BUFFER_SECONDS * VIDEO_FPS)
        self.video_writer = ThreadedVideoWriter()
        self.recording = False
        self.frames_since_detection = 0
        self.frames_since_start = 0
        self.current_video_number = None
        # Initialize next video number by checking AVI directory once
        self.next_video_number = self._init_video_number()
        
    def _init_video_number(self) -> int:
        """Find the next available video number by checking AVI directory once at startup."""
        i = 0
        while os.path.exists(os.path.join(VIDEO_SAVE_DIR, f"{i:05d}.avi")):
            i += 1
        return i

    def set_source(self, source: str) -> bool:
        """Set the video source and initialize capture."""
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            # Set frame rate explicitly
            self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
            return True
        return False
        
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the next frame from the video source."""
        if not self.cap or not self.cap.isOpened():
            return False, None
            
        ret, frame = self.cap.read()
        return ret, frame
        
    def get_next_video_number(self) -> int:
        """Get and increment the next video number."""
        number = self.next_video_number
        self.next_video_number += 1
        return number

    def add_to_buffer(self, frame: np.ndarray, annotated_frame: np.ndarray, debug_view: Optional[np.ndarray]) -> None:
        """Add frame, annotated frame and debug view to buffers."""
        self.frame_buffer.append(frame.copy())
        self.annotated_buffer.append(annotated_frame.copy())
        self.debug_buffer.append(debug_view.copy() if debug_view is not None else None)
        
    def start_recording(self, frame_size: tuple, debug_view: Optional[np.ndarray] = None) -> bool:
        """Start recording video with buffer."""
        if self.recording:
            return False
            
        # Get next available video number
        self.current_video_number = self.get_next_video_number()
        filename = os.path.join(VIDEO_SAVE_DIR, f"{self.current_video_number:05d}.avi")
            
        # Initialize video writer
        if self.video_writer.start(filename, VIDEO_FPS, frame_size):
            # Write buffered frames first, using the detection's debug view
            for annotated in self.annotated_buffer:
                if debug_view is not None:
                    h, w = annotated.shape[:2]
                    debug_h, debug_w = debug_view.shape[:2]
                    combined = np.zeros((max(h, debug_h), w + debug_w, 3), dtype=np.uint8)
                    combined[:debug_h, :debug_w] = debug_view
                    combined[:h, debug_w:] = annotated
                    self.video_writer.write(combined)
                else:
                    self.video_writer.write(annotated)
            self.recording = True
            self.frames_since_detection = 0
            return True
        return False
        
    def stop_recording(self) -> None:
        """Stop recording video."""
        if self.recording:
            self.recording = False
            self.video_writer.stop()
            self.frames_since_start = 0
            print("\nStopped recording")
            
    def update_recording(self, frame: np.ndarray, has_detection: bool = False) -> None:
        """Update recording with new frame."""
        if not self.recording:
            return
            
        if not self.video_writer.write(frame):
            print("Warning: Failed to write frame")
            
        self.frames_since_start += 1
            
        if has_detection:
            # Reset countdown on new detection
            self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1
            
        # Stop after POST_DETECTION_SECONDS with no detections
        if self.frames_since_detection >= VIDEO_FPS * POST_DETECTION_SECONDS:
            self.stop_recording()
                
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        self.stop_recording()
        self.frame_buffer.clear()
        self.annotated_buffer.clear()
        self.debug_buffer.clear() 