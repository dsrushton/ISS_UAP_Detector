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
    JPG_SAVE_DIR
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
        self.video_writer = ThreadedVideoWriter()
        self.recording = False
        self.frames_since_detection = 0
        self.current_video_number = None
        
    def set_source(self, source: str) -> bool:
        """Set the video source and initialize capture."""
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(source)
        return self.cap.isOpened()
        
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the next frame from the video source."""
        if not self.cap or not self.cap.isOpened():
            return False, None
            
        ret, frame = self.cap.read()
        if ret:
            self.frame_buffer.append(frame.copy())
        return ret, frame
        
    def get_next_video_number(self) -> int:
        """Find the next available video number by checking both video and jpg files."""
        i = 0
        # Check both AVI and JPG directories to find highest number
        while True:
            avi_exists = os.path.exists(os.path.join(VIDEO_SAVE_DIR, f"{i:05d}.avi"))
            jpg_exists = any(f.startswith(f"{i:05d}-") for f in os.listdir(JPG_SAVE_DIR))
            if not avi_exists and not jpg_exists:
                break
            i += 1
        return i

    def start_recording(self, frame_size: tuple) -> bool:
        """Start recording video with buffer."""
        if self.recording:
            return False
            
        # Get next available video number
        self.current_video_number = self.get_next_video_number()
        filename = os.path.join(VIDEO_SAVE_DIR, f"{self.current_video_number:05d}.avi")
            
        # Initialize video writer
        if self.video_writer.start(filename, VIDEO_FPS, frame_size):
            # Write buffered frames first
            for frame in self.frame_buffer:
                self.video_writer.write(frame)
            self.recording = True
            self.frames_since_detection = 0
            return True
        return False
        
    def stop_recording(self) -> None:
        """Stop recording video."""
        if self.recording:
            self.recording = False
            self.video_writer.stop()
            print("\nStopped recording")
            
    def update_recording(self, frame: np.ndarray, has_detection: bool = False) -> None:
        """Update recording with new frame."""
        if not self.recording:
            return
            
        if not self.video_writer.write(frame):
            print("Warning: Failed to write frame")
            
        if has_detection:
            # Reset countdown on new detection
            self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1
            
        # Stop after 2 seconds with no detections
        if self.frames_since_detection >= VIDEO_FPS * 2:
            self.stop_recording()
                
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        self.stop_recording()
        self.frame_buffer.clear() 