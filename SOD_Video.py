import cv2
import numpy as np
from collections import deque
import time
import os
from typing import Optional, Tuple
import logging

from SOD_Constants import (
    BUFFER_SECONDS, POST_DETECTION_SECONDS, 
    VIDEO_FPS, VIDEO_SAVE_DIR
)

class VideoManager:
    """Manages video buffering and recording."""
    
    def __init__(self):
        self.frame_buffer = deque(maxlen=BUFFER_SECONDS * VIDEO_FPS)
        self.is_recording = False
        self.writer = None
        self.counter = 0
        self.last_detection_time = 0
        self.current_video_path = None
        self.cap = None  # Initialize as None, will set up in set_source()
        
        # Ensure save directory exists
        os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
        
        # Get last counter value
        existing_files = [f for f in os.listdir(VIDEO_SAVE_DIR) if f.endswith('.avi')]
        if existing_files:
            numbers = [int(f.replace('.avi', '')) for f in existing_files]
            self.counter = max(numbers) + 1
    
    def add_to_buffer(self, frame: np.ndarray) -> None:
        """Add frame to rolling buffer."""
        self.frame_buffer.append(frame.copy())
    
    def start_recording(self, frame: np.ndarray, debug_view: np.ndarray = None) -> None:
        """Start new recording with buffered frames."""
        if self.is_recording:
            return
            
        h, w = frame.shape[:2]
        
        # If we have a debug view, create combined frame
        if debug_view is not None:
            debug_h, debug_w = debug_view.shape[:2]
            # Create wider frame to accommodate both views
            w = w + debug_w
        
        filename = f"{self.counter:05d}.avi"
        self.current_video_path = os.path.join(VIDEO_SAVE_DIR, filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(
            self.current_video_path,
            fourcc,
            VIDEO_FPS,
            (w, h)
        )
        
        # Write ALL buffered frames first
        buffered_frames = list(self.frame_buffer)  # Convert deque to list to preserve order
        for buffered_frame in buffered_frames:
            if debug_view is not None:
                # Create combined frame
                combined = np.zeros((h, w, 3), dtype=np.uint8)
                combined[0:debug_h, 0:debug_w] = debug_view
                combined[0:h, debug_w:] = buffered_frame
                self.writer.write(combined)
            else:
                self.writer.write(buffered_frame)
        
        # Write current frame
        if debug_view is not None:
            combined = np.zeros((h, w, 3), dtype=np.uint8)
            combined[0:debug_h, 0:debug_w] = debug_view
            combined[0:h, debug_w:] = frame
            self.writer.write(combined)
        else:
            self.writer.write(frame)
        
        self.is_recording = True
        self.last_detection_time = time.time()
        print(f"\nStarted recording: {filename} (with {len(buffered_frames)} buffered frames)")
    
    def update_recording(self, frame: np.ndarray, has_detection: bool, debug_view: np.ndarray = None) -> None:
        """Update ongoing recording."""
        if not self.is_recording:
            return
            
        current_time = time.time()
        
        # Write the frame
        if debug_view is not None:
            h, w = frame.shape[:2]
            debug_h, debug_w = debug_view.shape[:2]
            combined = np.zeros((h, w + debug_w, 3), dtype=np.uint8)
            combined[0:debug_h, 0:debug_w] = debug_view
            combined[0:h, debug_w:] = frame
            self.writer.write(combined)
        else:
            self.writer.write(frame)
        
        if has_detection:
            # Reset the timer on new detection
            self.last_detection_time = current_time
        elif current_time - self.last_detection_time > POST_DETECTION_SECONDS:
            # Stop recording if no detection for POST_DETECTION_SECONDS
            self.stop_recording()
    
    def stop_recording(self) -> None:
        """Stop current recording."""
        if not self.is_recording:
            return
            
        self.writer.release()
        self.writer = None
        self.is_recording = False
        self.counter += 1
        print(f"\nStopped recording: {self.current_video_path}")
        self.current_video_path = None
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.writer:
            self.writer.release()
        if self.cap:
            self.cap.release()
    
    def set_source(self, source: str) -> bool:
        """Set video source and initialize capture."""
        try:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                print(f"Failed to open video source: {source}")
                return False
            
            return True
        except Exception as e:
            print(f"Error setting video source: {e}")
            return False
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get frame from video source."""
        try:
            if self.cap is None:
                return False, None
            
            ret, frame = self.cap.read()
            if ret:
                self.add_to_buffer(frame)
            return ret, frame
        except Exception as e:
            logging.error(f"Error getting frame: {e}")
            return False, None 