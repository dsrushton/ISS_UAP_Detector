"""
Space Object Detection Capture Module
Handles saving of detections, burst captures, and frame management.
"""

import cv2
import os
import time
import queue
import threading
import numpy as np
from typing import Optional, Tuple
import json

from SOD_Constants import (
    JPG_SAVE_DIR,
    RAW_SUBDIR,
    SAVE_INTERVAL
)
from SOD_Utils import ensure_save_directory
from SOD_Detections import DetectionResults

class CaptureManager:
    """Manages frame capture and saving operations."""
    
    def __init__(self, save_dir: str = JPG_SAVE_DIR):
        """
        Initialize the capture manager.
        
        Args:
            save_dir: Directory for saving detections
        """
        self.save_dir = save_dir
        self.raw_dir = os.path.join(save_dir, RAW_SUBDIR)
        self.current_video_number = None
        self.current_jpg_suffix = 'a'
        self.last_save_time = 0
        
        # Setup save queue and worker thread
        self.save_queue = queue.Queue()
        self.save_thread = None
        self.is_running = False
        
        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

    def initialize(self) -> bool:
        """
        Initialize directories and start save worker thread.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Create directories if needed
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.raw_dir, exist_ok=True)
            
            # Start save worker thread
            self.is_running = True
            self.save_thread = threading.Thread(
                target=self._save_worker,
                daemon=True
            )
            self.save_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error initializing capture manager: {str(e)}")
            return False

    def _save_worker(self):
        """Worker thread for handling frame saves."""
        while self.is_running:
            try:
                # Get save data from queue
                save_data = self.save_queue.get(timeout=1.0)
                if save_data is None:  # Exit signal
                    break
                    
                image, save_path = save_data
                cv2.imwrite(save_path, image)
                print(f"Saved frame to {save_path}")
                
                self.save_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in save worker: {str(e)}")
                continue

    def start_new_video(self, video_number: int) -> None:
        """
        Start tracking a new video number.
        
        Args:
            video_number: The video number to track
        """
        self.current_video_number = video_number
        self.current_jpg_suffix = 'a'
        self.last_save_time = 0

    def get_next_filename(self, check_interval: bool = True) -> Optional[str]:
        """
        Get the next filename to use for saving, respecting save interval.
        
        Args:
            check_interval: Whether to enforce SAVE_INTERVAL
            
        Returns:
            str: Filename to use, or None if should not save yet
        """
        if self.current_video_number is None:
            return f"unmatched_{int(time.time())}.jpg"
            
        current_time = time.time()
        if check_interval and current_time - self.last_save_time < SAVE_INTERVAL:
            return None
            
        filename = f"{self.current_video_number:05d}-{self.current_jpg_suffix}.jpg"
        self.current_jpg_suffix = chr(ord(self.current_jpg_suffix) + 1)
        self.last_save_time = current_time
        return filename

    def save_detection(self, frame: np.ndarray, debug_view: np.ndarray = None, filename: str = None, check_interval: bool = True) -> None:
        """
        Save a detection frame with debug view if available.
        
        Args:
            frame: Frame to save
            debug_view: Debug visualization
            filename: Optional specific filename to use
            check_interval: Whether to enforce SAVE_INTERVAL
        """
        try:
            # Get filename if not provided
            if filename is None:
                filename = self.get_next_filename(check_interval)
                if filename is None:  # Too soon to save
                    return
            
            # Skip save if metadata indicates we should
            if hasattr(debug_view, 'metadata') and debug_view.metadata.get('skip_save'):
                print(f"\nSkipping save: {debug_view.metadata['skip_save']}")
                return
            
            # Create combined image if debug view provided
            if debug_view is not None:
                debug_h, debug_w = debug_view.shape[:2]
                frame_h, frame_w = frame.shape[:2]
                combined = np.zeros((frame_h, frame_w + debug_w, 3), dtype=np.uint8)
                # Put debug view on left, main frame on right
                combined[0:debug_h, 0:debug_w] = debug_view
                combined[0:frame_h, debug_w:] = frame
                save_image = combined
            else:
                save_image = frame
            
            save_path = os.path.join(self.save_dir, filename)
            self.save_queue.put((save_image, save_path))
            
        except Exception as e:
            print(f"Error saving detection: {str(e)}")

    def process_detections(self, frame: np.ndarray, detections: DetectionResults, debug_view: np.ndarray = None) -> None:
        """Process detections and save frames."""
        try:
            if self.is_paused():
                return
                
            if detections.anomalies:
                self.save_detection(frame, debug_view, check_interval=True)
                
        except Exception as e:
            print(f"Error processing detections: {str(e)}")

    def save_raw_frame(self, frame: np.ndarray) -> None:
        """
        Save raw frame without processing.
        
        Args:
            frame: Frame to save
        """
        try:
            # Generate timestamp-based filename
            timestamp = int(time.time() * 1000)  # Millisecond timestamp
            filename = f"raw_{timestamp}.jpg"
            save_path = os.path.join(self.raw_dir, filename)
            
            # Add to save queue
            self.save_queue.put((frame, save_path))
            
        except Exception as e:
            print(f"Error saving raw frame: {str(e)}")

    def start_burst_capture(self, frame_count: int = 100) -> None:
        """Start burst capture mode"""

    def process_burst(self, frame: np.ndarray) -> bool:
        """
        Process a frame during burst capture.
        
        Args:
            frame: Frame to process
        
        Returns:
            bool: True if burst is still active
        """
        if self.burst_remaining > 0:
            self.save_raw_frame(frame)
            self.burst_remaining -= 1
            
            if self.burst_remaining == 0:
                print("\nBurst save complete!")
                return False
            
            return True
        
        return False

    def pause_capture(self, duration: float = 5.0) -> None:
        """
        Pause capture for specified duration.
        
        Args:
            duration: Pause duration in seconds
        """
        self.pause_until = time.time() + duration

    def is_paused(self) -> bool:
        """
        Check if capture is currently paused.
        
        Returns:
            bool: True if capture is paused
        """
        return time.time() < self.pause_until

    def cleanup(self):
        """Clean up resources and stop worker thread."""
        self.is_running = False
        if self.save_thread and self.save_thread.is_alive():
            self.save_queue.put(None)  # Send exit signal
            self.save_thread.join(timeout=5.0)  # Wait up to 5 seconds
            
        # Clear queue
        while not self.save_queue.empty():
            try:
                self.save_queue.get_nowait()
            except queue.Empty:
                break

    def set_video_counter(self, counter: int) -> None:
        """Set current video counter and reset jpg suffix."""
        self.current_video_counter = counter
        self.current_jpg_suffix = 'a'

    def get_detection_count(self, video_number: int) -> int:
        """
        Get the number of detections for a specific video number.
        
        Args:
            video_number: The video number to count detections for
            
        Returns:
            int: Number of detections (0 if no detections found)
        """
        count = 0
        # Count files matching pattern XXXXX-[a-z].jpg
        for f in os.listdir(self.save_dir):
            if f.startswith(f"{video_number:05d}-") and f.endswith(".jpg"):
                count += 1
        return count


   