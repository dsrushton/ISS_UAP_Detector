"""
Video management and processing module.
Handles video capture, writing, and frame buffering.
"""

import cv2
import queue
import threading
import time
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from collections import deque
import os
from datetime import datetime

from SOD_Logger import StatusLogger as Logger
from SOD_Constants import (
    BUFFER_SECONDS,
    #VIDEO_FPS,
    VIDEO_SAVE_DIR,
    JPG_SAVE_DIR,
    POST_DETECTION_SECONDS,
    SAVE_INTERVAL
)

class ThreadedVideoWriter:
    """Handles video writing in a separate thread."""
    
    def __init__(self, max_queue_size: int = 300):
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.writer: Optional[cv2.VideoWriter] = None
        self.is_running = False
        self.thread = None
        self.current_filename = None
        self.frames_written = 0
        
    def start(self, filename: str, fps: float, frame_size: Tuple[int, int]) -> bool:
        """Start the video writer thread."""
        if self.is_running:
            return False
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
            
        # Initialize video writer with XVID codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        
        if not self.writer.isOpened():
            return False
            
        self.current_filename = filename
        self.frames_written = 0
        self.is_running = True
        
        # Start writer thread
        self.thread = threading.Thread(target=self._writer_thread, daemon=True)
        self.thread.start()
        
        return True
        
    def write(self, frame: np.ndarray) -> bool:
        """Add a frame to the queue for writing."""
        if not self.is_running:
            return False
            
        try:
            # Pass frame directly - no need to copy since the frame is only used for writing
            # and won't be modified by the writer thread
            self.frame_queue.put(frame, block=False)
            return True
        except queue.Full:
            return False
        except Exception as e:
            return False
            
    def _writer_thread(self):
        """Thread function that writes frames from the queue."""
        try:
            while self.is_running:
                try:
                    # Get frame from queue with timeout
                    # Using 0.5s timeout as a balance between responsiveness and reduced CPU usage
                    frame = self.frame_queue.get(timeout=0.5)
                    
                    # Write frame
                    if self.writer and frame is not None:
                        self.writer.write(frame)
                        self.frames_written += 1
                    
                    # Mark task as done
                    self.frame_queue.task_done()
                    
                except queue.Empty:
                    # No frames in queue, just continue
                    continue
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
        finally:
            pass
            
    def stop(self):
        """Stop the video writer thread."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            
        # Process remaining frames in queue
        remaining_frames = 0
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(block=False)
                if self.writer and frame is not None:
                    self.writer.write(frame)
                    remaining_frames += 1
                self.frame_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                pass
                
        # Release writer
        if self.writer:
            self.writer.release()
            self.writer = None
            
        self.current_filename = None

class VideoManager:
    """Manages video capture and writing operations."""
    
    def __init__(self, logger: Logger, fps: float = 30.0):
        """Initialize the video manager."""
        self.logger = logger
        self.fps = fps
        self.cap = None
        self.recording = False
        self.video_writer = ThreadedVideoWriter()
        self.next_video_number = self._init_video_number()
        
        # Calculate buffer size based on FPS and BUFFER_SECONDS
        buffer_size = int(fps * BUFFER_SECONDS)
        
        # Enforce minimum and maximum buffer sizes
        if buffer_size < 10:
            print(f"Warning: Calculated buffer size {buffer_size} is too small. Using minimum size of 10.")
            buffer_size = 10
        elif buffer_size > 1000:
            print(f"Warning: Calculated buffer size {buffer_size} is too large. Using maximum size of 1000.")
            buffer_size = 1000
            
        print(f"Initializing video buffer with size {buffer_size} frames ({BUFFER_SECONDS} seconds at {fps} FPS)")
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Track when recording started and last detection
        self.recording_start_time = 0
        self.last_detection_time = 0
        
        # Store the debug view for overlay in update_recording
        self.stored_debug_view = None
        
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
        return self.cap.isOpened()
        
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

    def add_to_buffer(self, combined_view: np.ndarray) -> None:
        """
        Add combined view frame to buffer.
        
        Frames are added to the right end of the deque (newest frames).
        When the buffer is full, the oldest frame is automatically removed from the left end.
        When we convert the deque to a list, the oldest frames will be at the beginning of the list.
        
        Args:
            combined_view: The combined view from display (debug view + annotated frame)
        """
        buffer_start = time.time()
        
        # Validate input
        if combined_view is None:
            #self.logger.log_error("Received None frame in add_to_buffer")
            return
            
        try:
            # Store the combined view
            self.frame_buffer.append(combined_view.copy())
            
            # Log operation time
            self.logger.log_operation_time('buffer_append', time.time() - buffer_start)
        except Exception as e:
            self.logger.log_error(f"Error adding frame to buffer: {str(e)}")
            

    def start_recording(self, frame_size: tuple, debug_view: Optional[np.ndarray] = None) -> bool:
        """Start recording video with buffer."""
        if self.recording:
            return False
            
        # Get next available video number
        self.current_video_number = self.get_next_video_number()
        filename = os.path.join(VIDEO_SAVE_DIR, f"{self.current_video_number:05d}.avi")
        
        # Check buffer status
        buffer_length = len(self.frame_buffer)
        buffer_duration = buffer_length / self.fps if self.fps > 0 else 0
        self.logger.log_iteration(True, True, f"Starting recording with {buffer_length} buffered frames ({buffer_duration:.1f} seconds)")
        
        if buffer_length == 0:
            self.logger.log_error("No frames in buffer! Check if add_to_buffer is being called.")
            # Try to continue anyway, but log the issue
        
        # Store the debug view for later use in update_recording
        if debug_view is not None:
            self.stored_debug_view = debug_view.copy()
        else:
            self.stored_debug_view = None
        
        # Initialize video writer
        if self.video_writer.start(filename, self.fps, frame_size):
            # Write buffered frames to video
            frames_written = 0
            
            # Convert buffer to list - oldest frames first
            buffer_list = list(self.frame_buffer)
            
            # Write all buffered frames to video
            for i, combined_frame in enumerate(buffer_list):
                if combined_frame is not None:
                    if self.video_writer.write(combined_frame):
                        frames_written += 1
                    else:
                        self.logger.log_error(f"Failed to write buffered frame {frames_written}")
            
            self.logger.log_iteration(True, True, f"Wrote {frames_written} buffered frames to video")
            
            # Start counting frames
            self.frames_since_start = frames_written
            self.frames_since_detection = 0
            self.recording = True
            return True
        else:
            self.logger.log_error("Failed to start video writer")
            return False
        
    def stop_recording(self) -> None:
        """Stop recording video."""
        if not self.recording:
            return
            
        try:
            recording_duration = self.frames_since_start / self.fps if self.fps > 0 else 0
            self.logger.log_iteration(True, False, f"Stopping recording after {self.frames_since_start} frames ({recording_duration:.1f} seconds)")
            self.recording = False
            self.video_writer.stop()
            
            # Clear the stored debug view
            self.stored_debug_view = None
        except Exception as e:
            self.logger.log_error(f"Error stopping recording: {str(e)}")
                
    def update_recording(self, frame: np.ndarray, has_detection: bool = False) -> None:
        """
        Update video recording with new frame.
        
        Args:
            frame: Frame to write to video (combined frame with debug view on left, annotated frame on right)
            has_detection: Whether this frame contains a detection
        """
        recording_start = time.time()
        
        if not self.recording or self.video_writer is None:
            self.logger.log_error("update_recording called but recording is not active")
            return
            
        if frame is None:
            self.logger.log_error("Received None frame in update_recording")
            return
            
        try:
            # We'll assume the frame is valid and proceed without the aspect ratio check
            # The previous check was causing frequent error messages
            
            # For the first 179 frames after the buffer, create a special view:
            # - Left half: Show the stored debug view (from when the detection occurred)
            # - Right half: Show only the right half of the combined frame (the annotated frame)
            if self.frames_since_start < 180 and self.stored_debug_view is not None:
                # Get the dimensions
                h, w = frame.shape[:2]
                half_w = w // 2
                
                # Create a new frame with the same dimensions as the combined frame
                modified_frame = np.zeros_like(frame)
                
                # Left half: Use the stored debug view (crop or pad if needed)
                if self.stored_debug_view.shape[0] != h or self.stored_debug_view.shape[1] != half_w:
                    # If dimensions don't match, create a properly sized debug view
                    debug_h, debug_w = self.stored_debug_view.shape[:2]
                    
                    # Create a black canvas of the correct size
                    sized_debug = np.zeros((h, half_w, 3), dtype=np.uint8)
                    
                    # Calculate dimensions for copying (to avoid out-of-bounds)
                    copy_h = min(h, debug_h)
                    copy_w = min(half_w, debug_w)
                    
                    # Copy the debug view (or a portion of it) to the canvas
                    sized_debug[0:copy_h, 0:copy_w] = self.stored_debug_view[0:copy_h, 0:copy_w]
                    
                    # Use the properly sized debug view
                    modified_frame[0:h, 0:half_w] = sized_debug
                else:
                    # Dimensions match, use as is
                    modified_frame[0:h, 0:half_w] = self.stored_debug_view
                
                # Right half: Use the right half of the combined frame (the annotated frame)
                modified_frame[0:h, half_w:w] = frame[0:h, half_w:w]
                
                # Write the modified frame to video
                if not self.video_writer.write(modified_frame):
                    self.logger.log_error("Failed to write frame in update_recording")
            else:
                # After 179 frames, use the combined frame as is
                if not self.video_writer.write(frame):
                    self.logger.log_error("Failed to write frame in update_recording")
            
            self.frames_since_start += 1
                
            if has_detection:
                # Reset countdown on new detection
                if self.frames_since_detection > 0:
                    self.logger.log_iteration(True, True, f"New detection at frame {self.frames_since_start}, resetting post-detection countdown")
                self.frames_since_detection = 0
            else:
                self.frames_since_detection += 1
                
            # Stop after POST_DETECTION_SECONDS with no detections
            frames_to_continue = int(self.fps * POST_DETECTION_SECONDS)
            if self.frames_since_detection >= frames_to_continue:
                self.logger.log_iteration(True, False, f"No detection for {POST_DETECTION_SECONDS} seconds, stopping recording")
                self.stop_recording()
                
            self.logger.log_operation_time('recording_write', time.time() - recording_start)
        except Exception as e:
            self.logger.log_error(f"Error in update_recording: {str(e)}")
                
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        self.stop_recording()
        self.frame_buffer.clear()

    def set_fps(self, fps: float):
        """Set the FPS for video recording and adjust buffer size."""
        if fps <= 0 or fps > 120:  # Sanity check
            print(f"Warning: Invalid FPS value ({fps}), using default of 30")
            fps = 30.0
            
        # Only update if FPS has changed
        if self.fps == fps:
            return
            
        old_fps = self.fps
        self.fps = fps
        
        # Recalculate buffer size based on new FPS
        new_buffer_size = int(BUFFER_SECONDS * fps)
        
        # Only recreate the buffer if the size has changed
        if new_buffer_size != self.frame_buffer.maxlen:
            print(f"Resizing video buffer from {self.frame_buffer.maxlen} frames ({self.frame_buffer.maxlen/old_fps:.1f}s) to {new_buffer_size} frames ({new_buffer_size/fps:.1f}s)")
            
            # Create a new buffer with the new size
            old_buffer = list(self.frame_buffer)
            self.frame_buffer = deque(maxlen=new_buffer_size)
            
            # Copy as many frames as will fit in the new buffer
            # Start with the newest frames (from the end of old_buffer)
            frames_to_copy = min(len(old_buffer), new_buffer_size)
            if frames_to_copy > 0:
                for i in range(frames_to_copy):
                    # Get frame from the end of old_buffer (newest frames first)
                    idx = len(old_buffer) - frames_to_copy + i
                    if idx >= 0 and idx < len(old_buffer):
                        self.frame_buffer.append(old_buffer[idx])
                
            print(f"Video buffer resized: {len(self.frame_buffer)}/{new_buffer_size} frames copied")
        
        print(f"Video FPS set to {fps}, buffer size: {self.frame_buffer.maxlen} frames ({BUFFER_SECONDS} seconds)")
        return self.fps

    def create_combined_frame(self, annotated_frame: np.ndarray, debug_view: np.ndarray, 
                             combined_buffer: Optional[np.ndarray] = None,
                             has_detection: bool = False) -> np.ndarray:
        """
        This method is kept for backward compatibility but should not be used.
        We expect to receive already combined frames from the display module.
        """
        # Just return the annotated_frame, which should already be a combined view
        return annotated_frame 