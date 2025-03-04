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
    #VIDEO_FPS,
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
            print(f"Error: Failed to open video writer for {filename}")
            return False
            
        self.current_filename = filename
        self.is_running = True
        self.frames_written = 0
        
        # Start worker thread
        self.thread = threading.Thread(target=self._write_frames)
        self.thread.daemon = True
        self.thread.start()
        return True
        
    def stop(self) -> None:
        """Stop the video writer thread."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            # Add None to signal thread to exit
            try:
                self.frame_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
                
            self.thread.join(timeout=3.0)
            
        # Release writer
        if self.writer:
            self.writer.release()
            self.writer = None
            
        print(f"Video saved: {self.current_filename} ({self.frames_written} frames)")
        self.current_filename = None
        
    def write(self, frame: np.ndarray) -> bool:
        """Add a frame to the queue for writing."""
        if not self.is_running:
            return False
            
        try:
            # Only copy if queue is not almost full
            if self.frame_queue.qsize() < self.frame_queue.maxsize - 10:
                self.frame_queue.put(frame, timeout=0.5)
                return True
            else:
                # If queue is almost full, skip frame to prevent stalling
                print("Warning: Video write queue almost full, skipping frame")
                return False
        except queue.Full:
            print("Warning: Video write queue is full")
            return False
            
    def _write_frames(self) -> None:
        """Worker thread that processes the frame queue."""
        while self.is_running:
            try:
                # Get frame with timeout
                frame = self.frame_queue.get(timeout=0.5)
                
                # Check for exit signal
                if frame is None:
                    break
                    
                # Write frame
                if self.writer and frame is not None:
                    self.writer.write(frame)
                    self.frames_written += 1
                    
                self.frame_queue.task_done()
                
            except queue.Empty:
                # Just continue if queue is empty
                continue
            except Exception as e:
                print(f"Error writing video frame: {str(e)}")
                
        # Process any remaining frames in the queue
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                if frame is not None and self.writer:
                    self.writer.write(frame)
                    self.frames_written += 1
                self.frame_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing remaining frames: {str(e)}")
                break

class VideoManager:
    """Manages video capture and writing operations."""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 54  # Default fallback FPS
        # Initialize with proper buffer size based on default FPS
        buffer_size = int(BUFFER_SECONDS * self.fps)
        self.frame_buffer = deque(maxlen=buffer_size)  # Initialize with correct size
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

    def add_to_buffer(self, frame: np.ndarray, annotated_frame: Optional[np.ndarray] = None, debug_view: Optional[np.ndarray] = None) -> None:
        """
        Add frame to buffer, optionally with annotations.
        
        Frames are added to the right end of the deque (newest frames).
        When the buffer is full, the oldest frame is automatically removed from the left end.
        When we convert the deque to a list, the oldest frames will be at the beginning of the list.
        """
        # Only add frames that have been fully processed (have annotations)
        if annotated_frame is None:
            return
            
        # Store all three components in a single buffer entry to keep them in sync
        # This is more efficient than maintaining separate buffers
        self.frame_buffer.append((
            frame.copy() if frame is not None else None,
            annotated_frame.copy() if annotated_frame is not None else None,
            debug_view.copy() if debug_view is not None else None
        ))

    def update_buffer_annotations(self, annotated_frame: np.ndarray, debug_view: Optional[np.ndarray] = None) -> None:
        """
        Update the most recent buffer entry with annotations.
        Note: This method is kept for backward compatibility but is no longer needed
        with the new buffer approach.
        """
        if len(self.frame_buffer) > 0:
            # Get the last entry
            raw_frame, _, _ = self.frame_buffer[-1]
            # Replace it with updated annotations
            self.frame_buffer[-1] = (
                raw_frame,
                annotated_frame.copy() if annotated_frame is not None else None,
                debug_view.copy() if debug_view is not None else None
            )

    def start_recording(self, frame_size: tuple, debug_view: Optional[np.ndarray] = None) -> bool:
        """Start recording video with buffer."""
        if self.recording:
            return False
            
        # Get next available video number
        self.current_video_number = self.get_next_video_number()
        filename = os.path.join(VIDEO_SAVE_DIR, f"{self.current_video_number:05d}.avi")
        
        # Initialize video writer
        if self.video_writer.start(filename, self.fps, frame_size):
            buffer_length = len(self.frame_buffer)
            print(f"\nWriting {buffer_length} buffered frames to video")
            
            # Pre-allocate a single combined frame buffer to reuse
            combined_buffer = None
            frames_written = 0
            
            # Convert buffer to list to ensure we process in correct order (oldest to newest)
            # The deque stores newest frames at the end, so we need to start from the beginning
            # of the list to get the oldest frames first
            buffer_list = list(self.frame_buffer)
            
            # Get the detection frame debug view (from the last frame in buffer)
            detection_debug_view = None
            if buffer_length > 0:
                _, _, frame_debug = buffer_list[-1]
                detection_debug_view = frame_debug if frame_debug is not None else debug_view
            
            # Write all buffered frames in chronological order (oldest to newest)
            # For each frame, use the main view from that frame but the debug view from the detection frame
            for i, (raw, annotated, _) in enumerate(buffer_list):
                # Skip frames without annotations
                if annotated is None:
                    continue
                
                frames_written += 1
                
                h, w = annotated.shape[:2]
                
                # Use the detection debug view for all frames
                if detection_debug_view is not None:
                    debug_h, debug_w = detection_debug_view.shape[:2]
                    
                    # Reuse or create combined buffer
                    if combined_buffer is None or combined_buffer.shape != (max(h, debug_h), w + debug_w, 3):
                        combined_buffer = np.zeros((max(h, debug_h), w + debug_w, 3), dtype=np.uint8)
                    else:
                        combined_buffer.fill(0)
                    
                    # Fill the combined buffer - detection debug view on left, current frame on right
                    combined_buffer[:debug_h, :debug_w] = detection_debug_view
                    combined_buffer[:h, debug_w:] = annotated
                    
                    # Write the combined frame
                    self.video_writer.write(combined_buffer.copy())
                else:
                    self.video_writer.write(annotated.copy())
            
            print(f"Successfully wrote {frames_written} buffered frames to video")
            
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
            
        # Write frame to video
        if not self.video_writer.write(frame):
            print("Warning: Failed to write frame")
            
        self.frames_since_start += 1
            
        if has_detection:
            # Reset countdown on new detection
            self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1
            
        # Stop after POST_DETECTION_SECONDS with no detections
        if self.frames_since_detection >= int(self.fps * POST_DETECTION_SECONDS):
            self.stop_recording()
                
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        self.stop_recording()
        self.frame_buffer.clear()

    def set_fps(self, fps: float):
        """Update FPS and buffer sizes."""
        old_fps = self.fps
        self.fps = max(1.0, fps)  # Ensure positive FPS
        
        # Calculate new buffer size
        buffer_size = int(BUFFER_SECONDS * self.fps)
        
        # Only resize buffers if FPS has changed significantly or buffer size is wrong
        if abs(old_fps - self.fps) > 0.1 or self.frame_buffer.maxlen != buffer_size:
            # Create new buffer with correct size and preserve existing content
            old_buffer = list(self.frame_buffer)
            self.frame_buffer = deque(maxlen=buffer_size)
            
            # Copy existing frames to new buffer
            for item in old_buffer:
                self.frame_buffer.append(item)
            
            print(f"Video FPS set to: {self.fps:.2f}, buffer size: {buffer_size} frames")
        
    def create_combined_frame(self, annotated_frame: np.ndarray, debug_view: np.ndarray, 
                             combined_buffer: Optional[np.ndarray] = None,
                             has_detection: bool = False) -> np.ndarray:
        """Create a combined frame with debug view and main view side by side.
        Reuses the provided buffer if possible to avoid memory allocations.
        
        Args:
            annotated_frame: The main view frame
            debug_view: The debug view frame
            combined_buffer: Optional pre-allocated buffer to reuse
            has_detection: Whether this frame has a detection
        
        Returns:
            Combined frame with debug view on left and main view on right
        """
        if debug_view is None:
            return annotated_frame
            
        h, w = annotated_frame.shape[:2]
        debug_h, debug_w = debug_view.shape[:2]
        
        # Reuse or create combined buffer
        if combined_buffer is None or combined_buffer.shape != (max(h, debug_h), w + debug_w, 3):
            combined_buffer = np.zeros((max(h, debug_h), w + debug_w, 3), dtype=np.uint8)
        else:
            combined_buffer.fill(0)
        
        # Always show both views
        combined_buffer[:debug_h, :debug_w] = debug_view
        combined_buffer[:h, debug_w:] = annotated_frame
        
        return combined_buffer 