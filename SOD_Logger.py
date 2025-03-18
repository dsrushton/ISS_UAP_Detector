"""
Space Object Detection Logger Module
Simplified logging of key performance metrics.
"""

import threading
import time
from datetime import datetime
import os
from typing import Dict
from collections import deque

class StatusLogger:
    """Handles simple logging of key system performance metrics."""
    
    def __init__(self, log_interval: int = 30):  # Report every 30 seconds
        self.is_running = False
        self.thread = None
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.start_time = time.time()
        
        # Frame counters
        self.detection_frame_count = 0
        self.stream_frame_count = 0
        self.last_detection_count = 0
        self.last_stream_count = 0
        self.last_count_time = time.time()
        
        # Performance tracking of key operations
        self.operation_times = {
            'total_iteration': deque(maxlen=100),  # Total time for detection loop
            'combined_view': deque(maxlen=100),    # Time for building combined view
        }
        
        # Ensure logs directory exists
        os.makedirs("./logs", exist_ok=True)
        
        # Create log file
        self.log_file = os.path.join("./logs", f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(self.log_file, 'w') as f:
            f.write(f"Performance log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
    
    def start_frame(self):
        """Start timing a new frame."""
        self.frame_start_time = time.time()
    
    def end_frame(self):
        """End timing for the current frame and update detection frame count."""
        if hasattr(self, 'frame_start_time') and self.frame_start_time > 0:
            total_time = time.time() - self.frame_start_time
            self.log_operation_time('total_iteration', total_time)
        
        # Increment detection frame count
        self.detection_frame_count += 1
        
    def start(self):
        """Start the logger thread."""
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._log_worker)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the logger thread."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
            
    def log_operation_time(self, operation: str, duration: float):
        """Log the duration of a key operation."""
        if operation in self.operation_times:
            self.operation_times[operation].append(duration)
        
    def log_stream_frame(self):
        """Increment stream frame counter."""
        self.stream_frame_count += 1
        
    def log_error(self, error_msg: str):
        """Log critical errors to console and file."""
        print(f"ERROR: {error_msg}")
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"[ERROR] {datetime.now()}: {error_msg}\n")
        except Exception as e:
            print(f"Failed to write to error log: {str(e)}")

    def ensure_running(self):
        """Ensure logger thread is running, restart if needed."""
        if not self.is_running or not self.thread or not self.thread.is_alive():
            print("Logger thread not running, restarting...")
            self.start()
            
    def set_expected_fps(self, fps: float):
        """Set the expected FPS - keeps method for backward compatibility."""
        pass  # Simplified version doesn't need this

    def log_memory_usage(self):
        """Kept for backward compatibility, does nothing in simplified version."""
        pass
            
    def log_iteration(self, success: bool, had_detection: bool = False, error_msg: str = None):
        """Legacy method kept for backward compatibility."""
        if not success and error_msg:
            self.log_error(error_msg)
    
    def _log_worker(self):
        """Worker thread that logs performance metrics."""
        while self.is_running:
            try:
                # Sleep for short interval to check if log is needed
                time.sleep(1.0)
                
                current_time = time.time()
                # Check if it's time to log performance metrics
                if current_time - self.last_log_time >= self.log_interval:
                    self._log_performance()
                    self.last_log_time = current_time
                    
            except Exception as e:
                print(f"Error in logger: {str(e)}")
                time.sleep(1.0)  # Prevent tight error loop
                
        print("Logger thread stopped")
        
    def _log_performance(self):
        """Log performance metrics."""
        try:
            current_time = time.time()
            time_since_last = current_time - self.last_count_time
            
            # Calculate detection framerate
            detection_frames = self.detection_frame_count - self.last_detection_count
            detection_fps = detection_frames / time_since_last if time_since_last > 0 else 0
            
            # Calculate stream framerate
            stream_frames = self.stream_frame_count - self.last_stream_count
            stream_fps = stream_frames / time_since_last if time_since_last > 0 else 0
            
            # Calculate average times for operations
            avg_total_time = 0
            if self.operation_times['total_iteration']:
                avg_total_time = sum(self.operation_times['total_iteration']) / len(self.operation_times['total_iteration'])
            
            avg_combined_view_time = 0
            if 'combined_view' in self.operation_times and self.operation_times['combined_view']:
                avg_combined_view_time = sum(self.operation_times['combined_view']) / len(self.operation_times['combined_view'])
            
            # Log to console
            print(f"\n--- Performance Metrics ({datetime.now().strftime('%H:%M:%S')}) ---")
            print(f"Detection loop: {detection_fps:.1f} FPS (avg time: {avg_total_time*1000:.1f} ms)")
            print(f"Combined view: {avg_combined_view_time*1000:.1f} ms per frame")
            print(f"Stream output: {stream_fps:.1f} FPS")
            
            # Log to file
            with open(self.log_file, 'a') as f:
                f.write(f"\n--- {datetime.now()} ---\n")
                f.write(f"Detection: {detection_fps:.1f} FPS (avg time: {avg_total_time*1000:.1f} ms)\n")
                f.write(f"Combined view: {avg_combined_view_time*1000:.1f} ms per frame\n")
                f.write(f"Stream: {stream_fps:.1f} FPS\n")
                
            # Update counters for next calculation
            self.last_detection_count = self.detection_frame_count
            self.last_stream_count = self.stream_frame_count
            self.last_count_time = current_time
            
        except Exception as e:
            print(f"Error logging performance: {str(e)}") 