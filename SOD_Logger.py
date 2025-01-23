"""
Space Object Detection Logger Module
Handles threaded logging of system status and performance metrics.
"""

import threading
import queue
import time
from datetime import datetime
import os
from typing import Dict
from collections import deque

class StatusLogger:
    """Handles threaded logging of system status."""
    
    def __init__(self, log_interval: int = 300):  # Log status every 5 minutes by default
        self.log_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.frame_count = 0
        self.detection_count = 0
        self.error_count = 0
        
        # Performance tracking
        self.operation_times: Dict[str, deque] = {
            'frame_read': deque(maxlen=100),
            'detection': deque(maxlen=100),
            'display_update': deque(maxlen=100),
            'debug_view': deque(maxlen=100),
            'video_buffer': deque(maxlen=100),
            'video_recording': deque(maxlen=100),
            'total_iteration': deque(maxlen=100),
            # Granular detection timing
            'rcnn_prep': deque(maxlen=100),
            'rcnn_inference': deque(maxlen=100),
            'rcnn_postprocess': deque(maxlen=100),
            'mask_creation': deque(maxlen=100),
            'contour_finding': deque(maxlen=100),
            'contour_analysis': deque(maxlen=100),
            'box_filtering': deque(maxlen=100),
            'brightness_check': deque(maxlen=100)
        }
        
        # Ensure logs directory exists
        os.makedirs("./logs", exist_ok=True)
        
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
            self.thread.join()
            
    def log_operation_time(self, operation: str, duration: float):
        """Log the duration of an operation."""
        if operation in self.operation_times:
            self.operation_times[operation].append(duration)
            
    def _calculate_stats(self, times: deque) -> tuple:
        """Calculate min, max, avg for a set of times."""
        if not times:
            return 0, 0, 0
        return min(times), max(times), sum(times) / len(times)
            
    def log_iteration(self, success: bool, had_detection: bool = False, error_msg: str = None):
        """Log the result of a processing iteration."""
        self.frame_count += 1
        if had_detection:
            self.detection_count += 1
        if not success:
            self.error_count += 1
            
        # Check if it's time to write a status update
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            # Calculate performance stats
            perf_stats = {}
            
            # Group operations for cleaner output
            operation_groups = {
                'Main Pipeline': ['frame_read', 'detection', 'display_update', 'debug_view', 
                                'video_buffer', 'video_recording', 'total_iteration'],
                'RCNN Operations': ['rcnn_prep', 'rcnn_inference', 'rcnn_postprocess'],
                'Detection Analysis': ['mask_creation', 'contour_finding', 'contour_analysis', 
                                     'box_filtering', 'brightness_check']
            }
            
            for group, operations in operation_groups.items():
                group_stats = {}
                for op in operations:
                    if op in self.operation_times and self.operation_times[op]:
                        min_time, max_time, avg_time = self._calculate_stats(self.operation_times[op])
                        group_stats[op] = {
                            'min': min_time * 1000,  # Convert to ms
                            'max': max_time * 1000,
                            'avg': avg_time * 1000
                        }
                if group_stats:
                    perf_stats[group] = group_stats
            
            self.log_queue.put({
                'timestamp': datetime.now(),
                'frames_processed': self.frame_count,
                'detections': self.detection_count,
                'errors': self.error_count,
                'last_error': error_msg,
                'performance': perf_stats
            })
            self.last_log_time = current_time
            
    def _log_worker(self):
        """Worker thread that processes the log queue."""
        log_file = f"./logs/status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        while self.is_running:
            try:
                status = self.log_queue.get(timeout=1)
                with open(log_file, 'a') as f:
                    f.write(f"\n--- Status Update {status['timestamp']} ---\n")
                    f.write(f"Frames Processed: {status['frames_processed']}\n")
                    f.write(f"Detections: {status['detections']}\n")
                    f.write(f"Errors: {status['errors']}\n")
                    if status['last_error']:
                        f.write(f"Last Error: {status['last_error']}\n")
                    
                    # Write performance stats by group
                    if 'performance' in status:
                        for group_name, group_stats in status['performance'].items():
                            f.write(f"\n{group_name}:\n")
                            for op, stats in group_stats.items():
                                f.write(f"{op:15} min: {stats['min']:6.2f}  max: {stats['max']:6.2f}  avg: {stats['avg']:6.2f}\n")
                    
                    f.write("----------------------------------------\n")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error writing log: {str(e)}")
                
    def log_error(self, error_msg: str):
        """Immediately log an error message."""
        self.error_count += 1
        with open(f"./logs/errors_{datetime.now().strftime('%Y%m%d')}.log", 'a') as f:
            f.write(f"{datetime.now()}: {error_msg}\n") 