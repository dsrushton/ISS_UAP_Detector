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
import psutil
import torch

class StatusLogger:
    """Handles threaded logging of system status."""
    
    def __init__(self, first_log_interval: int = 60, subsequent_log_interval: int = 300):  # First log at 1 minute, then every 5 minutes
        self.log_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        self.first_log_interval = first_log_interval  # 1 minute for first log
        self.subsequent_log_interval = subsequent_log_interval  # 5 minutes for subsequent logs
        self.first_log_done = False  # Track if first log has been done
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
            'brightness_check': deque(maxlen=100),
            # Debug view granular timing
            'debug_frame_copy': deque(maxlen=100),
            'debug_space_box': deque(maxlen=100),
            'debug_contours': deque(maxlen=100),
            'debug_anomalies': deque(maxlen=100),
            # Display granular timing
            'display_prep': deque(maxlen=100),
            'display_draw': deque(maxlen=100),
            'display_show': deque(maxlen=100)
        }
        
        # Memory tracking
        self.memory_usage: Dict[str, deque] = {
            'cpu_percent': deque(maxlen=100),
            'ram_used': deque(maxlen=100),
            'ram_percent': deque(maxlen=100),
            'gpu_allocated': deque(maxlen=100),
            'gpu_cached': deque(maxlen=100)
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
            
    def log_memory_usage(self):
        """Log current memory usage."""
        # CPU/RAM usage
        process = psutil.Process(os.getpid())
        self.memory_usage['cpu_percent'].append(process.cpu_percent())
        self.memory_usage['ram_used'].append(process.memory_info().rss / 1024 / 1024)  # MB
        self.memory_usage['ram_percent'].append(process.memory_percent())
        
        # GPU memory if available
        if torch.cuda.is_available():
            self.memory_usage['gpu_allocated'].append(torch.cuda.memory_allocated() / 1024 / 1024)  # MB
            self.memory_usage['gpu_cached'].append(torch.cuda.memory_reserved() / 1024 / 1024)  # MB
            
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
            
        # Log memory usage every iteration
        self.log_memory_usage()
            
        # Check if it's time to write a status update
        current_time = time.time()
        # Use first log interval if first log hasn't been done yet, otherwise use subsequent interval
        log_interval = self.first_log_interval if not self.first_log_done else self.subsequent_log_interval
        
        if current_time - self.last_log_time >= log_interval:
            # Calculate performance stats
            perf_stats = {}
            
            # Group operations for cleaner output
            operation_groups = {
                'Main Pipeline': ['frame_read', 'detection', 'display_update', 'debug_view', 
                                'video_buffer', 'video_recording', 'total_iteration'],
                'RCNN Operations': ['rcnn_prep', 'rcnn_inference', 'rcnn_postprocess'],
                'Detection Analysis': ['mask_creation', 'contour_finding', 'contour_analysis', 
                                     'box_filtering', 'brightness_check'],
                'Display Operations': ['display_prep', 'display_draw', 'display_show']
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
            
            # Calculate memory stats
            memory_stats = {}
            for metric, values in self.memory_usage.items():
                if values:
                    min_val, max_val, avg_val = self._calculate_stats(values)
                    memory_stats[metric] = {
                        'min': min_val,
                        'max': max_val,
                        'avg': avg_val
                    }
            
            self.log_queue.put({
                'timestamp': datetime.now(),
                'frames_processed': self.frame_count,
                'detections': self.detection_count,
                'errors': self.error_count,
                'last_error': error_msg,
                'performance': perf_stats,
                'memory': memory_stats
            })
            self.last_log_time = current_time
            self.first_log_done = True  # Mark that first log has been done
            
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
                        f.write("\nPerformance Metrics (ms):\n")
                        for group_name, group_stats in status['performance'].items():
                            f.write(f"\n{group_name}:\n")
                            for op, stats in group_stats.items():
                                f.write(f"{op:15} min: {stats['min']:6.2f}  max: {stats['max']:6.2f}  avg: {stats['avg']:6.2f}\n")
                    
                    # Write memory stats
                    if 'memory' in status:
                        f.write("\nMemory Usage:\n")
                        for metric, stats in status['memory'].items():
                            if metric.startswith('gpu'):
                                f.write(f"{metric:15} min: {stats['min']:6.1f}MB  max: {stats['max']:6.1f}MB  avg: {stats['avg']:6.1f}MB\n")
                            elif metric.endswith('percent'):
                                f.write(f"{metric:15} min: {stats['min']:6.1f}%   max: {stats['max']:6.1f}%   avg: {stats['avg']:6.1f}%\n")
                            else:
                                f.write(f"{metric:15} min: {stats['min']:6.1f}MB  max: {stats['max']:6.1f}MB  avg: {stats['avg']:6.1f}MB\n")
                    
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