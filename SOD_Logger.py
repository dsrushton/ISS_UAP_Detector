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
        self.start_time = time.time()  # Track when the logger started
        self.frame_count = 0
        self.detection_count = 0
        self.error_count = 0
        self.verbose_console = False  # Flag to control console output verbosity
        self.expected_fps = 60.0  # Default expected FPS
        self.current_frame_times = {}  # Store times for the current frame
        self.frame_start_time = 0  # Track when the current frame started
        
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
        
        # Create log file
        self.log_file = os.path.join("./logs", f"status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(self.log_file, 'w') as f:
            f.write(f"Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
    
    def start_frame(self):
        """Start timing a new frame."""
        self.current_frame_times = {}
        self.frame_start_time = time.time()
    
    def end_frame(self):
        """End timing for the current frame."""
        # Calculate total frame time
        if self.frame_start_time > 0:
            total_time = time.time() - self.frame_start_time
            self.log_operation_time('total_iteration', total_time)
        
        # Increment frame count
        self.frame_count += 1
        
    def start(self):
        """Start the logger thread."""
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._log_worker)
        self.thread.daemon = True
        self.thread.start()
        if self.verbose_console:
            print(f"Logger thread started")
        
    def stop(self):
        """Stop the logger thread."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)  # Add timeout to avoid hanging
            if self.thread.is_alive():
                print("Logger thread did not terminate within timeout")
            self.thread = None
            
    def log_operation_time(self, operation: str, duration: float):
        """Log the duration of an operation."""
        if operation in self.operation_times:
            self.operation_times[operation].append(duration)
        
        # Also store in current frame times
        self.current_frame_times[operation] = duration
            
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
        
        # Debug output for first log
        if not self.first_log_done and self.verbose_console:
            time_since_start = current_time - self.last_log_time
            if time_since_start > 30 and time_since_start % 10 < 1:  # Print every 10 seconds after 30 seconds
                print(f"Logger waiting for first log: {time_since_start:.1f}s / {self.first_log_interval}s")
        
        if current_time - self.last_log_time >= log_interval:
            if self.verbose_console:
                print(f"\nCreating log entry after {current_time - self.last_log_time:.1f}s")
            
            # Calculate performance stats
            perf_stats = {}
            
            # Calculate processing rate
            elapsed_time = current_time - self.start_time
            actual_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            fps_ratio = actual_fps / self.expected_fps if self.expected_fps > 0 else 0
            
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
                'memory': memory_stats,
                'processing_rate': {
                    'actual_fps': actual_fps,
                    'expected_fps': self.expected_fps,
                    'fps_ratio': fps_ratio,
                    'elapsed_time': elapsed_time
                }
            })
            self.last_log_time = current_time
            self.first_log_done = True  # Mark that first log has been done
            
            # Also write a simple message to the log file
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            status = "SUCCESS" if success else "FAILURE"
            detection = "DETECTION" if had_detection else "NO_DETECTION"
            
            with open(self.log_file, 'a') as f:
                log_line = f"[{status}] {timestamp}: {detection}"
                if error_msg:
                    log_line += f" - {error_msg}"
                f.write(log_line + "\n")
                
            # Print to console only for important events
            if had_detection or not success or error_msg:
                console_line = f"[{status}] {detection}"
                if error_msg:
                    console_line += f" - {error_msg}"
                print(console_line)
            
    def _log_worker(self):
        """Worker thread that processes the log queue."""
        log_file = f"./logs/status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Ensure logs directory exists
        try:
            os.makedirs("./logs", exist_ok=True)
        except Exception as e:
            print(f"ERROR: Failed to create logs directory: {str(e)}")
            return
            
        error_count = 0
        max_errors = 5
        last_error_time = time.time()
        
        while self.is_running:
            try:
                # Try to get an item from the queue with timeout
                status = self.log_queue.get(timeout=1)
                
                with open(log_file, 'a') as f:
                    f.write(f"\n--- Status Update {status['timestamp']} ---\n")
                    f.write(f"Frames Processed: {status['frames_processed']}\n")
                    f.write(f"Detections: {status['detections']}\n")
                    f.write(f"Errors: {status['errors']}\n")
                    if status['last_error']:
                        f.write(f"Last Error: {status['last_error']}\n")
                    
                    # Write processing rate information
                    if 'processing_rate' in status:
                        rate = status['processing_rate']
                        f.write(f"\nProcessing Rate:\n")
                        f.write(f"Actual FPS: {rate['actual_fps']:.2f}\n")
                        f.write(f"Expected FPS: {rate['expected_fps']:.2f}\n")
                        f.write(f"FPS Ratio: {rate['fps_ratio']:.2f} ({rate['fps_ratio']*100:.1f}% of expected)\n")
                        f.write(f"Elapsed Time: {rate['elapsed_time']:.1f} seconds\n")
                        f.write(f"Total Frames: {status['frames_processed']}\n")
                        f.write(f"Theoretical Frames at {rate['expected_fps']:.1f} FPS: {rate['elapsed_time'] * rate['expected_fps']:.0f}\n")
                        f.write(f"Frame Deficit: {(rate['elapsed_time'] * rate['expected_fps']) - status['frames_processed']:.0f}\n")
                        
                        # Add warning if processing is too slow
                        if rate['fps_ratio'] < 0.95:  # Less than 95% of expected rate
                            f.write(f"WARNING: Processing is slower than the expected frame rate!\n")
                            if rate['fps_ratio'] < 0.75:  # Less than 75% of expected rate
                                f.write(f"CRITICAL: Processing is significantly slower than required (below 75% of target)!\n")
                    
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
                            else:
                                f.write(f"{metric:15} min: {stats['min']:6.1f}%   max: {stats['max']:6.1f}%   avg: {stats['avg']:6.1f}%\n")
                    
                    f.write("-" * 40 + "\n")
                
                # Print to console as well (but only once per log)
                if self.verbose_console:
                    print(f"\n--- Status Update {status['timestamp']} ---")
                    print(f"Frames Processed: {status['frames_processed']}")
                    print(f"Detections: {status['detections']}")
                    print(f"Errors: {status['errors']}")
                    
                    # Print processing rate information
                    if 'processing_rate' in status:
                        rate = status['processing_rate']
                        print(f"\nProcessing Rate:")
                        print(f"Actual FPS: {rate['actual_fps']:.2f}")
                        print(f"Expected FPS: {rate['expected_fps']:.2f}")
                        print(f"FPS Ratio: {rate['fps_ratio']:.2f} ({rate['fps_ratio']*100:.1f}% of expected)")
                        print(f"Total Frames: {status['frames_processed']}")
                        print(f"Theoretical Frames at {rate['expected_fps']:.1f} FPS: {rate['elapsed_time'] * rate['expected_fps']:.0f}")
                        print(f"Frame Deficit: {(rate['elapsed_time'] * rate['expected_fps']) - status['frames_processed']:.0f}")
                        
                        # Add warning if processing is too slow
                        if rate['fps_ratio'] < 0.95:  # Less than 95% of expected rate
                            print(f"WARNING: Processing is slower than the expected frame rate!")
                            if rate['fps_ratio'] < 0.75:  # Less than 75% of expected rate
                                print(f"CRITICAL: Processing is significantly slower than required (below 75% of target)!")
                    
                    # Print performance metrics summary
                    if 'performance' in status:
                        print("\nPerformance Metrics (ms):")
                        for group_name, group_stats in status['performance'].items():
                            print(f"\n{group_name}:")
                            for op, stats in group_stats.items():
                                print(f"{op:15} min: {stats['min']:6.2f}  max: {stats['max']:6.2f}  avg: {stats['avg']:6.2f}")
                    
                    # Print memory usage summary
                    if 'memory' in status:
                        print("\nMemory Usage:")
                        for metric, stats in status['memory'].items():
                            if metric.startswith('gpu'):
                                print(f"{metric:15} min: {stats['min']:6.1f}MB  max: {stats['max']:6.1f}MB  avg: {stats['avg']:6.1f}MB")
                            else:
                                print(f"{metric:15} min: {stats['min']:6.1f}%   max: {stats['max']:6.1f}%   avg: {stats['avg']:6.1f}%")
                
                print("Added to log")
                self.log_queue.task_done()
                error_count = 0  # Reset error count on success
                
            except queue.Empty:
                continue
            except Exception as e:
                error_count += 1
                current_time = time.time()
                
                # Print detailed error
                if self.verbose_console:
                    print(f"ERROR in logger: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # Only log frequent errors once per minute to avoid spam
                if current_time - last_error_time > 60:
                    if self.verbose_console:
                        print(f"Logger has encountered {error_count} errors. Will continue trying.")
                    last_error_time = current_time
                
                # If too many errors occur, try to create a new log file
                if error_count >= max_errors:
                    if self.verbose_console:
                        print("Too many logger errors, creating new log file...")
                    log_file = f"./logs/status_{datetime.now().strftime('%Y%m%d_%H%M%S')}_recovery.log"
                    error_count = 0
                    
                # Short delay to avoid tight error loop
                time.sleep(0.5)
        
        if self.verbose_console:
            print("Logger worker thread exiting")
        
    def log_error(self, error_msg: str):
        """Immediately log an error message."""
        self.error_count += 1
        
        # Always print errors to console for visibility
        if self.verbose_console:
            print(f"ERROR: {error_msg}")
        
        try:
            # Ensure logs directory exists
            os.makedirs("./logs", exist_ok=True)
            
            # Write to error log file
            error_log_file = f"./logs/errors_{datetime.now().strftime('%Y%m%d')}.log"
            with open(error_log_file, 'a') as f:
                f.write(f"{datetime.now()}: {error_msg}\n")
                
            # Also write to main log file
            with open(self.log_file, 'a') as f:
                f.write(f"[ERROR] {datetime.now()}: {error_msg}\n")
        except Exception as e:
            if self.verbose_console:
                print(f"Failed to write to error log: {str(e)}")

    def ensure_running(self):
        """Ensure logger thread is running, restart if needed."""
        if not self.is_running or not self.thread or not self.thread.is_alive():
            print("Logger thread not running, restarting...")
            self.start()
            
    def set_expected_fps(self, fps: float):
        """Set the expected frames per second for rate calculations."""
        if fps > 0 and fps < 120:  # Sanity check
            self.expected_fps = fps 

    def get_performance_stats(self):
        """Get performance statistics."""
        stats = {'operations': {}, 'overall': {}}
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        
        # Calculate actual FPS
        actual_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate theoretical frames based on expected FPS
        theoretical_frames = int(elapsed_time * self.expected_fps)
        
        # Calculate frame deficit
        frame_deficit = theoretical_frames - self.frame_count
        
        # Calculate FPS ratio
        fps_ratio = actual_fps / self.expected_fps if self.expected_fps > 0 else 0
        
        # Calculate processing time per frame (from total_iteration)
        avg_processing_time = 0
        if 'total_iteration' in self.operation_times and self.operation_times['total_iteration']:
            avg_processing_time = sum(self.operation_times['total_iteration']) / len(self.operation_times['total_iteration'])
        
        # Calculate theoretical FPS based on processing time
        theoretical_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        # Calculate efficiency (ratio of actual FPS to theoretical FPS)
        efficiency = actual_fps / theoretical_fps if theoretical_fps > 0 else 0
        
        # Add overall stats
        stats['overall'] = {
            'fps': actual_fps,
            'expected_fps': self.expected_fps,
            'fps_ratio': fps_ratio,
            'elapsed_time': elapsed_time,
            'total_frames': self.frame_count,
            'theoretical_frames': theoretical_frames,
            'frame_deficit': frame_deficit,
            'detections': self.detection_count,
            'errors': self.error_count,
            'avg_processing_time': avg_processing_time * 1000,  # Convert to ms
            'theoretical_fps': theoretical_fps,
            'efficiency': efficiency
        }
        
        # Add operation stats
        stats['operations'] = {}
        for op_name, times in self.operation_times.items():
            if times:
                stats['operations'][op_name] = {
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
        
        return stats
        
    def print_performance(self):
        """Print performance statistics to console."""
        stats = self.get_performance_stats()
        overall = stats['overall']
        operations = stats['operations']
        
        # Log full performance report to file
        log_message = "\n===== PERFORMANCE REPORT =====\n"
        log_message += f"Frames Processed: {overall['total_frames']}\n"
        log_message += f"Detections: {overall['detections']}\n"
        log_message += f"Errors: {overall['errors']}\n"
        log_message += f"Frame Deficit: {overall['frame_deficit']}\n"
        log_message += f"Actual FPS: {overall['fps']:.2f}\n"
        log_message += f"Expected FPS: {overall['expected_fps']:.2f}\n"
        log_message += f"FPS Ratio: {overall['fps_ratio']:.2f} ({overall['fps_ratio']*100:.1f}% of expected)\n"
        log_message += f"Avg Processing Time: {overall['avg_processing_time']:.2f} ms\n"
        log_message += f"Theoretical FPS (based on processing time): {overall['theoretical_fps']:.2f}\n"
        log_message += f"Efficiency (actual/theoretical): {overall['efficiency']:.2f} ({overall['efficiency']*100:.1f}%)\n"
        log_message += f"Elapsed Time: {overall['elapsed_time']:.1f} seconds\n"
        log_message += f"Total Frames: {overall['total_frames']}\n"
        log_message += f"Theoretical Frames at {overall['expected_fps']:.1f} FPS: {overall['theoretical_frames']}\n"
        
        # Add warning if processing is too slow
        if overall['fps_ratio'] < 0.95:
            log_message += "\nWARNING: Processing is slower than expected (below 95% of target)\n"
        if overall['fps_ratio'] < 0.75:
            log_message += "\nCRITICAL: Processing is significantly slower than required (below 75% of target)\n"
        
        # Add warning if efficiency is low
        if overall['efficiency'] < 0.8:
            log_message += f"\nWARNING: Low efficiency ({overall['efficiency']*100:.1f}%) - There may be delays between frames\n"
        
        log_message += "\nOperation Times (ms):\n"
        log_message += f"{'Operation':<20} {'Avg':<10} {'Min':<10} {'Max':<10} {'Count':<10}\n"
        log_message += "-" * 60 + "\n"
        
        # Group operations for cleaner output
        operation_groups = {
            'Main Pipeline': ['frame_read', 'detection', 'display_update', 'debug_view', 
                             'video_buffer', 'video_recording', 'total_iteration'],
            'RCNN Operations': ['rcnn_inference', 'rcnn_process', 'rcnn_prep', 'rcnn_postprocess'],
            'Detection Analysis': ['contour_detection', 'anomaly_detection', 'mask_creation', 
                                  'contour_finding', 'contour_analysis', 'box_filtering', 'brightness_check'],
            'Display Operations': ['display_draw', 'combined_view', 'debug_frame_copy', 'debug_space_box', 
                                  'debug_contours', 'debug_anomalies'],
            'Buffer Operations': ['buffer_append', 'recording_write']
        }
        
        # Add grouped operations to log
        for group_name, op_names in operation_groups.items():
            log_message += f"\n{group_name}:\n"
            for op_name in op_names:
                if op_name in operations:
                    op = operations[op_name]
                    log_message += f"{op_name:<20} {op['avg']*1000:<10.2f} {op['min']*1000:<10.2f} {op['max']*1000:<10.2f} {op['count']:<10}\n"
                else:
                    log_message += f"{op_name:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}\n"
        
        # Add memory usage to log
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            log_message += f"\nMemory Usage: {memory_info.rss / (1024 * 1024):.1f} MB\n"
            log_message += f"CPU Usage: {psutil.cpu_percent(interval=0.1):.1f}%\n"
        except ImportError:
            log_message += "\nMemory usage not available (psutil not installed)\n"
            
        # Add the full log message to the queue
        self.log_queue.put(log_message)
        
        # Print only essential information to console
        if not self.verbose_console:
            # Print only a brief summary to console
            print(f"\nPerformance: {overall['fps']:.1f} FPS ({overall['fps_ratio']*100:.1f}% of target), {overall['total_frames']} frames, {overall['detections']} detections")
            print(f"Avg. processing time: {overall['avg_processing_time']:.2f} ms per frame (theoretical {overall['theoretical_fps']:.1f} FPS)")
            print(f"Efficiency: {overall['efficiency']*100:.1f}% (ratio of actual FPS to theoretical FPS)")
            
            # Add warning if processing is too slow
            if overall['fps_ratio'] < 0.75:
                print(f"CRITICAL: Processing speed at {overall['fps_ratio']*100:.1f}% of required")
            elif overall['fps_ratio'] < 0.95:
                print(f"WARNING: Processing speed at {overall['fps_ratio']*100:.1f}% of required")
                
            # Add warning if efficiency is low
            if overall['efficiency'] < 0.8:
                print(f"WARNING: Low efficiency - There may be delays between frames")
        else:
            # Print the full report to console if verbose mode is enabled
            print(log_message)
            
    def get_current_frame_times(self) -> Dict[str, float]:
        """Get the operation times for the current frame."""
        return self.current_frame_times.copy() 