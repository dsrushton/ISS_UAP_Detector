"""
Space Object Detection Streaming Module
Handles YouTube streaming functionality.
"""

import os
import subprocess
import cv2
import numpy as np
import time
import threading
import queue
import signal
from typing import Optional

# Import YouTube API Manager for broadcast rotation
try:
    from SOD_YouTubeAPI import YouTubeAPIManager, GOOGLE_API_AVAILABLE
except ImportError:
    GOOGLE_API_AVAILABLE = False

# Constants
DEFAULT_RTMP_URL = "rtmp://a.rtmp.youtube.com/live2"

# Try to import cycling settings from youtube_config
try:
    from youtube_config import (
        ENABLE_STREAM_CYCLING,
        STREAM_CYCLE_SECONDS,
        RESTART_DELAY_SECONDS
    )
    CYCLING_CONFIG_LOADED = True
except ImportError:
    # Default values if config doesn't exist
    ENABLE_STREAM_CYCLING = True
    STREAM_CYCLE_SECONDS = 11 * 3600 + 55 * 60  # 11h 55m in seconds
    RESTART_DELAY_SECONDS = 20  # Seconds to wait after stopping before restarting
    CYCLING_CONFIG_LOADED = False

class StreamManager:
    """Manages streaming video to YouTube."""
    
    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        """Initialize the stream manager with frame dimensions."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.stream_key = ""
        self.stream_url = ""
        self.is_streaming = False
        self.ffmpeg_process = None
        self.stream_thread = None
        self.stream_start_time = 0
        self.frames_sent = 0
        self.adapt_to_frame_size = False  # Default to not resizing frames
        self.use_software_encoding = False  # Default to hardware encoding
        self.target_fps = 59.94  # Target fps for streaming (slightly under 60 to avoid overruns)
        self.logger = None  # Reference to the logger
        
        # Create a queue for frames - increase buffer size to handle spikes
        self.frames_queue = queue.Queue(maxsize=300)  # Buffer ~5 seconds at 60fps
        
        # Add a black frame for emergencies when queue is empty
        self.black_frame = None
        
        # Last frame timestamp for rate limiting
        self.last_frame_time = 0
        self.frame_interval = 1.0 / self.target_fps
        
        # Auto-restart streaming variables
        self.auto_restart = ENABLE_STREAM_CYCLING
        self.stream_cycle_timer = None
        self.cycle_count = 0
        
        # Initialize YouTube API Manager for broadcast rotation if available
        self.youtube_api = None
        if GOOGLE_API_AVAILABLE:
            try:
                self.youtube_api = YouTubeAPIManager()
                if self.youtube_api.authorized:
                    # If API is available and authorized, try to load or create a stream
                    if self.youtube_api.load_or_create_stream():
                        # Try to get the stream key
                        api_stream_key = self.youtube_api.get_stream_key()
                        if api_stream_key:
                            self.stream_key = api_stream_key
                            print(f"Using stream key from YouTube API: {self.stream_key[:4]}...")
                    
                    # Create initial broadcast
                    self.youtube_api.create_broadcast()
            except Exception as e:
                print(f"Error initializing YouTube API: {str(e)}")
                self.youtube_api = None
        
        print(f"StreamManager initialized with frame size: {frame_width}x{frame_height}, target FPS: {self.target_fps}")
        if self.auto_restart:
            hours = STREAM_CYCLE_SECONDS // 3600
            minutes = (STREAM_CYCLE_SECONDS % 3600) // 60
            print(f"Auto-restart streaming: Enabled (will cycle every {hours}h {minutes}m)")
        else:
            print("Auto-restart streaming: Disabled")
            
        if self.youtube_api and self.youtube_api.authorized:
            print("YouTube API integration: Active (broadcasts will be properly rotated)")
        else:
            print("YouTube API integration: Not available (basic stream cycling only)")
    
    def set_software_encoding(self, use_software: bool) -> None:
        """Set whether to use software encoding instead of hardware encoding."""
        self.use_software_encoding = use_software
        print(f"Encoding mode: {'Software (CPU)' if use_software else 'Hardware (NVIDIA)'}")
    
    def set_logger(self, logger):
        """Set the logger reference for tracking stream frames."""
        self.logger = logger
    
    def _start_cycle_timer(self) -> None:
        """Start a timer to cycle the stream after the specified duration."""
        if not self.auto_restart:
            return
            
        if self.stream_cycle_timer:
            # Cancel any existing timer
            self.stream_cycle_timer.cancel()
            
        # Create a new timer for STREAM_CYCLE_SECONDS
        self.stream_cycle_timer = threading.Timer(STREAM_CYCLE_SECONDS, self._cycle_stream)
        self.stream_cycle_timer.daemon = True
        self.stream_cycle_timer.start()
        
        # Calculate the cycle end time
        cycle_end_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                       time.localtime(time.time() + STREAM_CYCLE_SECONDS))
        
        print(f"Stream cycle timer started. Stream will restart at {cycle_end_time}")
    
    def _cycle_stream(self) -> None:
        """Stop and restart the stream to work around YouTube's duration limits."""
        if not self.is_streaming:
            return
            
        self.cycle_count += 1
        print(f"\n*** CYCLING STREAM #{self.cycle_count} (duration limit reached) ***")
        
        # Save the current frames queue and stream key for restart
        current_frames_queue = self.frames_queue
        current_stream_key = self.stream_key
        
        # If using YouTube API, create a new broadcast before stopping the stream
        if self.youtube_api and self.youtube_api.authorized:
            print("Creating new YouTube broadcast for rotation...")
            success, broadcast_id = self.youtube_api.rotate_broadcast()
            if success:
                print(f"New broadcast created and bound to stream (ID: {broadcast_id})")
            else:
                print("Failed to create new broadcast, but will continue with stream restart")
        
        # Stop the stream gracefully (sending SIGINT to FFmpeg for clean shutdown)
        self._graceful_stop_streaming()
        
        # Wait briefly to ensure YouTube registers the end of the stream
        print(f"Waiting {RESTART_DELAY_SECONDS} seconds before restarting stream...")
        time.sleep(RESTART_DELAY_SECONDS)
        
        # Restart the stream with the same settings
        print("Restarting stream with the same settings...")
        self.stream_key = current_stream_key
        self.start_streaming(current_frames_queue)
    
    def _graceful_stop_streaming(self) -> bool:
        """Stop the active stream by sending SIGINT to FFmpeg for clean shutdown."""
        if not self.is_streaming:
            print("No active stream to stop")
            return True
            
        print("Gracefully stopping stream with SIGINT...")
        self.is_streaming = False
        
        # Wait for the streaming thread to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5.0)
            
        # Close the FFmpeg process with SIGINT for graceful shutdown
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            try:
                # On Windows, we need to use CTRL_C_EVENT
                if os.name == 'nt':
                    os.kill(self.ffmpeg_process.pid, signal.CTRL_C_EVENT)
                else:
                    # On Unix systems, use SIGINT
                    os.kill(self.ffmpeg_process.pid, signal.SIGINT)
                
                # Give FFmpeg time to shut down gracefully
                print("Waiting for FFmpeg to shut down gracefully...")
                self.ffmpeg_process.wait(timeout=10.0)
                
                # If it's still running, terminate it
                if self.ffmpeg_process.poll() is None:
                    print("FFmpeg did not exit after SIGINT, terminating...")
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process.wait(timeout=2.0)
                    
                    # If it's still running, kill it
                    if self.ffmpeg_process.poll() is None:
                        print("FFmpeg did not exit after termination, killing process...")
                        self.ffmpeg_process.kill()
                        self.ffmpeg_process.wait(timeout=1.0)
                
            except Exception as e:
                print(f"Error stopping FFmpeg process: {str(e)}")
                
            self.ffmpeg_process = None
            
        # Calculate stream duration
        if self.stream_start_time > 0:
            duration = time.time() - self.stream_start_time
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Stream stopped after {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
        print("Stream stopped successfully")
        return True
    
    def start_streaming(self, frames_queue: queue.Queue) -> bool:
        """Start streaming to YouTube using the configured stream key."""
        if self.is_streaming:
            print("Streaming is already active")
            return False
            
        if not self.stream_key:
            print("No stream key set - please set a valid YouTube stream key")
            return False
            
        # Set up the RTMP URL with the stream key
        self.stream_url = f"{DEFAULT_RTMP_URL}/{self.stream_key}"
        print(f"Starting stream to: {DEFAULT_RTMP_URL}/{self.stream_key[:4]}...")
        
        # Create a pipe for sending frames to FFmpeg
        try:
            # Choose between hardware and software encoding
            if not self.use_software_encoding:
                print("Attempting to use NVIDIA hardware encoding...")
                command = [
                    "ffmpeg",
                    '-f', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', f'{self.frame_width}x{self.frame_height}',
                    '-r', '60',  # Increase to 60fps to match YouTube's recommendation
                    '-i', '-',  # Read from stdin
                    # Add silent audio stream - required by YouTube
                    '-f', 'lavfi',
                    '-i', 'anullsrc=r=44100:cl=mono',
                    '-c:v', 'h264_nvenc',  # Use NVIDIA hardware encoding
                    '-preset', 'p1',       # Low latency preset
                    '-pix_fmt', 'yuv420p', # Required by YouTube
                    '-g', '120',           # Keyframe interval (2s at 60fps)
                    '-b:v', '6800k',       # Match YouTube's recommended bitrate
                    '-maxrate', '9000k',   # Higher maximum bitrate
                    '-bufsize', '13600k',  # Larger buffer size (2x bitrate)
                    '-qmin', '0',          # Minimum quantization level (0-51, lower is better quality)
                    '-qmax', '28',         # Maximum quantization level (lower than default for better quality)
                    # Audio encoding
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-ar', '44100',
                    '-f', 'flv',
                    self.stream_url
                ]
            else:
                print("Using software encoding (libx264)...")
                command = [
                    "ffmpeg",
                    '-f', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', f'{self.frame_width}x{self.frame_height}',
                    '-r', '60',  # Increase to 60fps to match YouTube's recommendation
                    '-i', '-',  # Read from stdin
                    # Add silent audio stream - required by YouTube
                    '-f', 'lavfi',
                    '-i', 'anullsrc=r=44100:cl=mono',
                    '-c:v', 'libx264',     # Software encoding
                    '-preset', 'fast',     # Better quality preset (slower than veryfast)
                    '-pix_fmt', 'yuv420p', # Required by YouTube
                    '-g', '120',           # Keyframe interval (2s at 60fps)
                    '-b:v', '6800k',       # Match YouTube's recommended bitrate
                    '-maxrate', '9000k',   # Higher maximum bitrate
                    '-bufsize', '13600k',  # Larger buffer size (2x bitrate)
                    '-crf', '18',          # Constant Rate Factor (18-23 is visually lossless, lower is better)
                    # Audio encoding
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-ar', '44100',
                    '-f', 'flv',
                    self.stream_url
                ]
            
            print(f"Starting FFmpeg with command: {' '.join(command)}")
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8  # Use a large buffer
            )
            
            # Start a thread to process frames from the queue
            self.is_streaming = True
            self.stream_start_time = time.time()
            self.frames_queue = frames_queue
            
            def stream_frames():
                # Initialize counters and timing variables
                frames_sent = 0
                last_report_time = time.time()
                black_frame_count = 0
                
                try:
                    while self.is_streaming:
                        # No frame timing enforcement - send frames as fast as we can get them
                        # This lets FFmpeg and YouTube handle the rate rather than our own timing
                        
                        # Try to get a frame, wait a bit longer if queue is empty to avoid black frames
                        try:
                            # Use a longer timeout (50ms instead of 10ms) to give more time for frames to arrive
                            queue_item = self.frames_queue.get(timeout=0.05)
                            
                            # Process the queue item based on its type
                            if isinstance(queue_item, tuple) and len(queue_item) == 3:
                                # New format: tuple of (type, frame/pool, index)
                                item_type, item_data, item_idx = queue_item
                                
                                if item_type == "buffer_idx" and item_data is not None:
                                    # Item is a buffer index in a frame pool
                                    frame_pool = item_data
                                    buffer_idx = item_idx
                                    
                                    # Get the frame from the pool
                                    frame = frame_pool.get_buffer(buffer_idx)
                                else:
                                    # Traditional frame
                                    frame = item_data
                            else:
                                # Legacy format: direct frame
                                frame = queue_item
                            
                            # Reset black frame counter when we get a real frame
                            if black_frame_count > 0:
                                print(f"Recovered after {black_frame_count} black frames")
                                black_frame_count = 0
                                
                            # Store this frame for potential reuse
                            self.last_good_frame = frame.copy()
                            
                        except queue.Empty:
                            # Always reuse the last good frame instead of using black frames
                            if hasattr(self, 'last_good_frame') and self.last_good_frame is not None:
                                frame = self.last_good_frame
                                if black_frame_count == 0:
                                    # First time we're reusing a frame
                                    black_frame_count += 1
                                    print("Reusing previous frame due to empty queue")
                                elif black_frame_count % 60 == 0:
                                    # Log every 60 frames to avoid spam
                                    print(f"Still reusing previous frame ({black_frame_count} times)")
                                else:
                                    black_frame_count += 1
                            else:
                                # Only use black frame if we have no previous frame at all
                                # This should only happen on the very first frame
                                if self.black_frame is None:
                                    self.black_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                                frame = self.black_frame
                                print("Using temporary black frame until first real frame arrives")
                                # Don't continue - we want to send this frame so FFmpeg starts properly
                                black_frame_count += 1
                            
                        # Write frame to FFmpeg process
                        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                            try:
                                self.ffmpeg_process.stdin.write(frame.tobytes())
                                frames_sent += 1
                                
                                # Periodic progress report
                                current_time = time.time()
                                if current_time - last_report_time >= 60.0:
                                    elapsed = current_time - self.stream_start_time
                                    fps = frames_sent / elapsed if elapsed > 0 else 0
                                    print(f"Streaming: {frames_sent} frames sent in {elapsed:.1f}s ({fps:.1f} fps)")
                                    last_report_time = current_time
                                    
                                    # Log the frame rate if logger is available
                                    if self.logger:
                                        self.logger.log_operation_time('streaming_fps', fps)
                            except BrokenPipeError as e:
                                print(f"Error: Broken pipe when writing to FFmpeg - {str(e)}")
                                self.is_streaming = False
                                break
                            except Exception as e:
                                print(f"Error writing frame to FFmpeg: {str(e)}")
                                continue
                        else:
                            # FFmpeg process terminated
                            if self.ffmpeg_process:
                                returncode = self.ffmpeg_process.returncode
                                stderr_output = self.ffmpeg_process.stderr.read().decode('utf-8', errors='replace')
                                print(f"FFmpeg process terminated with return code: {returncode}")
                                print("FFmpeg error output:")
                                for line in stderr_output.split('\n')[:20]:
                                    if line.strip():
                                        print(f"  {line.strip()}")
                            self.is_streaming = False
                            break
                except Exception as e:
                    print(f"Streaming thread exception: {str(e)}")
                finally:
                    print("Frame processing thread stopped")
                    
                    # Clean up FFmpeg process
                    if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                        try:
                            self.ffmpeg_process.stdin.close()
                            self.ffmpeg_process.terminate()
                            self.ffmpeg_process.wait(timeout=5)
                        except Exception as e:
                            print(f"Error closing FFmpeg process: {str(e)}")
                            
                    # Get any remaining FFmpeg output
                    if self.ffmpeg_process:
                        try:
                            stderr_output = self.ffmpeg_process.stderr.read().decode('utf-8', errors='replace')
                            if stderr_output:
                                print("FFmpeg final output:")
                                for line in stderr_output.split('\n')[:20]:
                                    if line.strip():
                                        print(f"  {line.strip()}")
                        except Exception as e:
                            print(f"Error reading FFmpeg output: {str(e)}")
            
            # Start the streaming thread
            self.stream_thread = threading.Thread(target=stream_frames)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
            # Always start the cycle timer if auto-restart is enabled
            if self.auto_restart:
                self._start_cycle_timer()
            
            print("Streaming started successfully")
            return True
            
        except Exception as e:
            print(f"Error starting stream: {str(e)}")
            self.is_streaming = False
            return False
    
    def stop_streaming(self) -> bool:
        """Stop the active stream."""
        # Cancel cycle timer if it exists
        if self.stream_cycle_timer:
            self.stream_cycle_timer.cancel()
            self.stream_cycle_timer = None
            
        return self._graceful_stop_streaming()
            
    def stream_frame(self, frame, make_copy: bool = False, frame_pool=None, buffer_idx=None) -> bool:
        """
        Add a frame to the streaming queue with optimized handling.
        
        Args:
            frame: The frame to stream, or None if using buffer_idx
            make_copy: Whether to make a copy of the frame (ignored if using buffer_idx)
            frame_pool: Reference to a FrameBufferPool if using buffer indices
            buffer_idx: Index of buffer in the frame_pool to use instead of direct frame
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_streaming:
            return False
        
        # Initialize black frame if needed (for emergency use)
        if self.black_frame is None:
            self.black_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            
        # No rate limiting - send every frame to maximize streaming rate
        current_time = time.time()
        self.last_frame_time = current_time

        try:
            # Get current queue size
            current_queue_size = self.frames_queue.qsize()
            
            # Calculate estimated time to process the current queue
            # Max latency threshold is 0.5 seconds
            MAX_LATENCY_THRESHOLD = 0.5  # 500ms max end-to-end latency
            
            # Calculate estimated queue processing time
            queue_time = current_queue_size * self.frame_interval
            
            if queue_time > MAX_LATENCY_THRESHOLD:
                # Queue would exceed latency threshold if we add this frame
                # Calculate how many frames to drop to get back under threshold
                frames_to_drop = int((queue_time - (MAX_LATENCY_THRESHOLD * 0.8)) / self.frame_interval)
                frames_to_drop = max(1, min(frames_to_drop, current_queue_size))  # At least 1, at most queue size
                
                # Drop oldest frames from the queue
                for _ in range(frames_to_drop):
                    try:
                        self.frames_queue.get_nowait()  # Remove oldest frame
                        self.frames_queue.task_done()
                    except queue.Empty:
                        break  # Queue emptied during dropping
                
                if frames_to_drop > 1:
                    print(f"Dropped {frames_to_drop} oldest frames to maintain latency (queue: {current_queue_size} frames, {queue_time:.2f}s)")

            # Determine what to queue based on inputs
            if frame_pool is not None and buffer_idx is not None:
                # Using frame buffer pool - queue the buffer index
                item_to_queue = ("buffer_idx", frame_pool, buffer_idx)
                self.frames_queue.put(item_to_queue, timeout=0.1)
            else:
                # Traditional frame queueing
                frame_to_queue = frame.copy() if make_copy else frame
                item_to_queue = ("frame", frame_to_queue, None)
                self.frames_queue.put(item_to_queue, timeout=0.1)
            
            self.frames_sent += 1

            # Log the frame for framerate tracking
            if self.logger:
                self.logger.log_stream_frame()

            return True

        except queue.Full:
            # Queue is full even after dropping frames
            # This should be rare now that we're proactively managing queue size
            return True  # Successfully handled (by dropping)
        except Exception as e:
            print(f"Error adding frame to stream queue: {str(e)}")
            return False
            
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_streaming()
        
    def set_adapt_to_frame_size(self, adapt: bool) -> None:
        """Set whether to adapt to incoming frame size or maintain fixed dimensions."""
        self.adapt_to_frame_size = adapt
        print(f"Adapt to frame size: {'Enabled' if adapt else 'Disabled'}")
        print(f"Current frame size: {self.frame_width}x{self.frame_height}")