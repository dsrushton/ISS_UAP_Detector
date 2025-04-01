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
from typing import Optional

# Constants
DEFAULT_RTMP_URL = "rtmp://a.rtmp.youtube.com/live2"

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
        
        print(f"StreamManager initialized with frame size: {frame_width}x{frame_height}, target FPS: {self.target_fps}")
    
    def set_software_encoding(self, use_software: bool) -> None:
        """Set whether to use software encoding instead of hardware encoding."""
        self.use_software_encoding = use_software
        print(f"Encoding mode: {'Software (CPU)' if use_software else 'Hardware (NVIDIA)'}")
    
    def set_logger(self, logger):
        """Set the logger reference for tracking stream frames."""
        self.logger = logger
    
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
                    # Add padding to maintain aspect ratio - only vertical padding for wider frames
                    '-vf', 'scale=-1:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black',
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
                    # Add padding to maintain aspect ratio - only vertical padding for wider frames
                    '-vf', 'scale=-1:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black',
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
                            frame = self.frames_queue.get(timeout=0.05)
                            
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
            
            print("Streaming started successfully")
            return True
            
        except Exception as e:
            print(f"Error starting stream: {str(e)}")
            self.is_streaming = False
            return False
    
    def stop_streaming(self) -> bool:
        """Stop the active stream."""
        if not self.is_streaming:
            print("No active stream to stop")
            return True
            
        print("Stopping stream...")
        self.is_streaming = False
        
        # Wait for the streaming thread to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5.0)
            
        # Close the FFmpeg process
        if self.ffmpeg_process:
            try:
                # Close stdin to signal end of input
                if self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.close()
                
                # Wait for the process to terminate
                self.ffmpeg_process.wait(timeout=5.0)
                
                # If it's still running, terminate it
                if self.ffmpeg_process.poll() is None:
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process.wait(timeout=2.0)
                    
                    # If it's still running, kill it
                    if self.ffmpeg_process.poll() is None:
                        self.ffmpeg_process.kill()
                        self.ffmpeg_process.wait(timeout=1.0)
                
            except Exception as e:
                print(f"Error stopping FFmpeg process: {str(e)}")
                
            self.ffmpeg_process = None
            
        # Calculate stream duration
        if self.stream_start_time > 0:
            duration = time.time() - self.stream_start_time
            print(f"Stream stopped after {duration:.1f} seconds")
            
        print("Stream stopped successfully")
        return True
            
    def stream_frame(self, frame: np.ndarray) -> bool:
        """Add a frame to the streaming queue with optimized handling."""
        if not self.is_streaming:
            return False
        
        # Initialize black frame if needed (for emergency use)
        if self.black_frame is None:
            self.black_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            
        # No rate limiting - send every frame to maximize streaming rate
        current_time = time.time()
        self.last_frame_time = current_time
            
        # Pre-resized frame optimization - check dimensions only once then use direct path
        needs_resize = frame.shape[0] != self.frame_height or frame.shape[1] != self.frame_width
        
        try:
            # Check if queue is getting full, if so drop frames to prevent lag
            if self.frames_queue.qsize() > 240:  # 4 seconds buffer, allow some accumulation
                # Dropping frames when queue gets too full
                return True  # Successfully handled (by dropping)
                
            # Fast path for correctly sized frames (most common case)
            if not needs_resize:
                self.frames_queue.put(frame.copy(), timeout=0.1)  # Use copy to avoid race conditions
                self.frames_sent += 1
                
                # Log the frame for framerate tracking
                if self.logger:
                    self.logger.log_stream_frame()
                
                return True
                
            # Slow path for frames that need resizing
            # Only log a warning every 300 frames to avoid spam
            if self.frames_sent % 300 == 0:
                print(f"Warning: Frame dimensions ({frame.shape[1]}x{frame.shape[0]}) don't match expected dimensions ({self.frame_width}x{self.frame_height})")
            
            # Resize the frame to match expected dimensions
            resized_frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            self.frames_queue.put(resized_frame, timeout=0.1)
            self.frames_sent += 1
            
            # Log the frame for framerate tracking
            if self.logger:
                self.logger.log_stream_frame()
            
            return True
            
        except queue.Full:
            # Queue is full, which means streaming is falling behind
            # This is normal during high CPU load
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