"""
Space Object Detection System - Main Module
Handles video streaming and orchestrates the detection, display, and capture processes.
"""

import cv2
import threading
import time
import os
from typing import Optional, List, Tuple
import numpy as np
import importlib
import logging
import queue
import sys

from SOD_Constants import (
    MAX_CONSECUTIVE_ERRORS, 
    BURST_CAPTURE_FRAMES, 
    CROPPED_WIDTH,
    TEST_IMAGE_PATH,
    RECONNECT_DELAY,
    VIDEO_SAVE_DIR,
    JPG_SAVE_DIR,
    RAW_SUBDIR,
    #VIDEO_FPS,
    BUFFER_SECONDS
)
from SOD_Utils import get_best_stream_url, crop_frame, ensure_save_directory
from SOD_Video import VideoManager
from SOD_Logger import StatusLogger
from SOD_Console import ParameterConsole
from SOD_Stream import StreamManager

class SpaceObjectDetectionSystem:
    """
    Main class that orchestrates the Space Object Detection system.
    Coordinates between detection, display, and capture modules.
    """
    
    def __init__(self):
        """Initialize the detection system and its components."""
        # Set FFmpeg log level to error only before any VideoCapture is created
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;error'
        
        # These will be imported and initialized here to avoid circular imports
        from SOD_Detections import SpaceObjectDetector
        from SOD_Display import DisplayManager
        from SOD_Capture import CaptureManager
        
        self.detector = SpaceObjectDetector()
        self.display = DisplayManager()
        self.capture = CaptureManager()
        self.logger = StatusLogger()
        self.console = None  # Will be initialized later
        self.stream = None  # Will be initialized after we know the frame dimensions
        
        # State tracking
        self.frame_count: int = 0
        self.burst_remaining: int = 0
        self.is_running: bool = False
        
        # Test image state
        self.test_images = []
        self.current_test_image = 0
        self.inject_test_frames = 0
        self.current_frame_is_test = False  # Flag to track if current frame is a test frame
        
        self.video = VideoManager()
        
        # Add combined frame buffer - initialize as None, will be created when needed
        self.combined_frame_buffer = None
        
        # Ensure required directories exist
        os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
        os.makedirs(JPG_SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(JPG_SAVE_DIR, RAW_SUBDIR), exist_ok=True)
        
        # Initialize state
        self.running = False
        self.frame_count = 0
        self.last_save_time = 0
        self.frame_display_start = 180  # Default value for test frame display duration
        
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize logger
            self.logger.start()
            
            # Disable OpenCV's default logger output to console
            opencv_logger = logging.getLogger('opencv')
            opencv_logger.setLevel(logging.ERROR)
            
            # Create custom handler for OpenCV logs
            class OpenCVLogHandler(logging.Handler):
                def __init__(self, logger):
                    super().__init__()
                    self.logger = logger
                    self.warning_counts = {}
                    self.last_log_time = time.time()
                    
                def emit(self, record):
                    # Check if message contains HTTP connection warning or stream timeout
                    if "HTTP connection" in record.msg or "Stream timeout" in record.msg:
                        # Extract warning type
                        warning_type = "HTTP connection warning" if "HTTP connection" in record.msg else "Stream timeout"
                        
                        # Increment count for this warning type
                        if warning_type in self.warning_counts:
                            self.warning_counts[warning_type] += 1
                        else:
                            self.warning_counts[warning_type] = 1
                        
                        # Log summary every 5 minutes
                        current_time = time.time()
                        if current_time - self.last_log_time > 300:  # 5 minutes
                            for wtype, count in self.warning_counts.items():
                                self.logger.log_error(f"Accumulated {count} {wtype}s in the last 5 minutes")
                            # Reset counters and timer
                            self.warning_counts = {}
                            self.last_log_time = current_time
            
            # Add custom handler to OpenCV logger
            handler = OpenCVLogHandler(self.logger)
            opencv_logger.addHandler(handler)
            
            # Disable OpenCV's default logger output to console
            os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
            
            # Initialize the model
            if not self.detector.initialize_model():
                print("Failed to initialize detector model")
                return False
                
            # Set logger for detector and display
            self.detector.set_logger(self.logger)
            self.display.set_logger(self.logger)
                
            # Initialize capture system
            if not self.capture.initialize():
                print("Failed to initialize capture system")
                return False
                
            # Load test images
            if not self.load_test_images():
                print("Warning: Could not load test images")
            
            # Launch parameter console in a separate thread
            if self.console is None:
                self.console = ParameterConsole()
                # Start console in a separate thread
                console_thread = threading.Thread(target=self.console.start)
                console_thread.daemon = True  # Make thread daemon so it exits with main program
                console_thread.start()
                time.sleep(0.5)  # Give console time to initialize
            
            return True
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            return False

    def toggle_streaming(self) -> None:
        """Toggle streaming state and update display."""
        if not self.stream.is_streaming:
            # Prompt for stream key if not already set
            if not self.stream.stream_key:
                # Use hardcoded stream key for testing
                stream_key = "3qsu-m42f-vp02-9w0r-f42a"  # Hardcoded for testing
                self.stream.stream_key = stream_key
                print(f"Using stream key: {stream_key[:4]}...{stream_key[-4:]}")
            
            print("\nAttempting to start YouTube stream...")
            if self.stream.start_streaming(self.stream.frames_queue):
                self.display.set_streaming(True)
                print("\nStream started - check YouTube Studio")
                print("Note: It may take 60-90 seconds for YouTube to show the stream")
            else:
                print("\nFailed to start streaming - check console for details")
                print("Press 'i' to run streaming troubleshooting")
        else:
            print("\nStopping YouTube stream...")
            self.stream.stop_streaming()
            self.display.set_streaming(False)

    def process_frame(self, frame: np.ndarray, avoid_boxes: List[Tuple[int, int, int, int]] = None) -> Optional[bool]:
        """
        Process a single frame.
        
        Returns:
            None if error occurred
            False if should quit
            True if processed successfully
        """
        try:
            iteration_start = time.time()
            
            # Get latest constants
            import SOD_Constants as const
            
            # Reset test frame flag
            self.current_frame_is_test = False
            
            # Handle test frame injection
            if self.inject_test_frames > 0:
                if self.current_test_image < len(self.test_images):
                    frame = self.test_images[self.current_test_image].copy()
                    
                    # Mark this as a test frame
                    self.current_frame_is_test = True
                    current_img_num = self.current_test_image + 1  # 1-based index for display
                    
                    # Calculate remaining time for this test frame
                    remaining_seconds = round(self.inject_test_frames / 60)  # Approximate seconds at 60 FPS
                    
                    # Only show message at the start and then every second
                    if self.inject_test_frames == self.frame_display_start:  # First frame of the sequence
                        print(f"\nInjecting test frame {current_img_num}/{len(self.test_images)} - displaying for ~3 seconds")
                    elif self.inject_test_frames % 60 == 0:  # Print update every second
                        print(f"Test frame {current_img_num}/{len(self.test_images)} - {remaining_seconds} seconds remaining")
                    
                    # Only decrement counter and move to next image after showing this one for the full duration
                    self.inject_test_frames -= 1
                    
                    # Move to next image when done with current one
                    if self.inject_test_frames == 0:
                        self.current_test_image += 1
                        self.inject_test_frames = self.frame_display_start
                        print(f"\nMoving to test frame {self.current_test_image + 1}/{len(self.test_images)}")
            
            # Crop frame - but skip for test frames since they're already cropped
            if not self.current_frame_is_test:
                frame = frame[:, const.get_value('CROP_LEFT'):-const.get_value('CROP_RIGHT')]  # Crop left and right sides
            
            # Run detection
            detection_start = time.time()
            detections = self.detector.process_frame(frame, self.display.avoid_boxes, is_test_frame=self.current_frame_is_test)
            self.logger.log_operation_time('detection', time.time() - detection_start)
            
            if detections is None:
                self.logger.log_operation_time('total_iteration', time.time() - iteration_start)
                return True
            
            # Update display
            display_start = time.time()
            annotated_frame = self.display.draw_detections(frame, detections)
            
            # Create debug view if needed
            debug_view = None
            if 'space' in detections.rcnn_boxes:
                space_data = []
                space_data.append((
                    detections.rcnn_boxes['space'],
                    detections.contours,
                    detections.anomalies,
                    detections.metadata
                ))
                debug_view = self.display.create_debug_view(frame, space_data)
            
            # Create combined view
            combined_frame = self.display.create_combined_view(annotated_frame, debug_view)
            
            # Update video buffer with combined frame
            self.video.update_buffer_annotations(combined_frame, debug_view)
            
            # Handle recording based on detections
            has_detection = bool(detections.anomalies)
            if has_detection and not self.video.recording:  # Only start recording if not already recording
                # Use fixed dimensions since both frames are 939x720
                frame_size = (939 * 2, 720)  # Width is 939*2 for combined view, height is 720
                if self.video.start_recording(frame_size, debug_view):
                    print(f"\nStarted recording: {self.video.current_video_number:05d}.avi")
                    # Save first detection frame
                    self.capture.start_new_video(self.video.current_video_number)
                    self.capture.save_detection(annotated_frame, debug_view, check_interval=False)
            
            # Stream frame if streaming is active
            if self.stream.is_streaming:
                # Resize the combined frame to match the expected dimensions for streaming
                if combined_frame.shape[1] != self.stream.frame_width or combined_frame.shape[0] != self.stream.frame_height:
                    stream_frame = cv2.resize(combined_frame, (self.stream.frame_width, self.stream.frame_height))
                else:
                    stream_frame = combined_frame
                # Send the frame to the stream manager
                self.stream.stream_frame(stream_frame)
            
            # Display the combined frame in a single window
            cv2.imshow('ISS Object Detection', combined_frame)
            
            # Update frame count
            self.frame_count += 1
            
            # Log timing information
            self.logger.log_operation_time('display', time.time() - display_start)
            self.logger.log_operation_time('total_iteration', time.time() - iteration_start)

            # Handle burst capture
            if self.burst_remaining > 0:
                self.capture.save_raw_frame(frame)
                self.burst_remaining -= 1
            
            # Use a 1ms timeout for key checks to minimize display latency
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                self._cleanup()
                return False
            elif key == ord('t'):
                self.inject_test_frames = 1  # Inject just one test frame at a time
            elif key == ord('b'):
                if self.inject_test_frames > 0:
                    # Exit test frame mode
                    self.inject_test_frames = 0
                    self.current_test_image = 0
                    print("\nExiting test frame mode")
                else:
                    # Start burst capture
                    self.burst_remaining = BURST_CAPTURE_FRAMES
                    print("\nStarting burst capture...")
            elif key == ord('c'):
                self.display.avoid_boxes = []  # Clear avoid boxes
                print("\nCleared avoid boxes")
            
            return True
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None

    def _attempt_connection(self, source: str, is_youtube: bool = False) -> bool:
        """Attempt to establish a connection to the video source."""
        try:
            if is_youtube:
                print("\nAttempting to get fresh stream URL...")
                stream_url = get_best_stream_url(source)
                if not stream_url:
                    print("Failed to get stream URL")
                    return False
                source = stream_url
            
            print("Initializing video capture...")
            # Set FFmpeg log level to error only and suppress connection warnings
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;error'
            os.environ['OPENCV_FFMPEG_DEBUG'] = '0'
            os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
            
            # Additional options to suppress HTTP connection warnings
            ffmpeg_options = [
                'protocol_whitelist;file,http,https,tcp,tls,crypto',
                'loglevel;error',
                'max_delay;500000',
                'reconnect;1',
                'reconnect_streamed;1',
                'reconnect_delay_max;5'
            ]
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = '|'.join(ffmpeg_options)
            
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                print("Failed to open video capture")
                return False
                
            # Get actual frame rate from capture
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                print("Warning: Invalid frame rate detected, using default 30 fps")
                fps = 30
            
            # Update RCNN cycle and video parameters based on actual fps
            #self.detector.set_rcnn_cycle(int(fps))  # Run RCNN once per second
            self.video.set_fps(fps)  # Update video recording fps
            
            # Get frame dimensions
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize StreamManager with correct dimensions
            if self.stream is None:
                self.stream = StreamManager(frame_width=frame_width, frame_height=frame_height)
            
            print("Successfully connected to video source")
            return True
            
        except Exception as e:
            print(f"Connection attempt failed: {str(e)}")
            return False

    def _setup_display(self):
        """Set up display windows and callbacks."""
        cv2.namedWindow('ISS Object Detection')
        cv2.setMouseCallback('ISS Object Detection', self.display._mouse_callback)

    def process_video_stream(self, source: str, is_youtube: bool = False) -> None:
        """Process video stream from source."""
        if not self._attempt_connection(source, is_youtube):
            print("Failed to connect to video source")
            return
            
        self._setup_display()
        self.is_running = True
        
        # Get the frame rate for timing control
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:  # Sanity check
            fps = 30.0  # Default to 30 fps if invalid
            
        # Calculate frame interval in seconds
        frame_interval = 1.0 / fps
        
        # Variables for frame rate control
        frame_count = 0
        start_time = time.time()
        buffer_level = 0  # Track buffer level to adjust timing
        last_frame_time = start_time
        
        # Main processing loop
        consecutive_errors = 0
        try:
            while self.is_running:
                iteration_start = time.time()
                
                # Calculate elapsed time and expected frame count
                elapsed_time = iteration_start - start_time
                expected_frame_count = int(elapsed_time / frame_interval)
                
                # If we're ahead of schedule, wait
                if frame_count > expected_frame_count:
                    # Calculate time to wait
                    wait_time = frame_interval - (iteration_start - last_frame_time)
                    if wait_time > 0:
                        time.sleep(wait_time)
                        
                # Read frame
                read_start = time.time()
                ret, frame = self.cap.read()
                self.logger.log_operation_time('frame_read', time.time() - read_start)
                
                if not ret:
                    # Adjust buffer level down if we're not getting frames
                    buffer_level = max(0, buffer_level - 1)
                    
                    # If buffer is depleted, handle error
                    if buffer_level <= 0:
                        consecutive_errors += 1
                        print(f"Error reading frame ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS})")
                        
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            print("Too many consecutive errors, reconnecting...")
                            if not self._attempt_connection(source, is_youtube):
                                print("Failed to reconnect, exiting")
                                break
                            consecutive_errors = 0
                            
                        # Short delay before retry
                        time.sleep(0.1)
                        continue
                else:
                    # Reset error counter and update buffer level
                    consecutive_errors = 0
                    buffer_level = min(10, buffer_level + 1)  # Cap at 10
                    
                    # Process frame
                    if not self.process_frame(frame):
                        break
                        
                    # Update frame count and timing
                    frame_count += 1
                    last_frame_time = time.time()
                    
                    # Periodically reset timing to prevent drift
                    if frame_count % 300 == 0:  # Every ~5 seconds at 60fps
                        start_time = time.time()
                        frame_count = 0
                
                # Check for user input (non-blocking)
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('+'):
                    print("\nStarting burst capture...")
                    self.burst_remaining = self.BURST_CAPTURE_FRAMES
                elif key == ord('s'):
                    self.toggle_streaming()
                elif key == ord('i'):
                    # Run streaming troubleshooting
                    print("\nTroubleshooting key pressed - running diagnostics...")
                    self.stream.troubleshoot_streaming()
                elif key == ord('b'):
                    # Start burst capture
                    self.capture.start_burst_capture()
                    print("\nStarted burst capture mode")
                    
        except Exception as e:
            print(f"Error processing video stream: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up resources before reconnection attempt."""
        # Stop video recording if active
        if self.video.recording:
            self.video.stop_recording()
            
        # Release video capture
        if hasattr(self, 'cap'):
            self.cap.release()
            delattr(self, 'cap')

    def cleanup(self) -> None:
        """Final cleanup of all resources."""
        self.is_running = False
        
        # Stop the console
        if self.console:
            self.console.stop()
            
        # Stop the logger
        if self.logger:
            self.logger.stop()
            
        # Release video capture
        if hasattr(self, 'cap'):
            self.cap.release()
            
        # Clean up managers
        self.capture.cleanup()
        self.video.cleanup()
        self.display.cleanup()
        self.detector.cleanup()
        self.stream.cleanup()  # Add stream cleanup
        
        # Destroy windows
        cv2.destroyAllWindows()

    def run(self) -> None:
        """Main run loop with mode selection."""
        try:
            print("\nSpace Object Detection System")
            print("----------------------------")
            print("Select mode:")
            print("v - Process video file")
            print("l - Process live feed")
            
            mode = input("Enter mode (v/l): ").lower()
            
            if mode == 'v':
                print("\nEnter video file path:")
                video_path = input("> ").strip()
                if not os.path.isfile(video_path):
                    print(f"Error: {video_path} not found")
                    return
                self.process_video(video_path)
            elif mode == 'l':
                print("\nStarting live feed processing...")
                self.process_live_feed()
            else:
                print("Invalid mode selected")
                
        except Exception as e:
            print(f"Error in run: {e}")
        finally:
            self.cleanup()
            
    def process_live_feed(self) -> None:
        """Process live feed from camera."""
        print("\nProcessing live feed. Press 'q' to quit.")
        print("Key commands:")
        print("  1 - Toggle streaming (start/stop)")
        print("  2 - Run streaming troubleshooting")
        print("  4 - Start burst capture")
        print("  5 - Pause capture for 5 seconds")
        
        # Main processing loop
        while True:
            # Get frame from video source
            ret, frame = self.video.get_frame()
            if not ret or frame is None:
                print("Error: Failed to get frame from video source")
                time.sleep(0.1)
                continue
                
            # Process the frame
            detection_result = self.process_frame(frame)
            
            # Check for key presses - use a shorter wait time for better responsiveness
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.toggle_streaming()
            elif key == ord('2'):
                # Run streaming troubleshooting
                print("\nTroubleshooting key pressed - running diagnostics...")
                self.stream.troubleshoot_streaming()
            elif key == ord('4'):
                # Start burst capture
                self.capture.start_burst_capture()
                print("\nStarted burst capture mode")
            elif key == ord('5'):
                # Pause capture for 5 seconds
                self.capture.pause_capture()
                print("\nPaused capture for 5 seconds")

    def load_test_images(self) -> bool:
        """Load all test images from the Test_Image_Collection directory."""
        try:
            self.test_images = []
            test_dir = os.path.dirname(TEST_IMAGE_PATH)
            
            # Get all jpg files in the test directory
            test_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
            
            if not test_files:
                print("No test images found in Test_Image_Collection directory")
                return False
                
            # Load each image
            for filename in test_files:
                img_path = os.path.join(test_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Get rightmost 939x720 pixels
                    h, w = img.shape[:2]
                    x1 = max(0, w - 939)  # Start x coordinate for cropping
                    img = img[:720, x1:]  # Crop to 939x720 from the right
                    self.test_images.append(img)
                    print(f"Loaded test image: {img_path}")
                else:
                    print(f"Failed to load test image: {img_path}")
            
            print(f"Loaded {len(self.test_images)} test images")
            return len(self.test_images) > 0
            
        except Exception as e:
            print(f"Error loading test images: {e}")
            return False

    def start_test_injection(self, frames: int = 10) -> None:
        """Start test frame injection."""
        if self.test_images:
            # Set frames to approximately 3 seconds worth of frames (at 60 FPS)
            self.inject_test_frames = 180  # ~3 seconds at 60 FPS
            self.frame_display_start = 180  # Track the total frames for percentage calculation
            print(f"\nStarting test frame injection - each frame will display for ~3 seconds")
        else:
            if self.load_test_images():
                self.start_test_injection(frames)
            else:
                print("No test images available")

def main():
    """Main entry point for the Space Object Detection system."""
    try:
        # Initialize the system
        sod = SpaceObjectDetectionSystem()
        if not sod.initialize():
            print("Failed to initialize system")
            return

        print("\nSpace Object Detection System")
        print("----------------------------")
        print("Saving both video (.avi) and images (.jpg) for each detection")
        print("Videos include 3s pre-detection buffer")
        
        # YouTube ISS live stream URL
        youtube_url = 'https://www.youtube.com/watch?v=jKHvbJe9c_Y'
        
        print("\nConnecting to ISS live feed...")
        sod.process_video_stream(youtube_url, is_youtube=True)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        if 'sod' in locals():
            sod.cleanup()

if __name__ == "__main__":
    main()