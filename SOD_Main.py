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
    BUFFER_SECONDS,
    POST_DETECTION_SECONDS,
    SAVE_INTERVAL
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
        self.display = None  # Will be initialized later
        self.capture = CaptureManager()
        self.logger = None  # Will be initialized in initialize()
        self.console = None  # Will be initialized later
        self.stream = None  # Will be initialized after we know the frame dimensions
        self.video = None   # Will be initialized in initialize() after logger
        
        # State tracking
        self.frame_count: int = 0
        self.burst_remaining: int = 0
        self.is_running: bool = False
        self.fps: float = 60.0  # Default FPS value
        
        # Test image state
        self.test_images = []
        self.current_test_image = 0
        self.inject_test_frames = 0
        self.frame_display_start = 180  # Default to 3 seconds at 60 FPS
        self.frames_per_test_image = 0
        self.current_image_frames = 0
        self.current_frame_is_test = False  # Flag to track if current frame is a test frame
        
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
        
    def initialize(self) -> bool:
        """Initialize the system components."""
        try:
            # Initialize logger first
            from SOD_Logger import StatusLogger
            self.logger = StatusLogger()
            
            # Start the logger thread
            self.logger.start()
            
            # Set verbose mode for debugging
            self.logger.verbose_console = False  # Set to True for more detailed logging
            
            print("Initializing Space Object Detection System...")
            
            # Initialize video manager
            from SOD_Video import VideoManager
            self.video = VideoManager(self.logger)
            
            # Set up display manager
            from SOD_Display import DisplayManager
            self.display = DisplayManager()
            self.display.set_logger(self.logger)
            
            # Set up detector
            self.detector.set_logger(self.logger)
            
            # Initialize stream manager (will be set up when needed)
            self.stream = None
            
            # Set up OpenCV logging to our logger
            class OpenCVLogHandler(logging.Handler):
                def __init__(self, logger):
                    super().__init__()
                    self.logger = logger
                    self.last_warning_time = 0
                    self.warning_count = 0
                    
                def emit(self, record):
                    # Check if message contains HTTP connection warning or stream timeout
                    msg = self.format(record)
                    if "CURL" in msg or "error" in msg.lower() or "fail" in msg.lower():
                        # Rate limit these warnings to avoid spam
                        current_time = time.time()
                        if current_time - self.last_warning_time > 5.0:  # Only log every 5 seconds
                            self.logger.log_error(f"OpenCV: {msg}")
                            self.last_warning_time = current_time
                            self.warning_count = 1
                        else:
                            self.warning_count += 1
                            if self.warning_count % 10 == 0:  # Log every 10th warning in a burst
                                self.logger.log_error(f"OpenCV: {msg} (repeated {self.warning_count} times)")
            
            # Set up OpenCV logging
            cv_logger = logging.getLogger('opencv')
            cv_logger.setLevel(logging.WARNING)
            cv_logger.addHandler(OpenCVLogHandler(self.logger))
            
            # Load test images
            self.load_test_images()
            
            # Initialize capture manager
            self.capture.initialize()
            
            # Initialize parameter console
            self.console = ParameterConsole()
            self.console.start()
            
            # Initialize model
            if not self.detector.initialize_model():
                self.logger.log_error("Failed to initialize detection model")
                return False
                
            # Create output directories
            os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
            os.makedirs(JPG_SAVE_DIR, exist_ok=True)
            
            # Set up display
            self._setup_display()
            
            print("Initialization complete")
            return True
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def toggle_streaming(self) -> None:
        """Toggle streaming state and update display."""
        # Track last streaming message time
        if not hasattr(self, 'last_streaming_message_time'):
            self.last_streaming_message_time = 0
            
        current_time = time.time()
        should_print = (current_time - self.last_streaming_message_time) >= 60  # Only print once per minute
        
        if not self.stream.is_streaming:
            # Prompt for stream key if not already set
            if not self.stream.stream_key:
                # Use hardcoded stream key for testing
                stream_key = "3qsu-m42f-vp02-9w0r-f42a"  # Hardcoded for testing
                self.stream.stream_key = stream_key
                if should_print:
                    print(f"Using stream key: {stream_key[:4]}...{stream_key[-4:]}")
            
            if should_print:
                print("\nAttempting to start YouTube stream...")
                
            if self.stream.start_streaming(self.stream.frames_queue):
                self.display.set_streaming(True)
                if should_print:
                    print("\nStream started - check YouTube Studio")
                    print("Note: It may take 60-90 seconds for YouTube to show the stream")
                    self.last_streaming_message_time = current_time
            else:
                if should_print:
                    print("\nFailed to start streaming - check console for details")
                    self.last_streaming_message_time = current_time
        else:
            if should_print:
                print("\nStopping YouTube stream...")
                self.last_streaming_message_time = current_time
                
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
            
            # Start frame timing in logger
            self.logger.start_frame()
            
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
                        print(f"\nInjecting test frame {current_img_num}/{len(self.test_images)} - displaying for 1 second")
                    elif self.inject_test_frames % 60 == 0:  # Print update every second
                        print(f"Test frame {current_img_num}/{len(self.test_images)} - {remaining_seconds} seconds remaining")
                    
                    # Only decrement counter and move to next image after showing this one for the full duration
                    self.inject_test_frames -= 1
                    
                    # Move to next image when done with current one
                    if self.inject_test_frames == 0:
                        self.current_test_image += 1
                        
                        # Check if we've gone through all test images
                        if self.current_test_image >= len(self.test_images):
                            # Reset test frame injection and return to main feed
                            self.inject_test_frames = 0
                            self.current_test_image = 0
                            print(f"\nCompleted test frame cycle, returning to main feed")
                        else:
                            # Continue to next test image
                            self.inject_test_frames = self.frame_display_start
                            print(f"\nMoving to test frame {self.current_test_image + 1}/{len(self.test_images)}")
            
            # Crop frame - but skip for test frames since they're already cropped
            if not self.current_frame_is_test:
                frame = frame[:, const.get_value('CROP_LEFT'):-const.get_value('CROP_RIGHT')]  # Crop left and right sides
            
            # Check if we have avoid boxes from the display manager
            if avoid_boxes is None and self.display:
                avoid_boxes = self.display.avoid_boxes
            
            # Run detection
            detection_start = time.time()
            detections = self.detector.process_frame(frame, avoid_boxes, is_test_frame=self.current_frame_is_test)
            self.logger.log_operation_time('detection', time.time() - detection_start)
            
            if detections is None:
                self.logger.log_error("Detection failed")
                self.logger.log_operation_time('total_iteration', time.time() - iteration_start)
                self.logger.end_frame()
                return True
            
            # Update display
            display_start = time.time()
            annotated_frame = self.display.draw_detections(frame, detections)
            
            # Create debug view if needed
            debug_view = None
            
            # Check for special cases first (darkness, no feed, no space)
            if detections.darkness_detected or 'nofeed' in detections.rcnn_boxes:
                # For darkness or no feed, create a debug message view
                # The create_debug_view method now handles these cases automatically
                debug_view = self.display.create_debug_view(frame, [])
            elif 'space' in detections.rcnn_boxes:
                # Normal case with space regions
                space_data = []
                # Use combined space boxes from metadata if available, otherwise use raw RCNN boxes
                space_boxes = detections.metadata.get('all_space_boxes', detections.rcnn_boxes['space'])
                space_data.append((
                    space_boxes,
                    detections.contours,
                    detections.anomalies,
                    detections.metadata,
                    detections.space_mask if hasattr(detections, 'space_mask') else None,
                    detections.space_contours if hasattr(detections, 'space_contours') else None
                ))
                debug_view = self.display.create_debug_view(frame, space_data)
            else:
                # No space regions found
                debug_view = self.display.create_debug_view(frame, [])
            
            self.logger.log_operation_time('debug_view', time.time() - display_start)
            
            # Create combined view
            combined_frame = self.display.create_combined_view(annotated_frame, debug_view)
            
            # Handle recording based on detections
            has_detection = bool(detections.anomalies)
            
            # Add frame to video buffer
            if self.video:
                self.video.add_to_buffer(combined_frame)
            
            # Start recording if detection found and not already recording
            if has_detection and self.video and not self.video.recording:
                # Use fixed dimensions since both frames are 939x720
                frame_size = (combined_frame.shape[1], combined_frame.shape[0])
                if self.video.start_recording(frame_size, debug_view):
                    print(f"\nStarted recording: {self.video.current_video_number:05d}.avi")
                    # Save first detection frame
                    if self.capture:
                        self.capture.start_new_video(self.video.current_video_number)
                        self.capture.save_detection(annotated_frame, debug_view, check_interval=False)
            # Process detections for saving JPGs with incremented suffixes
            # This ensures that subsequent detections during the same video recording
            # are saved with incremented suffixes (-a, -b, -c, etc.)
            elif has_detection and self.video and self.video.recording and self.capture:
                # Only process if we're within the POST_DETECTION_SECONDS window
                # but after the SAVE_s     fv
                # 
                # 
                # VAL to avoid saving frames too frequently
                current_time = time.time()
                if (self.video.frames_since_detection / self.fps < POST_DETECTION_SECONDS and 
                    current_time - self.capture.last_save_time >= SAVE_INTERVAL):
                    # Process the detection to save a JPG with the next suffix
                    self.capture.process_detections(annotated_frame, detections, debug_view)
            
            # Update recording if active
            if self.video and self.video.recording:
                self.video.update_recording(combined_frame, has_detection)
            
            # Stream frame if streaming is active
            if self.stream and self.stream.is_streaming:
                # Only resize if dimensions don't match to avoid unnecessary CPU usage
                if combined_frame.shape[1] != self.stream.frame_width or combined_frame.shape[0] != self.stream.frame_height:
                    # Use a faster interpolation method for streaming since quality is less critical
                    stream_frame = cv2.resize(combined_frame, 
                                             (self.stream.frame_width, self.stream.frame_height),
                                             interpolation=cv2.INTER_NEAREST)
                else:
                    # Pass the frame directly without copying if dimensions already match
                    stream_frame = combined_frame
                    
                # Send the frame to the stream manager (no need to copy, StreamManager handles this)
                self.stream.stream_frame(stream_frame)
            
            # Display the combined frame using DisplayManager with optional rate limiting
            # Only display every X frames to reduce display overhead (e.g., 2 = 30fps display at 60fps processing)
            display_rate = 1  # Set to 1 for every frame, 2 for every other frame, etc.
            key = self.display.show_frame(combined_frame, rate_limit=display_rate)
            
            # Update frame count
            self.frame_count += 1
            
            # Log timing information
            self.logger.log_operation_time('display', time.time() - display_start)
            self.logger.log_operation_time('total_iteration', time.time() - iteration_start)
            
            # End frame timing in logger
            self.logger.end_frame()

            # Handle burst capture
            if hasattr(self, 'burst_remaining') and self.burst_remaining > 0:
                if self.capture:
                    self.capture.save_raw_frame(frame)
                self.burst_remaining -= 1
            
            # Process key commands
            if key == ord('q'):
                print("\nQuitting...")
                self._cleanup()
                return False
            elif key == ord('t'):
                if self.inject_test_frames > 0:
                    # If already in test mode, stop it
                    self.inject_test_frames = 0
                    self.current_test_image = 0
                    print("\nStopped test frame cycle, returning to main feed")
                else:
                    # Start cycling through all test frames
                    self.start_test_injection()
            elif key == ord('b'):
                if self.inject_test_frames > 0:
                    # Exit test frame mode
                    self.inject_test_frames = 0
                    self.current_test_image = 0
                    print("\nExiting test frame mode")
                else:
                    # Start burst capture
                    if not hasattr(self, 'burst_remaining'):
                        self.burst_remaining = 0
                    self.burst_remaining = const.get_value('BURST_CAPTURE_FRAMES')
                    print("\nStarting burst capture...")
            elif key == ord('c'):
                if self.display:
                    self.display.avoid_boxes = []  # Clear avoid boxes
                    print("\nCleared avoid boxes")
            elif key == ord('s'):
                self.toggle_streaming()
            
            return True
            
        except Exception as e:
            # Log the error and return None to indicate failure
            error_msg = f"Error processing frame: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_error(error_msg)
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
            if fps <= 0 or fps > 120:  # Sanity check
                fps = 60.0  # Default to 60 fps if invalid
                print("Could not detect valid FPS from source, defaulting to 60 FPS")
            
            # Update RCNN cycle and video parameters based on actual fps
            #self.detector.set_rcnn_cycle(int(fps))  # Run RCNN once per second
            self.video.set_fps(fps)  # Update video recording fps
            
            # Get frame dimensions from source (for reference only)
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Source video dimensions: {frame_width}x{frame_height}")
            
            # Initialize StreamManager with fixed 1920x1080 dimensions for padded output
            if self.stream is None:
                # Always use 1920x1080 for stream dimensions to match our padded output
                self.stream = StreamManager(frame_width=1920, frame_height=1080)
                # Set the logger reference
                if hasattr(self, 'logger') and self.logger:
                    self.stream.set_logger(self.logger)
                print("Stream manager initialized with fixed 1920x1080 dimensions")
            
            print("Successfully connected to video source")
            return True
            
        except Exception as e:
            print(f"Connection attempt failed: {str(e)}")
            return False

    def _setup_display(self):
        """Set up display windows and callbacks."""
        # Note: Window initialization and callbacks are now handled by DisplayManager
        # This method now just ensures the DisplayManager is properly referenced
        
        # Check if display manager needs initialization
        if not self.display:
            from SOD_Display import DisplayManager
            self.display = DisplayManager()
            
        # Make sure the display has a logger reference
        if self.logger:
            self.display.set_logger(self.logger)

    def process_video_stream(self, source: str, is_youtube: bool = False) -> None:
        """Process video stream from source."""
        if not self._attempt_connection(source, is_youtube):
            print("Failed to connect to video source")
            return
            
        self._setup_display()
        self.is_running = True
        
        # Get the frame rate for timing calculations and video manager
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:  # Sanity check
            fps = 60.0  # Default to 60 fps if invalid
            print("Could not detect valid FPS from source, defaulting to 60 FPS")
            
        # Update the system FPS
        self.fps = fps
        
        # Set the FPS for the video manager to ensure proper buffer size
        print(f"Setting video manager FPS to {fps}")
        self.video.set_fps(fps)
        
        # Set the expected FPS in the logger for rate calculations
        self.logger.set_expected_fps(fps)
        
        # Variables for metrics tracking
        frame_count = 0
        start_time = time.time()
        last_logger_check = start_time
        
        # Main processing loop
        consecutive_errors = 0
        try:
            while self.is_running:
                # Start timing the main loop cycle
                main_loop_start = time.time()
                
                # Periodically check if logger is still running (every 5 minutes)
                current_time = time.time()
                if current_time - last_logger_check > 300:
                    self.logger.ensure_running()
                    last_logger_check = current_time
                
                # Read frame
                read_start = time.time()
                ret, frame = self.cap.read()
                self.logger.log_operation_time('frame_read', time.time() - read_start)
                
                if not ret:
                    consecutive_errors += 1
                    print(f"Error reading frame ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS})")
                    
                    # Log the iteration with error
                    self.logger.log_iteration(False, False, f"Error reading frame ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS})")
                    
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
                    # Reset error counter
                    consecutive_errors = 0
                    
                    # Process frame - no frame limiting
                    process_result = self.process_frame(frame, self.display.avoid_boxes)
                    
                    # Check if we should quit
                    if process_result is None:
                        # Error occurred
                        self.logger.log_iteration(False, False, "Error processing frame")
                    elif isinstance(process_result, bool) and not process_result:
                        print("Quit requested")
                        break
                    else:
                        # Successful processing
                        self.logger.log_iteration(True, True, "Processed successfully")
                        
                    # Update frame count for metrics only
                    frame_count += 1
                
                # Record the main loop cycle time
                main_loop_time = time.time() - main_loop_start
                if self.logger:
                    self.logger.log_operation_time('main_loop_cycle', main_loop_time)
                
                # Key handling is now done in process_frame
                
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
        """Clean up resources."""
        print("Cleaning up resources...")
        
        try:
            # Clean up video capture
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                
            # Clean up stream
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop_streaming()
                print("Stream stopped")
                
            # Clean up video manager
            if hasattr(self, 'video') and self.video:
                self.video.cleanup()
                print("Video manager cleaned up")
                
            # Clean up display
            if hasattr(self, 'display') and self.display:
                self.display.cleanup()
                print("Display cleaned up")
                
            # Clean up detector
            if hasattr(self, 'detector') and self.detector:
                self.detector.cleanup()
                print("Detector cleaned up")
                
            # Clean up logger
            if hasattr(self, 'logger') and self.logger:
                print("Stopping logger...")
                self.logger.stop()
                
            # Clean up console
            if hasattr(self, 'console') and self.console:
                print("Stopping parameter console...")
                self.console.stop()
                
            # Destroy all OpenCV windows
            cv2.destroyAllWindows()
            
            print("Cleanup complete")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            import traceback
            traceback.print_exc()

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
        print("  s - Toggle streaming (start/stop)")
        print("  t - Cycle through test frames (1 second each)")
        print("  b - Toggle burst capture mode")
        print("  c - Clear avoid boxes")
        print("  q - Quit")
        
        # Main processing loop
        while True:
            # Start timing the main loop cycle
            main_loop_start = time.time()
            
            # Get frame from video source
            ret, frame = self.video.get_frame()
            if not ret or frame is None:
                print("Error: Failed to get frame from video source")
                time.sleep(0.1)
                continue
                
            # Process the frame
            process_result = self.process_frame(frame)
            
            # Check the result
            if process_result is None:
                # Error occurred
                print("Error processing frame")
                time.sleep(0.1)
                continue
            elif process_result is False:
                # Quit requested
                print("Quit requested")
                break
            
            # Key handling is now done in process_frame, no need for additional waitKey here
            
            # Record the main loop cycle time
            main_loop_time = time.time() - main_loop_start
            if self.logger:
                self.logger.log_operation_time('main_loop_cycle', main_loop_time)

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
            # Set the number of frames to display each test image
            if frames == 1:
                # Just show one frame
                self.inject_test_frames = 1
                self.frame_display_start = 1
            else:
                # Display each test image for 1 second (at current FPS)
                frames_per_image = int(self.fps)  # 1 second worth of frames at current FPS
                self.inject_test_frames = frames_per_image
                self.frame_display_start = frames_per_image
            
            self.current_test_image = 0  # Start from the first image
            print(f"\nStarting test frame injection - showing test image 1/{len(self.test_images)}")
            print(f"Each test frame will display for 1 second, then return to main feed after cycle")
        else:
            if self.load_test_images():
                self.start_test_injection(frames)

def main():
    """Main entry point for the Space Object Detection system."""
    try:
        # Ensure logs directory exists
        logs_dir = "./logs"
        if not os.path.exists(logs_dir):
            print(f"Creating logs directory: {logs_dir}")
            os.makedirs(logs_dir, exist_ok=True)
        
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
        import traceback
        traceback.print_exc()
    finally:
        if 'sod' in locals():
            sod.cleanup()

if __name__ == "__main__":
    main()