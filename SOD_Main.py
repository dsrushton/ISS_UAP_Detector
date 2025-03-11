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
        self.frame_display_start = 0
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

    def process_frame(self, frame: np.ndarray, avoid_boxes: List[Tuple[int, int, int, int]] = None) -> Tuple[bool, bool]:
        """
        Process a single frame for object detection.
        
        Args:
            frame: The frame to process
            avoid_boxes: List of boxes to avoid (x, y, w, h)
            
        Returns:
            Tuple of (has_detection, quit_requested)
            - has_detection: True if any objects were detected
            - quit_requested: True if user requested to quit
            - None if an error occurred
        """
        frame_start = time.time()
        
        try:
            # Start frame timing in logger
            self.logger.start_frame()
            
            # Check if we have avoid boxes from the display manager
            if avoid_boxes is None and self.display:
                avoid_boxes = self.display.avoid_boxes
                
                # Convert avoid boxes from (x1, y1, x2, y2) format to the format expected by the detector
                if avoid_boxes:
                    # Log the avoid boxes being used
                    self.logger.log_info(f"Using {len(avoid_boxes)} avoid boxes")
                    for i, box in enumerate(avoid_boxes):
                        self.logger.log_info(f"Avoid box {i+1}: {box}")
                
            # Handle test frames differently
            is_test_frame = False
            if self.inject_test_frames > 0:
                is_test_frame = True
                self.inject_test_frames -= 1
                
                # Use a test image if available
                if self.test_images:
                    # Get the current test image
                    test_frame = self.test_images[self.current_test_image]
                    
                    # Replace the current frame with the test frame
                    frame = test_frame.copy()
                    
                    # Increment the counter for current image frames
                    self.current_image_frames += 1
                    
                    # If we've shown this image for the required duration, move to the next image
                    if self.current_image_frames >= self.frames_per_test_image:
                        self.current_test_image = (self.current_test_image + 1) % len(self.test_images)
                        self.current_image_frames = 0
                        
                        # If we've cycled through all images, print a message
                        if self.current_test_image == 0:
                            print(f"Completed one full cycle through all {len(self.test_images)} test images")
                    
                    # Display current test image and progress
                    images_remaining = (self.inject_test_frames / self.frames_per_test_image)
                    print(f"Test image {self.current_test_image + 1}/{len(self.test_images)}: {images_remaining:.1f} seconds remaining")
                else:
                    print("No test images available. Run with --load-test-images first.")
                    self.inject_test_frames = 0
                
            # Crop frame to standard size if needed
            if frame.shape[1] > 1000:  # Only crop if width is large enough
                crop_start = time.time()
                # Crop to 939x720 (standard size for processing)
                h, w = frame.shape[:2]
                # Center crop
                start_x = (w - 939) // 2
                frame = frame[0:720, start_x:start_x+939]
                self.logger.log_operation_time('crop', time.time() - crop_start)
                
            # Run detection
            detect_start = time.time()
            detection_results = self.detector.process_frame(frame, avoid_boxes, is_test_frame)
            self.logger.log_operation_time('detect', time.time() - detect_start)
            
            if detection_results is None:
                self.logger.log_error("Detection failed")
                return None
                
            # Check for 'nofeed' or darkness - stop recording if active
            if (detection_results.darkness_detected or 'nofeed' in detection_results.rcnn_boxes) and self.video and self.video.recording:
                event_type = "darkness" if detection_results.darkness_detected else "nofeed"
                self.logger.log_iteration(True, False, f"Stopping recording due to {event_type} detection")
                self.video.stop_recording()
                
            # Create debug view
            debug_start = time.time()
            debug_view = None
            if self.display:
                # Get space data for debug view
                space_data = []
                if 'space' in detection_results.rcnn_boxes and detection_results.rcnn_boxes['space']:
                    space_data.append((
                        detection_results.rcnn_boxes['space'],
                        detection_results.contours,
                        detection_results.anomalies,
                        detection_results.metadata,
                        detection_results.space_mask,
                        detection_results.space_contours
                    ))
                    
                # Create debug view
                debug_view = self.display.create_debug_view(frame, space_data)
            self.logger.log_operation_time('debug_view', time.time() - debug_start)
            
            # Draw detections on frame
            draw_start = time.time()
            annotated_frame = None
            if self.display:
                annotated_frame = self.display.draw_detections(frame, detection_results)
            else:
                # If no display manager, use the original frame
                annotated_frame = frame.copy()
            self.logger.log_operation_time('draw', time.time() - draw_start)
            
            # Create combined view
            combined_start = time.time()
            combined_view = None
            try:
                if self.display:
                    combined_view = self.display.create_combined_view(annotated_frame, debug_view)
                self.logger.log_operation_time('combined_view', time.time() - combined_start)
            except Exception as e:
                self.logger.log_error(f"Error creating combined view: {str(e)}")
                combined_view = None
            
            # Check for detections
            has_detection = False
            if detection_results.anomalies or any(len(boxes) > 0 for boxes in detection_results.rcnn_boxes.values()):
                has_detection = True

                # Process detections for still image capture
                if self.capture:
                    self.capture.process_detections(frame, detection_results, debug_view)
                
            # Add frame to video buffer
            buffer_start = time.time()
        
            if self.video and combined_view is not None:
                # Always add to buffer - we need the context before detections
                try:
                    self.video.add_to_buffer(combined_view)
                except Exception as e:
                    self.logger.log_error(f"Error adding to buffer: {str(e)}")
            self.logger.log_operation_time('buffer', time.time() - buffer_start)
            
            # Start recording if detection found and not already recording
            # Only start recording if not in a 'nofeed' or darkness state
            if has_detection and self.video and not self.video.recording and not detection_results.darkness_detected and 'nofeed' not in detection_results.rcnn_boxes:
                record_start = time.time()
                # Use the combined view dimensions for the video
                frame_size = (combined_view.shape[1], combined_view.shape[0])
                if self.video.start_recording(frame_size, debug_view):
                    self.logger.log_iteration(True, True, "Started recording due to detection")
                    
                    # Sync the capture manager with the video number
                    if self.capture and self.video.current_video_number is not None:
                        self.capture.start_new_video(self.video.current_video_number)
                        
                self.logger.log_operation_time('record_start', time.time() - record_start)
                
            # Update recording if active
            if self.video and self.video.recording:
                record_update_start = time.time()
                self.video.update_recording(combined_view, has_detection)
                self.logger.log_operation_time('record_update', time.time() - record_update_start)
                
            # Update streaming if active
            if self.stream and self.stream.is_streaming and combined_view is not None:
                stream_start = time.time()
                self.stream.stream_frame(combined_view)
                self.logger.log_operation_time('stream', time.time() - stream_start)
                
            # Log total frame processing time
            self.logger.log_operation_time('total', time.time() - frame_start)
            
            # End frame timing in logger
            self.logger.end_frame()
            
            # Handle keyboard input
            quit_requested = False
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                quit_requested = True
            elif key == ord('t'):
                # Inject test frame
                print("Test frame injection requested")
                self.start_test_injection()  # Use the updated method
            elif key == ord('b'):
                # Start burst capture
                print("Burst capture requested")
                if self.video and not self.video.recording:
                    frame_size = (combined_view.shape[1], combined_view.shape[0])
                    if self.video.start_recording(frame_size, debug_view):
                        self.logger.log_iteration(True, True, "Started recording due to manual request")
            elif key == ord('c'):
                # Clear avoid boxes
                if self.display:
                    self.display.avoid_boxes = []
                    print("Avoid boxes cleared")
                    
            # Display the frame
            if self.display and combined_view is not None:
                cv2.imshow('ISS Object Detection', combined_view)
                # Force window to update (this is separate from the key handling waitKey)
                cv2.waitKey(1)
                
            return (has_detection, quit_requested)
            
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
                print("Stream manager initialized with fixed 1920x1080 dimensions")
            
            print("Successfully connected to video source")
            return True
            
        except Exception as e:
            print(f"Connection attempt failed: {str(e)}")
            return False

    def _setup_display(self):
        """Set up display windows and callbacks."""
        cv2.namedWindow('ISS Object Detection', cv2.WINDOW_NORMAL)
        # Set a fixed window size to match our 1920x1080 padded output
        cv2.resizeWindow('ISS Object Detection', 1920, 1080)
        cv2.setMouseCallback('ISS Object Detection', self.display._mouse_callback)
        
        # Create a blank initial frame to ensure the window is visible
        blank_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Add text to the blank frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(blank_frame, 'Initializing...', (int(1920/2) - 200, int(1080/2)), 
                    font, 2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow('ISS Object Detection', blank_frame)
        # Force window to update and be visible
        cv2.waitKey(1)

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
            fps = 60.0  # Default to 60 fps if invalid
            print("Could not detect valid FPS from source, defaulting to 60 FPS")
            
        # Update the system FPS
        self.fps = fps
            
        # Set the FPS for the video manager to ensure proper buffer size
        print(f"Setting video manager FPS to {fps}")
        self.video.set_fps(fps)
            
        # Set the expected FPS in the logger for rate calculations
        self.logger.set_expected_fps(fps)
            
        # Calculate frame interval in seconds
        frame_interval = 1.0 / fps
        
        # Variables for frame rate control
        frame_count = 0
        start_time = time.time()
        buffer_level = 0  # Track buffer level to adjust timing
        last_frame_time = start_time
        last_logger_check = start_time
        
        # Main processing loop
        consecutive_errors = 0
        try:
            while self.is_running:
                iteration_start = time.time()
                
                # Periodically check if logger is still running (every 60 seconds)
                current_time = time.time()
                if current_time - last_logger_check > 300:  # Check every 5 minutes
                    self.logger.ensure_running()
                    last_logger_check = current_time
                
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
                    # Reset error counter and update buffer level
                    consecutive_errors = 0
                    buffer_level = min(10, buffer_level + 1)  # Cap at 10
                    
                    # Process frame
                    process_result = self.process_frame(frame, self.display.avoid_boxes)
                    
                    # Check if we should quit
                    if process_result is None:
                        # Error occurred
                        self.logger.log_iteration(False, False, "Error processing frame")
                    elif isinstance(process_result, tuple) and len(process_result) == 2:
                        # Unpack the tuple
                        has_detection, quit_requested = process_result
                        
                        if quit_requested:
                            print("Quit requested")
                            break
                            
                        # Successful processing with detection info
                        self.logger.log_iteration(True, has_detection)
                        
                        # Periodically check buffer size
                        if frame_count % 300 == 0:  # Every ~5 seconds
                            buffer_size = len(self.video.frame_buffer)
                            expected_size = min(int(BUFFER_SECONDS * fps), self.video.frame_buffer.maxlen)
                            if buffer_size < expected_size * 0.5 and frame_count > expected_size:
                                print(f"WARNING: Video buffer size ({buffer_size}) is less than expected ({expected_size})")
                    else:
                        # Unexpected return value
                        self.logger.log_iteration(False, False, f"Unexpected return value from process_frame: {process_result}")
                    
                    # Update frame count and timing
                    frame_count += 1
                    last_frame_time = time.time()
                
                # Check for user input (non-blocking)
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    print("\nQuitting from process_video_stream...")
                    break
                elif key == ord('+'):
                    print("\nStarting burst capture...")
                    self.burst_remaining = BURST_CAPTURE_FRAMES
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
            # Calculate frames needed for 1 second per test image
            frames_per_image = int(self.fps)  # 1 second worth of frames at current FPS
            self.inject_test_frames = len(self.test_images) * frames_per_image
            self.frame_display_start = self.inject_test_frames
            self.current_test_image = 0  # Start from the first image
            self.frames_per_test_image = frames_per_image
            self.current_image_frames = 0  # Counter for frames shown for current image
            print(f"\nStarting test frame injection - cycling through {len(self.test_images)} images, 1 second each")
        else:
            if self.load_test_images():
                self.start_test_injection(frames)
            else:
                print("No test images available")

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