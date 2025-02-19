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

from SOD_Constants import (
    MAX_CONSECUTIVE_ERRORS, 
    BURST_CAPTURE_FRAMES, 
    CROPPED_WIDTH,
    TEST_IMAGE_PATH,
    RECONNECT_DELAY,
    VIDEO_SAVE_DIR,
    #VIDEO_FPS,
    BUFFER_SECONDS
)
from SOD_Utils import get_best_stream_url, crop_frame
from SOD_Video import VideoManager
from SOD_Logger import StatusLogger
from SOD_Console import ParameterConsole

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
        
        # State tracking
        self.frame_count: int = 0
        self.burst_remaining: int = 0
        self.is_running: bool = False
        
        # Test image state
        self.test_images = []
        self.current_test_image = 0
        self.inject_test_frames = 0
        
        self.video = VideoManager()
        
        # Add combined frame buffer
        self.combined_frame_buffer = None
        
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Start the logger
            self.logger.start()
            
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
                console_thread = threading.Thread(target=self.console.run)
                console_thread.daemon = True  # Make thread daemon so it exits with main program
                console_thread.start()
                time.sleep(0.5)  # Give console time to initialize
            
            return True
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            return False

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
            
            # Handle test frame injection
            if self.inject_test_frames > 0:
                if self.current_test_image < len(self.test_images):
                    frame = self.test_images[self.current_test_image].copy()
                    self.current_test_image += 1
                    if self.current_test_image >= len(self.test_images):
                        self.current_test_image = 0
                    self.inject_test_frames -= 1
                else:
                    self.inject_test_frames = 0
            
            # Crop frame
            frame = frame[:, const.get_value('CROP_LEFT'):-const.get_value('CROP_RIGHT')]  # Crop left and right sides
            
            # Run detection
            detection_start = time.time()
            detections = self.detector.process_frame(frame, self.display.avoid_boxes)
            self.logger.log_operation_time('detection', time.time() - detection_start)
            
            if detections is None:
                self.logger.log_iteration(False, error_msg="Failed to process frame")
                return True
            
            # Update display
            display_start = time.time()
            annotated_frame = self.display.draw_detections(frame, detections)
            self.logger.log_operation_time('display_update', time.time() - display_start)
            
            # Create debug view if we have space regions
            debug_view = None
            if 'space' in detections.rcnn_boxes and not detections.darkness_detected and 'nofeed' not in detections.rcnn_boxes:
                debug_start = time.time()
                
                # Use the detection results we already have
                space_data = []
                if 'space' in detections.rcnn_boxes:
                    # Pass the raw RCNN boxes for proper merging
                    space_data.append((
                        detections.rcnn_boxes['space'],  # Raw boxes
                        detections.contours,
                        detections.anomalies,
                        detections.metadata
                    ))
                
                # Create debug view with all space boxes
                debug_view = self.display.create_debug_view(frame, space_data)
                self.logger.log_operation_time('debug_view', time.time() - debug_start)
            elif detections.darkness_detected:
                debug_view = self.display.draw_darkness_overlay(np.zeros((720, CROPPED_WIDTH, 3), dtype=np.uint8))
            elif 'nofeed' in detections.rcnn_boxes:
                debug_view = self.display.draw_nofeed_overlay(np.zeros((720, CROPPED_WIDTH, 3), dtype=np.uint8))
            
            # Add to video buffer
            buffer_start = time.time()
            self.video.add_to_buffer(frame, annotated_frame, debug_view)
            self.logger.log_operation_time('video_buffer', time.time() - buffer_start)
            
            # Handle video recording
            recording_start = time.time()
            if detections.anomalies and not self.video.recording and not detections.metadata.get('skip_save'):
                # Start new recording with combined frame size
                frame_size = (frame.shape[1] + (debug_view.shape[1] if debug_view is not None else 0),
                             max(frame.shape[0], debug_view.shape[0] if debug_view is not None else 0))
                if self.video.start_recording(frame_size, debug_view):
                    print(f"\nStarted recording: {self.video.current_video_number:05d}.avi")
                    # Start tracking this video number for JPGs
                    self.capture.start_new_video(self.video.current_video_number)
                    # Save first detection frame without interval check
                    self.capture.save_detection(annotated_frame, debug_view, check_interval=False)

            # Update recording if active
            if self.video.recording:
                if debug_view is not None:
                    # Pre-allocate combined frame buffer if needed
                    h, w = frame.shape[:2]
                    debug_h, debug_w = debug_view.shape[:2]
                    if (self.combined_frame_buffer is None or 
                        self.combined_frame_buffer.shape != (max(h, debug_h), w + debug_w, 3)):
                        self.combined_frame_buffer = np.zeros((max(h, debug_h), w + debug_w, 3), dtype=np.uint8)
                    
                    # Reuse combined frame buffer
                    self.combined_frame_buffer.fill(0)
                    self.combined_frame_buffer[:debug_h, :debug_w] = debug_view
                    self.combined_frame_buffer[:h, debug_w:] = annotated_frame
                    self.video.update_recording(self.combined_frame_buffer, detections.anomalies)
                    
                    # Save new detection frame if anomaly detected and not skipped
                    if detections.anomalies and not detections.metadata.get('skip_save'):
                        self.capture.save_detection(annotated_frame, debug_view)
                else:
                    self.video.update_recording(annotated_frame, detections.anomalies)
            self.logger.log_operation_time('video_recording', time.time() - recording_start)

            # Log successful iteration
            self.logger.log_iteration(True, had_detection=bool(detections.anomalies))
            self.logger.log_operation_time('total_iteration', time.time() - iteration_start)

            # Handle burst capture
            if self.burst_remaining > 0:
                self.capture.save_raw_frame(frame)
                self.burst_remaining -= 1
            
            # Display frames and handle user input with minimal delay
            cv2.imshow('Main View', annotated_frame)
            if debug_view is not None:
                cv2.imshow('Debug View', debug_view)
            
            # Use a 1ms timeout for key checks to minimize display latency
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                self._cleanup()
                return False
            elif key == ord('t'):
                self.inject_test_frames = 100
            elif key == ord('b'):
                self.burst_remaining = BURST_CAPTURE_FRAMES
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
            self.cap = cv2.VideoCapture(source)
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
                
            print("Successfully connected to video source")
            return True
            
        except Exception as e:
            print(f"Connection attempt failed: {str(e)}")
            return False

    def _setup_display(self):
        """Set up display windows and callbacks."""
        cv2.namedWindow('Main View')
        cv2.setMouseCallback('Main View', self.display._mouse_callback)

    def process_video_stream(self, source: str, is_youtube: bool = False) -> None:
        """Main processing loop for video input."""
        if not self.initialize():
            return
            
        self.is_running = True
        consecutive_errors = 0
        backoff_time = RECONNECT_DELAY
        
        # Initial display setup
        self._setup_display()
        
        while self.is_running:
            # Attempt connection
            if not self._attempt_connection(source, is_youtube):
                print(f"Connection failed (attempt {consecutive_errors + 1})")
                consecutive_errors += 1
                
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"Too many consecutive errors ({consecutive_errors})")
                    print(f"Backing off for {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, 60)
                    self._cleanup()
                else:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                continue
            
            # Reset error counters on successful connection
            consecutive_errors = 0
            backoff_time = RECONNECT_DELAY
            
            # Main frame processing loop
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("\nFailed to read frame")
                    print("Capture state:", "Opened" if self.cap.isOpened() else "Closed")
                    break  # Break to outer loop for reconnection
                
                # Process frame
                result = self.process_frame(frame)
                if result is None:
                    print("Frame processing error")
                    continue
                elif result is False:
                    print("User initiated shutdown")
                    self.is_running = False
                    break
            
            # Clean up before reconnection attempt
            if self.is_running:
                print("\nLost connection - cleaning up and preparing to reconnect...")
                self._cleanup()
                time.sleep(5)
        
        # Final cleanup
        print("\nShutting down...")
        cv2.destroyAllWindows()  # Only destroy windows at final shutdown
        self.cleanup()

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
        """Process live video feed."""
        try:
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\nFailed to get frame")
                    break
                
                # Add frame to buffer
                self.video.add_to_buffer(frame)
                
                # Process frame
                predictions = self.detector.process_frame(frame)
                if predictions is None:
                    continue
                
                # Update display
                annotated_frame = self.display.draw_detections(frame, predictions)
                
                # Handle recording based on detections
                has_detection = bool(predictions.anomalies)
                if has_detection and not self.video.is_recording:
                    self.video.start_recording(frame)
                elif self.video.is_recording:
                    self.video.update_recording(frame, has_detection)
                
                # Show frame
                cv2.imshow('Live Feed', annotated_frame)
                
                # Handle user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error processing live feed: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def load_test_images(self) -> bool:
        """Load all test images."""
        try:
            self.test_images = []
            img = cv2.imread(TEST_IMAGE_PATH)
            if img is not None:
                # Get rightmost 939x720 pixels
                h, w = img.shape[:2]
                x1 = max(0, w - 939)  # Start x coordinate for cropping
                img = img[:720, x1:]  # Crop to 939x720 from the right
                self.test_images.append(img)
                print(f"Loaded test image: {TEST_IMAGE_PATH}")
                return True
            else:
                print(f"Failed to load test image: {TEST_IMAGE_PATH}")
                return False
            
        except Exception as e:
            print(f"Error loading test images: {e}")
            return False

    def start_test_injection(self, frames: int = 10) -> None:
        """Start test frame injection."""
        if self.test_images:
            self.inject_test_frames = frames
            print(f"\nStarting {frames} frame test injection")
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
        youtube_url = 'https://www.youtube.com/watch?v=OCem0E-0Q6Y'
        
        print("\nConnecting to ISS live feed...")
        sod.process_video_stream(youtube_url, is_youtube=True)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        if 'sod' in locals():
            sod.cleanup()

if __name__ == "__main__":
    main()