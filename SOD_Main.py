"""
Space Object Detection System - Main Module
Handles video streaming and orchestrates the detection, display, and capture processes.
"""

import cv2
import threading
import time
import os
from typing import Optional
import numpy as np

from SOD_Constants import (
    MAX_CONSECUTIVE_ERRORS, 
    BURST_CAPTURE_FRAMES, 
    CROPPED_WIDTH,
    TEST_IMAGE_PATH,
    RECONNECT_DELAY,
    VIDEO_SAVE_DIR,
    VIDEO_FPS,
    BUFFER_SECONDS
)
from SOD_Utils import get_best_stream_url, crop_frame
from SOD_Video import VideoManager
from SOD_Logger import StatusLogger

class SpaceObjectDetectionSystem:
    """
    Main class that orchestrates the Space Object Detection system.
    Coordinates between detection, display, and capture modules.
    """
    
    def __init__(self):
        """Initialize the detection system and its components."""
        # These will be imported and initialized here to avoid circular imports
        from SOD_Detections import SpaceObjectDetector
        from SOD_Display import DisplayManager
        from SOD_Capture import CaptureManager
        
        self.detector = SpaceObjectDetector()
        self.display = DisplayManager()
        self.capture = CaptureManager()
        self.logger = StatusLogger()
        
        # State tracking
        self.frame_count: int = 0
        self.burst_remaining: int = 0
        self.is_running: bool = False
        
        # Test image state
        self.test_images = []
        self.current_test_image = 0
        self.inject_test_frames = 0
        
        self.video = VideoManager()
        
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
                
            # Set logger for detector
            self.detector.set_logger(self.logger)
                
            # Initialize capture system
            if not self.capture.initialize():
                print("Failed to initialize capture system")
                return False
                
            # Load test images
            if not self.load_test_images():
                print("Warning: Could not load test images")
            
            return True
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            return False

    def process_frame(self, frame: np.ndarray) -> Optional[bool]:
        """
        Process a single frame.
        
        Returns:
            None if error occurred
            False if should quit
            True if processed successfully
        """
        try:
            iteration_start = time.time()
            
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
            
            # Crop frame if needed
            if frame.shape[1] != CROPPED_WIDTH:
                frame = crop_frame(frame)
            
            # Run detection
            detection_start = time.time()
            detections = self.detector.process_frame(frame, self.display.avoid_boxes)
            self.logger.log_operation_time('detection', time.time() - detection_start)
            
            if detections is None:
                self.logger.log_iteration(False, error_msg="Failed to process frame")
                print("\nError processing frame - continuing to next frame")
                return True
            
            # Update display
            display_start = time.time()
            annotated_frame = self.display.draw_detections(frame, detections)
            self.logger.log_operation_time('display_update', time.time() - display_start)
            
            # Create debug view if we have a space region
            debug_view = None
            if detections.space_box is not None:
                debug_start = time.time()
                x1, y1, x2, y2 = detections.space_box
                roi = frame[y1:y2, x1:x2]
                debug_view = self.display.create_debug_view(
                    roi, 
                    detections.contours,
                    detections.space_box,
                    detections.anomalies,
                    detections.metadata
                )
                self.logger.log_operation_time('debug_view', time.time() - debug_start)
            
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
                    # Combine frame and debug view for recording
                    h, w = frame.shape[:2]
                    debug_h, debug_w = debug_view.shape[:2]
                    combined = np.zeros((max(h, debug_h), w + debug_w, 3), dtype=np.uint8)
                    # Put debug view on left, main frame on right
                    combined[:debug_h, :debug_w] = debug_view
                    combined[:h, debug_w:] = annotated_frame
                    self.video.update_recording(combined, detections.anomalies)
                    
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
                self.capture.save_frame(frame, debug_view)
                self.burst_remaining -= 1
            
            # Display frames
            cv2.imshow('Main View', annotated_frame)
            if debug_view is not None:
                cv2.imshow('Debug View', debug_view)
            
            # Handle user input
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

    def process_video_stream(self, source: str, is_youtube: bool = False) -> None:
        """
        Main processing loop for video input.
        
        Args:
            source: Path to video file or YouTube URL
            is_youtube: Whether source is a YouTube URL
        """
        # Initialize components
        if not self.initialize():
            return
            
        self.is_running = True
        
        while self.is_running:  # Outer loop for reconnection attempts
            try:
                # Handle YouTube URLs
                if is_youtube:
                    stream_url = get_best_stream_url(source)
                    if not stream_url:
                        print("\nFailed to get stream URL. Retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    source = stream_url

                # Initialize video capture
                print("\nConnecting to video source...")
                if not self.video.set_source(source):
                    print("\nError: Could not open video source. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue

                # Main processing loop
                consecutive_errors = 0
                backoff_time = RECONNECT_DELAY
                
                while self.is_running:
                    try:
                        # Get fresh stream URL for YouTube
                        url = get_best_stream_url(source)
                        self.cap = cv2.VideoCapture(url)
                    except Exception as e:
                        print(f"Error: {str(e)}")
                        consecutive_errors += 1
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            print(f"\nToo many consecutive errors ({consecutive_errors}). Attempting reconnection...")
                            time.sleep(backoff_time)
                            backoff_time = min(backoff_time * 2, 60)  # Double backoff time, max 60 seconds
                            self._cleanup()  # Ensure resources are cleaned up before reconnecting
                        else:
                            time.sleep(1)  # Brief pause before retry
                        continue
                        
                    if not self.cap.isOpened():
                        raise RuntimeError("Failed to open video capture")
                        
                    print("Connected to ISS live feed")
                    consecutive_errors = 0  # Reset error count on successful connection
                    backoff_time = RECONNECT_DELAY  # Reset backoff time
                    
                    while self.is_running:
                        # Read frame
                        ret, frame = self.cap.read()
                        if not ret:
                            raise RuntimeError("Failed to read frame")
                        
                        # Process frame
                        result = self.process_frame(frame)
                        if result is None:
                            consecutive_errors += 1
                            continue
                        elif result is False:
                            self.is_running = False  # User quit
                            break
                        
                    # Clean up current capture before reconnection attempt
                    self.video.cap.release()
                    if self.is_running:  # Only sleep if we're retrying
                        time.sleep(5)
                    
            except Exception as e:
                print(f"\nError processing video stream: {str(e)}")
                self.is_running = False
            
        # Final cleanup
        cv2.destroyAllWindows()
        self.cleanup()

    def _cleanup(self):
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
        if hasattr(self, 'save_thread') and self.save_thread is not None:
            self.save_thread.join()

    def cleanup(self) -> None:
        """Clean up resources and perform any necessary shutdown tasks."""
        self.is_running = False
        self.capture.cleanup()
        self.display.cleanup()
        self.detector.cleanup()

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