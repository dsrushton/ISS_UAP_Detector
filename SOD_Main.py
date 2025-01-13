"""
Space Object Detection System - Main Module
Handles video streaming and orchestrates the detection, display, and capture processes.
"""

import cv2
import threading
from typing import Optional
import time
import os

from SOD_Constants import MAX_CONSECUTIVE_ERRORS, BURST_CAPTURE_FRAMES
from SOD_Utils import get_best_stream_url, crop_frame
from SOD_Video import VideoManager

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
        
        # State tracking
        self.frame_count: int = 0
        self.burst_remaining: int = 0
        self.is_running: bool = False
        
        # Test frame injection
        self.test_image = None
        self.inject_test_frames = 0
        
        self.video = VideoManager()
        
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize the model
            if not self.detector.initialize_model():
                print("Failed to initialize detector model")
                return False
                
            # Initialize capture system
            if not self.capture.initialize():
                print("Failed to initialize capture system")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            return False

    def load_test_image(self, path: str) -> bool:
        """Load test image for injection."""
        try:
            self.test_image = cv2.imread(path)
            if self.test_image is not None:
                self.test_image = cv2.resize(self.test_image, (1280, 720))
                return True
        except Exception as e:
            print(f"Error loading test image: {e}")
        return False

    def process_frame(self, frame) -> Optional[bool]:
        """
        Process a single frame through the detection pipeline.
        
        Args:
            frame: Video frame to process
            
        Returns:
            Optional[bool]: True to continue, False to stop, None on error
        """
        try:
            # Handle test frame injection
            if self.inject_test_frames > 0 and self.test_image is not None:
                frame = self.test_image.copy()
                print(f"\nTest frame {self.inject_test_frames}/10")
                self.inject_test_frames -= 1
            
            # Crop the frame
            frame = crop_frame(frame)
            
            # Run detection
            predictions = self.detector.process_frame(frame)
            if predictions is None:
                return None
                
            # Update display with detections
            annotated_frame = self.display.draw_detections(frame, predictions)
            debug_view = None
            
            # Create debug view if there's a space region
            if predictions.space_box is not None:
                x1, y1, x2, y2 = predictions.space_box
                space_roi = frame[y1:y2, x1:x2]
                
                # Get contours from space region
                gray = cv2.cvtColor(space_roi, cv2.COLOR_BGR2GRAY)
                background = cv2.GaussianBlur(gray, (21, 21), 0)
                diff = cv2.absdiff(gray, background)
                _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Convert anomaly coordinates to ROI space
                roi_anomalies = []
                if predictions.anomalies:
                    for ax, ay, aw, ah in predictions.anomalies:
                        # Convert to ROI coordinates
                        roi_x = ax - x1
                        roi_y = ay - y1
                        roi_anomalies.append((roi_x, roi_y, aw, ah))
                
                # Create debug view with ROI-space anomalies
                debug_view = self.display.create_debug_view(
                    space_roi, 
                    contours,
                    roi_anomalies if roi_anomalies else None,
                    predictions.metadata
                )
                if debug_view is not None:
                    cv2.imshow('Space Debug View', debug_view)
            
            # Show main detection view
            cv2.imshow('Space Object Detection', annotated_frame)
            
            # Handle burst capture if active
            if self.burst_remaining > 0:
                self.capture.save_raw_frame(frame)
                self.burst_remaining -= 1
                if self.burst_remaining == 0:
                    print("\nBurst save complete!")
            
            # Handle detections - always do both video and image
            if predictions.anomalies:
                # Start/update video recording
                if not self.video.is_recording:
                    self.video.start_recording(annotated_frame, debug_view)
                    # Set the video counter for jpg naming
                    self.capture.set_video_counter(self.video.counter)
                self.video.update_recording(annotated_frame, True, debug_view)
                
                # Save image
                self.capture.process_detections(annotated_frame, predictions, debug_view)
            elif self.video.is_recording:
                # Update ongoing video recording
                self.video.update_recording(annotated_frame, False, debug_view)
            
            # Add frame to buffer for video recording
            self.video.add_to_buffer(annotated_frame)
            
            # Increment frame counter
            self.frame_count += 1
            
            # Handle user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            elif key == ord('f'):
                print("\nStarting test image injection!")
                self.inject_test_frames = 10
                # Load test image if not already loaded
                if self.test_image is None:
                    self.load_test_image(r"C:\Users\dsrus\OneDrive\Pictures\sprites1.jpg")
            elif key == ord(' '):
                print("\nStarting 100 frame burst save to raw directory!")
                self.burst_remaining = BURST_CAPTURE_FRAMES
                
            return True
            
        except Exception as e:
            print(f"\nError processing frame: {str(e)}")
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
                
                while self.is_running:
                    # Read frame
                    ret, frame = self.video.get_frame()
                    if not ret:
                        consecutive_errors += 1
                        print(f"\nFrame read error ({consecutive_errors})")
                        if consecutive_errors > MAX_CONSECUTIVE_ERRORS:
                            print("\nToo many consecutive errors. Attempting to reconnect in 5 seconds...")
                            break  # Break inner loop to attempt reconnection
                        continue
                        
                    # Reset error counter on successful read
                    consecutive_errors = 0
                    
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