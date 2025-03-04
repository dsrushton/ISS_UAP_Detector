"""
Space Object Detection Display Module
Handles visualization of detection results and debug information.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from SOD_Constants import (
    CLASS_COLORS, CLASS_NAMES, CROPPED_WIDTH,
    DEBUG_VIEW_ENABLED, CONTOUR_COLOR, ANOMALY_BOX_COLOR,
    AVOID_BOX_COLOR, AVOID_BOX_THICKNESS, GAUSSIAN_BLUR_SIZE
)
from SOD_Detections import DetectionResults

class DisplayManager:
    """Manages visualization of detection results and debug information."""
    
    def __init__(self):
        self.debug_view = None
        self.last_save_time = 0
        self.logger = None
        self.is_streaming = False
        
        # Cache for shared computations
        self.last_space_mask = None
        self.last_space_contours = None
        self.last_frame_shape = None
        
        # Pre-allocated buffers
        self.debug_buffer = None
        self.debug_mask_3ch = None
        
        # Avoid box state
        self.avoid_boxes = []
        self.drawing_avoid_box = False
        self.avoid_start_pos = None
        self.current_avoid_box = None
        
        # Set up mouse callback
        cv2.namedWindow('Main View')
        cv2.setMouseCallback('Main View', self._mouse_callback)
        self._update_window_title()
    
    def set_logger(self, logger):
        """Set the logger instance."""
        self.logger = logger

    def set_streaming(self, is_streaming: bool) -> None:
        """Set streaming state and update window title."""
        self.is_streaming = is_streaming
        self._update_window_title()

    def _update_window_title(self) -> None:
        """Update the window title with streaming status."""
        title = "Main View"
        if self.is_streaming:
            title = "🔴 STREAMING - " + title
        cv2.setWindowTitle('Main View', title)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for avoid box drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing avoid box
            self.drawing_avoid_box = True
            self.avoid_start_pos = (x, y)
            self.current_avoid_box = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # Update current box while drawing
            if self.drawing_avoid_box:
                self.current_avoid_box = (
                    self.avoid_start_pos[0], self.avoid_start_pos[1],
                    x - self.avoid_start_pos[0], y - self.avoid_start_pos[1]
                )
                
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing avoid box
            if self.drawing_avoid_box:
                self.drawing_avoid_box = False
                if self.avoid_start_pos and abs(x - self.avoid_start_pos[0]) > 5 and abs(y - self.avoid_start_pos[1]) > 5:
                    # Convert to x1,y1,x2,y2 format and ensure positive width/height
                    x1 = min(self.avoid_start_pos[0], x)
                    y1 = min(self.avoid_start_pos[1], y)
                    x2 = max(self.avoid_start_pos[0], x)
                    y2 = max(self.avoid_start_pos[1], y)
                    self.avoid_boxes.append((x1, y1, x2, y2))
                self.current_avoid_box = None
                self.avoid_start_pos = None

    def _compute_space_mask_and_contours(self, boxes: list, frame: np.ndarray) -> tuple:
        """Compute space mask and contours for visualization."""
        # Create a mask for all space regions
        h, w = frame.shape[:2]
        space_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw all space boxes on the mask
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(space_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Find contours of space regions
        space_contours, _ = cv2.findContours(space_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Return mask and contours directly without computing dark mask
        # This improves performance by skipping unnecessary processing
        return space_mask, space_contours

    def _clear_cached_computations(self):
        """Clear cached computations when they should be recomputed."""
        self.last_space_mask = None
        self.last_space_contours = None
        self.last_frame_shape = None

    def _ensure_debug_buffers(self, frame_shape: tuple):
        """Ensure debug buffers are allocated with correct dimensions."""
        h, w = frame_shape[:2]
        
        if (self.debug_buffer is None or 
            self.debug_buffer.shape[0] != h or 
            self.debug_buffer.shape[1] != w):
            self.debug_buffer = np.zeros((h, w, 3), dtype=np.uint8)
            self.debug_mask_3ch = np.zeros((h, w, 3), dtype=np.uint8)
            print(f"Debug buffers allocated with shape {self.debug_buffer.shape}")

    def create_debug_view(self, frame: np.ndarray, space_data: list) -> np.ndarray:
        """Create debug visualization using the same structure as main view."""
        if not DEBUG_VIEW_ENABLED or not space_data:
            return np.zeros((720, CROPPED_WIDTH, 3), dtype=np.uint8)
            
        # Ensure buffers are allocated
        self._ensure_debug_buffers(frame.shape)
        
        # Reset debug buffer
        copy_start = time.time()
        self.debug_buffer.fill(0)
        
        # Get raw boxes from first entry
        boxes, contours, anomalies, metadata = space_data[0]
        
        # Use shared computation for space mask and contours, passing the frame
        space_mask, space_contours = self._compute_space_mask_and_contours(boxes, frame)
        
        # Create 3-channel mask for copying
        self.debug_mask_3ch = cv2.merge([space_mask, space_mask, space_mask])
        
        # Copy frame content for space regions (faster than np.where)
        np.copyto(self.debug_buffer, frame, where=(self.debug_mask_3ch > 0))
        self.logger.log_operation_time('debug_frame_copy', time.time() - copy_start)
        
        space_start = time.time()
        # Draw only the external space contours in blue
        for i, contour in enumerate(space_contours):
            if i < len(boxes):  # Only draw external space contours
                color = CLASS_COLORS['space']  # Blue for space
                cv2.drawContours(self.debug_buffer, [contour], -1, color, 2)
        
        # Remove the dark mask overlay to improve performance
        self.logger.log_operation_time('debug_space_box', time.time() - space_start)
        
        anomaly_start = time.time()
        # Draw detection contours - these are already in absolute coordinates
        if contours is not None:
            # Draw each contour directly - they're already in global coordinates
            cv2.drawContours(self.debug_buffer, contours, -1, CONTOUR_COLOR, 1)
        self.logger.log_operation_time('debug_contours', time.time() - anomaly_start)
        
        anomaly_start = time.time()
        # Draw anomaly boxes and metrics
        if anomalies is not None and metadata is not None and 'anomaly_metrics' in metadata:
            for anomaly_idx, (x, y, w, h) in enumerate(anomalies):
                # Draw rectangle with a small padding
                cv2.rectangle(self.debug_buffer, 
                            (x - 2, y - 2),
                            (x + w + 2, y + h + 2),
                            ANOMALY_BOX_COLOR, 2)
                
                if anomaly_idx < len(metadata['anomaly_metrics']):
                    metric = metadata['anomaly_metrics'][anomaly_idx]
                    if isinstance(metric, dict) and 'obj_brightness' in metric and 'contrast' in metric:
                        text = f"B:{metric['obj_brightness']:.1f} C:{metric['contrast']:.1f}"
                        cv2.putText(self.debug_buffer, text, 
                                  (x, y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, ANOMALY_BOX_COLOR, 1)
        self.logger.log_operation_time('debug_anomalies', time.time() - anomaly_start)
        
        return self.debug_buffer

    def draw_detections(self, frame: np.ndarray, detections: DetectionResults) -> np.ndarray:
        """Draw detection boxes and labels on frame."""
        # Clear cached computations at start of new frame
        self._clear_cached_computations()
        
        # Create a copy of the frame for annotations
        annotated_frame = frame.copy()
        
        # Handle special cases first
        if detections.darkness_detected:
            return self.draw_darkness_overlay(annotated_frame)
        elif 'nofeed' in detections.rcnn_boxes:
            return self.draw_nofeed_overlay(annotated_frame)
        
        draw_start = time.time()
        # Draw RCNN detections
        for class_name, boxes in detections.rcnn_boxes.items():
            # Handle space boxes separately
            if class_name == 'space':
                if boxes and len(boxes) > 0:  # Ensure we have boxes
                    color = CLASS_COLORS[class_name]
                    scores = detections.rcnn_scores[class_name]
                    
                    # Use shared computation for space mask and contours, passing the frame
                    mask, contours = self._compute_space_mask_and_contours(boxes, frame)
                    
                    # Draw labels for each box
                    for i, box in enumerate(boxes):
                        if i < len(scores):  # Ensure we have a matching score
                            x1, y1 = map(int, box[:2])
                            label_text = f"{class_name}: {scores[i]:.2f}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            thickness = 1
                            text_y = y1 + 20
                            text_x = x1 + 5
                            cv2.putText(annotated_frame, label_text, (text_x, text_y), font, font_scale, color, thickness)
                    
                    # Draw only the external space contours (blue)
                    # The first len(boxes) contours are the external space contours
                    for i, contour in enumerate(contours):
                        if i < len(boxes):  # Only draw external space contours
                            cv2.drawContours(annotated_frame, [contour], -1, color, 2)
                continue
                
            scores = detections.rcnn_scores[class_name]
            color = CLASS_COLORS[class_name]
            
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label
                label_text = f"{class_name}: {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, font, font_scale, thickness
                )
                
                # Position label
                if class_name == 'lf':
                    text_y = y1 - 5  # Above the box
                else:
                    text_y = y1 + text_height + 5  # Below the box
                text_x = x1 + 5
                
                # Draw label background
                cv2.rectangle(
                    annotated_frame,
                    (text_x - 2, text_y - text_height - 2),
                    (text_x + text_width + 2, text_y + 2),
                    (0, 0, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    color,
                    thickness
                )
        
        # Draw avoid boxes
        if self.avoid_boxes:
            for box in self.avoid_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), AVOID_BOX_COLOR, AVOID_BOX_THICKNESS)
                cv2.putText(annotated_frame, "AVOID", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, AVOID_BOX_COLOR, 1)
        
        # Draw current avoid box being drawn
        if self.drawing_avoid_box and self.current_avoid_box:
            x, y, w, h = self.current_avoid_box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), AVOID_BOX_COLOR, AVOID_BOX_THICKNESS)
        
        self.logger.log_operation_time('display_draw', time.time() - draw_start)
        return annotated_frame

    def draw_darkness_overlay(self, frame: np.ndarray, darkness_ratio: float = 0) -> np.ndarray:
        """
        Draw darkness overlay on frame.
        
        Args:
            frame: Input frame
            darkness_ratio: Ratio of frame covered by darkness (unused)
            
        Returns:
            Frame with darkness overlay
        """
        # Draw "DARKNESS" text centered
        text = "DARKNESS"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3
        color = (128, 0, 128)  # Purple
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (frame.shape[1] - text_width) // 2
        text_y = (frame.shape[0] + text_height) // 2
        
        # Draw text shadow
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0,0,0), thickness+2)
        # Draw text
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return frame

    def draw_nofeed_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw nofeed overlay on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with nofeed overlay
        """
        # Draw "NO FEED" text centered
        text = "NO FEED"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3
        color = (128, 0, 128)  # Purple
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (frame.shape[1] - text_width) // 2
        text_y = (frame.shape[0] + text_height) // 2
        
        # Draw text shadow
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0,0,0), thickness+2)
        # Draw text
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return frame

    def create_combined_view(self, frame: np.ndarray, debug: np.ndarray) -> np.ndarray:
        """Create combined view with debug information."""
        if self.debug_view is None:
            return frame
            
        debug_h, debug_w = self.debug_view.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        
        # Create combined frame
        combined = np.zeros((frame_h, frame_w + debug_w, 3), dtype=np.uint8)
        
        # Add debug view on left
        combined[0:debug_h, 0:debug_w] = self.debug_view
        
        # Add main frame on right
        combined[0:frame_h, debug_w:] = frame
        
        return combined

    def cleanup(self):
        """Clean up any display resources."""
        cv2.destroyAllWindows()