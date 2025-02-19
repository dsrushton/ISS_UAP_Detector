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
    AVOID_BOX_COLOR, AVOID_BOX_THICKNESS
)
from SOD_Detections import DetectionResults

class DisplayManager:
    """Manages visualization of detection results and debug information."""
    
    def __init__(self):
        self.debug_view = None
        self.last_save_time = 0
        self.logger = None
        
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
    
    def set_logger(self, logger):
        """Set the logger instance."""
        self.logger = logger

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

    def _compute_space_mask_and_contours(self, frame_shape: tuple, boxes: list) -> tuple:
        """Compute space mask and contours if needed or return cached version."""
        if (self.last_space_mask is not None and 
            self.last_frame_shape == frame_shape):
            return self.last_space_mask, self.last_space_contours
            
        # Create a new mask for all space regions
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        
        # Fill all space boxes in the mask
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # Fill
            
        # Find external contours of the combined regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Cache results
        self.last_space_mask = mask
        self.last_space_contours = contours
        self.last_frame_shape = frame_shape
        
        return mask, contours

    def _clear_cached_computations(self):
        """Clear cached computations when they should be recomputed."""
        self.last_space_mask = None
        self.last_space_contours = None
        self.last_frame_shape = None

    def _ensure_debug_buffers(self, shape: tuple):
        """Ensure debug buffers are allocated with correct shape."""
        if (self.debug_buffer is None or 
            self.debug_buffer.shape != shape):
            self.debug_buffer = np.zeros(shape, dtype=np.uint8)
            self.debug_mask_3ch = np.zeros(shape, dtype=np.uint8)

    def create_debug_view(self, frame: np.ndarray, space_data: list) -> np.ndarray:
        """Create debug visualization using the same structure as main view."""
        if not DEBUG_VIEW_ENABLED or not space_data:
            return np.zeros((720, CROPPED_WIDTH, 3), dtype=np.uint8)
            
        # Ensure buffers are allocated
        self._ensure_debug_buffers(frame.shape)
        
        # Reset debug buffer
        self.debug_buffer.fill(0)
        
        # Get raw boxes from first entry
        boxes, contours, anomalies, metadata = space_data[0]
        
        # Use shared computation for space mask and contours
        mask, space_contours = self._compute_space_mask_and_contours(frame.shape, boxes)
        
        # Convert mask to 3 channels for efficient copying
        self.debug_mask_3ch[:] = 0
        self.debug_mask_3ch[:, :, 0] = mask
        self.debug_mask_3ch[:, :, 1] = mask
        self.debug_mask_3ch[:, :, 2] = mask
        
        # Copy frame content for space regions (faster than np.where)
        np.copyto(self.debug_buffer, frame, where=(self.debug_mask_3ch > 0))
        
        # Draw only the external contours
        cv2.drawContours(self.debug_buffer, space_contours, -1, CLASS_COLORS['space'], 2)
        
        # Draw detection contours - now in absolute coordinates
        if contours is not None:
            cv2.drawContours(self.debug_buffer, contours, -1, CONTOUR_COLOR, 1)
        
        # Draw anomaly boxes and metrics
        if anomalies is not None and metadata is not None and 'anomaly_metrics' in metadata:
            for anomaly_idx, (x, y, w, h) in enumerate(anomalies):
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
        
        return self.debug_buffer

    def draw_detections(self, frame: np.ndarray, detections: DetectionResults) -> np.ndarray:
        """Draw detection boxes and labels on frame."""
        # Clear cached computations at start of new frame
        self._clear_cached_computations()
        
        annotated_frame = frame.copy()
        
        # Handle special cases first
        if detections.darkness_detected:
            return self.draw_darkness_overlay(annotated_frame)
        elif 'nofeed' in detections.rcnn_boxes:
            return self.draw_nofeed_overlay(annotated_frame)
        
        # Draw RCNN detections
        for class_name, boxes in detections.rcnn_boxes.items():
            # Handle space boxes separately
            if class_name == 'space':
                if boxes and len(boxes) > 0:  # Ensure we have boxes
                    color = CLASS_COLORS[class_name]
                    scores = detections.rcnn_scores[class_name]
                    
                    # Use shared computation for space mask and contours
                    mask, contours = self._compute_space_mask_and_contours(frame.shape, boxes)
                    
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
                    
                    # Draw only the external contours
                    cv2.drawContours(annotated_frame, contours, -1, color, 2)
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