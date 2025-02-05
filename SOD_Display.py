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

    def create_debug_view(self, frame: np.ndarray, space_data: list) -> np.ndarray:
        """
        Create debug visualization for multiple space boxes.
        
        Args:
            frame: Full frame for extracting ROIs
            space_data: List of tuples (space_box, contours, anomalies, metadata)
        """
        if not DEBUG_VIEW_ENABLED or not space_data:
            return np.zeros((720, CROPPED_WIDTH, 3), dtype=np.uint8)
        
        t0 = time.time()
        
        # Create black background for debug view
        debug_view = np.zeros_like(frame)
        t1 = time.time()
        if self.logger:
            self.logger.log_operation_time('debug_frame_copy', t1 - t0)
        
        # Create a mask for all space regions
        space_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # First, create the mask for all space regions
        for space_box, _, _, _ in space_data:
            x1, y1, x2, y2 = map(int, space_box)
            cv2.rectangle(space_mask, (x1, y1), (x2, y2), 255, -1)
            
        # Find external contours of the combined regions
        contours, _ = cv2.findContours(space_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        # Copy frame content for space regions
        debug_view = np.where(space_mask[:, :, np.newaxis] == 255, frame, debug_view)
        
        # Draw the unified space box borders
        cv2.drawContours(debug_view, contours, -1, (0, 255, 0), 1)
        
        # Now draw all debug information on top
        for idx, (space_box, contours, anomalies, metadata) in enumerate(space_data):
            x1, y1, x2, y2 = map(int, space_box)  # Ensure integer coordinates
            
            t2 = time.time()
            # Draw space box label
            cv2.putText(debug_view, f"Space {idx + 1}", 
                       (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            t3 = time.time()
            if self.logger:
                self.logger.log_operation_time('debug_space_box', t3 - t2)
            
            # Draw contours directly
            if contours is not None and len(contours) > 0:
                t4 = time.time()
                cv2.drawContours(debug_view, contours, -1, CONTOUR_COLOR, 1, offset=(x1, y1))
                t5 = time.time()
                if self.logger:
                    self.logger.log_operation_time('debug_contours', t5 - t4)
            
            # Draw anomalies and metrics
            if anomalies is not None and metadata is not None and 'anomaly_metrics' in metadata:
                t6 = time.time()
                for anomaly_idx, (x, y, w, h) in enumerate(anomalies):
                    # Draw bounding box - expanded by 2 pixels in each direction
                    cv2.rectangle(debug_view, 
                                (x - 2, y - 2),  # Top-left expanded
                                (x + w + 2, y + h + 2),  # Bottom-right expanded
                                ANOMALY_BOX_COLOR, 2)
                    
                    # Find matching metrics
                    if 'anomaly_metrics' in metadata and anomaly_idx < len(metadata['anomaly_metrics']):
                        metric = metadata['anomaly_metrics'][anomaly_idx]
                        if isinstance(metric, dict) and 'obj_brightness' in metric and 'contrast' in metric:
                            text = f"B:{metric['obj_brightness']:.1f} C:{metric['contrast']:.1f}"
                            cv2.putText(debug_view, text, 
                                      (x, y - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, ANOMALY_BOX_COLOR, 1)
                t7 = time.time()
                if self.logger:
                    self.logger.log_operation_time('debug_anomalies', t7 - t6)
        
        t8 = time.time()
        if self.logger:
            self.logger.log_operation_time('debug_view', t8 - t0)
        return debug_view

    def draw_detections(self, frame: np.ndarray, detections: DetectionResults) -> np.ndarray:
        """Draw detection boxes and labels on frame."""
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
                    
                    # Create a mask for all space regions
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    
                    # Fill all space boxes in the mask
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # Fill
                        
                        # Draw labels for each box
                        if i < len(scores):  # Ensure we have a matching score
                            label_text = f"{class_name}: {scores[i]:.2f}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            thickness = 1
                            text_y = y1 + 20
                            text_x = x1 + 5
                            cv2.putText(annotated_frame, label_text, (text_x, text_y), font, font_scale, color, thickness)
                    
                    # Find external contours of the combined regions
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw only the external contours to create unified borders without internals
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