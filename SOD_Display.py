"""
Space Object Detection Display Module
Handles visualization of detection results and debug information.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

from SOD_Constants import (
    CLASS_COLORS, CLASS_NAMES, CROPPED_WIDTH,
    DEBUG_VIEW_ENABLED, CONTOUR_COLOR, ANOMALY_BOX_COLOR
)
from SOD_Detections import DetectionResults

class DisplayManager:
    """Manages visualization of detection results and debug information."""
    
    def __init__(self):
        self.debug_view = None
        self.last_save_time = 0
        
    def create_debug_view(self, roi: np.ndarray, contours: list, space_box: tuple, anomalies=None, metadata=None) -> np.ndarray:
        """Create debug visualization with contours and anomalies."""
        if not DEBUG_VIEW_ENABLED:
            return np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
        
        # Get dimensions
        roi_h, roi_w = roi.shape[:2]
        debug_view = np.zeros((roi_h, CROPPED_WIDTH, 3), dtype=np.uint8)
        x_offset = (CROPPED_WIDTH - roi_w) // 2  # Center the ROI
        
        # Copy ROI to debug view
        debug_view[:, x_offset:x_offset+roi_w] = roi.copy()
        
        # Check for nofeed detection
        if metadata is not None and metadata.get('nofeed_detected', False):
            return debug_view
        
        # Create overlay for contours and anomalies
        overlay = np.zeros_like(debug_view)
        roi_section = overlay[:, x_offset:x_offset+roi_w]
        
        # Draw contours in green
        if contours is not None and len(contours) > 0:
            cv2.drawContours(roi_section, contours, -1, CONTOUR_COLOR, 1)
        
        # Draw anomalies and metrics
        if anomalies is not None and metadata is not None and 'anomaly_metrics' in metadata:
            metrics_lookup = {}
            for m in metadata['anomaly_metrics']:
                pos = m['position']
                roi_x, roi_y = pos[0] - space_box[0], pos[1] - space_box[1]
                metrics_lookup[(roi_x, roi_y)] = m
            
            for x, y, w, h in anomalies:
                # Adjust coordinates to ROI space
                roi_x = x - space_box[0]
                roi_y = y - space_box[1]
                
                # Draw bounding box
                cv2.rectangle(roi_section, 
                            (roi_x, roi_y), 
                            (roi_x + w, roi_y + h), 
                            ANOMALY_BOX_COLOR, 2)
                
                # Find matching metrics
                metric = metrics_lookup.get((roi_x, roi_y))
                if metric:
                    text = f"B:{metric['obj_brightness']:.1f} C:{metric['contrast']:.1f}"
                    cv2.putText(roi_section, text, 
                              (roi_x, roi_y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, ANOMALY_BOX_COLOR, 1)
        
        # Blend overlay with debug view
        alpha = 0.7
        cv2.addWeighted(debug_view, alpha, overlay, 1 - alpha, 0, debug_view)
        
        return debug_view

    def draw_detections(self, frame: np.ndarray, detections: DetectionResults) -> np.ndarray:
        """Draw detection boxes and labels on frame."""
        annotated_frame = frame.copy()
        
        # Handle special cases first
        if detections.darkness_detected:
            return self.draw_darkness_overlay(annotated_frame)
        elif 'nofeed' in detections.rcnn_boxes:
            return self.draw_nofeed_overlay(annotated_frame)
        
        # Draw RCNN detections only (no anomaly boxes)
        for class_name, boxes in detections.rcnn_boxes.items():
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