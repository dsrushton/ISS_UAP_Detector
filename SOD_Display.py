"""
Space Object Detection Display Module
Handles visualization of detection results and debug information.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

from SOD_Constants import CLASS_COLORS, CLASS_NAMES
from SOD_Detections import DetectionResults

class DisplayManager:
    """Manages visualization of detection results and debug information."""
    
    def __init__(self):
        self.debug_view = None
        self.last_save_time = 0
        
    def create_debug_view(self, roi: np.ndarray, contours: list, anomalies: list = None, metadata: dict = None) -> np.ndarray:
        """Create debug visualization with contours and anomalies."""
        # Create debug view same width as main frame, height from ROI
        h, w = roi.shape[:2]
        debug_view = np.zeros((h, roi.shape[1], 3), dtype=np.uint8)  # Full width, ROI height
        debug_view[:, :w] = roi  # Copy ROI into left portion
        overlay = debug_view.copy()
        
        # If nofeed is detected, return blank debug view
        if metadata and metadata.get('nofeed_detected'):
            return np.zeros_like(debug_view)
        
        # Draw all contours in green (thinner lines)
        cv2.drawContours(debug_view, contours, -1, (0, 255, 0), 1)
        
        # Draw anomaly boxes in red with transparency
        if anomalies:
            for x, y, w, h in anomalies:
                # Draw directly in ROI coordinates
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
                
                # Add metrics if available
                if metadata and 'anomaly_metrics' in metadata:
                    for metric in metadata['anomaly_metrics']:
                        if metric['position'] == (x, y, w, h):
                            text = f"B:{metric['obj_brightness']:.1f} C:{metric['contrast']:.1f}"
                            cv2.putText(overlay, text, (x, y - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Blend overlay with original
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, debug_view, 1 - alpha, 0, debug_view)
        
        # Add text indicating this is debug view
        cv2.putText(debug_view, "Space Region Debug View", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return debug_view

    def draw_detections(self, frame: np.ndarray, detections: DetectionResults) -> np.ndarray:
        """
        Draw detection boxes and labels on frame.
        
        Args:
            frame: Input frame
            detections: Detection results containing boxes and metadata
            
        Returns:
            Frame with annotations
        """
        annotated_frame = frame.copy()
        
        # Handle special cases first
        if detections.darkness_detected:
            return self.draw_darkness_overlay(annotated_frame)
        elif 'nofeed' in detections.rcnn_boxes:  # Check for nofeed detection
            return self.draw_nofeed_overlay(annotated_frame)
        
        # Draw RCNN detections
        for class_name, boxes in detections.rcnn_boxes.items():
            scores = detections.rcnn_scores[class_name]
            color = CLASS_COLORS[class_name]
            
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                
                if class_name == 'panel':
                    # Draw actual panel box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw expanded exclusion zone with dashed lines
                    width = x2 - x1
                    height = y2 - y1
                    ex1 = x1                    # Left stays same
                    ey1 = int(y1 - height * 0.10)  # Top expands up
                    ex2 = int(x2 + width * 0.10)   # Right expands
                    ey2 = y2                    # Bottom stays same
                    
                    # Draw dashed lines for exclusion zone
                    for i in range(ex1, ex2, 10):
                        cv2.line(annotated_frame, (i, ey1), (min(i+5, ex2), ey1), color, 1)
                    for i in range(ey1, ey2, 10):
                        cv2.line(annotated_frame, (ex2, i), (ex2, min(i+5, ey2)), color, 1)
                else:
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
        """Draw NO FEED overlay on frame."""
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
        """
        Create combined view with debug information.
        
        Args:
            frame: Original frame
            debug: Debug view
            
        Returns:
            Combined visualization frame
        """
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