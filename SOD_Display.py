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
        """Initialize the display manager."""
        # Initialize state variables
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
        
        # Avoid box state
        self.avoid_boxes = []
        self.drawing_avoid_box = False
        self.avoid_start_pos = None
        self.current_avoid_box = None
        
        # Update window title
        self._update_window_title()
        
        # Set up mouse callback
        cv2.namedWindow('ISS Object Detection')
        cv2.setMouseCallback('ISS Object Detection', self._mouse_callback)
        
        # Set initial window size and title
        self._set_window_size()
        self._update_window_title()
    
    def set_logger(self, logger):
        """Set the logger instance."""
        self.logger = logger

    def set_streaming(self, is_streaming: bool) -> None:
        """Set streaming state and update window title."""
        self.is_streaming = is_streaming
        self._update_window_title()

    def _set_window_size(self):
        """Set the window to the correct size."""
        # Use 1920x1080 for the window size to match our padded output
        cv2.resizeWindow('ISS Object Detection', 1920, 1080)
        # No need for waitKey here, it will be handled in the main loop

    def _update_window_title(self) -> None:
        """Update the window title with streaming status."""
        if self.is_streaming:
            title = "[STREAMING] ISS Object Detection"  # Streaming indicator
        else:
            title = "ISS Object Detection"  # Normal title
        cv2.setWindowTitle('ISS Object Detection', title)
        # We don't need to reset the window size every time the title changes
        # This was causing the window to resize when streaming starts

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for avoid box drawing."""
        # Check if we have padding offsets defined
        has_padding = hasattr(self, 'x_padding_offset') and hasattr(self, 'y_padding_offset')
        
        # Determine if the click is in the main view (right side)
        # The combined view has the debug view on the left and the main view on the right
        # We need to account for padding if it exists
        debug_width = CROPPED_WIDTH  # Width of debug view (939 pixels)
        
        if has_padding:
            # If we have padding, we need to check if the click is within the actual main view area
            # The main view starts at x_padding_offset + debug_width
            main_view_start = self.x_padding_offset + debug_width
            
            # Check if click is in the main view area
            if x < main_view_start or x >= main_view_start + debug_width:
                # Click is outside the main view area
                return
                
            # Adjust coordinates to be relative to the main view (removing padding)
            main_x = x - main_view_start
            main_y = y - self.y_padding_offset
        else:
            # No padding, use the simple check
            if x < debug_width:
                # Ignore clicks in the left half (debug view)
                return
                
            # Adjust x coordinate to be relative to the main view
            main_x = x - debug_width
            main_y = y
            
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing avoid box
            self.drawing_avoid_box = True
            self.avoid_start_pos = (main_x, main_y)  # Store coordinates relative to main view
            self.current_avoid_box = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # Update current box while drawing
            if self.drawing_avoid_box:
                self.current_avoid_box = (
                    self.avoid_start_pos[0], self.avoid_start_pos[1],
                    main_x - self.avoid_start_pos[0], main_y - self.avoid_start_pos[1]
                )
                
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing avoid box
            if self.drawing_avoid_box:
                self.drawing_avoid_box = False
                if self.avoid_start_pos and abs(main_x - self.avoid_start_pos[0]) > 5 and abs(main_y - self.avoid_start_pos[1]) > 5:
                    # Convert to x1,y1,x2,y2 format and ensure positive width/height
                    x1 = min(self.avoid_start_pos[0], main_x)
                    y1 = min(self.avoid_start_pos[1], main_y)
                    x2 = max(self.avoid_start_pos[0], main_x)
                    y2 = max(self.avoid_start_pos[1], main_y)
                    self.avoid_boxes.append((x1, y1, x2, y2))
                    print(f"Added avoid box: ({x1}, {y1}, {x2}, {y2})")
                self.current_avoid_box = None
                self.avoid_start_pos = None

    def _compute_space_mask_and_contours(self, boxes: list, frame: np.ndarray) -> tuple:
        """
        Compute space mask and contours for visualization.
        
        NOTE: This is a fallback method that should rarely be called - normally these
        should be computed by the detection module and passed to the display module.
        This is only used if the masks/contours are missing for some reason.
        """
        # Create a mask for all space regions - no caching, just compute directly
        h, w = frame.shape[:2]
        space_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Skip if no boxes
        if not boxes or len(boxes) == 0:
            return space_mask, []
        
        # Draw all space boxes on the mask
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            space_mask[y1:y2, x1:x2] = 255  # Faster than cv2.rectangle for filled rectangles
        
        # Find contours of space regions
        space_contours, _ = cv2.findContours(space_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        return space_mask, space_contours

    def _clear_cached_computations(self):
        """Clear any buffers that might need resetting between frames."""
        # Since we're no longer caching, this is mostly a no-op
        # Only the buffers are retained between frames for memory efficiency
        pass

    def _ensure_debug_buffers(self, frame_shape: tuple):
        """Ensure debug buffers are allocated with correct dimensions."""
        h, w = frame_shape[:2]
        
        # Only allocate new buffers if dimensions have changed or buffers don't exist
        if (self.debug_buffer is None or 
            self.debug_buffer.shape[0] != h or 
            self.debug_buffer.shape[1] != w):
            # Allocate new buffers
            self.debug_buffer = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            # Clear the existing buffer
            self.debug_buffer.fill(0)
            
        # The buffers don't need to be cleared here because they'll be reset in the
        # specific create_debug_view method with fill(0) or overwritten entirely

    def create_debug_message_view(self, message: str) -> np.ndarray:
        """Create a black debug view with centered text message.
        
        Args:
            message: Message to display, can contain \n for line breaks
            
        Returns:
            Black debug view with centered text
        """
        # Create or reuse a black canvas buffer
        if not hasattr(self, 'message_buffer') or self.message_buffer is None:
            self.message_buffer = np.zeros((720, CROPPED_WIDTH, 3), dtype=np.uint8)
        else:
            # Clear the existing buffer
            self.message_buffer.fill(0)
        
        # Handle multiline messages
        lines = message.split('\n')
        
        # Calculate vertical position for centered text block
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (255, 255, 255)  # White text
        
        # Calculate total height of all text lines to center vertically
        line_heights = []
        for line in lines:
            (_, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            line_heights.append(text_height + 10)  # Add 10px padding between lines
            
        total_height = sum(line_heights)
        start_y = (720 - total_height) // 2
        
        # Draw each line centered horizontally
        current_y = start_y
        for i, line in enumerate(lines):
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            text_x = (CROPPED_WIDTH - text_width) // 2
            
            # Draw text with shadow for better visibility
            # Draw shadow
            cv2.putText(self.message_buffer, line, (text_x+2, current_y+2), font, font_scale, (0, 0, 0), thickness+1)
            # Draw main text
            cv2.putText(self.message_buffer, line, (text_x, current_y), font, font_scale, color, thickness)
            
            # Move to next line
            current_y += line_heights[i]
            
        return self.message_buffer

    def create_debug_view(self, frame: np.ndarray, space_data: list) -> np.ndarray:
        """Create debug visualization using the same structure as main view."""
        if not DEBUG_VIEW_ENABLED:
            return np.zeros((720, CROPPED_WIDTH, 3), dtype=np.uint8)
            
        # Handle special cases where we need to show a message instead of space data
        if hasattr(self, 'latest_detections') and self.latest_detections is not None:
            # Check for darkness case
            if self.latest_detections.darkness_detected:
                return self.create_debug_message_view("No \"Space\" regions to analyze currently\n\nTry back soon\n\n\n\nISS In Darkness")
                
            # Check for no feed case
            if 'nofeed' in self.latest_detections.rcnn_boxes:
                return self.create_debug_message_view("No \"Space\" regions to analyze currently\n\nTry back soon\n\n\n\nStream Interrupted")
                
        # Check if space_data is missing or empty (no space boxes to process)
        if not space_data:
            return self.create_debug_message_view("No \"Space\" regions to analyze currently\n\nTry back soon\n\n\n\n")
            
        #debug_start = time.time()
        
        # Get data from first entry
        boxes, contours, anomalies, metadata, space_mask, space_contours = space_data[0]
        
        # Ensure buffers are allocated
        self._ensure_debug_buffers(frame.shape)
        
        # Copy raw frame content for space regions
        np.copyto(self.debug_buffer, frame, where=space_mask[:, :, None] > 0)
        
        #space_start = time.time()
        # Step 2: Draw the space region outline (blue)
        cv2.drawContours(self.debug_buffer, space_contours, -1, CLASS_COLORS['space'], 1)  # Reduced thickness to 1
        
        # Step 2.5: Draw the space bounding boxes from RCNN detections
        if boxes and len(boxes) > 0:
            # Check if these are the original RCNN boxes or the boxes are part of a multi-faced polygon
            are_multi_faced = False
            if space_data and len(space_data) > 0:
                # Check if we have multiple boxes that form a multi-faced polygon
                are_multi_faced = len(boxes) > 1
                    
            # For multi-faced polygons, don't draw the individual boxes to avoid interior borders
            # Instead rely on the space_contours drawn above for the external border
            if not are_multi_faced:
                # Only draw box outlines for single regions
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    # Use standard color for single boxes
                    box_color = (255, 165, 0)  # Orange color
                    cv2.rectangle(self.debug_buffer, (x1, y1), (x2, y2), box_color, 1)
            else:
                # For multi-faced, draw a label on the first box
                if boxes:
                    x1, y1, _, _ = map(int, boxes[0])
                    cv2.putText(self.debug_buffer, "multi-region", (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        
        #self.logger.log_operation_time('debug_space_box', time.time() - space_start)
        
        #anomaly_start = time.time()
        # Step 3: Draw detection contours directly on the debug buffer, but only within the space mask
        if contours is not None and len(contours) > 0:
            # Skip creating a new mask - use the existing space_mask directly
            for contour in contours:
                # Check if contour is within the space mask - use a faster approach
                # Get bounding rect of contour
                x, y, w, h = cv2.boundingRect(contour)
                # Check if the bounding rect overlaps with space mask
                roi = space_mask[y:y+h, x:x+w]
                if roi.size > 0 and np.any(roi > 0):
                    # Draw the contour with a more visible color and thickness
                    cv2.drawContours(self.debug_buffer, [contour], -1, CONTOUR_COLOR, 1)  # Reduced thickness to 1
        
        # Step 4: Draw anomaly boxes and metrics
        if anomalies is not None and len(anomalies) > 0:
            # Make sure anomaly boxes are drawn with high visibility
            for anomaly_idx, (x, y, w, h) in enumerate(anomalies):
                # Draw rectangle with a small padding and increased thickness
                cv2.rectangle(self.debug_buffer, 
                            (x - 2, y - 2),
                            (x + w + 2, y + h + 2),
                            ANOMALY_BOX_COLOR, 3)  # Increased thickness to 3
                
                # Add metrics text if available
                if metadata is not None and 'anomaly_metrics' in metadata and anomaly_idx < len(metadata['anomaly_metrics']):
                    metric = metadata['anomaly_metrics'][anomaly_idx]
                    if isinstance(metric, dict) and 'obj_brightness' in metric and 'contrast' in metric:
                        text = f"B:{metric['obj_brightness']:.1f} C:{metric['contrast']:.1f}"
                        # Draw text with better visibility
                        # First draw a black outline
                        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                            cv2.putText(self.debug_buffer, text, 
                                      (x + dx, y - 5 + dy),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        # Then draw the text in the anomaly color
                        cv2.putText(self.debug_buffer, text, 
                                  (x, y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANOMALY_BOX_COLOR, 1)
        
        # Avoid boxes are only drawn in the main view, not in debug view
        
        #self.logger.log_operation_time('debug_anomalies', time.time() - anomaly_start)
        #self.logger.log_operation_time('debug_view', time.time() - debug_start)
        
        return self.debug_buffer

    def draw_detections(self, frame: np.ndarray, detections: DetectionResults) -> np.ndarray:
        """Draw detection boxes and labels on frame."""
        # Clear cached computations at start of new frame
        self._clear_cached_computations()
        
        # Store the latest detections for use in other methods
        self.latest_detections = detections
        
        # Handle special cases first - in these cases we don't need a frame copy
        if detections.darkness_detected:
            return self.draw_darkness_overlay(frame)
        elif 'nofeed' in detections.rcnn_boxes:
            return self.draw_nofeed_overlay(frame)
        
        # Create drawing buffer if not already created or if size doesn't match
        if not hasattr(self, 'drawing_buffer') or self.drawing_buffer is None or self.drawing_buffer.shape != frame.shape:
            self.drawing_buffer = np.zeros_like(frame)
        
        # Copy the frame only once - reuse the drawing buffer
        np.copyto(self.drawing_buffer, frame)
        
        draw_start = time.time()
        # Draw RCNN detections
        for class_name, boxes in detections.rcnn_boxes.items():
            # Handle space boxes separately
            if class_name == 'space':
                if boxes and len(boxes) > 0:  # Ensure we have boxes
                    color = CLASS_COLORS[class_name]
                    scores = detections.rcnn_scores[class_name]
                    
                    # Use boxes from metadata if available
                    # Note: All boxes are kept as original instead of being combined into one large box
                    multi_region = False
                    display_boxes = boxes
                    if 'all_space_boxes' in detections.metadata:
                        display_boxes = detections.metadata['all_space_boxes']
                        multi_region = len(display_boxes) > 1
                    
                    # For multi-region, draw the contours instead of individual boxes
                    if multi_region and hasattr(detections, 'space_contours') and detections.space_contours:
                        # Draw the external contours in a distinct color
                        box_color = (0, 200, 255)  # Cyan-blue for multi-region
                        cv2.drawContours(self.drawing_buffer, detections.space_contours, -1, box_color, 2)
                        
                        # Add a label for the multi-region
                        if display_boxes:
                            x1, y1, _, _ = map(int, display_boxes[0])
                            label_text = f"{class_name}: multi-region"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            thickness = 1
                            text_y = y1 + 20
                            text_x = x1 + 5
                            cv2.putText(self.drawing_buffer, label_text, (text_x, text_y), font, font_scale, box_color, thickness)
                    else:
                        # For single regions, draw the boxes as before
                        for i, box in enumerate(display_boxes):
                            x1, y1, x2, y2 = map(int, box)
                            # Draw the space box rectangle
                            cv2.rectangle(self.drawing_buffer, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label
                            if i < len(scores):
                                label_text = f"{class_name}: {scores[i]:.2f}"
                            else:
                                label_text = f"{class_name}"
                                
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            thickness = 1
                            text_y = y1 + 20
                            text_x = x1 + 5
                            cv2.putText(self.drawing_buffer, label_text, (text_x, text_y), font, font_scale, color, thickness)
                continue
                
            scores = detections.rcnn_scores[class_name]
            color = CLASS_COLORS[class_name]
            
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(self.drawing_buffer, (x1, y1), (x2, y2), color, 2)
                
                # Use the same simple label style as space class
                label_text = f"{class_name}: {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                
                # Position label consistently
                text_y = y1 + 20  # Same position as space labels
                text_x = x1 + 5
                
                # Draw label text directly without background
                cv2.putText(
                    self.drawing_buffer,
                    label_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    color,
                    thickness
                )
        
        # DO NOT draw anomaly boxes (contour detections) in the main view
        # These will only be shown in the debug view
        # The following code is commented out to prevent drawing contour detection boxes
        # for box in detections.anomalies:
        #     x, y, w, h = map(int, box)
        #     cv2.rectangle(self.drawing_buffer, (x, y), (x + w, y + h), ANOMALY_BOX_COLOR, 2)
        
        # Draw avoid boxes
        if self.avoid_boxes:
            for box in self.avoid_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(self.drawing_buffer, (x1, y1), (x2, y2), AVOID_BOX_COLOR, AVOID_BOX_THICKNESS)
                cv2.putText(self.drawing_buffer, "AVOID", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, AVOID_BOX_COLOR, 1)
        
        # Draw current avoid box being drawn
        if self.drawing_avoid_box and self.current_avoid_box:
            x, y, w, h = self.current_avoid_box
            cv2.rectangle(self.drawing_buffer, (x, y), (x + w, y + h), AVOID_BOX_COLOR, AVOID_BOX_THICKNESS)
        
        #self.logger.log_operation_time('display_draw', time.time() - draw_start)
        return self.drawing_buffer

    def draw_darkness_overlay(self, frame: np.ndarray, darkness_ratio: float = 0) -> np.ndarray:
        """
        Draw darkness overlay on frame.
        
        Args:
            frame: Input frame
            darkness_ratio: Ratio of frame covered by darkness (unused)
            
        Returns:
            Frame with darkness overlay
        """
        # Create overlay buffer if needed
        if not hasattr(self, 'overlay_buffer') or self.overlay_buffer is None or self.overlay_buffer.shape != frame.shape:
            self.overlay_buffer = np.zeros_like(frame)
        
        # Copy the frame to our buffer instead of modifying the original
        np.copyto(self.overlay_buffer, frame)
        
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
        cv2.putText(self.overlay_buffer, text, (text_x, text_y), font, font_scale, (0,0,0), thickness+2)
        # Draw text
        cv2.putText(self.overlay_buffer, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return self.overlay_buffer

    def draw_nofeed_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw nofeed overlay on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with nofeed overlay
        """
        # Create overlay buffer if needed
        if not hasattr(self, 'overlay_buffer') or self.overlay_buffer is None or self.overlay_buffer.shape != frame.shape:
            self.overlay_buffer = np.zeros_like(frame)
        
        # Copy the frame to our buffer instead of modifying the original
        np.copyto(self.overlay_buffer, frame)
        
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
        cv2.putText(self.overlay_buffer, text, (text_x, text_y), font, font_scale, (0,0,0), thickness+2)
        # Draw text
        cv2.putText(self.overlay_buffer, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return self.overlay_buffer

    def create_combined_view(self, frame: np.ndarray, debug: np.ndarray) -> np.ndarray:
        """
        Create a combined view with debug view on left and main view on right.
        Always returns a padded 1920x1080 frame.
        
        Args:
            frame: The main view frame (annotated with detections)
            debug: The debug view frame
            
        Returns:
            Combined view with debug on left, main view on right, padded to 1920x1080
        """
        try:
            combined_start = time.time()
            
            # Validate inputs
            if frame is None:
                return np.zeros((1080, 1920, 3), dtype=np.uint8)
                
            # If no debug view, just return the padded frame
            if debug is None:
                return self._pad_to_1920x1080(frame)
                
            # Get frame dimensions
            frame_h, frame_w = frame.shape[:2]
            debug_h, debug_w = debug.shape[:2]
            
            # Ensure debug view is the same height as frame
            if debug_h != frame_h:
                # Resize debug view to match frame height
                debug = cv2.resize(debug, (int(debug_w * frame_h / debug_h), frame_h))
                debug_h, debug_w = debug.shape[:2]
                
            # Create combined view buffer if needed or size has changed
            if not hasattr(self, 'combined_buffer') or self.combined_buffer is None or self.combined_buffer.shape[:2] != (frame_h, frame_w + debug_w):
                self.combined_buffer = np.zeros((frame_h, frame_w + debug_w, 3), dtype=np.uint8)
                
            # Store dimensions for padding calculations
            self.x_padding_offset = (1920 - (frame_w + debug_w)) // 2
            self.y_padding_offset = (1080 - frame_h) // 2
                
            # Copy debug view to left side and frame to right side
            self.combined_buffer[:, :debug_w] = debug
            self.combined_buffer[:, debug_w:] = frame
            
            # Draw streaming indicator (red circle) if streaming
            if self.is_streaming:
                # Always show the streaming indicator when streaming is active
                cv2.circle(self.combined_buffer, (30, 30), 15, (0, 0, 255), -1)
            
            # Pad the combined buffer to 1920x1080
            padded_combined = self._pad_to_1920x1080(self.combined_buffer)

            # Only draw avoid box if actively drawing to save processing time
            if self.drawing_avoid_box and self.current_avoid_box:
                x, y, w, h = self.current_avoid_box

                # Draw on the right half (main view) with adjusted coordinates
                cv2.rectangle(padded_combined,
                             (x + 939 + self.x_padding_offset, y + self.y_padding_offset),
                             (x + w + 939 + self.x_padding_offset, y + h + self.y_padding_offset),
                             AVOID_BOX_COLOR, AVOID_BOX_THICKNESS)

            return padded_combined
            
        except Exception as e:
            # Try to return just the frame if possible
            try:
                if frame is not None:
                    return self._pad_to_1920x1080(frame)
            except:
                pass
                
            # Last resort - return a black frame of the right size
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def _pad_to_1920x1080(self, frame: np.ndarray) -> np.ndarray:
        """
        Pad the frame with black bars to achieve 1920x1080 resolution.
        This preserves the original aspect ratio without distortion.
        """
        try:
            padded_start = time.time()
            
            if frame is None:
                self.logger.log_error("_pad_to_1920x1080 received None for frame")
                return np.zeros((1080, 1920, 3), dtype=np.uint8)
                
            h, w = frame.shape[:2]
            
            # Create a black canvas of 1920x1080 - reuse buffer if possible
            if not hasattr(self, 'padded_buffer') or self.padded_buffer is None or self.padded_buffer.shape != (1080, 1920, 3):
                self.padded_buffer = np.zeros((1080, 1920, 3), dtype=np.uint8)
            else:
                # Clear the buffer instead of creating a new one
                self.padded_buffer.fill(0)
            
            # Calculate centering offsets
            x_offset = (1920 - w) // 2
            y_offset = (1080 - h) // 2
            
            # Store offsets for later use
            self.x_padding_offset = x_offset
            self.y_padding_offset = y_offset
            
            # Copy the frame to the center of the canvas
            self.padded_buffer[y_offset:y_offset+h, x_offset:x_offset+w] = frame
            
            # Log the time taken for padding
            #self.logger.log_operation_time('padded_view', time.time() - padded_start)
            
            return self.padded_buffer
            
        except Exception as e:
            self.logger.log_error(f"Error in _pad_to_1920x1080: {str(e)}")
                
            # Last resort - return a black frame of the right size
            try:
                return np.zeros((1080, 1920, 3), dtype=np.uint8)
            except:
                return None

    def update_stream(self, combined_frame: np.ndarray) -> None:
        """Update the RTMP stream with the latest frame."""
        if not self.is_streaming or self.stream_writer is None:
            return
            
        try:
            # Write the frame directly to the stream since it's already padded
            self.stream_writer.write(combined_frame)
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_error(f"Error updating stream: {str(e)}")

    def show_frame(self, frame: np.ndarray, rate_limit: int = 1) -> int:
        """
        Display the frame in the window, optionally limiting update rate.
        
        Args:
            frame: Frame to display
            rate_limit: Only update display every rate_limit frames (1=every frame)
        
        Returns:
            Key code from pollKey
        """
        # Simple frame counter for rate limiting
        if not hasattr(self, 'display_counter'):
            self.display_counter = 0
        self.display_counter += 1
        
        # Only update display if we're on the right frame count
        if rate_limit <= 1 or self.display_counter % rate_limit == 0:
            cv2.imshow('ISS Object Detection', frame)
        
        # Always check for key input regardless of display update
        return cv2.pollKey() & 0xFF

    def cleanup(self):
        """Clean up any display resources."""
        cv2.destroyAllWindows()

    def draw_space_boxes_on_padded(self, padded_frame: np.ndarray, detections: DetectionResults) -> None:
        """
        Draw space boxes on the padded frame with adjusted coordinates.
        This ensures space boxes are visible on the 1920x1080 padded frame.
        """
        if not hasattr(self, 'x_padding_offset') or not hasattr(self, 'y_padding_offset'):
            return
            
        # Check if we have space boxes
        if 'space' not in detections.rcnn_boxes or not detections.rcnn_boxes['space']:
            return
            
        # Determine which boxes to use (combined or raw)
        if 'all_space_boxes' in detections.metadata and detections.metadata['all_space_boxes']:
            boxes = detections.metadata['all_space_boxes']
            multi_region = len(boxes) > 1
        else:
            boxes = detections.rcnn_boxes['space']
            multi_region = False
            
        color = CLASS_COLORS['space']
        
        # Draw space contours on the padded frame
        if hasattr(detections, 'space_contours') and detections.space_contours:
            # Adjust contours for padding
            adjusted_contours = []
            for contour in detections.space_contours:
                # Create a copy of the contour
                adjusted_contour = contour.copy()
                # Adjust x and y coordinates
                adjusted_contour[:, :, 0] += self.x_padding_offset + 939  # Add offset and shift to right half
                adjusted_contour[:, :, 1] += self.y_padding_offset
                adjusted_contours.append(adjusted_contour)
                
            # Draw the adjusted contours with appropriate style
            contour_color = (0, 200, 255) if multi_region else color
            contour_thickness = 2 if multi_region else 1
            cv2.drawContours(padded_frame, adjusted_contours, -1, contour_color, contour_thickness)
            
            # For multi-region, add a label and skip drawing boxes
            if multi_region:
                if boxes:
                    x1, y1 = map(int, boxes[0][:2])
                    padded_x1 = x1 + self.x_padding_offset + 939
                    padded_y1 = y1 + self.y_padding_offset
                    
                    label_text = "space: multi-region"
                    cv2.putText(padded_frame, label_text, 
                              (padded_x1 + 5, padded_y1 + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, contour_color, 1)
                return  # Skip drawing individual boxes
        
        # Only draw individual boxes for single regions
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            # Adjust coordinates for padding
            padded_x1 = x1 + self.x_padding_offset + 939
            padded_y1 = y1 + self.y_padding_offset
            padded_x2 = x2 + self.x_padding_offset + 939
            padded_y2 = y2 + self.y_padding_offset
            
            # Draw rectangle
            cv2.rectangle(padded_frame, (padded_x1, padded_y1), (padded_x2, padded_y2), color, 2)
            
            # Draw label
            if 'space' in detections.rcnn_scores and i < len(detections.rcnn_scores['space']):
                label_text = f"space: {detections.rcnn_scores['space'][i]:.2f}"
            else:
                label_text = "space"
                
            cv2.putText(padded_frame, label_text, 
                      (padded_x1 + 5, padded_y1 + 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)