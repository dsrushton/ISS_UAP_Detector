"""
Space Object Detection Module
Handles both RCNN model detection and first principles analysis.
"""

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import time
import threading
import queue
import importlib
import math

# Import all constants at module level
import SOD_Constants as const
from SOD_Constants import (
    DEVICE, MODEL_PATH, CLASS_NAMES, CLASS_THRESHOLDS
)

def gpu_crop_frame(frame: np.ndarray, left: int, right: int) -> np.ndarray:
    """
    Crop a frame using GPU when CUDA is available, otherwise fallback to CPU.
    
    Args:
        frame: Input frame (numpy array)
        left: Number of pixels to crop from left
        right: Number of pixels to crop from right
        
    Returns:
        Cropped frame
    """
    # Check specifically for OpenCV CUDA support
    has_opencv_cuda = False
    try:
        # Check if cv2.cuda.getCudaEnabledDeviceCount exists and returns at least 1
        has_opencv_cuda = hasattr(cv2, 'cuda') and callable(getattr(cv2.cuda, 'getCudaEnabledDeviceCount', None)) and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        # If any error occurs during check, assume no CUDA support
        has_opencv_cuda = False
    
    # If no OpenCV CUDA support, fall back to CPU cropping
    if not has_opencv_cuda:
        return frame[:, left:-right]
        
    try:
        # Create a GPU Mat from the numpy frame
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # Calculate the right boundary
        width = frame.shape[1]
        right_boundary = width - right
        
        # Use colRange to crop the frame horizontally (much faster than CPU cropping)
        gpu_cropped = gpu_frame.colRange(left, right_boundary)
        
        # Download the result back to CPU
        cropped_frame = gpu_cropped.download()
        
        return cropped_frame
    except Exception as e:
        # Don't print the error - we've already checked for CUDA support
        # but there might be other CUDA-related errors we want to silently handle
        return frame[:, left:-right]

@dataclass
class DetectionResults:
    """Container for detection results."""
    
    def __init__(self, frame_number: int = None, **kwargs):
        """Initialize detection results with optional kwargs."""
        self.rcnn_boxes = kwargs.get('rcnn_boxes', {})      # Dict of class_name -> list of boxes
        self.rcnn_scores = kwargs.get('rcnn_scores', {})    # Dict of class_name -> list of scores
        self.anomalies = kwargs.get('anomalies', [])        # List of (x,y,w,h) tuples
        self.space_box = kwargs.get('space_box', None)      # Space region box
        self.darkness_detected = kwargs.get('darkness_detected', False)  # Track if darkness is detected
        self.metadata = kwargs.get('metadata', {})          # Store anomaly metrics and other metadata
        self.contours = kwargs.get('contours', [])          # Store contours for debug view
        self.frame_number = frame_number                    # Track frame number if provided
        self.space_mask = kwargs.get('space_mask', None)    # Store the space mask for display
        self.space_contours = kwargs.get('space_contours', [])  # Store space contours for display
        
    def add_rcnn_detection(self, class_name: str, box: list, score: float):
        """Add an RCNN detection."""
        if class_name not in self.rcnn_boxes:
            self.rcnn_boxes[class_name] = []
            self.rcnn_scores[class_name] = []
        self.rcnn_boxes[class_name].append(box)
        self.rcnn_scores[class_name].append(score)

    def add_anomalies(self, anomalies: list):
        """Add anomaly detections."""
        self.anomalies = anomalies
        
    def add_metadata(self, metadata: dict):
        """Add metadata to results."""
        self.metadata = metadata

class SpaceObjectDetector:
    """Handles object detection using RCNN and first principles analysis."""
    
    def __init__(self):
        """Initialize detector with frame cycle for RCNN."""
        self.model = None
        self.transform = None
        # Cache constants instead of importing repeatedly
        self.refresh_constants()
        self.frame_count = 0
        self.logger = None
        
        # Initialize with empty results
        self.last_rcnn_results = {
            'boxes': {},
            'scores': {}
        }
        self.last_rcnn_frame = -1

        # Add cached border data
        self.cached_space_contours = None
        self.cached_space_boxes = None
        self.cached_space_mask = None
        self.cached_border_margin = None
        self.last_border_update_frame = -1
        
        # Add cached darkness detection state
        self.cached_darkness_detected = False
        self.cached_darkness_area_ratio = 0.0
        self.last_darkness_update_frame = -1
        
        # Initialize transform
        self.transform = T.Compose([
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.consecutive_lf = 0
        self.lf_pause_until = 0  # Track when to resume after lens flares
        self.is_test_frame = False  # Flag to track if current frame is a test frame

        # Setup RCNN threading
        self.rcnn_queue = queue.Queue(maxsize=1)  # Only keep one pending frame
        self.rcnn_worker_running = True
        self.rcnn_worker_thread = threading.Thread(target=self._rcnn_worker)
        self.rcnn_worker_thread.daemon = True
        self.rcnn_worker_thread.start()

    def refresh_constants(self):
        """Refresh constant values from SOD_Constants."""
        # Core parameters
        self.rcnn_cycle = const.RCNN_DETECTION_CYCLE
        
        # Detection thresholds
        self.filter_iss = const.FILTER_ISS
        self.filter_panel = const.FILTER_PANEL
        self.filter_lf = const.FILTER_LF
        self.max_lens_flares = const.MAX_LENS_FLARES
        self.darkness_area_threshold = const.DARKNESS_AREA_THRESHOLD
        
        # Contour analysis parameters
        self.min_contour_dimension = const.MIN_CONTOUR_DIMENSION
        self.max_contour_width = const.MAX_CONTOUR_WIDTH
        self.max_contour_height = const.MAX_CONTOUR_HEIGHT
        self.min_contour_area = const.MIN_CONTOUR_AREA
        self.max_aspect_ratio = const.MAX_ASPECT_RATIO
        self.border_margin = const.BORDER_MARGIN
        self.max_valid_detections = const.MAX_VALID_DETECTIONS
        self.max_contours_per_frame = const.MAX_CONTOURS_PER_FRAME
        
        # Brightness parameters
        self.min_object_brightness = const.MIN_OBJECT_BRIGHTNESS
        self.max_object_brightness = const.MAX_OBJECT_BRIGHTNESS
        self.max_bg_brightness = const.MAX_BG_BRIGHTNESS
        self.min_contrast = const.MIN_CONTRAST
        self.dark_region_threshold = const.DARK_REGION_THRESHOLD
        
        # Processing parameters
        self.gaussian_blur_size = const.GAUSSIAN_BLUR_SIZE
        self.morph_kernel_size = const.MORPH_KERNEL_SIZE

    def set_logger(self, logger):
        """Set the logger instance."""
        self.logger = logger

    def detect_anomalies(self, frame: np.ndarray, space_boxes: list, avoid_boxes: list = None) -> DetectionResults:
        """Find relevant contour distinctions in space region."""
        try:
            results = DetectionResults()
            
            if not space_boxes or len(space_boxes) == 0:
                return results
                
            # Use cached space contours if available and not too old
            use_cached_data = (
                self.cached_space_contours is not None and 
                self.cached_space_mask is not None and
                self.last_border_update_frame > 0 and
                self.frame_count - self.last_border_update_frame < self.rcnn_cycle
            )
            
            if use_cached_data:
                space_contours = self.cached_space_contours
                space_mask = self.cached_space_mask
                border_margin = self.cached_border_margin
            else:
                # Create a combined space mask for all space boxes if cached data not available
                h, w = frame.shape[:2]
                space_mask = np.zeros((h, w), dtype=np.uint8)
                
                # Ensure space_boxes is a list of boxes, not a single box
                if not isinstance(space_boxes, list):
                    space_boxes = [space_boxes]
                
                # Draw all space boxes on the mask
                for box in space_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(space_mask, (x1, y1), (x2, y2), 255, -1)
                
                # If we have avoid boxes, remove them from the space mask
                if avoid_boxes and len(avoid_boxes) > 0:
                    for box in avoid_boxes:
                        # Check if box is in the correct format (x1, y1, x2, y2)
                        if len(box) == 4:
                            x1, y1, x2, y2 = map(int, box)
                            # Draw the avoid box in black (0) to remove it from the space mask
                            cv2.rectangle(space_mask, (x1, y1), (x2, y2), 0, -1)
                
                # Find external contour of all space regions combined
                space_contours, _ = cv2.findContours(space_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Clear mask and redraw only the external contour
                space_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(space_mask, space_contours, -1, 255, -1)
                
                # Use cached border margin from class attributes
                border_margin = self.border_margin
                
                # Cache the computed values for future frames
                #self.cached_space_contours = space_contours
                self.cached_space_mask = space_mask
                self.cached_border_margin = border_margin
                self.last_border_update_frame = self.frame_count
            
            # Store the space mask and contours in results
            results.space_mask = space_mask
            results.space_contours = space_contours
            
            # Get the bounding box of the combined space region
            x1, y1, w1, h1 = cv2.boundingRect(space_mask)
            x2, y2 = x1 + w1, y1 + h1
            
            # Store the space box in results
            results.space_box = (x1, y1, x2, y2)
            
            # Extract ROI for faster processing
            roi = frame[y1:y2, x1:x2]
            roi_mask = space_mask[y1:y2, x1:x2]
            
            if roi.size == 0 or roi_mask.size == 0:
                return results
                
            # FIXED SCALE FACTOR FOR PERFORMANCE IMPROVEMENT
            # Apply a fixed downscaling to large ROIs
            scale_factor = 0.6  # Process at half resolution
            original_shape = roi.shape[:2]  # Store original shape for later scaling back
            
            # Resize ROI and mask for faster processing
            if roi.shape[0] > 0 and roi.shape[1] > 0:  # Ensure valid dimensions
                roi = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                roi_mask = cv2.resize(roi_mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            
            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                roi_max = np.max(roi, axis=2)  # Use max RGB value for better detection
            else:
                roi_max = roi
            
            # Get blur size and thresholds from class attributes
            blur_size = self.gaussian_blur_size
            dark_threshold = self.dark_region_threshold
            min_brightness = self.min_object_brightness
            morph_size = self.morph_kernel_size

            # --- Hybrid Masking Logic Start ---
            mask_start = time.time()

            # Optional: Gaussian Blur (Uncomment roi_blurred = roi_max line if removing blur)
            # Apply Gaussian Blur to reduce noise before thresholding
            #roi_blurred = cv2.GaussianBlur(roi_max, (blur_size, blur_size), 0)
            roi_blurred = roi_max # Use this line if skipping GaussianBlur

            # 1. Create Dark Mask: Identify initial dark regions
            _, dark_mask_initial = cv2.threshold(roi_blurred, dark_threshold, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((morph_size, morph_size), np.uint8)
            dark_mask_opened = cv2.morphologyEx(dark_mask_initial, cv2.MORPH_OPEN, kernel)
            dark_mask_final = cv2.bitwise_and(dark_mask_opened, roi_mask) # Limit to original space region

            # 2. Create Precise Dark Region Mask from Contours
            dark_contours, _ = cv2.findContours(dark_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dark_region_mask = np.zeros_like(roi_mask) # Start with black canvas
            cv2.drawContours(dark_region_mask, dark_contours, -1, 255, -1) # Fill contours

            # 3. Create Bright Mask: Identify bright pixels
            _, bright_mask_initial = cv2.threshold(roi_blurred, min_brightness, 255, cv2.THRESH_BINARY)
            bright_mask_dilated = cv2.morphologyEx(bright_mask_initial, cv2.MORPH_DILATE, kernel)
            # Note: bright_mask is not limited to roi_mask here

            # 4. Final Search Mask: Find bright pixels within the precise dark regions
            final_search_mask = cv2.bitwise_and(bright_mask_dilated, dark_region_mask)

            # --- Hybrid Masking Logic End ---

            # self.logger.log_operation_time('mask_creation', time.time() - mask_start)

            # Find contours directly on the final search mask
            contour_start = time.time()
            contours, _ = cv2.findContours(final_search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Limit number of contours processed
            if len(contours) > self.max_contours_per_frame:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.max_contours_per_frame]
                results.metadata['max_contours_reached'] = True
            #self.logger.log_operation_time('contour_finding', time.time() - contour_start)
            
            # Store all contours for visualization (will convert to absolute later)
            local_contours = []
            for cont in contours:
                local_contours.append(cont.copy())
            
            anomalies = []
            anomaly_metrics = []
            valid_detections = 0
            
            # Scaling factor for metrics and coordinates to match original image
            scale_back = 1.0 / scale_factor
            
            # Process each contour
            analysis_start = time.time()
            
            for contour in contours:
                # Skip if we've already found max valid detections
                if valid_detections >= self.max_valid_detections:
                    break
                
                # Get contour properties - do this first for ultra-fast rejection
                x, y, w, h = cv2.boundingRect(contour)
                
                # Scale dimensions for comparison with thresholds
                scaled_w = w * scale_back
                scaled_h = h * scale_back
                
                # Check if contour has enough points to be valid
                if len(contour) < 10:  # Minimum 3 points needed for a valid contour
                    continue
                
                # Ultra-fast size check before any other processing - use scaled dimensions
                if (scaled_w < self.min_contour_dimension or scaled_h < self.min_contour_dimension or 
                    scaled_w > self.max_contour_width or scaled_h > self.max_contour_height):
                    continue
                
                # Convert to global coordinates for later use - account for scaling
                abs_x = int((x * scale_back) + x1)
                abs_y = int((y * scale_back) + y1)
                
                # Create binary mask for exact object boundaries
                obj_mask = np.zeros_like(final_search_mask)
                cv2.drawContours(obj_mask, [contour], -1, 255, -1)
                
                # Check if object passes brightness and contrast requirements first
                is_valid, metrics = self.analyze_object(roi_max, obj_mask, w, h, contour)
                if not is_valid:
                    continue
                
                # Scale metrics back to original dimensions
                if 'width' in metrics:
                    metrics['width'] *= scale_back
                if 'height' in metrics:
                    metrics['height'] *= scale_back
                if 'area' in metrics:
                    metrics['area'] *= (scale_back * scale_back)
                
                # Skip if contrast is too low or negative
                if metrics['contrast'] < self.min_contrast:
                    continue
                
                # Only now check if the contour is too close to the border
                # This is more expensive but we do it only for valid detections
                too_close_to_border = False
                
                # Get bounding rect of this contour in global coordinates
                # We already have x, y, w, h from above
                cx_global = abs_x
                cy_global = abs_y
                
                for space_contour in space_contours:
                    # Get bounding rect of space contour
                    sx, sy, sw, sh = cv2.boundingRect(space_contour)
                    
                    # Calculate minimum distance between boxes
                    # If boxes are far apart, the contour can't be close to the border
                    min_dist_x = max(0, max(cx_global - (sx + sw), sx - (cx_global + int(w * scale_back))))
                    min_dist_y = max(0, max(cy_global - (sy + sh), sy - (cy_global + int(h * scale_back))))
                    min_dist = math.sqrt(min_dist_x**2 + min_dist_y**2)
                    
                    # If minimum distance is greater than border margin, definitely not too close
                    if min_dist > border_margin:
                        continue
                    
                    # Use inset rectangle with border margin to check if contour is too close to border
                    # Create inset space box by applying border_margin to all sides
                    inset_x1 = sx + border_margin
                    inset_y1 = sy + border_margin
                    inset_x2 = (sx + sw) - border_margin
                    inset_y2 = (sy + sh) - border_margin
                    
                    # OPTIMIZATION: Instead of expensive point-by-point check, we create an inset
                    # rectangle shrunk by BORDER_MARGIN on all sides. Any contour that's not completely
                    # inside this inset rectangle is considered too close to the border.
                    # This is much faster than checking the distance of each contour point to the space border.
                    
                    # Skip if inset rectangle is invalid (too small)
                    if inset_x1 >= inset_x2 or inset_y1 >= inset_y2:
                        too_close_to_border = True
                        break
                    
                    # Check if contour is entirely within the inset rectangle
                    # If not, it's too close to the border
                    scaled_w_int = int(w * scale_back)
                    scaled_h_int = int(h * scale_back)
                    if not (cx_global >= inset_x1 and (cx_global + scaled_w_int) <= inset_x2 and
                            cy_global >= inset_y1 and (cy_global + scaled_h_int) <= inset_y2):
                        too_close_to_border = True
                        break
                
                if too_close_to_border:
                    continue
                
                # Only collect filter boxes if we have a potentially valid detection
                filter_boxes = []
                
                # Add avoid boxes
                if avoid_boxes:
                    filter_boxes.extend(avoid_boxes)
                
                # Add RCNN boxes for filtering (lf, iss, panel)
                for box_type in ['lf', 'iss', 'panel']:
                    if box_type in self.last_rcnn_results['boxes']:
                        # Check if this box type should be filtered
                        if ((box_type == 'iss' and not self.filter_iss) or
                            (box_type == 'panel' and not self.filter_panel) or
                            (box_type == 'lf' and not self.filter_lf)):
                            continue
                        boxes = self.last_rcnn_results['boxes'][box_type]
                        filter_boxes.extend(boxes)
                
                # Check if anomaly overlaps with any filtering box (in global coordinates)
                overlaps = False
                scaled_w_int = int(w * scale_back)
                scaled_h_int = int(h * scale_back)
                for i, (fx1, fy1, fx2, fy2) in enumerate(filter_boxes):
                    if (abs_x < fx2 and abs_x + scaled_w_int > fx1 and
                        abs_y < fy2 and abs_y + scaled_h_int > fy1):
                        overlaps = True
                        break
                
                if overlaps:
                    continue
                
                # Store metrics for this anomaly
                metrics['position'] = (abs_x, abs_y)  # Already in global coordinates
                metrics['width'] = scaled_w_int
                metrics['height'] = scaled_h_int
                metrics['area'] = scaled_w_int * scaled_h_int
                
                anomalies.append((abs_x, abs_y, scaled_w_int, scaled_h_int))
                anomaly_metrics.append(metrics)
                valid_detections += 1
            
            #self.logger.log_operation_time('contour_analysis', time.time() - analysis_start)
            
            # Now convert contours to absolute coordinates after all analysis is done
            global_contours = []
            for cont in local_contours:
                # Create a copy of the contour and scale it back to original size
                global_cont = cont.copy().astype(np.float32)
                global_cont = global_cont * scale_back
                
                # Convert back to integer and shift to global coordinates
                global_cont = global_cont.astype(np.int32)
                global_cont[:, :, 0] += x1
                global_cont[:, :, 1] += y1
                global_contours.append(global_cont)
            
            # Store the global contours in the results
            results.contours = global_contours
            
            # Store results
            results.metadata['anomaly_metrics'] = anomaly_metrics
            results.metadata['total_contours'] = len(contours)
            results.metadata['valid_detections'] = valid_detections
            results.add_anomalies(anomalies)
            
            return results
            
        except Exception as e:
            print(f"Error in detect_anomalies: {e}")
            return DetectionResults()
    
    def analyze_object(self, frame: np.ndarray, binary_mask: np.ndarray, w: int, h: int, contour: np.ndarray = None) -> bool:
        """Analyze if object passes detection criteria"""
        try:
            # ULTRA-FAST SIZE REJECTION - Check dimensions first before any other operations
            # These are the most common rejection criteria and fastest to check
            if (w < self.min_contour_dimension or h < self.min_contour_dimension or 
                w > self.max_contour_width or h > self.max_contour_height):
                return False, {'width': w, 'height': h}
            
            # Use provided contour if available, otherwise find it
            if contour is None:
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return False, {}
                contour = max(contours, key=cv2.contourArea)
            
            # Calculate area and aspect ratio - still relatively fast operations
            area = cv2.contourArea(contour)
            aspect_ratio = float(w)/h if h != 0 else 0
            
            # Create basic metrics dict with just the fast calculations
            metrics = {
                'area': area,
                'aspect_ratio': aspect_ratio,
                'width': w,
                'height': h
            }
            
            # SECOND LEVEL REJECTION - Area and aspect ratio
            if area < self.min_contour_area:
                return False, metrics
            
            if aspect_ratio > self.max_aspect_ratio or aspect_ratio < (1/self.max_aspect_ratio):
                return False, metrics
            
            # THIRD LEVEL - More expensive operations only if we passed the quick checks
            
            # Use the original binary mask instead of creating a new one
            # This ensures the mask dimensions match the frame dimensions
            mask = binary_mask.copy()
            
            # Convert frame to max RGB if not already grayscale - do this before creating bg_mask
            if len(frame.shape) == 3:
                frame_max = np.max(frame, axis=2)  # Use max RGB value
            else:
                frame_max = frame
            
            # Get object brightness first - this is faster than creating the bg_mask
            obj_brightness = cv2.mean(frame_max, mask=mask)[0]
            
            # FOURTH LEVEL REJECTION - Object brightness check before creating bg_mask
            if not (self.min_object_brightness < obj_brightness < self.max_object_brightness):
                metrics['obj_brightness'] = obj_brightness
                return False, metrics
            
            # Now create the background mask for contrast check
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            bg_mask = cv2.dilate(mask, kernel, iterations=2)
            bg_mask = cv2.bitwise_xor(bg_mask, mask)  # Creates ring around object
            
            # Get background brightness
            bg_brightness = cv2.mean(frame_max, mask=bg_mask)[0]
            contrast = obj_brightness - bg_brightness
            
            # Update metrics with brightness values
            metrics.update({
                'obj_brightness': obj_brightness,
                'bg_brightness': bg_brightness,
                'contrast': contrast
            })
            
            # FINAL CHECKS - Background brightness and contrast
            if bg_brightness > self.max_bg_brightness:
                return False, metrics
                
            if contrast < self.min_contrast:
                return False, metrics
            
            return True, metrics
            
        except Exception as e:
            print(f"Error in analyze_object: {e}")
            return False, {}

    def _combine_overlapping_boxes(self, boxes: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        """
        Combine overlapping space boxes into a set of non-overlapping boxes.
        
        Instead of merging boxes into a single larger box, we maintain the original boxes
        but remove any interior borders when drawing the mask, creating multi-faced polygons.
        
        Args:
            boxes: List of (x1, y1, x2, y2) boxes
            
        Returns:
            List of original boxes that will be treated as a combined polygon
        """
        if not boxes or len(boxes) <= 1:
            return boxes
            
        # For multi-faced polygons, we don't need to modify the boxes themselves
        # They should be passed as-is to the mask creation function that will handle
        # drawing them without internal borders
        return boxes

    def _update_darkness_state(self, frame: np.ndarray) -> None:
        """Update cached darkness detection state based on current frame and RCNN results."""
        darkness_start = time.time()
        self.cached_darkness_detected = False
        self.cached_darkness_area_ratio = 0.0
        
        # Check for darkness using 'td' boxes from RCNN results
        if 'td' in self.last_rcnn_results['boxes']:
            frame_area = frame.shape[0] * frame.shape[1]
            for box in self.last_rcnn_results['boxes']['td']:
                x1, y1, x2, y2 = box
                td_area = (x2 - x1) * (y2 - y1)
                td_ratio = td_area / frame_area
                if td_ratio > self.darkness_area_threshold:
                    self.cached_darkness_detected = True
                    self.cached_darkness_area_ratio = td_ratio
                    break
        
        # Update the last update frame number
        self.last_darkness_update_frame = self.frame_count
        
        # Log timing for darkness detection update
        #if self.logger:
         #   self.logger.log_operation_time('darkness_update', time.time() - darkness_start)

    def process_frame(self, frame: np.ndarray, avoid_boxes: List[Tuple[int, int, int, int]] = None, is_test_frame: bool = False, crop_left: int = None, crop_right: int = None) -> Optional[DetectionResults]:
        """Process frame through detection pipeline."""
        try:
            self.frame_count += 1
            is_rcnn_frame = (self.frame_count % self.rcnn_cycle) == 0
            results = DetectionResults(frame_number=self.frame_count)
            
            # Store test frame status
            self.is_test_frame = is_test_frame
            results.metadata['test_frame'] = is_test_frame
            
            # Get crop values if not provided
            if crop_left is None:
                crop_left = const.get_value('CROP_LEFT')
            if crop_right is None:
                crop_right = const.get_value('CROP_RIGHT')
            
            # Run RCNN on regular cycle OR if this is a test frame
            if is_rcnn_frame or is_test_frame:
                # Instead of running RCNN synchronously, push it to the queue
                try:
                    # Only send new frames if queue isn't full
                    if not self.rcnn_queue.full():
                        if not is_test_frame:
                            # For non-test frames, send the frame with crop parameters
                            self.rcnn_queue.put_nowait((frame, crop_left, crop_right))
                        else:
                            # Test frames are already cropped
                            self.rcnn_queue.put_nowait(frame)
                        results.metadata['rcnn_frame'] = True
                    else:
                        results.metadata['rcnn_frame'] = False
                        results.metadata['rcnn_queue_full'] = True
                except Exception as e:
                    print(f"Error queueing frame: {e}")
                    results.metadata['rcnn_frame'] = False
            
            # Use the latest available RCNN results
            results.rcnn_boxes = self.last_rcnn_results.get('boxes', {})
            results.rcnn_scores = self.last_rcnn_results.get('scores', {})
            
            # Check for darkness - use cached values or update if RCNN just ran or it's the first time
            use_cached_darkness = (
                self.last_darkness_update_frame > 0 and
                self.frame_count - self.last_darkness_update_frame < self.rcnn_cycle
            )
            
            if use_cached_darkness:
                # Use cached darkness detection state
                results.darkness_detected = self.cached_darkness_detected
                if self.cached_darkness_detected:
                    results.metadata['darkness_area_ratio'] = self.cached_darkness_area_ratio
            else:
                # Update darkness detection state
                if not is_test_frame:
                    # Crop frame for darkness detection
                    cropped_frame = gpu_crop_frame(frame, crop_left, crop_right)
                    self._update_darkness_state(cropped_frame)
                else:
                    # Test frames are already cropped
                    self._update_darkness_state(frame)
                
                # Use newly calculated values
                results.darkness_detected = self.cached_darkness_detected
                if self.cached_darkness_detected:
                    results.metadata['darkness_area_ratio'] = self.cached_darkness_area_ratio
            
            # Quick exit for darkness or no feed cases
            if results.darkness_detected or 'nofeed' in results.rcnn_boxes:
                return results
            
            # Check lens flare count
            if 'lf' in self.last_rcnn_results['boxes']:
                lf_count = len(self.last_rcnn_results['boxes']['lf'])
                if lf_count >= self.max_lens_flares:
                    results.metadata['skip_save'] = 'too_many_lens_flares'
            
            # Only proceed with anomaly detection if we have a space region
            if 'space' in results.rcnn_boxes and results.rcnn_boxes['space']:
                # Get all space boxes and combine overlapping ones
                space_boxes = self._combine_overlapping_boxes(results.rcnn_boxes['space'])
                
                # Skip processing if no valid space boxes
                if not space_boxes:
                    return results
                    
                # Sort combined boxes by y1 coordinate (highest first)
                space_boxes = sorted(space_boxes, key=lambda box: box[1])
                
                # Store the combined box for display and processing
                if space_boxes:
                    results.space_box = space_boxes[0]  # Main combined box for display
                    results.metadata['all_space_boxes'] = space_boxes  # Store all boxes for reference
                
                    # Update cached border data if this is an RCNN frame or if boxes have changed
                    space_boxes_changed = not self._are_space_boxes_same(space_boxes)
                    
                    if is_rcnn_frame or is_test_frame or space_boxes_changed:
                        # Crop frame if needed
                        if not is_test_frame:
                            cropped_frame = gpu_crop_frame(frame, crop_left, crop_right)
                            self._update_cached_border_data(cropped_frame, space_boxes)
                        else:
                            self._update_cached_border_data(frame, space_boxes)
                        self.last_border_update_frame = self.frame_count
                    
                    # Track all anomalies and metrics
                    all_anomalies = []
                    all_metrics = []
                    
                    # Process the main combined space box using cached border data
                    if not is_test_frame:
                        # Crop frame for detect_anomalies
                        cropped_frame = gpu_crop_frame(frame, crop_left, crop_right)
                        box_results = self.detect_anomalies(cropped_frame, space_boxes, avoid_boxes)
                    else:
                        # Test frames are already cropped
                        box_results = self.detect_anomalies(frame, space_boxes, avoid_boxes)
                    
                    # Store all results from the combined box
                    if box_results.anomalies:
                        all_anomalies.extend(box_results.anomalies)
                        if 'anomaly_metrics' in box_results.metadata:
                            all_metrics.extend(box_results.metadata['anomaly_metrics'])
                    
                    # Store contours from the combined box
                    results.contours = box_results.contours
                    
                    # Store space mask and contours from the combined box
                    results.space_mask = box_results.space_mask
                    results.space_contours = box_results.space_contours
                    
                    # Only process additional space boxes if:
                    # 1. There's more than one space box
                    # 2. No anomalies were found yet (optimization)
                    if len(space_boxes) > 1 and not all_anomalies:
                        # Process additional space boxes if they don't overlap with the main one
                        for space_box in space_boxes[1:]:
                            # Check if this box overlaps with the main box
                            main_x1, main_y1, main_x2, main_y2 = results.space_box
                            x1, y1, x2, y2 = space_box
                            
                            # If boxes don't overlap, process this one too
                            if (x2 < main_x1 or x1 > main_x2 or 
                                y2 < main_y1 or y1 > main_y2):
                                # Pass the space box as a list to avoid 'numpy.int32' object is not iterable error
                                if not is_test_frame:
                                    # Crop frame for detect_anomalies
                                    cropped_frame = gpu_crop_frame(frame, crop_left, crop_right)
                                    box_results = self.detect_anomalies(cropped_frame, [space_box], avoid_boxes)
                                else:
                                    # Test frames are already cropped
                                    box_results = self.detect_anomalies(frame, [space_box], avoid_boxes)
                                
                                if box_results.anomalies:
                                    all_anomalies.extend(box_results.anomalies)
                                    if 'anomaly_metrics' in box_results.metadata:
                                        all_metrics.extend(box_results.metadata['anomaly_metrics'])
                    
                    # Update final results
                    results.anomalies = all_anomalies
                    results.metadata['anomaly_metrics'] = all_metrics
            
            return results
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup(self):
        """Clean up resources."""
        # Stop the worker thread
        self.rcnn_worker_running = False
        if hasattr(self, 'rcnn_worker_thread'):
            self.rcnn_worker_thread.join(timeout=1.0)
            
        # Clear model
        self.model = None

    def initialize_model(self) -> bool:
        """Initialize the RCNN model."""
        try:
            # Initialize RCNN model
            self.model = fasterrcnn_resnet50_fpn(weights=None)
            
            # Update box predictor for our number of classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES))
            
            # Load model weights
            self.model.to(DEVICE)
            self.model.load_state_dict(torch.load(
                MODEL_PATH,
                map_location=DEVICE,
                weights_only=True
            ))
            self.model.eval()
            
            print(f"Model initialized on {DEVICE}")
            print(f"Number of classes: {len(CLASS_NAMES)}")
            print(f"Class names: {CLASS_NAMES}")
            
            return True
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False

    def _run_rcnn_detection(self, frame: np.ndarray, is_gpu_tensor: bool = False) -> Dict:
        """Run RCNN detection on a frame."""
        start = time.time()
        
        # Log initial memory state
        #self.logger.log_memory_usage()
        
        if is_gpu_tensor:
            # Frame is already a GPU tensor, use it directly
            img_tensor = frame
        else:
            # Convert BGR numpy array directly to tensor (HWC -> CHW)
            img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            img_tensor = self.transform(img_tensor).unsqueeze(0).to(DEVICE)
        #self.logger.log_operation_time('rcnn_prep', time.time() - start)
        
        # Log memory after tensor creation
        # self.logger.log_memory_usage()
        
        # Run detection
        inference_start = time.time()
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
        #self.logger.log_operation_time('rcnn_inference', time.time() - inference_start)
        
        # Log memory after inference
        #self.logger.log_memory_usage()
        
        # Process predictions
        postprocess_start = time.time()
        # Keep tensors on GPU - don't convert to CPU
        boxes = predictions['boxes']
        labels = predictions['labels']
        scores = predictions['scores']

        
        
        results = {'boxes': {}, 'scores': {}}
        
        # Process all predictions together
        for i in range(len(boxes)):
            box = boxes[i].tolist()  # Convert individual tensor to list when needed
            label = labels[i].item()  # Get scalar value from tensor
            score = scores[i].item()  # Get scalar value from tensor
            
            class_name = CLASS_NAMES[label]
            threshold = CLASS_THRESHOLDS.get(class_name, 0.5)
            
            if score > threshold:
                if class_name not in results['boxes']:
                    results['boxes'][class_name] = []
                    results['scores'][class_name] = []
                    
                results['boxes'][class_name].append(box)
                results['scores'][class_name].append(score)
        
        #self.logger.log_operation_time('rcnn_postprocess', time.time() - postprocess_start)
        
        # We're keeping everything on GPU, so don't explicitly clean up
        # Just delete Python references to allow garbage collection when needed
        if not is_gpu_tensor:
            del img_tensor
        del predictions
        torch.cuda.empty_cache()  
        
        # Log final memory state
        #self.logger.log_memory_usage()
        
        return results

    def _rcnn_worker(self):
        """Background worker thread to run RCNN inference without blocking the main loop."""
        import time
        last_refresh_time = 0
        refresh_interval = 5.0  # Refresh constants every 5 seconds
        
        # Check for PyTorch CUDA
        use_gpu_model = torch.cuda.is_available()
        if use_gpu_model:
            print("Using GPU for RCNN model inference")
        
        # Check specifically for OpenCV CUDA support
        has_opencv_cuda = False
        try:
            # Check if cv2.cuda.getCudaEnabledDeviceCount exists and returns at least 1
            has_opencv_cuda = hasattr(cv2, 'cuda') and callable(getattr(cv2.cuda, 'getCudaEnabledDeviceCount', None)) and cv2.cuda.getCudaEnabledDeviceCount() > 0
            if has_opencv_cuda:
                print("Using OpenCV CUDA for frame cropping")
            else:
                print("OpenCV CUDA not available, using CPU cropping")
        except Exception:
            # If any error occurs during check, assume no CUDA support
            has_opencv_cuda = False
            print("Error checking OpenCV CUDA support, using CPU cropping")
        
        while self.rcnn_worker_running:
            try:
                # Get the current time
                current_time = time.time()
                
                # Refresh constants periodically
                if current_time - last_refresh_time > refresh_interval:
                    self.refresh_constants()
                    last_refresh_time = current_time
                
                # Get the next frame from the queue
                try:
                    frame_data = self.rcnn_queue.get(timeout=1)
                    if isinstance(frame_data, tuple) and len(frame_data) == 3:
                        # If we receive a tuple, it contains (frame, left, right)
                        frame, left, right = frame_data
                        
                        # Use GPU cropping only if OpenCV CUDA is available
                        if has_opencv_cuda:
                            cropped_frame = gpu_crop_frame(frame, left, right)
                            predictions = self._run_rcnn_detection(cropped_frame)
                        else:
                            # Fall back to CPU cropping
                            cropped_frame = frame[:, left:-right]
                            predictions = self._run_rcnn_detection(cropped_frame)
                    else:
                        # Backward compatibility - just a frame with no crop info
                        frame = frame_data
                        predictions = self._run_rcnn_detection(frame)
                except queue.Empty:
                    continue
                
                if frame_data is None:
                    break

                # Store results
                self.last_rcnn_results = predictions
                self.last_rcnn_frame = self.frame_count
                self.rcnn_queue.task_done()
                
                # Update darkness detection state when RCNN results are refreshed
                # Only do this in the worker thread to avoid blocking the main thread
                if isinstance(frame_data, tuple) and len(frame_data) == 3:
                    # Use the cropped frame for darkness detection
                    self._update_darkness_state(cropped_frame)
                else:
                    # Use the original frame
                    self._update_darkness_state(frame)
                
                # We're keeping everything on GPU, so don't explicitly clean up
                # Just delete Python references to allow garbage collection when needed
            except Exception as e:
                print(f"Error in RCNN worker thread: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)  # Prevent tight error loop

    def set_rcnn_cycle(self, fps: int):
        """Update the RCNN detection cycle based on frame rate."""
        # Refresh all constants
        self.refresh_constants()
        # RCNN cycle is now cached as self.rcnn_cycle

    def force_rcnn_detection(self, frame: np.ndarray) -> None:
        """Force RCNN detection on a specific frame (for test frames)."""
        try:
            # Push frame to RCNN queue with high priority - no need to copy as the frame won't be modified
            if self.rcnn_queue.empty():
                self.rcnn_queue.put_nowait(frame)
            else:
                # Clear queue and add new frame
                try:
                    self.rcnn_queue.get_nowait()  # Remove existing frame
                    self.rcnn_queue.put_nowait(frame)
                except queue.Empty:
                    pass
        except Exception as e:
            print(f"Error forcing RCNN detection: {e}")

    def _are_space_boxes_same(self, new_boxes):
        """Compare new space boxes with cached ones to determine if we need to recompute border data."""
        if self.cached_space_boxes is None or len(new_boxes) != len(self.cached_space_boxes):
            return False
            
        # Compare boxes - allow small tolerance for rounding errors
        for i, new_box in enumerate(new_boxes):
            old_box = self.cached_space_boxes[i]
            
            # Check if any corner differs by more than 2 pixels
            if (abs(new_box[0] - old_box[0]) > 2 or 
                abs(new_box[1] - old_box[1]) > 2 or 
                abs(new_box[2] - old_box[2]) > 2 or 
                abs(new_box[3] - old_box[3]) > 2):
                return False
                
        return True
        
    def _update_cached_border_data(self, frame, space_boxes):
        """Update cached border data for efficient border filtering."""
        # Store boxes for future comparison
        self.cached_space_boxes = space_boxes.copy()
        
        # Cache border margin since it's needed for border filtering
        self.cached_border_margin = self.border_margin
        
        # Create a fresh space mask - this is needed for creating space contours
        h, w = frame.shape[:2]
        space_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw all space boxes on the mask - keeping original boxes
        for box in space_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(space_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Find external contours directly - this properly excludes internal borders
        external_contours, _ = cv2.findContours(space_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Clear mask and redraw only the external contours
        space_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(space_mask, external_contours, -1, 255, -1)
        
        # Store the space mask and contours for reuse
        self.cached_space_mask = space_mask
        self.cached_space_contours = external_contours

    def crop_and_prepare_for_rcnn(self, frame: np.ndarray, left: int, right: int) -> torch.Tensor:
        """
        Crop frame on GPU and prepare it for RCNN processing.
        
        Args:
            frame: Input frame
            left: Pixels to crop from left
            right: Pixels to crop from right
            
        Returns:
            PyTorch tensor ready for RCNN inference
        """
        # Check specifically for OpenCV CUDA support
        has_opencv_cuda = False
        try:
            # Check if cv2.cuda.getCudaEnabledDeviceCount exists and returns at least 1
            has_opencv_cuda = hasattr(cv2, 'cuda') and callable(getattr(cv2.cuda, 'getCudaEnabledDeviceCount', None)) and cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            # If any error occurs during check, assume no CUDA support
            has_opencv_cuda = False
        
        # Fallback to CPU processing if no OpenCV CUDA support
        if not has_opencv_cuda:
            cropped = frame[:, left:-right]
            img_tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0
            img_tensor = self.transform(img_tensor).unsqueeze(0).to(DEVICE)
            return img_tensor
        
        try:
            # Crop on GPU
            cropped = gpu_crop_frame(frame, left, right)
            
            # Convert to tensor and prepare for RCNN
            img_tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0
            img_tensor = self.transform(img_tensor).unsqueeze(0).to(DEVICE)
            return img_tensor
        except Exception:
            # Silently fall back to CPU processing without error messages
            cropped = frame[:, left:-right]
            img_tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0
            img_tensor = self.transform(img_tensor).unsqueeze(0).to(DEVICE)
            return img_tensor

class FastRCNNPredictor(torch.nn.Module):
    """Simple FastRCNN prediction head."""
    
    def __init__(self, in_channels: int, num_classes: int):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = torch.nn.Linear(in_channels, num_classes)
        self.bbox_pred = torch.nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas