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

# Only import non-changing constants at module level
from SOD_Constants import (
    DEVICE, MODEL_PATH, CLASS_NAMES, CLASS_THRESHOLDS
)

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
        # Get latest constants
        import SOD_Constants as const
        self.rcnn_cycle = const.RCNN_DETECTION_CYCLE  # Use dynamic constant
        self.frame_count = 0
        self.logger = None
        
        # Initialize with empty results
        self.last_rcnn_results = {
            'boxes': {},
            'scores': {}
        }
        self.last_rcnn_frame = -1
        
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

    def set_logger(self, logger):
        """Set the logger instance."""
        self.logger = logger

    def detect_anomalies(self, frame: np.ndarray, space_boxes: list, avoid_boxes: list = None) -> DetectionResults:
        """Find relevant contour distinctions in space region."""
        try:
            results = DetectionResults()
            
            if not space_boxes or len(space_boxes) == 0:
                return results
                
            # Get latest constants
            import SOD_Constants as const
            
            # Create a combined space mask for all space boxes
            h, w = frame.shape[:2]
            space_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Draw all space boxes on the mask
            for box in space_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(space_mask, (x1, y1), (x2, y2), 255, -1)
            
            # Find external contour of all space regions combined
            space_contours, _ = cv2.findContours(space_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Clear mask and redraw only the external contour
            space_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(space_mask, space_contours, -1, 255, -1)
            
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
                
            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                roi_max = np.max(roi, axis=2)  # Use max RGB value for better detection
            else:
                roi_max = roi
                
            # SIMPLIFIED APPROACH: Focus only on bright objects for faster processing
            mask_start = time.time()
            blur_size = const.get_value('GAUSSIAN_BLUR_SIZE')
            blurred = cv2.GaussianBlur(roi_max, (blur_size, blur_size), 0)
            
            # Create only the bright mask - skip dark mask and contrast mask
            _, bright_mask = cv2.threshold(blurred, const.get_value('MIN_OBJECT_BRIGHTNESS'), 255, cv2.THRESH_BINARY)
            
            # Limit to space region
            bright_mask = cv2.bitwise_and(bright_mask, roi_mask)
            
            # Apply morphological operations
            morph_size = const.get_value('MORPH_KERNEL_SIZE')
            kernel = np.ones((morph_size, morph_size), np.uint8)
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_DILATE, kernel)
            
            self.logger.log_operation_time('mask_creation', time.time() - mask_start)
            
            # Find contours directly on the bright mask
            contour_start = time.time()
            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Limit number of contours processed
            max_contours = const.get_value('MAX_CONTOURS_PER_FRAME')
            if len(contours) > max_contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]
                results.metadata['max_contours_reached'] = True
            self.logger.log_operation_time('contour_finding', time.time() - contour_start)
            
            # Store all contours for visualization (will convert to absolute later)
            local_contours = []
            for cont in contours:
                local_contours.append(cont.copy())
            
            anomalies = []
            anomaly_metrics = []
            valid_detections = 0
            
            # Process each contour
            analysis_start = time.time()
            max_valid = const.get_value('MAX_VALID_DETECTIONS')
            border_margin = const.get_value('BORDER_MARGIN')
            min_contrast = const.get_value('MIN_CONTRAST')
            
            for contour in contours:
                # Skip if we've already found max valid detections
                if valid_detections >= max_valid:
                    break
                
                # Get contour properties - do this first for ultra-fast rejection
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if contour has enough points to be valid
                if len(contour) < 3:  # Minimum 3 points needed for a valid contour
                    continue
                
                # Ultra-fast size check before any other processing
                min_dim = const.get_value('MIN_CONTOUR_DIMENSION')
                max_width = const.get_value('MAX_CONTOUR_WIDTH')
                max_height = const.get_value('MAX_CONTOUR_HEIGHT')
                if (w < min_dim or h < min_dim or w > max_width or h > max_height):
                    continue
                
                # Convert to global coordinates for filtering
                abs_x = x + x1
                abs_y = y + y1
                
                # Check if any part of the contour is too close to space contour
                too_close_to_border = False
                for space_contour in space_contours:
                    # Ensure contour has proper shape and enough points
                    if len(space_contour) < 3:
                        continue
                        
                    # Get contour points and ensure proper shape
                    contour_points = contour.reshape(-1, 2)  # This handles both single and multi-point contours
                    
                    if len(contour_points) < 3:  # Double-check after reshape
                        continue
                    
                    # Check each point against the space contour
                    for point in contour_points:
                        dist = cv2.pointPolygonTest(space_contour, (float(point[0] + x1), float(point[1] + y1)), True)
                        if abs(dist) < border_margin:
                            too_close_to_border = True
                            break
                    if too_close_to_border:
                        break
                
                if too_close_to_border:
                    continue
                
                # Create binary mask for exact object boundaries
                obj_mask = np.zeros_like(bright_mask)
                cv2.drawContours(obj_mask, [contour], -1, 255, -1)
                
                # Only proceed if analyze_object approves this contour
                is_valid, metrics = self.analyze_object(roi_max, obj_mask, w, h, contour)
                if not is_valid:
                    continue
                
                # Skip if contrast is too low or negative
                if metrics['contrast'] < min_contrast:
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
                        if ((box_type == 'iss' and not const.get_value('FILTER_ISS')) or
                            (box_type == 'panel' and not const.get_value('FILTER_PANEL')) or
                            (box_type == 'lf' and not const.get_value('FILTER_LF'))):
                            continue
                        boxes = self.last_rcnn_results['boxes'][box_type]
                        filter_boxes.extend(boxes)
                
                # Check if anomaly overlaps with any filtering box (in global coordinates)
                overlaps = False
                for i, (fx1, fy1, fx2, fy2) in enumerate(filter_boxes):
                    if (abs_x < fx2 and abs_x + w > fx1 and
                        abs_y < fy2 and abs_y + h > fy1):
                        overlaps = True
                        break
                
                if overlaps:
                    continue
                
                # Store metrics for this anomaly
                metrics['position'] = (abs_x, abs_y)  # Already in global coordinates
                metrics['width'] = w
                metrics['height'] = h
                metrics['area'] = w * h
                
                anomalies.append((abs_x, abs_y, w, h))
                anomaly_metrics.append(metrics)
                valid_detections += 1
            
            self.logger.log_operation_time('contour_analysis', time.time() - analysis_start)
            
            # Now convert contours to absolute coordinates after all analysis is done
            global_contours = []
            for cont in local_contours:
                # Create a copy of the contour and shift it to global coordinates
                global_cont = cont.copy()
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
            # Get latest constants
            import SOD_Constants as const
            
            # ULTRA-FAST SIZE REJECTION - Check dimensions first before any other operations
            # These are the most common rejection criteria and fastest to check
            min_dim = const.get_value('MIN_CONTOUR_DIMENSION')
            max_width = const.get_value('MAX_CONTOUR_WIDTH')
            max_height = const.get_value('MAX_CONTOUR_HEIGHT')
            
            if (w < min_dim or h < min_dim or w > max_width or h > max_height):
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
            min_area = const.get_value('MIN_CONTOUR_AREA')
            max_ratio = const.get_value('MAX_ASPECT_RATIO')
            
            if area < min_area:
                return False, metrics
            
            if aspect_ratio > max_ratio or aspect_ratio < (1/max_ratio):
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
            min_obj_bright = const.get_value('MIN_OBJECT_BRIGHTNESS')
            max_obj_bright = const.get_value('MAX_OBJECT_BRIGHTNESS')
            
            if not (min_obj_bright < obj_brightness < max_obj_bright):
                metrics['obj_brightness'] = obj_brightness
                return False, metrics
            
            # Now create the background mask for contrast check
            morph_size = const.get_value('MORPH_KERNEL_SIZE')
            kernel = np.ones((morph_size, morph_size), np.uint8)
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
            max_bg_bright = const.get_value('MAX_BG_BRIGHTNESS')
            min_contrast = const.get_value('MIN_CONTRAST')
            
            if bg_brightness > max_bg_bright:
                return False, metrics
                
            if contrast < min_contrast:
                return False, metrics
            
            return True, metrics
            
        except Exception as e:
            print(f"Error in analyze_object: {e}")
            return False, {}

    def _combine_overlapping_boxes(self, boxes: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        """
        Combine overlapping space boxes into larger boxes.
        
        Args:
            boxes: List of (x1, y1, x2, y2) boxes
            
        Returns:
            List of combined boxes
        """
        if not boxes:
            return []
            
        # Convert to numpy array for easier manipulation
        boxes = np.array(boxes)
        combined_boxes = []
        used = set()
        
        for i, box1 in enumerate(boxes):
            if i in used:
                continue
                
            # Start with current box
            x1, y1, x2, y2 = box1
            used.add(i)
            
            # Keep expanding box while we find overlaps
            changed = True
            while changed:
                changed = False
                for j, box2 in enumerate(boxes):
                    if j in used:
                        continue
                        
                    # Check if boxes overlap
                    ox1 = max(x1, box2[0])
                    oy1 = max(y1, box2[1])
                    ox2 = min(x2, box2[2])
                    oy2 = min(y2, box2[3])
                    
                    if ox1 < ox2 and oy1 < oy2:
                        # Boxes overlap, expand current box
                        x1 = min(x1, box2[0])
                        y1 = min(y1, box2[1])
                        x2 = max(x2, box2[2])
                        y2 = max(y2, box2[3])
                        used.add(j)
                        changed = True
            
            combined_boxes.append((x1, y1, x2, y2))
        
        return combined_boxes

    def process_frame(self, frame: np.ndarray, avoid_boxes: List[Tuple[int, int, int, int]] = None, is_test_frame: bool = False) -> Optional[DetectionResults]:
        """Process frame through detection pipeline."""
        try:
            # Get latest constants
            import SOD_Constants as const
            
            self.frame_count += 1
            is_rcnn_frame = (self.frame_count % self.rcnn_cycle) == 0
            results = DetectionResults(frame_number=self.frame_count)
            
            # Check for "No Feed" text using specific pixel detection
            # if self.is_no_feed_frame(frame):
            #     results.rcnn_boxes['nofeed'] = [(0, 0, frame.shape[1], frame.shape[0])]
            #     results.metadata['nofeed_detected_by_pixel'] = True
            #     return results
            
            # Run RCNN on regular cycle OR if this is a test frame
            if is_rcnn_frame or is_test_frame:
                # Instead of running RCNN synchronously, push it to the queue
                try:
                    self.rcnn_queue.put_nowait(frame.copy())
                    results.metadata['rcnn_frame'] = True
                    if is_test_frame:
                        results.metadata['test_frame'] = True
                except Exception:
                    results.metadata['rcnn_frame'] = False
            
            # Use the latest available RCNN results
            results.rcnn_boxes = self.last_rcnn_results.get('boxes', {})
            results.rcnn_scores = self.last_rcnn_results.get('scores', {})
            
            # Check for darkness
            if 'td' in self.last_rcnn_results['boxes']:
                frame_area = frame.shape[0] * frame.shape[1]
                for box in self.last_rcnn_results['boxes']['td']:
                    x1, y1, x2, y2 = box
                    td_area = (x2 - x1) * (y2 - y1)
                    if td_area > frame_area * const.get_value('DARKNESS_AREA_THRESHOLD'):
                        results.darkness_detected = True
                        break
            
            # Check for darkness or no feed from RCNN
            if results.darkness_detected or 'nofeed' in results.rcnn_boxes:
                return results
            
            # Check lens flare count
            if 'lf' in self.last_rcnn_results['boxes']:
                lf_count = len(self.last_rcnn_results['boxes']['lf'])
                if lf_count >= const.get_value('MAX_LENS_FLARES'):
                    results.metadata['skip_save'] = 'too_many_lens_flares'
            
            # Only proceed with anomaly detection if we have a space region
            if 'space' in results.rcnn_boxes:
                # Get all space boxes and combine overlapping ones
                space_boxes = self._combine_overlapping_boxes(results.rcnn_boxes['space'])
                
                # Sort combined boxes by y1 coordinate (highest first)
                space_boxes = sorted(space_boxes, key=lambda box: box[1])
                
                # Store the combined box for display and processing
                results.space_box = space_boxes[0]  # Main combined box for display
                results.metadata['all_space_boxes'] = space_boxes  # Store all boxes for reference
                
                # Track all anomalies and metrics
                all_anomalies = []
                all_metrics = []
                
                # Process the main combined space box
                box_results = self.detect_anomalies(frame, space_boxes, avoid_boxes)
                
                # Store all results from the combined box
                if box_results.anomalies:
                    all_anomalies.extend(box_results.anomalies)
                    if 'anomaly_metrics' in box_results.metadata:
                        all_metrics.extend(box_results.metadata['anomaly_metrics'])
                
                # Store contours from the combined box
                results.contours = box_results.contours
                
                # Process additional space boxes if they don't overlap with the main one
                for space_box in space_boxes[1:]:
                    # Check if this box overlaps with the main box
                    main_x1, main_y1, main_x2, main_y2 = results.space_box
                    x1, y1, x2, y2 = space_box
                    
                    # If boxes don't overlap, process this one too
                    if (x2 < main_x1 or x1 > main_x2 or 
                        y2 < main_y1 or y1 > main_y2):
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

    def _run_rcnn_detection(self, frame: np.ndarray) -> Dict:
        """Run RCNN detection on a frame."""
        start = time.time()
        
        # Log initial memory state
        self.logger.log_memory_usage()
        
        # Convert BGR numpy array directly to tensor (HWC -> CHW)
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        img_tensor = self.transform(img_tensor).unsqueeze(0).to(DEVICE)
        self.logger.log_operation_time('rcnn_prep', time.time() - start)
        
        # Log memory after tensor creation
        self.logger.log_memory_usage()
        
        # Run detection
        inference_start = time.time()
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
        self.logger.log_operation_time('rcnn_inference', time.time() - inference_start)
        
        # Log memory after inference
        self.logger.log_memory_usage()
        
        # Process predictions
        postprocess_start = time.time()
        boxes = predictions['boxes'].cpu().numpy().astype(np.int32)
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        results = {'boxes': {}, 'scores': {}}
        
        # Process all predictions together
        for box, label, score in zip(boxes, labels, scores):
            class_name = CLASS_NAMES[label]
            threshold = CLASS_THRESHOLDS.get(class_name, 0.5)
            
            if score > threshold:
                if class_name not in results['boxes']:
                    results['boxes'][class_name] = []
                    results['scores'][class_name] = []
                    
                results['boxes'][class_name].append(box)
                results['scores'][class_name].append(score)
        
        self.logger.log_operation_time('rcnn_postprocess', time.time() - postprocess_start)
        
        # Clean up GPU memory
        del img_tensor
        del predictions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log final memory state
        self.logger.log_memory_usage()
        
        return results

    def _rcnn_worker(self):
        """Background worker thread to run RCNN inference without blocking the main loop."""
        import time
        while self.rcnn_worker_running:
            try:
                frame = self.rcnn_queue.get(timeout=1)
            except Exception:
                continue

            if frame is None:
                break

            # Run RCNN detection on the provided frame
            predictions = self._run_rcnn_detection(frame)
            self.last_rcnn_results = predictions
            self.last_rcnn_frame = self.frame_count
            self.rcnn_queue.task_done()
            
            # Clean up GPU memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def set_rcnn_cycle(self, fps: int):
        """Update the RCNN detection cycle based on frame rate."""
        # Get latest constants
        import SOD_Constants as const
        self.rcnn_cycle = const.RCNN_DETECTION_CYCLE  # Use dynamic constant

    def force_rcnn_detection(self, frame: np.ndarray) -> None:
        """Force RCNN detection on a specific frame (for test frames)."""
        try:
            # Push frame to RCNN queue with high priority
            if self.rcnn_queue.empty():
                self.rcnn_queue.put_nowait(frame.copy())
            else:
                # Clear queue and add new frame
                try:
                    self.rcnn_queue.get_nowait()  # Remove existing frame
                    self.rcnn_queue.put_nowait(frame.copy())
                except queue.Empty:
                    pass
        except Exception as e:
            print(f"Error forcing RCNN detection: {e}")

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