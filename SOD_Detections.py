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

        # Setup RCNN threading
        self.rcnn_queue = queue.Queue(maxsize=1)  # Only keep one pending frame
        self.rcnn_worker_running = True
        self.rcnn_worker_thread = threading.Thread(target=self._rcnn_worker)
        self.rcnn_worker_thread.daemon = True
        self.rcnn_worker_thread.start()

    def set_logger(self, logger):
        """Set the logger instance."""
        self.logger = logger

    def detect_anomalies(self, frame: np.ndarray, space_box: tuple, avoid_boxes: List[Tuple[int, int, int, int]] = None) -> DetectionResults:
        """Find relevant contour distinctions in space region."""
        try:
            # Get latest constants
            import SOD_Constants as const
            
            results = DetectionResults()
            x1, y1, x2, y2 = space_box  # This is in global coordinates
            
            roi = frame[y1:y2, x1:x2]
            
            # Create a mask for the exact space region using the actual space contour
            space_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            if hasattr(self, 'last_rcnn_results') and 'boxes' in self.last_rcnn_results:
                space_boxes = self.last_rcnn_results['boxes'].get('space', [])
                for box in space_boxes:
                    # Convert global coordinates to ROI coordinates
                    sx1, sy1, sx2, sy2 = box
                    local_x1 = max(0, sx1 - x1)
                    local_y1 = max(0, sy1 - y1)
                    local_x2 = min(x2-x1, sx2 - x1)
                    local_y2 = min(y2-y1, sy2 - y1)
                    cv2.rectangle(space_mask, (local_x1, local_y1), (local_x2, local_y2), 255, -1)
                
                # Find external contour of all space regions combined
                space_contours, _ = cv2.findContours(space_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Clear mask and redraw only the external contour
                space_mask.fill(0)
                cv2.drawContours(space_mask, space_contours, -1, 255, -1)
            
            # Convert to max RGB if needed
            if len(roi.shape) == 3:
                roi_max = np.max(roi, axis=2)
            else:
                roi_max = roi
            
            # Apply Gaussian blur to reduce noise
            mask_start = time.time()
            blur_size = const.get_value('GAUSSIAN_BLUR_SIZE')
            blurred = cv2.GaussianBlur(roi_max, (blur_size, blur_size), 0)
            
            # Create binary mask for dark background
            _, dark_mask = cv2.threshold(blurred, const.get_value('MAX_BG_BRIGHTNESS'), 255, cv2.THRESH_BINARY_INV)
            dark_mask = cv2.bitwise_and(dark_mask, space_mask)  # Limit to space region
            
            # # Check if dark area has any 100x100 region that's completely dark
            # dark_region_size = const.get_value('MIN_DARK_REGION_SIZE')
            # kernel = np.ones((dark_region_size, dark_region_size), np.uint8)
            # valid_region_mask = cv2.erode(dark_mask, kernel, iterations=1)
            
            # valid_dark_region = cv2.countNonZero(valid_region_mask) > 0
            
            # if not valid_dark_region:
            #     # Return empty results if no valid dark region found
            #     results.metadata['reason'] = 'no_valid_dark_region'
            #     return results
            
            # Create binary mask for bright objects
            _, bright_mask = cv2.threshold(blurred, const.get_value('MIN_OBJECT_BRIGHTNESS'), 255, cv2.THRESH_BINARY)
            bright_mask = cv2.bitwise_and(bright_mask, space_mask)  # Limit to space region
            
            # Combine masks to get regions where bright objects meet dark background
            morph_size = const.get_value('MORPH_KERNEL_SIZE')
            kernel = np.ones((morph_size, morph_size), np.uint8)
            bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
            dark_mask = cv2.dilate(dark_mask, kernel, iterations=1)
            
            # Get intersection of dilated masks
            edge_mask = cv2.bitwise_and(bright_mask, dark_mask)
            edge_mask = cv2.bitwise_and(edge_mask, space_mask)  # Ensure we stay within space region
            
            # Clean up the mask
            edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
            self.logger.log_operation_time('mask_creation', time.time() - mask_start)
            
            # Find contours using the bright mask
            contour_start = time.time()
            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Limit number of contours processed
            max_contours = const.get_value('MAX_CONTOURS_PER_FRAME')
            if len(contours) > max_contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]
                results.metadata['max_contours_reached'] = True
            self.logger.log_operation_time('contour_finding', time.time() - contour_start)
            
            # Store all contours for visualization (will convert to absolute later)
            results.contours = contours
            
            anomalies = []
            anomaly_metrics = []
            valid_detections = 0
            
            # Get frame dimensions for border check
            frame_height, frame_width = roi.shape[:2]
            
            # Process each contour
            analysis_start = time.time()
            max_valid = const.get_value('MAX_VALID_DETECTIONS')
            border_margin = const.get_value('BORDER_MARGIN')
            min_contrast = const.get_value('MIN_CONTRAST')
            
            for contour in contours:
                # Skip if we've already found max valid detections
                if valid_detections >= max_valid:
                    break
                
                # Get contour properties
                x, y, w, h = cv2.boundingRect(contour)
                
                # Convert to global coordinates for filtering
                abs_x = x + x1
                abs_y = y + y1
                
                # Check if any part of the contour is too close to space contour
                too_close_to_border = False
                for space_contour in space_contours:
                    # Test each point in the contour
                    contour_points = contour.squeeze()
                    if len(contour_points.shape) == 1:  # Single point
                        contour_points = contour_points.reshape(1, 2)
                    for point in contour_points:
                        dist = cv2.pointPolygonTest(space_contour, (float(point[0]), float(point[1])), True)
                        if abs(dist) < border_margin:
                            too_close_to_border = True
                            break
                    if too_close_to_border:
                        break
                
                if too_close_to_border:
                    continue
                
                # Create binary mask for exact object boundaries
                obj_mask = np.zeros_like(edge_mask)
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
            results.contours = [cont + np.array([x1, y1])[np.newaxis, np.newaxis, :] for cont in contours]
            
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
            
            # Use provided contour if available, otherwise find it
            if contour is None:
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return False, {}
                contour = max(contours, key=cv2.contourArea)
            
            x, y, w, h = cv2.boundingRect(contour)
            
            metrics = {
                'area': cv2.contourArea(contour),
                'aspect_ratio': float(w)/h if h != 0 else 0,
                'width': w,
                'height': h
            }
            
            # Get size requirements
            min_dim = const.get_value('MIN_CONTOUR_DIMENSION')
            max_width = const.get_value('MAX_CONTOUR_WIDTH')
            max_height = const.get_value('MAX_CONTOUR_HEIGHT')
            min_area = const.get_value('MIN_CONTOUR_AREA')
            max_ratio = const.get_value('MAX_ASPECT_RATIO')
            
            # STRICT SIZE REQUIREMENTS FIRST
            if (w < min_dim or h < min_dim or
                w > max_width or h > max_height):
                return False, metrics
            
            if metrics['area'] < min_area:
                return False, metrics
            
            if metrics['aspect_ratio'] > max_ratio or metrics['aspect_ratio'] < (1/max_ratio):
                return False, metrics
            
            # Create precise masks for object and background
            mask = np.zeros(binary_mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Create dilated ring around object for background measurement
            morph_size = const.get_value('MORPH_KERNEL_SIZE')
            kernel = np.ones((morph_size, morph_size), np.uint8)
            bg_mask = cv2.dilate(mask, kernel, iterations=2)
            bg_mask = cv2.bitwise_xor(bg_mask, mask)  # Creates ring around object
            
            # Convert frame to max RGB if not already grayscale
            if len(frame.shape) == 3:
                frame_max = np.max(frame, axis=2)  # Use max RGB value
            else:
                frame_max = frame
            
            # Get precise brightness measurements
            obj_brightness = cv2.mean(frame_max, mask=mask)[0]
            bg_brightness = cv2.mean(frame_max, mask=bg_mask)[0]
            contrast = obj_brightness - bg_brightness
            
            metrics.update({
                'obj_brightness': obj_brightness,
                'bg_brightness': bg_brightness,
                'contrast': contrast
            })
            
            # Get brightness thresholds
            min_obj_bright = const.get_value('MIN_OBJECT_BRIGHTNESS')
            max_obj_bright = const.get_value('MAX_OBJECT_BRIGHTNESS')
            max_bg_bright = const.get_value('MAX_BG_BRIGHTNESS')
            min_contrast = const.get_value('MIN_CONTRAST')
            
            # Brightness checks
            if not (min_obj_bright < obj_brightness < max_obj_bright):
                return False, metrics
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

    def process_frame(self, frame: np.ndarray, avoid_boxes: List[Tuple[int, int, int, int]] = None) -> Optional[DetectionResults]:
        """Process frame through detection pipeline."""
        try:
            # Get latest constants
            import SOD_Constants as const
            
            self.frame_count += 1
            is_rcnn_frame = (self.frame_count % self.rcnn_cycle) == 0
            results = DetectionResults(frame_number=self.frame_count)
            
            if is_rcnn_frame:
                # Instead of running RCNN synchronously, push it to the queue
                try:
                    self.rcnn_queue.put_nowait(frame.copy())
                    results.metadata['rcnn_frame'] = True
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
            
            # Check for darkness or no feed
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
                box_results = self.detect_anomalies(frame, results.space_box, avoid_boxes)
                
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
                        box_results = self.detect_anomalies(frame, space_box, avoid_boxes)
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
        
        # Convert BGR numpy array directly to tensor (HWC -> CHW)
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        img_tensor = self.transform(img_tensor).unsqueeze(0).to(DEVICE)
        self.logger.log_operation_time('rcnn_prep', time.time() - start)
        
        # Run detection
        inference_start = time.time()
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
        self.logger.log_operation_time('rcnn_inference', time.time() - inference_start)
        
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

    def set_rcnn_cycle(self, fps: int):
        """Update the RCNN detection cycle based on frame rate."""
        # Get latest constants
        import SOD_Constants as const
        self.rcnn_cycle = const.RCNN_DETECTION_CYCLE  # Use dynamic constant

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