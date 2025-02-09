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

from SOD_Constants import (
    DEVICE, MODEL_PATH, CLASS_NAMES, CLASS_THRESHOLDS,
    MIN_BRIGHTNESS, MAX_BRIGHTNESS, CROPPED_WIDTH,
    RCNN_DETECTION_CYCLE, MIN_CONTOUR_WIDTH, MIN_CONTOUR_HEIGHT,
    MAX_CONTOUR_WIDTH, MAX_CONTOUR_HEIGHT, MIN_CONTOUR_AREA,
    MAX_ASPECT_RATIO, MAX_BG_BRIGHTNESS, MIN_CONTRAST, MAX_LENS_FLARES,
    DARKNESS_AREA_THRESHOLD, MIN_DARK_REGION_SIZE, GAUSSIAN_BLUR_SIZE,
    MORPH_KERNEL_SIZE, BORDER_MARGIN, MAX_VALID_DETECTIONS, MAX_CONTOURS_PER_FRAME
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
    
    def __init__(self, rcnn_cycle: int = RCNN_DETECTION_CYCLE):
        """Initialize detector with frame cycle for RCNN."""
        self.model = None
        self.transform = None
        self.rcnn_cycle = rcnn_cycle
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
        results = DetectionResults()
        x1, y1, x2, y2 = space_box
        roi = frame[y1:y2, x1:x2]
        
        # Convert to max RGB if needed
        if len(roi.shape) == 3:
            roi_max = np.max(roi, axis=2)
        else:
            roi_max = roi
            
        # Apply Gaussian blur to reduce noise
        mask_start = time.time()
        blurred = cv2.GaussianBlur(roi_max, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
        
        # Create binary mask for dark background
        _, dark_mask = cv2.threshold(blurred, MAX_BG_BRIGHTNESS, 255, cv2.THRESH_BINARY_INV)
        
        # Check if dark area has any 100x100 region that's completely dark
        kernel = np.ones((MIN_DARK_REGION_SIZE, MIN_DARK_REGION_SIZE), np.uint8)
        valid_region_mask = cv2.erode(dark_mask, kernel, iterations=1)
        
        valid_dark_region = cv2.countNonZero(valid_region_mask) > 0
        
        if not valid_dark_region:
            # Return empty results if no valid dark region found
            results.metadata['reason'] = 'no_valid_dark_region'
            return results
        
        # Create binary mask for bright objects
        _, bright_mask = cv2.threshold(blurred, MIN_BRIGHTNESS, 255, cv2.THRESH_BINARY)
        
        # Combine masks to get regions where bright objects meet dark background
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
        dark_mask = cv2.dilate(dark_mask, kernel, iterations=1)
        
        # Get intersection of dilated masks
        edge_mask = cv2.bitwise_and(bright_mask, dark_mask)
        
        # Clean up the mask
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
        self.logger.log_operation_time('mask_creation', time.time() - mask_start)
        
        # Find contours using the bright mask
        contour_start = time.time()
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Limit number of contours processed
        if len(contours) > MAX_CONTOURS_PER_FRAME:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CONTOURS_PER_FRAME]
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
        for contour in contours:
            # Skip if we've already found max valid detections
            if valid_detections >= MAX_VALID_DETECTIONS:
                break
            
            # Get contour properties
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if detection touches frame border
            if (x <= BORDER_MARGIN or 
                y <= BORDER_MARGIN or 
                x + w >= frame_width - BORDER_MARGIN or 
                y + h >= frame_height - BORDER_MARGIN):
                continue
            
            # Create binary mask for exact object boundaries
            obj_mask = np.zeros_like(edge_mask)
            cv2.drawContours(obj_mask, [contour], -1, 255, -1)
            
            # Only proceed if analyze_object approves this contour
            is_valid, metrics = self.analyze_object(roi_max, obj_mask, w, h, contour)
            if not is_valid:
                continue
            
            # Skip if contrast is too low or negative
            if metrics['contrast'] < MIN_CONTRAST:
                continue
            
            # Store metrics for this anomaly
            metrics['position'] = (x + x1, y + y1)  # Convert to frame coordinates
            metrics['width'] = w
            metrics['height'] = h
            metrics['area'] = w * h
            
            # Convert to absolute coordinates
            abs_x = x + x1
            abs_y = y + y1
            
            # Skip if in any avoid box
            box_start = time.time()
            if avoid_boxes:
                in_avoid = False
                for avoid_box in avoid_boxes:
                    ax1, ay1, ax2, ay2 = avoid_box
                    if (abs_x < ax2 and abs_x + w > ax1 and
                        abs_y < ay2 and abs_y + h > ay1):
                        in_avoid = True
                        break
                if in_avoid:
                    continue
            self.logger.log_operation_time('box_filtering', time.time() - box_start)
            
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
    
    def analyze_object(self, frame: np.ndarray, binary_mask: np.ndarray, w: int, h: int, contour: np.ndarray = None) -> bool:
        """Analyze if object passes detection criteria"""
        try:
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
            
            # STRICT SIZE REQUIREMENTS FIRST
            if (w < MIN_CONTOUR_WIDTH or h < MIN_CONTOUR_HEIGHT or
                w > MAX_CONTOUR_WIDTH or h > MAX_CONTOUR_HEIGHT):
                return False, metrics
            
            if metrics['area'] < MIN_CONTOUR_AREA:
                return False, metrics
            
            if metrics['aspect_ratio'] > MAX_ASPECT_RATIO or metrics['aspect_ratio'] < (1/MAX_ASPECT_RATIO):
                return False, metrics
            
            # Create precise masks for object and background
            mask = np.zeros(binary_mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Create dilated ring around object for background measurement
            kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
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
            
            # Brightness checks
            if not (MIN_BRIGHTNESS < obj_brightness < MAX_BRIGHTNESS):
                return False, metrics
            if bg_brightness > MAX_BG_BRIGHTNESS:
                return False, metrics
            if contrast < MIN_CONTRAST:
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
                    if td_area > frame_area * DARKNESS_AREA_THRESHOLD:
                        results.darkness_detected = True
                        break
            
            # Check for darkness or no feed
            if results.darkness_detected or 'nofeed' in results.rcnn_boxes:
                return results
            
            # Check lens flare count
            if 'lf' in self.last_rcnn_results['boxes']:
                lf_count = len(self.last_rcnn_results['boxes']['lf'])
                if lf_count >= MAX_LENS_FLARES:
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
                
                # Get lens flare boxes for filtering
                lf_boxes = self.last_rcnn_results['boxes'].get('lf', [])
                
                # Get ISS and panel boxes for filtering
                panel_boxes = self.last_rcnn_results['boxes'].get('panel', [])
                iss_boxes = self.last_rcnn_results['boxes'].get('iss', [])
                
                # If we have lens flare boxes, filter anomalies that overlap with them
                if (lf_boxes or panel_boxes or iss_boxes) and all_anomalies:
                    filtered_anomalies = []
                    filtered_metrics = []
                    
                    for i, (ax, ay, aw, ah) in enumerate(all_anomalies):
                        overlaps = False
                        
                        # Check lens flare overlap
                        for lf_box in lf_boxes:
                            lf_x1, lf_y1, lf_x2, lf_y2 = lf_box
                            if (ax < lf_x2 and ax + aw > lf_x1 and
                                ay < lf_y2 and ay + ah > lf_y1):
                                overlaps = True
                                break
                        
                        # Check ISS overlap
                        #if not overlaps:
                          #  for iss_box in iss_boxes:
                                #iss_x1, iss_y1, iss_x2, iss_y2 = iss_box
                                #if (ax < iss_x2 and ax + aw > iss_x1 and
                                #    ay < iss_y2 and ay + ah > iss_y1):
                                #    overlaps = True
                                #    break
                        
                        # Check panel overlap
                        if not overlaps:
                            for panel_box in panel_boxes:
                                panel_x1, panel_y1, panel_x2, panel_y2 = panel_box
                                if (ax < panel_x2 and ax + aw > panel_x1 and
                                    ay < panel_y2 and ay + ah > panel_y1):
                                    overlaps = True
                                    break
                        
                        if not overlaps:
                            filtered_anomalies.append(all_anomalies[i])
                            if all_metrics:
                                filtered_metrics.append(all_metrics[i])
                    
                    # Update anomalies and metrics after filtering
                    all_anomalies = filtered_anomalies
                    all_metrics = filtered_metrics
                
                # Update final results
                results.anomalies = all_anomalies
                results.metadata['anomaly_metrics'] = all_metrics
            
            return results
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None

    def cleanup(self):
        """Clean up resources."""
        self.rcnn_worker_running = False
        # Signal the worker thread to exit
        try:
            self.rcnn_queue.put(None, timeout=1)
        except Exception:
            pass
        if hasattr(self, 'rcnn_worker_thread'):
            self.rcnn_worker_thread.join()
        self.model = None
        torch.cuda.empty_cache()

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
        self.rcnn_cycle = max(1, fps)  # Run RCNN once per second
        print(f"RCNN cycle set to: {self.rcnn_cycle} frames")

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