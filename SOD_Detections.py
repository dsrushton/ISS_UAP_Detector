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

from SOD_Constants import (
    DEVICE, MODEL_PATH, CLASS_NAMES, CLASS_THRESHOLDS,
    MIN_BRIGHTNESS, MAX_BRIGHTNESS, MIN_PIXEL_DIM, CROPPED_WIDTH,
    RCNN_DETECTION_CYCLE, MIN_CONTOUR_WIDTH, MIN_CONTOUR_HEIGHT,
    MAX_CONTOUR_WIDTH, MAX_CONTOUR_HEIGHT, MIN_CONTOUR_AREA,
    MAX_ASPECT_RATIO, MAX_BG_BRIGHTNESS, MIN_CONTRAST, MAX_LENS_FLARES
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
        
        # Initialize with empty results
        self.last_rcnn_results = {
            'boxes': {},
            'scores': {}
        }
        self.last_rcnn_frame = -1
        
        # Initialize transform
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.consecutive_lf = 0
        self.lf_pause_until = 0  # Track when to resume after lens flares

    def analyze_object(self, gray_frame: np.ndarray, binary_mask: np.ndarray, w: int, h: int) -> bool:
        """Analyze if object passes detection criteria"""
        # STRICT SIZE REQUIREMENTS FIRST
        # Minimum dimensions
        if w < MIN_CONTOUR_WIDTH or h < MIN_CONTOUR_HEIGHT:
            return False
        
        # Maximum dimensions    
        if w > MAX_CONTOUR_WIDTH or h > MAX_CONTOUR_HEIGHT:
            return False
        
        # Area check - much more permissive for elongated objects
        aspect = max(w, h) / min(w, h)
        min_area = MIN_CONTOUR_AREA if aspect > 3 else MIN_CONTOUR_AREA * 1.5
        
        # Use the binary mask for exact object boundaries
        obj_pixels = cv2.bitwise_and(gray_frame, gray_frame, mask=binary_mask)
        area = np.count_nonzero(binary_mask)
        if area < min_area:
            return False
        
        # Aspect ratio check - Allow very elongated objects
        if aspect > MAX_ASPECT_RATIO or aspect < (1.0 / MAX_ASPECT_RATIO):
            return False
        
        # Get brightness using exact binary mask boundaries
        masked_pixels = obj_pixels[binary_mask > 0]
        if len(masked_pixels) == 0:
            return False
        obj_brightness = float(np.mean(masked_pixels))
        
        # Get background from dilated region outside mask
        kernel = np.ones((3,3), np.uint8)
        bg_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        bg_mask = cv2.bitwise_xor(bg_mask, binary_mask)
        bg_pixels = cv2.bitwise_and(gray_frame, gray_frame, mask=bg_mask)
        
        # Check if we have valid background pixels
        masked_bg = bg_pixels[bg_mask > 0]
        if len(masked_bg) == 0:
            return False
        bg_brightness = float(np.mean(masked_bg))
        
        contrast = abs(obj_brightness - bg_brightness)
        
        # Brightness checks - Using constants
        if not (MIN_BRIGHTNESS < obj_brightness < MAX_BRIGHTNESS):
            return False
        if bg_brightness > MAX_BG_BRIGHTNESS:
            return False
        if contrast < MIN_CONTRAST:
            return False
        
        return True

    def detect_anomalies(self, frame: np.ndarray, space_box: tuple) -> DetectionResults:
        results = DetectionResults()
        x1, y1, x2, y2 = space_box
        roi = frame[y1:y2, x1:x2]
        
        # Convert ROI to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Create masks for brightness detection
        _, lower_mask = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
        _, upper_mask = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(lower_mask, upper_mask)
        
        # Clean up mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results.contours = contours
        
        # Limit number of contours processed
        if len(contours) > 10:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            results.metadata['max_contours_reached'] = True
        
        anomalies = []
        anomaly_metrics = []
        valid_detections = 0
        
        # Get frame dimensions for border check
        frame_height, frame_width = roi.shape[:2]
        BORDER_MARGIN = 5  # How many pixels from border to consider "touching"
        
        # Process each contour
        for contour in contours:
            # Skip if we've already found 4 valid detections
            if valid_detections >= 4:
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
            obj_mask = np.zeros_like(gray)
            cv2.drawContours(obj_mask, [contour], -1, 255, -1)
            
            # Only proceed if analyze_object approves this contour
            if not self.analyze_object(gray, obj_mask, w, h):
                continue
            
            # Calculate metrics for display
            obj_pixels = cv2.bitwise_and(gray, gray, mask=obj_mask)
            obj_brightness = np.mean(obj_pixels[obj_mask > 0])
            
            # Calculate background brightness from dilated region outside mask
            bg_mask = cv2.dilate(obj_mask, kernel, iterations=2)
            bg_mask = cv2.bitwise_xor(bg_mask, obj_mask)
            bg_pixels = cv2.bitwise_and(gray, gray, mask=bg_mask)
            bg_brightness = np.mean(bg_pixels[bg_mask > 0])
            
            contrast = abs(obj_brightness - bg_brightness)
            
            # Store metrics for this anomaly
            metrics = {
                'position': (x + x1, y + y1),
                'obj_brightness': obj_brightness,
                'bg_brightness': bg_brightness,
                'contrast': contrast,
                'width': w,
                'height': h,
                'area': w * h
            }
            
            anomalies.append((x + x1, y + y1, w, h))
            anomaly_metrics.append(metrics)
            valid_detections += 1
        
        # Store results - preserve any existing metadata
        if 'anomaly_metrics' not in results.metadata:
            results.metadata['anomaly_metrics'] = []
        results.metadata['anomaly_metrics'] = anomaly_metrics
        
        # Add detection statistics
        results.metadata['total_contours'] = len(contours)
        results.metadata['valid_detections'] = valid_detections
        
        results.add_anomalies(anomalies)
        
        return results

    def process_frame(self, frame: np.ndarray) -> Optional[DetectionResults]:
        """Process frame through detection pipeline."""
        try:
            self.frame_count += 1
            is_rcnn_frame = (self.frame_count % self.rcnn_cycle) == 0
            results = DetectionResults(frame_number=self.frame_count)
            
            if is_rcnn_frame:
                self.last_rcnn_results = self._run_rcnn_detection(frame)
                self.last_rcnn_frame = self.frame_count
                results.metadata['rcnn_frame'] = True
            
            # Copy RCNN results
            results.rcnn_boxes = self.last_rcnn_results['boxes']
            results.rcnn_scores = self.last_rcnn_results['scores']
            
            # Check for darkness
            if 'td' in self.last_rcnn_results['boxes']:
                frame_area = frame.shape[0] * frame.shape[1]
                for box in self.last_rcnn_results['boxes']['td']:
                    x1, y1, x2, y2 = box
                    td_area = (x2 - x1) * (y2 - y1)
                    if td_area > frame_area * 0.40:  # Darkness threshold
                        results.darkness_detected = True
                        break
            
            # Check for darkness or no feed
            if results.darkness_detected or 'nofeed' in results.rcnn_boxes:
                return results
            
            # Check lens flare count
            skip_contour_detection = False
            if 'lf' in self.last_rcnn_results['boxes']:
                lf_count = len(self.last_rcnn_results['boxes']['lf'])
                if lf_count >= MAX_LENS_FLARES:
                    skip_contour_detection = True
                    results.metadata['skipped_contours_lf'] = True
            
            # Only proceed with anomaly detection if we have a space region and not skipping due to lens flares
            if 'space' in results.rcnn_boxes and not skip_contour_detection:
                space_boxes = sorted(results.rcnn_boxes['space'], 
                                  key=lambda box: box[1])  # Sort by y1 coordinate
                
                # Store the highest box for display purposes
                results.space_box = space_boxes[0]
                
                # Track all anomalies across all space boxes
                all_anomalies = []
                all_metrics = []
                all_contours = []
                seen_positions = set()  # Track positions to avoid duplicate detections
                
                # Process each space box
                for space_box in space_boxes:
                    # Run anomaly detection on this box
                    box_results = self.detect_anomalies(frame, space_box)
                    
                    # Check each anomaly to avoid duplicates (if boxes overlap)
                    for i, (x, y, w, h) in enumerate(box_results.anomalies):
                        # Create a position key (using center point)
                        pos_key = (x + w//2, y + h//2)
                        
                        # Skip if we've seen a detection at this position
                        if pos_key in seen_positions:
                            continue
                            
                        seen_positions.add(pos_key)
                        all_anomalies.append((x, y, w, h))
                        
                        if 'anomaly_metrics' in box_results.metadata:
                            all_metrics.append(box_results.metadata['anomaly_metrics'][i])
                    
                    # Store contours only from the top box for display
                    if np.array_equal(space_box, results.space_box):
                        all_contours = box_results.contours
                
                # Get lens flare boxes for filtering
                lf_boxes = self.last_rcnn_results['boxes'].get('lf', [])
                
                # If we have lens flare boxes, filter anomalies that overlap with them
                if lf_boxes and all_anomalies:
                    filtered_anomalies = []
                    filtered_metrics = []
                    
                    for i, (ax, ay, aw, ah) in enumerate(all_anomalies):
                        overlaps_lf = False
                        for lf_box in lf_boxes:
                            # Check for overlap
                            lf_x1, lf_y1, lf_x2, lf_y2 = lf_box
                            if (ax < lf_x2 and ax + aw > lf_x1 and
                                ay < lf_y2 and ay + ah > lf_y1):
                                overlaps_lf = True
                                break
                        
                        if not overlaps_lf:
                            filtered_anomalies.append((ax, ay, aw, ah))
                            if len(all_metrics) > i:
                                filtered_metrics.append(all_metrics[i])
                    
                    # Update anomalies and metrics after filtering
                    all_anomalies = filtered_anomalies
                    all_metrics = filtered_metrics
                
                # Update final results
                results.anomalies = all_anomalies
                results.contours = all_contours
                if all_metrics:
                    results.metadata['anomaly_metrics'] = all_metrics
                results.metadata['total_space_boxes'] = len(space_boxes)
            
            return results
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None

    def cleanup(self):
        """Clean up resources."""
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
        # Convert frame for RCNN
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img_pil).unsqueeze(0).to(DEVICE)
        
        # Run detection
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
        
        # Process predictions
        boxes = predictions['boxes'].cpu().numpy().astype(np.int32)
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        # Process predictions with priority
        priority_classes = ['panel', 'iss']  # Process these first
        results = {'boxes': {}, 'scores': {}}
        
        # Handle priority classes first
        for class_name in priority_classes:
            mask = labels == CLASS_NAMES.index(class_name)
            class_boxes = boxes[mask]
            class_scores = scores[mask]
            
            if len(class_scores) > 0:
                results['boxes'][class_name] = []
                results['scores'][class_name] = []
                
                for box, score in zip(class_boxes, class_scores):
                    if score > CLASS_THRESHOLDS.get(class_name, 0.5):
                        results['boxes'][class_name].append(box)
                        results['scores'][class_name].append(score)
        
        # Then handle remaining classes
        for box, label, score in zip(boxes, labels, scores):
            class_name = CLASS_NAMES[label]
            if class_name in priority_classes:
                continue  # Already handled
            
            threshold = CLASS_THRESHOLDS.get(class_name, 0.5)
            if score > threshold:
                if class_name not in results['boxes']:
                    results['boxes'][class_name] = []
                    results['scores'][class_name] = []
                    
                results['boxes'][class_name].append(box)
                results['scores'][class_name].append(score)
        
        return results


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