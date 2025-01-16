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
    MIN_BRIGHTNESS, MAX_BRIGHTNESS, MIN_PIXEL_DIM, CROPPED_WIDTH
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
    
    def __init__(self, rcnn_cycle: int = 10):
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

    def analyze_object(self, contour: np.ndarray, gray_frame: np.ndarray, binary_mask: np.ndarray) -> tuple[bool, dict]:
        """Analyze if object passes detection criteria"""
        x, y, w, h = cv2.boundingRect(contour)
        metrics = {
            'area': cv2.contourArea(contour),
            'aspect_ratio': float(w)/h if h != 0 else 0,
            'width': w,
            'height': h
        }
        
        # STRICT SIZE REQUIREMENTS FIRST
        # Minimum dimensions
        if max(w, h) < 4:  # Increased from 5 to 6
            return False, metrics
        
        # Maximum dimensions    
        if w > 120 or h > 120:  # Keep max size the same
            return False, metrics
        
        # Area check - much more permissive for elongated objects
        aspect = max(w, h) / min(w, h)
        min_area = 8 if aspect > 3 else 12  # Increased from 8/12 to 12/16
        if metrics['area'] < min_area:
            return False, metrics
        
        # Aspect ratio check - Allow very elongated objects
        if metrics['aspect_ratio'] > 8 or metrics['aspect_ratio'] < 0.125:  # Keep same ratio
            return False, metrics
        
        # Use the binary mask for exact object boundaries
        mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Get brightness using exact binary mask boundaries
        obj_brightness = cv2.mean(gray_frame, mask=mask)[0]
        
        # Get background from dilated region outside mask
        kernel = np.ones((3,3), np.uint8)
        bg_mask = cv2.dilate(mask, kernel, iterations=2)
        bg_mask = cv2.bitwise_xor(bg_mask, mask)
        bg_brightness = cv2.mean(gray_frame, mask=bg_mask)[0]
        
        contrast = obj_brightness - bg_brightness
        
        metrics.update({
            'obj_brightness': obj_brightness,
            'bg_brightness': bg_brightness,
            'contrast': contrast
        })
        
        # Brightness checks - Expanded ranges based on empirical data
        if not (12 < obj_brightness < 240):  # Increased minimum from 8 to 12
            return False, metrics
        if bg_brightness > 35:  # More permissive background (increased from 25 to 35)
            return False, metrics
        if contrast < 6:  # Keep same contrast requirement
            return False, metrics
        
        return True, metrics

    def detect_anomalies(self, frame: np.ndarray, space_box: tuple) -> DetectionResults:
        """Detect anomalies in the space region of a frame."""
        results = DetectionResults()
        results.space_box = space_box
        
        # Copy RCNN results from last detection
        results.rcnn_boxes = self.last_rcnn_results['boxes']
        results.rcnn_scores = self.last_rcnn_results['scores']
        
        if space_box is None:
            return results
            
        # Extract space region
        x1, y1, x2, y2 = space_box
        roi = frame[y1:y2, x1:x2]
        
        # Convert to grayscale and preprocess
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Create masks for brightness "pocket"
        _, lower_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        _, upper_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        pocket_mask = cv2.bitwise_and(lower_mask, upper_mask)
        
        # Clean up the mask
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(pocket_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Limit number of contours to process
        MAX_CONTOURS = 10  # Maximum number of contours to process
        if len(contours) > MAX_CONTOURS:
            # Sort contours by area and keep the largest ones
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CONTOURS]
            results.metadata['contours_limited'] = True
        
        results.contours = contours  # Store contours for debug view
        
        # Store metrics for valid anomalies
        anomalies = []
        anomaly_metrics = []
        valid_detections = 0
        MAX_DETECTIONS = 5  # Maximum number of valid detections to allow
        
        for contour in contours:
            if valid_detections >= MAX_DETECTIONS:
                results.metadata['detections_limited'] = True
                break
                
            # Call analyze_object first
            is_valid, metrics = self.analyze_object(contour, gray, pocket_mask)
            if not is_valid:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create binary mask for exact object boundaries
            obj_mask = np.zeros_like(gray)
            cv2.drawContours(obj_mask, [contour], -1, 255, -1)
            
            # Analyze object properties
            obj_pixels = gray[obj_mask == 255]
            bg_pixels = gray[obj_mask == 0]
            
            if len(obj_pixels) == 0 or len(bg_pixels) == 0:
                continue
                
            obj_brightness = np.mean(obj_pixels)
            bg_brightness = np.mean(bg_pixels)
            contrast = abs(obj_brightness - bg_brightness)
            
            # Skip if object doesn't meet criteria
            if not (12 < obj_brightness < 240):
                continue
            if bg_brightness > 35:
                continue
            if contrast < 6:
                continue
            
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
            print(f"Debug - Anomaly metrics: {metrics}")
            
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
                if lf_count >= 3:
                    skip_contour_detection = True
                    results.metadata['skipped_contours_lf'] = True
            
            # Only proceed with anomaly detection if we have a space region and not skipping due to lens flares
            if 'space' in results.rcnn_boxes and not skip_contour_detection:
                space_boxes = sorted(results.rcnn_boxes['space'], 
                                  key=lambda box: box[1])  # Sort by y1 coordinate
                space_box = space_boxes[0]  # Use highest space box
                
                # Get lens flare boxes if present (2 or fewer)
                lf_boxes = []
                if 'lf' in results.rcnn_boxes:
                    lf_count = len(self.last_rcnn_results['boxes']['lf'])
                    if lf_count <= 2:
                        lf_boxes = results.rcnn_boxes['lf']
                
                # Run anomaly detection
                anomaly_results = self.detect_anomalies(frame, space_box)
                
                # If we have lens flare boxes, filter anomalies that overlap with them
                if lf_boxes and anomaly_results.anomalies:
                    filtered_anomalies = []
                    filtered_metrics = []
                    
                    for i, (ax, ay, aw, ah) in enumerate(anomaly_results.anomalies):
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
                            if 'anomaly_metrics' in anomaly_results.metadata:
                                filtered_metrics.append(anomaly_results.metadata['anomaly_metrics'][i])
                    
                    # Update results with filtered anomalies
                    anomaly_results.anomalies = filtered_anomalies
                    if 'anomaly_metrics' in anomaly_results.metadata:
                        anomaly_results.metadata['anomaly_metrics'] = filtered_metrics
                
                # Update results
                results.anomalies = anomaly_results.anomalies
                results.contours = anomaly_results.contours
                results.metadata.update(anomaly_results.metadata)
                results.space_box = space_box
            
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