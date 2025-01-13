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
    frame_number: int
    rcnn_boxes: Dict[str, List[Tuple[int, int, int, int]]]
    rcnn_scores: Dict[str, List[float]]
    anomalies: List[Tuple[int, int, int, int]]
    darkness_detected: bool
    is_rcnn_frame: bool
    metadata: Optional[Dict[str, Any]] = None
    space_box: Optional[Tuple[int, int, int, int]] = None

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
        
        # Testing/debug state
        self.inject_test_frames = 0
        self.test_image = None
        
        # Initialize transform
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_test_image(self, path: str) -> bool:
        """Load test image for injection."""
        try:
            self.test_image = cv2.imread(path)
            if self.test_image is not None:
                self.test_image = cv2.resize(self.test_image, (1280, 720))
                return True
        except Exception as e:
            print(f"Error loading test image: {e}")
        return False

    def start_test_injection(self, frames: int = 10) -> None:
        """Start test frame injection."""
        if self.test_image is not None:
            self.inject_test_frames = frames
            print(f"\nStarting {frames} frame test injection!")

    def _analyze_object(self, contour: np.ndarray, gray_frame: np.ndarray) -> tuple[bool, dict]:
        """Analyze if object passes detection criteria"""
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip tiny objects without printing
        if w < 5 or h < 5:
            return False, {}
        
        rect_area = w * h
        contour_area = cv2.contourArea(contour)
        
        metrics = {
            'contour_area': contour_area,
            'rect_area': rect_area,
            'aspect_ratio': float(w)/h if h != 0 else 0,
            'width': w,
            'height': h
        }
        
        # Debug print metrics only for objects >= 5x5
        print(f"\nAnalyzing object:")
        print(f"- Size: {w}x{h}")
        print(f"- Rect Area: {rect_area}")
        print(f"- Contour Area: {contour_area}")
        print(f"- Area Ratio: {contour_area/rect_area if rect_area > 0 else 0}")
        print(f"- Aspect ratio: {metrics['aspect_ratio']}")
        
        # Adjusted checks for elongated objects
        if w > 100 or h > 100:  # Allow larger but not huge
            print("Failed: Too large")
            return False, metrics
        
        if contour_area < 12:  # Reduced minimum area
            print("Failed: Area too small")
            return False, metrics
        
        # More lenient aspect ratio for elongated objects
        if metrics['aspect_ratio'] > 8 or metrics['aspect_ratio'] < 0.125:  # Allow more elongated objects
            print("Failed: Bad aspect ratio")
            return False, metrics
        
        # Get brightness metrics
        mask = np.zeros(gray_frame.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        obj_brightness = cv2.mean(gray_frame, mask=mask)[0]
        
        kernel = np.ones((3,3), np.uint8)
        bg_mask = cv2.dilate(mask, kernel, iterations=2)
        bg_mask = cv2.bitwise_xor(bg_mask, mask)
        bg_brightness = cv2.mean(gray_frame, mask=bg_mask)[0]
        contrast = obj_brightness - bg_brightness
        
        print(f"- Brightness: {obj_brightness}")
        print(f"- Background: {bg_brightness}")
        print(f"- Contrast: {contrast}")
        
        if not (12 < obj_brightness < 100):
            print("Failed: Brightness out of range")
            return False, metrics
        if bg_brightness > 40:
            print("Failed: Background too bright")
            return False, metrics
        if contrast < 12:
            print("Failed: Insufficient contrast")
            return False, metrics
        
        print("PASSED ALL CHECKS")
        return True, metrics

    def _detect_anomalies(self, frame: np.ndarray, space_box: tuple, iss_boxes: list = None, sun_boxes: list = None) -> tuple:
        """Find anomalies in space region"""
        x1, y1, x2, y2 = space_box
        roi = frame[y1:y2, x1:x2]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Debug print ROI stats
        print(f"\nAnalyzing space region: {roi.shape}")
        
        # Create mask for dark regions (true space)
        _, dark_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        
        # Apply dark mask to gray image
        masked_gray = cv2.bitwise_and(gray, gray, mask=dark_mask)
        
        background = cv2.GaussianBlur(masked_gray, (21, 21), 0)
        diff = cv2.absdiff(masked_gray, background)
        
        _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug print contours found
        print(f"Found {len(contours)} potential objects")
        
        anomalies = []
        metadata = {'anomaly_metrics': []}
        
        # Get all ISS boxes if not provided
        if iss_boxes is None:
            iss_boxes = self.last_rcnn_results['boxes'].get('iss', [])
        
        # Get panel boxes
        panel_boxes = self.last_rcnn_results['boxes'].get('panel', [])
        
        # Get loss of feed boxes
        lf_boxes = self.last_rcnn_results['boxes'].get('lf', [])
        
        # Process contours
        valid_contour_count = 0
        for contour in contours:
            # Skip if we already have 3 or more valid detections
            if valid_contour_count >= 3:
                break
            
            x, y, w, h = cv2.boundingRect(contour)
            abs_x, abs_y = x1 + x, y1 + y
            
            # Skip if in any ISS box
            if iss_boxes:
                in_iss = False
                for iss_box in iss_boxes:
                    if (abs_x < iss_box[2] and abs_x + w > iss_box[0] and
                        abs_y < iss_box[3] and abs_y + h > iss_box[1]):
                        in_iss = True
                        break
                if in_iss:
                    continue
            
            # Skip if in any panel box
            if panel_boxes:
                in_panel = False
                for panel_box in panel_boxes:
                    # Expand panel box 10% up and right
                    x1, y1, x2, y2 = panel_box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Expand box
                    expanded_box = [
                        x1,                    # Left stays same
                        y1 - (height * 0.10),  # Top expands up
                        x2 + (width * 0.10),   # Right expands
                        y2                     # Bottom stays same
                    ]
                    
                    # Check against expanded box
                    if (abs_x < expanded_box[2] and abs_x + w > expanded_box[0] and
                        abs_y < expanded_box[3] and abs_y + h > expanded_box[1]):
                        in_panel = True
                        break
                if in_panel:
                    continue
            
            # Skip if in any loss of feed box
            if lf_boxes:
                in_lf = False
                for lf_box in lf_boxes:
                    if (abs_x < lf_box[2] and abs_x + w > lf_box[0] and
                        abs_y < lf_box[3] and abs_y + h > lf_box[1]):
                        in_lf = True
                        break
                if in_lf:
                    continue
            
            # Analyze object
            passes_analysis, metrics = self._analyze_object(contour, gray)
            if passes_analysis:
                print(f"Object passed analysis: {metrics}")  # Debug metrics
                anomalies.append((abs_x, abs_y, w, h))
                metrics['position'] = (abs_x, abs_y, w, h)
                metadata['anomaly_metrics'].append(metrics)
                valid_contour_count += 1
        
        # If we have too many detections, clear them all
        if valid_contour_count >= 3:
            anomalies = []
            metadata['anomaly_metrics'] = []
            metadata['too_many_detections'] = True
        
        return anomalies, metadata

    def process_frame(self, frame: np.ndarray) -> Optional[DetectionResults]:
        """Process frame through detection pipeline."""
        try:
            self.frame_count += 1
            
            # Debug print frame dimensions
            print(f"Processing frame with shape: {frame.shape}")
            
            # Handle test frame injection
            if self.inject_test_frames > 0 and self.test_image is not None:
                frame = self.test_image.copy()
                print(f"\nTest frame {self.inject_test_frames}/10")
                self.inject_test_frames -= 1
            
            # Ensure frame is cropped before processing
            if frame.shape[1] != CROPPED_WIDTH:
                from SOD_Utils import crop_frame
                frame = crop_frame(frame)
            
            is_rcnn_frame = (self.frame_count % self.rcnn_cycle) == 0
            metadata = {'anomaly_metrics': []}
            
            # Run RCNN if needed
            if is_rcnn_frame:
                self.last_rcnn_results = self._run_rcnn_detection(frame)
                print(f"RCNN results: {self.last_rcnn_results['boxes'].keys()}")  # Debug boxes found
                self.last_rcnn_frame = self.frame_count
                metadata['rcnn_frame'] = True
            
            # Check for darkness
            darkness_detected = False
            if 'td' in self.last_rcnn_results['boxes']:
                frame_area = frame.shape[0] * frame.shape[1]
                for box in self.last_rcnn_results['boxes']['td']:
                    x1, y1, x2, y2 = box
                    td_area = (x2 - x1) * (y2 - y1)
                    # if td_area > frame_area * 0.40:  # Commenting out darkness pause
                    #     darkness_detected = True
                    #     metadata['darkness_area_ratio'] = td_area / frame_area
                    #     break
                    # Still track the ratio for debugging
                    metadata['darkness_area_ratio'] = td_area / frame_area
            
            # Run anomaly detection if not dark
            anomalies = []
            space_box = None
            if not darkness_detected and 'space' in self.last_rcnn_results['boxes']:
                print(f"Found space region")
                for space_box in self.last_rcnn_results['boxes']['space']:
                    detected_anomalies, anomaly_metadata = self._detect_anomalies(
                        frame,
                        space_box,
                        self.last_rcnn_results['boxes'].get('iss', []),
                        self.last_rcnn_results['boxes'].get('sun', [])
                    )
                    anomalies.extend(detected_anomalies)
                    metadata.update(anomaly_metadata)
            
            # Update metadata - ensure anomaly_metrics exists
            metadata.update({
                'nofeed_detected': 'nofeed' in self.last_rcnn_results['boxes']
            })
            
            return DetectionResults(
                frame_number=self.frame_count,
                rcnn_boxes=self.last_rcnn_results['boxes'],
                rcnn_scores=self.last_rcnn_results['scores'],
                anomalies=anomalies,
                darkness_detected=darkness_detected,
                is_rcnn_frame=is_rcnn_frame,
                metadata=metadata,
                space_box=space_box
            )
            
        except Exception as e:
            print(f"Error processing frame: {e}")
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