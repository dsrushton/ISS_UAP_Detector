# SOD_Constants.py
"""
Shared constants for the Space Object Detection system.
Contains all configuration values, class definitions, and thresholds.
"""

import torch

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Path
MODEL_PATH = r"C:\Users\dsrus\Desktop\Workspace\UAP_Python_2\trainJan10_bb-12e-Full-NoMoon-NoColorJitter\best_map_model.pth"

# Class Definitions
CLASS_NAMES = [
    'background',  # 0
    'space',       # 1
    'earth',       # 2
    'iss',         # 3
    'lf',          # 4
    'td',          # 5
    'sun',         # 6
    'nofeed',      # 7
    'panel'        # 8
]

# Class Colors (B,G,R format)
CLASS_COLORS = {
    'background': (255, 255, 255),  # White
    'space': (0, 0, 255),          # Blue
    'earth': (255, 0, 0),          # Red
    'iss': (0, 255, 0),            # Green
    'lf': (0, 255, 255),           # Cyan
    'td': (255, 0, 255),           # Magenta
    'sun': (128, 0, 128),          # Purple
    'nofeed': (255, 255, 0),       # Yellow
    'panel': (0, 255, 0)           # Same as ISS (Green)
}

# Detection Thresholds
CLASS_THRESHOLDS = {
    'space': 0.75,
    'earth': 0.75,
    'iss': 0.75,
    'lf': 0.75,
    'td': 0.50,
    'sun': 0.75,
    'nofeed': 0.25,
    'panel': 0.75
}

# Frame Processing
CROP_LEFT = 165    # Pixels to crop from left
CROP_RIGHT = 176   # Pixels to crop from right
CROPPED_WIDTH = 1280 - CROP_LEFT - CROP_RIGHT  # = 939 pixels

# Anomaly Detection Parameters
MIN_PIXEL_DIM = 6
MIN_BRIGHTNESS = 15
MAX_BRIGHTNESS = 240
SAVE_INTERVAL = 2.0       # Minimum seconds between saves

# Video Processing
MAX_CONSECUTIVE_ERRORS = 30
BURST_CAPTURE_FRAMES = 100

# Save Directory Structure
DEFAULT_SAVE_DIR = "C:/Users/dsrus/Desktop/Workspace/MTLiens/ISS_UAP_Detector/Detections/JPG"
RAW_SUBDIR = "Burst_raw"

# Video Recording Settings
BUFFER_SECONDS = 3
POST_DETECTION_SECONDS = 2
VIDEO_FPS = 30
VIDEO_SAVE_DIR = "C:/Users/dsrus/Desktop/Workspace/MTLiens/ISS_UAP_Detector/Detections/AVI"