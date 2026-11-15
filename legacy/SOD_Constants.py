# SOD_Constants.py
"""
Shared constants for the Space Object Detection system.
Contains all configuration values, class definitions, and thresholds.
"""

import torch
import sys
import threading
import queue

class Constants:
    """Singleton class to hold all constants."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Constants, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize default values."""
        self._main_thread = threading.current_thread()
        self._pending_updates = {}
        
        # Device Configuration
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Filter Box Controls
        self.FILTER_ISS = False
        self.FILTER_PANEL = True
        self.FILTER_LF = True

        # Model Path
        self.MODEL_PATH = "./best_map_model.pth"

        # RCNN Parameters
        self.RCNN_DETECTION_CYCLE = 120  # Run RCNN every 2 seconds (at 54 fps)

        # Class Definitions
        self.CLASS_NAMES = [
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
        self.CLASS_COLORS = {
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
        self.CLASS_THRESHOLDS = {
            'space': 0.80,
            'earth': 0.75,
            'iss': 0.75,
            'lf': 0.99,
            'td': 0.25,
            'sun': 0.75,
            'nofeed': 0.25,
            'panel': 0.75
        }

        # Frame Processing
        self.CROP_LEFT = 165    # Pixels to crop from left
        self.CROP_RIGHT = 176   # Pixels to crop from right
        self.CROPPED_WIDTH = 1280 - self.CROP_LEFT - self.CROP_RIGHT  # = 939 pixels

        # Video Processing
        self.MAX_CONSECUTIVE_ERRORS = 30
        self.BURST_CAPTURE_FRAMES = 100

        # Save Directory Structure
        self.VIDEO_SAVE_DIR = "./Detections/AVI"
        self.JPG_SAVE_DIR = "./Detections/JPG"
        self.RAW_SUBDIR = "Burst_raw"

        # Test Image Path
        self.TEST_IMAGE_PATH = "./Test_Image_Collection/000512.jpg"

        # Recording Settings
        self.BUFFER_SECONDS = 3
        self.POST_DETECTION_SECONDS = 2.1
        self.SAVE_INTERVAL = 2.0  # Seconds between .jpg saves

        # RCNN Controls
        self.MAX_LENS_FLARES = 3  # Maximum number of lens flares before skipping contour detection
        self.DARKNESS_AREA_THRESHOLD = 0.25  # Total darkness area threshold
        self.MAX_RCNN_BOXES = 10

        # Anomaly Detection Parameters
        self.MIN_OBJECT_BRIGHTNESS = 10 
        self.MAX_OBJECT_BRIGHTNESS = 200
        self.GAUSSIAN_BLUR_SIZE = 5      # Size of Gaussian blur kernel (must be odd)
        self.MORPH_KERNEL_SIZE = 1    # Size of morphological operation kernel
        self.MAX_BG_BRIGHTNESS = 23
        self.MIN_CONTRAST = 23
        self.MIN_CONTOUR_DIMENSION = 5   # Minimum width/height for contours
        self.MAX_CONTOUR_WIDTH = 100
        self.MAX_CONTOUR_HEIGHT = 100
        self.MIN_CONTOUR_AREA = 25
        self.MAX_ASPECT_RATIO = 20.0
        self.DARK_REGION_THRESHOLD = 10  # Maximum brightness for dark region detection

        # Sliding Window Parameters
        self.MIN_DARK_REGION_SIZE = 100

        # Filtering Options
        self.BORDER_MARGIN = 3  # How many pixels from space box border to consider "touching"
        self.MAX_VALID_DETECTIONS = 2  # Maximum number of valid detections per frame
        self.MAX_CONTOURS_PER_FRAME = 6  # Maximum number of contours to process per frame

        # Display Parameters
        self.DEBUG_VIEW_ENABLED = True
        self.CONTOUR_COLOR = (0, 255, 0)  # Green
        self.ANOMALY_BOX_COLOR = (0, 0, 255)  # Bright Red
        self.NOFEED_TEXT_COLOR = (128, 0, 128)  # Purple
        self.DARKNESS_OVERLAY_COLOR = (0, 0, 0)  # Black

        # Avoid Box Parameters
        self.AVOID_BOX_COLOR = (128, 0, 128)  # Blue
        self.AVOID_BOX_THICKNESS = 2

        # Save Parameters
        self.RECONNECT_DELAY = 5.0  # Seconds to wait before reconnecting

    def get_value(self, name: str):
        """Thread-safe getter for constant values."""
        with self._lock:
            return getattr(self, name)

    def update_value(self, name: str, value):
        """Thread-safe update for constant values."""
        with self._lock:
            # Update instance attribute
            setattr(self, name, value)
            # Update module-level attribute for compatibility
            setattr(sys.modules[__name__], name, value)

# Create the singleton instance
const = Constants()

# Add all constants to the module's namespace
module = sys.modules[__name__]
for name in dir(const):
    if not name.startswith('_'):
        setattr(module, name, getattr(const, name))

def update_value(name: str, value):
    """Update a constant value."""
    if hasattr(const, name):
        const.update_value(name, value)
    else:
        raise AttributeError(f"No constant named {name}")

def get_value(name: str):
    """Get a constant value."""
    return const.get_value(name)