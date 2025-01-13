# SOD_Utils.py
"""
Utility functions shared across the Space Object Detection system.
"""

import cv2
import numpy as np
import subprocess
from PIL import Image
import os

def get_best_stream_url(youtube_url: str) -> str:
    """
    Get the highest quality stream URL from a YouTube link using yt-dlp.
    
    Args:
        youtube_url (str): YouTube video URL
        
    Returns:
        str: Direct stream URL or None if failed
    """
    try:
        result = subprocess.run(
            ['yt-dlp', '-f', 'best', '-g', youtube_url],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error fetching stream URL: {e.stderr}")
        return None

def crop_frame(frame: np.ndarray) -> np.ndarray:
    """
    Crop frame to remove unwanted edges.
    
    Args:
        frame (np.array): Input frame
        
    Returns:
        np.array: Cropped frame
    """
    from SOD_Constants import CROP_LEFT, CROP_RIGHT
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Ensure crop values don't exceed frame width
    left = min(CROP_LEFT, w//4)  # Limit to 1/4 of width
    right = min(CROP_RIGHT, w//4)
    
    # Return cropped frame
    return frame[:, left:w-right]

def calculate_intersection(box1, box2):
    """
    Calculate intersection area between two bounding boxes.
    
    Args:
        box1 (tuple): (x1, y1, x2, y2) of first box
        box2 (tuple): (x1, y1, x2, y2) of second box
        
    Returns:
        float: Intersection area
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0
    
    return (x2 - x1) * (y2 - y1)

def ensure_save_directory(path: str) -> str:
    """
    Ensure save directories exist and return the latest counter.
    
    Args:
        base_dir (str): Base directory for saves
        
    Returns:
        int: Next available counter value
    """
    from SOD_Constants import RAW_SUBDIR
    
    # Ensure main directory exists
    os.makedirs(path, exist_ok=True)
    
    # Ensure raw subdirectory exists
    raw_dir = os.path.join(path, RAW_SUBDIR)
    os.makedirs(raw_dir, exist_ok=True)
    
    # Find the latest counter value
    counter = 0
    existing_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    if existing_files:
        numbers = []
        for f in existing_files:
            try:
                num = int(f.replace('.jpg', ''))
                numbers.append(num)
            except ValueError:
                continue
        if numbers:
            counter = max(numbers) + 1
            
    return counter