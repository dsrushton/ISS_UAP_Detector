# SOD_Utils.py
"""
Utility functions shared across the Space Object Detection system.
"""

import cv2
import numpy as np
import subprocess
from PIL import Image
import os
from typing import Optional, Tuple
import re

from SOD_Constants import (
    RAW_SUBDIR,
    VIDEO_SAVE_DIR,
    JPG_SAVE_DIR
)

def get_best_stream_url(youtube_url: str) -> str:
    """
    Get the stream URL from a YouTube link using yt-dlp.
    
    Args:
        youtube_url (str): YouTube video URL
        
    Returns:
        str: Direct stream URL or None if failed
    """
    if not youtube_url or not isinstance(youtube_url, str):
        print("Invalid YouTube URL provided")
        return None
    
    try:
        # Use format 136 (720p MP4 video) which we originally used
        print("Getting stream URL using format 136 (720p MP4)")
        cmd = ['yt-dlp', '-f', '136', '-g', '--quiet', '--no-warnings', 
               '--no-progress', '--no-check-certificates']
                
        # Add the YouTube URL
        cmd.append(youtube_url)
        
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, check=True,
            timeout=30  # Add timeout
        )
        
        url = result.stdout.strip()
        if not url:
            print("Empty URL returned from yt-dlp")
            return None
            
        print("Successfully retrieved 720p MP4 video stream URL")
        return url
        
    except subprocess.TimeoutExpired:
        print("Timeout while fetching stream URL")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error fetching 720p MP4 stream URL: {e.stderr}")
        # Try a fallback to 'best' format
        try:
            print("Trying fallback with 'best' format...")
            cmd = ['yt-dlp', '-f', 'best', '-g', '--quiet', '--no-warnings', 
                  '--no-progress', '--no-check-certificates']
                
            # Add the YouTube URL
            cmd.append(youtube_url)
            
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, check=True,
                timeout=30
            )
            url = result.stdout.strip()
            if not url:
                print("Empty URL returned from best format fallback")
                return None
            print("Successfully retrieved best available stream URL")
            return url
        except Exception as fallback_e:
            print(f"Fallback also failed: {str(fallback_e)}")
            return None
    except Exception as e:
        print(f"Unexpected error fetching stream URL: {str(e)}")
        return None

def crop_frame(frame: np.ndarray) -> np.ndarray:
    """
    Crop frame to remove unwanted edges.
    
    Args:
        frame (np.array): Input frame
        
    Returns:
        np.array: Cropped frame or None if invalid input
    """
    from SOD_Constants import CROP_LEFT, CROP_RIGHT, CROPPED_WIDTH
    
    if frame is None or not isinstance(frame, np.ndarray):
        print("Invalid frame provided to crop_frame")
        return None
        
    # Get frame dimensions
    try:
        h, w = frame.shape[:2]
    except (AttributeError, IndexError):
        print("Invalid frame dimensions")
        return None
        
    # Validate frame width
    if w <= (CROP_LEFT + CROP_RIGHT):
        print("Frame too narrow for cropping")
        return None
        
    # Ensure crop values don't exceed frame width
    left = min(CROP_LEFT, w//4)  # Limit to 1/4 of width
    right = min(CROP_RIGHT, w//4)
    
    # Validate resulting width matches expected
    cropped = frame[:, left:w-right]
    if cropped.shape[1] != CROPPED_WIDTH:
        print(f"Warning: Cropped width {cropped.shape[1]} != expected {CROPPED_WIDTH}")
        
    return cropped

def ensure_save_directory(path: str) -> int:
    """
    Ensure save directories exist and return the latest counter.
    
    Args:
        path (str): Base directory for saves
        
    Returns:
        int: Next available counter value, or 0 if error
    """
    if not path or not isinstance(path, str):
        print("Invalid path provided")
        return 0
        
    try:
        # Clean and normalize path
        path = os.path.normpath(os.path.expanduser(path))
        
        # Ensure main directory exists
        os.makedirs(path, exist_ok=True)
        
        # Ensure raw subdirectory exists
        raw_dir = os.path.join(path, RAW_SUBDIR)
        os.makedirs(raw_dir, exist_ok=True)
        
        # Find the latest counter value
        counter = 0
        if not os.path.exists(path):
            print(f"Failed to create directory: {path}")
            return 0
            
        existing_files = [f for f in os.listdir(path) 
                         if f.endswith('.jpg') and f.replace('.jpg', '').isdigit()]
        
        if existing_files:
            numbers = [int(f.replace('.jpg', '')) for f in existing_files]
            if numbers:
                counter = max(numbers) + 1
                
        return counter
        
    except (OSError, IOError) as e:
        print(f"Error ensuring save directory: {str(e)}")
        return 0
    except Exception as e:
        print(f"Unexpected error in ensure_save_directory: {str(e)}")
        return 0

def init_save_dir(path: str) -> int:
    """Initialize save directory and get next counter value."""
    try:
        path = os.path.normpath(os.path.expanduser(path))
        
        # Ensure main directories exist
        os.makedirs(path, exist_ok=True)
        os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)  # Create AVI directory
        os.makedirs(JPG_SAVE_DIR, exist_ok=True)    # Create JPG directory
        
        # Ensure raw subdirectory exists
        raw_dir = os.path.join(path, RAW_SUBDIR)
        os.makedirs(raw_dir, exist_ok=True)
        
        # Find the latest counter value
        counter = 0
        return counter
    except Exception as e:
        print(f"Error initializing save directories: {str(e)}")
        return 0