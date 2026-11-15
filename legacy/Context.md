# Space Object Detection System

## Project Overview - Especially important.
This system is designed to detect and analyze objects in space footage, from the International Space Station (ISS) live feed. It combines deep learning (RCNN) with first principles analysis to identify various space objects and anomalies. OUR AIM IS TO DETECT OBJECTS IN THE 'space' RCNN BOUNDING BOX USING CONTOUR AND BRIGHTNESS ANALYSIS. THERE ARE MANY KNOWN PROSAIC OBJECTS TO AVOID EG. 'iss', 'lf', 'panel', 'sun' WITHIN THAT SPACE THAT CAN LOOK SIMILAR SO WE'RE RIGOUROUS IN OUR BOUNDING BOX DETECTION AWARENESS, AND ABOVE ALL SEARCHING FOR SOLID OBJECTS AGAINST THE BLACK BACKGROUND OF 'space'. 

--- A realtime parameter dashboard is a mdeium-term priority. And Test_Image_Collection short-term.

## System Architecture

### Core Components

1. **SOD_Main.py**
   - Entry point for the application
   - Handles video stream initialization
   - Orchestrates interaction between other modules
   - Manages the main processing loop
   - Provides mode selection (live feed/video file)

2. **SOD_Detection.py**
   - Implements RCNN model for object detection
   - Performs first principles analysis
   - Runs detection every 10 frames for efficiency
   - Handles anomaly detection in space regions

3. **SOD_Display.py**
   - Manages visualization of detections
   - Creates debug views for analysis
   - Handles text overlays and box drawing
   - Provides combined view capabilities

4. **SOD_Capture.py**
   - Handles saving of detections
   - Manages burst capture mode
   - Implements thread-safe saving queue
   - Maintains detection counter and directories

5. **SOD_Video.py**
   - Manages video recording functionality
   - Maintains 3-second rolling buffer
   - Handles automatic recording on detection
   - Saves .avi files with buffered context

### Support Files

1. **SOD_Constants.py**
   - Shared constants and configurations
   - Class definitions and thresholds
   - Color mappings
   - File paths and settings
   - Video recording parameters

2. **SOD_Utils.py**
   - Utility functions
   - Frame processing helpers
   - YouTube URL processing
   - Directory management

## Key Features

### Detection Capabilities
- Space regions
- Earth
- ISS components
- Light flashes (LF)
- Total darkness (TD)
- Sun
- Feed interruptions
- Anomalies

### Analysis Methods
- RCNN-based object detection
- Contour analysis
- Brightness/contrast evaluation
- Size and aspect ratio filtering
- Motion detection

### Operation Modes
- Live YouTube stream processing
- Local video file analysis
- Burst capture
- Debug visualization
- Video recording with pre-detection buffer

### Recording Features
- 3-second pre-detection buffer
- Automatic recording on anomaly detection
- 2.1-second post-detection continuation
- Numbered .avi file output
- Efficient frame buffering
- Continuous debug view throughout recording

## File Organization
```
project/
├── SOD_Main.py         # Main entry point
├── SOD_Detection.py    # Detection logic
├── SOD_Display.py      # Visualization
├── SOD_Capture.py      # Save management
├── SOD_Video.py        # Video recording
├── SOD_Constants.py    # Shared constants
├── SOD_Utils.py        # Utility functions
├── CONTEXT.md          # This file
└── requirements.txt    # Dependencies
```

## Workflow
1. User selects processing mode (live/video)
2. Video stream is initialized
3. Frames are buffered continuously
4. RCNN detection runs every 10 frames
5. First principles analysis runs on each frame
6. Detections trigger video recording
7. Recording continues until detection ends
8. Debug information displayed throughout

## Technical Notes
- RCNN model runs every 10 frames to maintain performance
- 3-second rolling buffer for context preservation
- Thread-safe saving systems
- Efficient memory management
- Robust error handling

## Dependencies
- PyTorch for RCNN model
- OpenCV for image/video processing
- yt-dlp for YouTube stream access
- numpy for numerical operations

## Usage
Primary usage is monitoring ISS live feed, with:
- Known object detection
- Anomaly identification
- Event capturing with context
- Scientific analysis
- Video evidence preservation

## Development Notes
- Use constants from SOD_Constants.py for consistency
- Maintain existing threshold values for reliable detection
- Follow established error handling patterns
- Preserve debug visualization capabilities
- Consider buffer size vs memory usage

Fixed Issues:
- Working on improving detection accuracy and visualization
- Implemented changes to save both video (.avi) and image (.jpg) captures
- Ensured proper handling of the 3-second buffer and 2.1-second post-detection recording
- Fixed issues with burst captures and raw frame saving
- Addressing initialization problems in CaptureManager
- Removed red masking overlay for improved performance
- Optimized debug view persistence throughout recording

## Detection Priorities
- Primary focus is detecting solid objects within RCNN-identified 'space' regions
- Multiple validation layers prevent false positives:
  1. RCNN identifies broad regions and known objects
  2. Contour detection finds potential anomalies
  3. Strict brightness/contrast analysis filters prosaic artifacts
  4. Cross-reference with known object locations (ISS, panels, etc.)

## Lens Flare Handling
- Critical for maintaining detection quality
- Three-tiered approach:
  1. Single/dual lens flares: Continue detection but filter overlapping regions
  2. Three+ lens flares: Pause detection for 10 seconds (too much visual noise)
  3. Track consecutive lens flare frames to prevent false positives

## Detection Thresholds
- Must preserve established values:
  - Darkness threshold: 40% frame area
  - Minimum contrast: 7 units
  - Object brightness: 12-240 range
  - Background brightness: < 35

## System Integrity
- Critical to maintain all detection features
- No arbitrary removal of established checks
- Preserve both permissive initial detection and strict filtering
- Keep all metadata for analysis and debugging
- Maintain synchronized file naming and organization

## Recent Updates (January 2024):
- Improved contour filtering:
  1. Added border margin check (5px) to filter edge detections
  2. Implemented filtering for contours overlapping with lens flares and panels
  3. Added metadata tracking for skipped saves
- Enhanced visualization:
  1. Increased color vibrancy in debug view (alpha 0.9)
  2. Expanded bounding boxes by 2px for better visibility
  3. Standardized bright red color for detections
- Detection improvements:
  1. Now searching all space boxes while displaying highest one
  2. Added duplicate detection prevention using center points
  3. Improved handling of multiple lens flares
  4. Switched to max RGB value for brightness detection (better detection of blue/red objects)
- Configuration updates:
  1. Moved lens flare threshold (MAX_LENS_FLARES = 3) to constants
  2. Updated save interval to 2.0 seconds
  3. Improved test image handling with proper 939x720 cropping
  4. Added automatic creation of required directories (AVI/JPG/Burst_raw)
  5. Reduced MAX_VALID_DETECTIONS to 4 for better quality control
  6. Removed red masking overlay for improved performance
  7. Extended debug view persistence throughout recording period

## Recent Optimizations
1. Adjusted RCNN cycle to match framerate (54 frames)
2. Balanced detection parameters for accuracy vs. speed
3. Optimized video buffer operations
4. Threaded RCNN
5. Added 100x100 sliding window filter
6. Removed unnecessary dark mask processing
7. Streamlined debug view creation
8. Improved memory usage in video recording

## Current Status:
- Continuing to refine detection algorithms for best captures
- Hunting down bugs and inefficiency
- Debug view now persists throughout entire recording period
- Improved performance by removing unnecessary processing
- Maintaining consistent visualization across all frames

### Performance Metrics
- Frame Processing: 12-13ms average
- Display Updates: <1ms
- Video Operations: <1ms
- Detection Analysis: 6-7ms for mask creation
- RCNN Operations: ~95ms (once per second)

Three things to be built out for deployment:
1. Near perfect capture, almost there. 
2. Better error handling, can't crash. - done
3. Real-time parameter dashboard.
4. Multiple space window display. - done
5. Livestream - seem like a simple process
6. Test Image Collection system
7. Enhanced logging system

Recent Additions:
1. Real-time Console for in-program Constants alteration
2. Enhanced Logging
3. Improved debug view persistence
4. Optimized video recording system
5. Better memory management


By tracking overlap of bounding boxes/detections, could build a better detector. 