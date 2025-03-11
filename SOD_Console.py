"""
Space Object Detection Console
Provides real-time control over detection parameters.
"""

import tkinter as tk
from tkinter import ttk
import importlib
import sys
import time
import threading
import queue

class ParameterConsole:
    def __init__(self):
        """Initialize the parameter console."""
        self.root = None
        self.is_running = False
        self.current_values = {}  # Store current values in memory
        self.update_queue = queue.Queue()
        self._main_thread = None
        
    def initialize(self):
        """Initialize the GUI components."""
        if self.root is not None:
            return
            
        self.root = tk.Tk()
        self.root.title("SOD Parameter Console")
        self.root.lift()  # Bring window to front
        self.root.attributes('-topmost', True)  # Keep window on top
        
        # Store main thread reference
        self._main_thread = threading.current_thread()
        
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create sections
        self.create_filter_controls(main_frame, 0)
        self.create_rcnn_controls(main_frame, 1)
        self.create_anomaly_detection(main_frame, 2)
        
        # Add apply and reset buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        apply_btn = ttk.Button(btn_frame, text="Apply Changes", command=self.apply_changes)
        apply_btn.pack(side=tk.LEFT, padx=5)
        
        reset_btn = ttk.Button(btn_frame, text="Reset", command=self.load_current_values)
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Store original values
        self.original_values = {}
        self.load_current_values()
        
        # Configure window
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.resizable(False, False)
        
        # Schedule periodic update processing
        self._schedule_update_processing()
        
    def _schedule_update_processing(self):
        """Schedule periodic processing of constant updates."""
        if self.is_running and self.root is not None:
            try:
                # Process any pending updates in the queue
                while not self.update_queue.empty():
                    try:
                        update_func = self.update_queue.get_nowait()
                        if callable(update_func):
                            update_func()
                        self.update_queue.task_done()
                    except queue.Empty:
                        break
                
                # Schedule next check
                self.root.after(100, self._schedule_update_processing)
            except Exception as e:
                print(f"Error processing updates: {e}")

    def _queue_gui_update(self, func):
        """Queue a GUI update to be executed in the main thread."""
        if threading.current_thread() is self._main_thread:
            func()
        else:
            self.update_queue.put(func)
            
    def create_filter_controls(self, parent, row):
        """Create Filter Controls section."""
        frame = ttk.LabelFrame(parent, text="Filter Box Controls", padding="5")
        frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Filter checkboxes
        self.filter_iss_var = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Filter ISS", variable=self.filter_iss_var).grid(row=0, column=0, sticky=tk.W)
        
        self.filter_panel_var = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Filter Panels", variable=self.filter_panel_var).grid(row=1, column=0, sticky=tk.W)
        
        self.filter_lf_var = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Filter Lens Flares", variable=self.filter_lf_var).grid(row=2, column=0, sticky=tk.W)
        
    def create_rcnn_controls(self, parent, row):
        """Create RCNN Controls section."""
        frame = ttk.LabelFrame(parent, text="RCNN Controls", padding="5")
        frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Max Lens Flares
        ttk.Label(frame, text="MAX_LENS_FLARES:").grid(row=0, column=0, sticky=tk.W)
        self.max_lf_var = tk.IntVar()
        ttk.Entry(frame, textvariable=self.max_lf_var, width=10).grid(row=0, column=1, padx=5)
        
        # Darkness Area Threshold
        ttk.Label(frame, text="DARKNESS_AREA_THRESHOLD:").grid(row=1, column=0, sticky=tk.W)
        self.darkness_var = tk.DoubleVar()
        ttk.Entry(frame, textvariable=self.darkness_var, width=10).grid(row=1, column=1, padx=5)
        
        # Max RCNN Boxes
        ttk.Label(frame, text="MAX_RCNN_BOXES:").grid(row=2, column=0, sticky=tk.W)
        self.max_boxes_var = tk.IntVar()
        ttk.Entry(frame, textvariable=self.max_boxes_var, width=10).grid(row=2, column=1, padx=5)
        
    def create_anomaly_detection(self, parent, row):
        """Create Anomaly Detection section."""
        frame = ttk.LabelFrame(parent, text="Anomaly Detection", padding="5")
        frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        params = [
            ("MIN_OBJECT_BRIGHTNESS", "min_bright_var", tk.IntVar),
            ("MAX_OBJECT_BRIGHTNESS", "max_bright_var", tk.IntVar),
            ("GAUSSIAN_BLUR_SIZE", "blur_var", tk.IntVar),
            ("MORPH_KERNEL_SIZE", "morph_var", tk.IntVar),
            ("MAX_BG_BRIGHTNESS", "max_bg_var", tk.IntVar),
            ("MIN_CONTRAST", "min_contrast_var", tk.IntVar),
            ("MIN_CONTOUR_DIMENSION", "min_contour_var", tk.IntVar),
            ("DARK_REGION_THRESHOLD", "dark_region_var", tk.IntVar)
        ]
        
        for i, (text, var_name, var_type) in enumerate(params):
            ttk.Label(frame, text=f"{text}:").grid(row=i, column=0, sticky=tk.W)
            setattr(self, var_name, var_type())
            ttk.Entry(frame, textvariable=getattr(self, var_name), width=10).grid(row=i, column=1, padx=5)
            
    def load_current_values(self):
        """Load current values from SOD_Constants."""
        def _load():
            try:
                import SOD_Constants as const
                importlib.reload(const)  # Reload to get fresh values
                
                # Filter Controls
                self.filter_iss_var.set(const.FILTER_ISS)
                self.filter_panel_var.set(const.FILTER_PANEL)
                self.filter_lf_var.set(const.FILTER_LF)
                
                # RCNN Controls
                self.max_lf_var.set(const.MAX_LENS_FLARES)
                self.darkness_var.set(const.DARKNESS_AREA_THRESHOLD)
                self.max_boxes_var.set(const.MAX_RCNN_BOXES)
                
                # Anomaly Detection
                self.min_bright_var.set(const.MIN_OBJECT_BRIGHTNESS)
                self.max_bright_var.set(const.MAX_OBJECT_BRIGHTNESS)
                self.blur_var.set(const.GAUSSIAN_BLUR_SIZE)
                self.morph_var.set(const.MORPH_KERNEL_SIZE)
                self.max_bg_var.set(const.MAX_BG_BRIGHTNESS)
                self.min_contrast_var.set(const.MIN_CONTRAST)
                self.min_contour_var.set(const.MIN_CONTOUR_DIMENSION)
                self.dark_region_var.set(const.DARK_REGION_THRESHOLD)
                
                # Store original values
                self.original_values = {
                    'MAX_LENS_FLARES': const.MAX_LENS_FLARES,
                    'DARKNESS_AREA_THRESHOLD': const.DARKNESS_AREA_THRESHOLD,
                    'MAX_RCNN_BOXES': const.MAX_RCNN_BOXES,
                    'MIN_OBJECT_BRIGHTNESS': const.MIN_OBJECT_BRIGHTNESS,
                    'MAX_OBJECT_BRIGHTNESS': const.MAX_OBJECT_BRIGHTNESS,
                    'GAUSSIAN_BLUR_SIZE': const.GAUSSIAN_BLUR_SIZE,
                    'MORPH_KERNEL_SIZE': const.MORPH_KERNEL_SIZE,
                    'MAX_BG_BRIGHTNESS': const.MAX_BG_BRIGHTNESS,
                    'MIN_CONTRAST': const.MIN_CONTRAST,
                    'MIN_CONTOUR_DIMENSION': const.MIN_CONTOUR_DIMENSION,
                    'DARK_REGION_THRESHOLD': const.DARK_REGION_THRESHOLD
                }
                
                # Initialize current values with original values
                self.current_values = self.original_values.copy()
                
            except Exception as e:
                print(f"Error loading values: {str(e)}")
        
        self._queue_gui_update(_load)
        
    def apply_changes(self):
        """Apply changes to in-memory constants."""
        def _apply():
            try:
                # Get all new values first
                new_values = {
                    # Filter Controls
                    'FILTER_ISS': self.filter_iss_var.get(),
                    'FILTER_PANEL': self.filter_panel_var.get(),
                    'FILTER_LF': self.filter_lf_var.get(),
                    
                    # Existing values
                    'MAX_LENS_FLARES': self.max_lf_var.get(),
                    'DARKNESS_AREA_THRESHOLD': self.darkness_var.get(),
                    'MAX_RCNN_BOXES': self.max_boxes_var.get(),
                    'MIN_OBJECT_BRIGHTNESS': self.min_bright_var.get(),
                    'MAX_OBJECT_BRIGHTNESS': self.max_bright_var.get(),
                    'GAUSSIAN_BLUR_SIZE': self.blur_var.get(),
                    'MORPH_KERNEL_SIZE': self.morph_var.get(),
                    'MAX_BG_BRIGHTNESS': self.max_bg_var.get(),
                    'MIN_CONTRAST': self.min_contrast_var.get(),
                    'MIN_CONTOUR_DIMENSION': self.min_contour_var.get(),
                    'DARK_REGION_THRESHOLD': self.dark_region_var.get()
                }
                
                # Validate values
                if new_values['GAUSSIAN_BLUR_SIZE'] % 2 == 0:
                    print("Warning: GAUSSIAN_BLUR_SIZE must be odd")
                    return
                    
                # Update current values in memory
                self.current_values.update(new_values)
                
                # Update the constants using the new update_value function
                import SOD_Constants as const
                for name, value in new_values.items():
                    const.update_value(name, value)
                
            except Exception as e:
                print(f"Error updating parameters: {str(e)}")
                self.load_current_values()
        
        self._queue_gui_update(_apply)
        
    def on_closing(self):
        """Handle window closing event."""
        self.is_running = False
        if self.root:
            self.root.destroy()
            self.root = None
        
    def start(self):
        """Start the console in a separate thread."""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Run the console in a separate thread
        console_thread = threading.Thread(target=self._run_console, daemon=True)
        console_thread.start()
        
    def _run_console(self):
        """Run the console in a separate thread."""
        try:
            # Initialize GUI components
            self.initialize()
            
            # Start the main loop
            if self.root:
                self.root.mainloop()
        except Exception as e:
            print(f"Error in console thread: {str(e)}")
        finally:
            self.is_running = False
                
    def stop(self):
        """Stop the console."""
        self.is_running = False
        if self.root:
            self.root.quit()
            self.root = None

def main():
    console = ParameterConsole()
    console.start()

if __name__ == "__main__":
    main() 