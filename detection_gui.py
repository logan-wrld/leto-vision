#!/usr/bin/env python3
"""
Object Detection GUI - User interface for camera object detection
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
from typing import Optional
import json

from camera_detector import ObjectDetectionController


class DetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ubiquiti Camera Object Detection")
        self.root.geometry("1400x900")
        
        # Controller
        self.controller = ObjectDetectionController()
        
        # GUI variables
        self.stream_url = tk.StringVar(value="rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp")
        self.resolution = tk.StringVar(value="1280x720")
        self.weights_path = tk.StringVar(value="yolov4.weights")
        self.config_path = tk.StringVar(value="yolov4.cfg")
        self.classes_path = tk.StringVar(value="coco.names")
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        self.nms_threshold = tk.DoubleVar(value=0.4)
        self.detection_enabled = tk.BooleanVar(value=True)
        self.save_detections = tk.BooleanVar(value=False)
        
        # State variables
        self.is_running = False
        self.current_frame = None
        self.update_thread = None
        self.saved_frames = 0
        
        self.setup_gui()
        self.update_stats()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        self.setup_control_panel(main_frame)
        
        # Right panel - Video display and stats
        self.setup_display_panel(main_frame)
        
    def setup_control_panel(self, parent):
        """Setup the control panel"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.configure(width=350)
        
        row = 0
        
        # Stream settings
        ttk.Label(control_frame, text="Stream URL:").grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        stream_entry = ttk.Entry(control_frame, textvariable=self.stream_url, width=40)
        stream_entry.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1
        
        # Resolution
        ttk.Label(control_frame, text="Resolution:").grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        resolution_combo = ttk.Combobox(control_frame, textvariable=self.resolution, 
                                       values=["2688x1512", "1920x1080", "1344x756", "1280x720", "640x360"],
                                       state="readonly", width=15)
        resolution_combo.grid(row=row, column=0, sticky=tk.W, pady=(0, 10))
        row += 1
        
        # Model files section
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                              sticky=(tk.W, tk.E), pady=10)
        row += 1
        ttk.Label(control_frame, text="Model Files", font=('TkDefaultFont', 9, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 10))
        row += 1
        
        # Weights file
        ttk.Label(control_frame, text="Weights file:").grid(row=row, column=0, sticky=tk.W, pady=(0, 2))
        row += 1
        weights_frame = ttk.Frame(control_frame)
        weights_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        weights_frame.columnconfigure(0, weight=1)
        ttk.Entry(weights_frame, textvariable=self.weights_path, width=25).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(weights_frame, text="Browse", width=8,
                  command=lambda: self.browse_file(self.weights_path, "Weights files", "*.weights")).grid(row=0, column=1, padx=(5, 0))
        row += 1
        
        # Config file
        ttk.Label(control_frame, text="Config file:").grid(row=row, column=0, sticky=tk.W, pady=(0, 2))
        row += 1
        config_frame = ttk.Frame(control_frame)
        config_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        config_frame.columnconfigure(0, weight=1)
        ttk.Entry(config_frame, textvariable=self.config_path, width=25).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(config_frame, text="Browse", width=8,
                  command=lambda: self.browse_file(self.config_path, "Config files", "*.cfg")).grid(row=0, column=1, padx=(5, 0))
        row += 1
        
        # Classes file
        ttk.Label(control_frame, text="Classes file:").grid(row=row, column=0, sticky=tk.W, pady=(0, 2))
        row += 1
        classes_frame = ttk.Frame(control_frame)
        classes_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        classes_frame.columnconfigure(0, weight=1)
        ttk.Entry(classes_frame, textvariable=self.classes_path, width=25).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(classes_frame, text="Browse", width=8,
                  command=lambda: self.browse_file(self.classes_path, "Names files", "*.names")).grid(row=0, column=1, padx=(5, 0))
        row += 1
        
        # Detection settings
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                              sticky=(tk.W, tk.E), pady=10)
        row += 1
        ttk.Label(control_frame, text="Detection Settings", font=('TkDefaultFont', 9, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 10))
        row += 1
        
        # Confidence threshold
        ttk.Label(control_frame, text="Confidence Threshold:").grid(row=row, column=0, sticky=tk.W, pady=(0, 2))
        row += 1
        conf_frame = ttk.Frame(control_frame)
        conf_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Scale(conf_frame, from_=0.1, to=1.0, variable=self.confidence_threshold, 
                 orient=tk.HORIZONTAL, length=200, command=self.on_threshold_change).grid(row=0, column=0)
        self.conf_label = ttk.Label(conf_frame, text="0.50")
        self.conf_label.grid(row=0, column=1, padx=(10, 0))
        row += 1
        
        # NMS threshold
        ttk.Label(control_frame, text="NMS Threshold:").grid(row=row, column=0, sticky=tk.W, pady=(0, 2))
        row += 1
        nms_frame = ttk.Frame(control_frame)
        nms_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Scale(nms_frame, from_=0.1, to=1.0, variable=self.nms_threshold, 
                 orient=tk.HORIZONTAL, length=200, command=self.on_nms_change).grid(row=0, column=0)
        self.nms_label = ttk.Label(nms_frame, text="0.40")
        self.nms_label.grid(row=0, column=1, padx=(10, 0))
        row += 1
        
        # Options
        ttk.Checkbutton(control_frame, text="Enable Object Detection", 
                       variable=self.detection_enabled, command=self.toggle_detection).grid(
                           row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        ttk.Checkbutton(control_frame, text="Save Detection Frames", 
                       variable=self.save_detections).grid(row=row, column=0, sticky=tk.W, pady=(0, 10))
        row += 1
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        self.start_button = ttk.Button(button_frame, text="Start Detection", command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E))
        row += 1
        
        # Additional buttons
        button_frame2 = ttk.Frame(control_frame)
        button_frame2.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        button_frame2.columnconfigure(0, weight=1)
        button_frame2.columnconfigure(1, weight=1)
        
        ttk.Button(button_frame2, text="Save Frame", command=self.save_current_frame).grid(
            row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(button_frame2, text="Reset Stats", command=self.reset_stats).grid(
            row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E))
        
    def setup_display_panel(self, parent):
        """Setup the display panel"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Video display
        video_frame = ttk.LabelFrame(display_frame, text="Camera Feed", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, text="Camera feed will appear here")
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Stats display
        stats_frame = ttk.LabelFrame(display_frame, text="Statistics", padding="5")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_frame.configure(height=200)
        
        # Create notebook for different stat tabs
        self.stats_notebook = ttk.Notebook(stats_frame)
        self.stats_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Camera stats tab
        camera_stats_frame = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(camera_stats_frame, text="Camera")
        self.camera_stats_text = scrolledtext.ScrolledText(camera_stats_frame, height=8, width=50)
        self.camera_stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Detection stats tab
        detection_stats_frame = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(detection_stats_frame, text="Detections")
        self.detection_stats_text = scrolledtext.ScrolledText(detection_stats_frame, height=8, width=50)
        self.detection_stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Console tab
        console_frame = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(console_frame, text="Console")
        self.console_text = scrolledtext.ScrolledText(console_frame, height=8, width=50)
        self.console_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure stats frame
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
        
    def browse_file(self, variable, title, filetypes):
        """Browse for a file"""
        filename = filedialog.askopenfilename(title=title, filetypes=[(title, filetypes), ("All files", "*.*")])
        if filename:
            variable.set(filename)
    
    def on_threshold_change(self, value):
        """Handle confidence threshold change"""
        val = float(value)
        self.conf_label.config(text=f"{val:.2f}")
        if hasattr(self.controller, 'set_confidence_threshold'):
            self.controller.set_confidence_threshold(val)
    
    def on_nms_change(self, value):
        """Handle NMS threshold change"""
        val = float(value)
        self.nms_label.config(text=f"{val:.2f}")
        if hasattr(self.controller, 'set_nms_threshold'):
            self.controller.set_nms_threshold(val)
    
    def toggle_detection(self):
        """Toggle object detection on/off"""
        self.controller.set_detection_enabled(self.detection_enabled.get())
        self.log_to_console(f"Object detection {'enabled' if self.detection_enabled.get() else 'disabled'}")
    
    def start_detection(self):
        """Start the detection process"""
        if self.is_running:
            return
            
        # Validate model files
        if not self.validate_model_files():
            return
        
        self.log_to_console("Initializing detection...")
        
        # Initialize camera
        if not self.controller.initialize_camera(self.stream_url.get(), self.resolution.get()):
            messagebox.showerror("Error", "Failed to connect to camera stream")
            return
        
        # Initialize detector
        if not self.controller.initialize_detector(
            self.weights_path.get(), 
            self.config_path.get(), 
            self.classes_path.get()
        ):
            messagebox.showerror("Error", "Failed to load YOLO model")
            return
        
        # Set thresholds
        self.controller.set_confidence_threshold(self.confidence_threshold.get())
        self.controller.set_nms_threshold(self.nms_threshold.get())
        self.controller.set_detection_enabled(self.detection_enabled.get())
        
        # Start update thread
        self.is_running = True
        self.update_thread = threading.Thread(target=self.update_video, daemon=True)
        self.update_thread.start()
        
        # Update UI
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        self.log_to_console("Detection started successfully!")
    
    def stop_detection(self):
        """Stop the detection process"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.controller.stop()
        
        # Update UI
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.video_label.config(image="", text="Camera feed stopped")
        
        self.log_to_console("Detection stopped")
    
    def validate_model_files(self) -> bool:
        """Validate that model files exist"""
        files = [
            (self.weights_path.get(), "Weights file"),
            (self.config_path.get(), "Config file"),
            (self.classes_path.get(), "Classes file")
        ]
        
        for filepath, description in files:
            if not os.path.exists(filepath):
                messagebox.showerror("Error", f"{description} not found: {filepath}")
                return False
        
        return True
    
    def update_video(self):
        """Update video feed in separate thread"""
        while self.is_running:
            try:
                ret, frame, detections = self.controller.get_frame()
                
                if ret and frame is not None:
                    # Convert frame to display format
                    self.current_frame = frame.copy()
                    
                    # Resize frame for display
                    display_frame = self.resize_frame_for_display(frame)
                    
                    # Convert to PIL Image and then to PhotoImage
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update GUI in main thread
                    self.root.after(0, self.update_video_label, photo)
                    
                    # Save frame if requested
                    if self.save_detections.get() and detections:
                        self.save_detection_frame(frame, detections)
                    
                    # Log detections
                    if detections:
                        detection_text = f"Frame: {len(detections)} objects detected - "
                        detection_text += ", ".join([f"{det['class']}: {det['confidence']:.2f}" for det in detections])
                        self.root.after(0, self.log_to_console, detection_text)
                
                time.sleep(0.033)  # ~30 FPS max
                
            except Exception as e:
                self.root.after(0, self.log_to_console, f"Error in video update: {e}")
                break
    
    def resize_frame_for_display(self, frame, max_width=800, max_height=600):
        """Resize frame to fit display area"""
        height, width = frame.shape[:2]
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        
        return frame
    
    def update_video_label(self, photo):
        """Update video label with new frame"""
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo  # Keep a reference
    
    def save_detection_frame(self, frame, detections):
        """Save frame with detections"""
        try:
            os.makedirs("detections", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"detections/detection_{timestamp}_{self.saved_frames:04d}.jpg"
            cv2.imwrite(filename, frame)
            self.saved_frames += 1
        except Exception as e:
            self.log_to_console(f"Error saving frame: {e}")
    
    def save_current_frame(self):
        """Save current frame manually"""
        if self.current_frame is not None:
            try:
                os.makedirs("saved_frames", exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"saved_frames/frame_{timestamp}.jpg"
                cv2.imwrite(filename, self.current_frame)
                self.log_to_console(f"Frame saved: {filename}")
                messagebox.showinfo("Saved", f"Frame saved as {filename}")
            except Exception as e:
                self.log_to_console(f"Error saving frame: {e}")
                messagebox.showerror("Error", f"Failed to save frame: {e}")
        else:
            messagebox.showwarning("Warning", "No frame to save")
    
    def reset_stats(self):
        """Reset detection statistics"""
        if hasattr(self.controller, 'detector') and self.controller.detector:
            self.controller.detector.reset_stats()
        self.saved_frames = 0
        self.log_to_console("Statistics reset")
    
    def log_to_console(self, message):
        """Log message to console tab"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.console_text.insert(tk.END, log_message)
        self.console_text.see(tk.END)
        
        # Keep only last 1000 lines
        lines = int(self.console_text.index('end-1c').split('.')[0])
        if lines > 1000:
            self.console_text.delete('1.0', '100.0')
    
    def update_stats(self):
        """Update statistics display"""
        if self.is_running:
            # Update camera stats
            camera_stats = self.controller.get_camera_stats()
            camera_text = "Camera Statistics:\n"
            camera_text += f"  FPS: {camera_stats.get('fps', 0):.1f}\n"
            camera_text += f"  Frames Processed: {camera_stats.get('frames_processed', 0)}\n"
            camera_text += f"  Resolution: {camera_stats.get('resolution', 'N/A')}\n"
            camera_text += f"  Status: {'Running' if camera_stats.get('is_running', False) else 'Stopped'}\n"
            camera_text += f"  Saved Frames: {self.saved_frames}\n"
            
            self.camera_stats_text.delete('1.0', tk.END)
            self.camera_stats_text.insert('1.0', camera_text)
            
            # Update detection stats
            detection_stats = self.controller.get_detection_stats()
            detection_text = "Detection Statistics:\n"
            detection_text += f"  Total Detections: {detection_stats.get('total_detections', 0)}\n"
            detection_text += f"  Detection Enabled: {self.detection_enabled.get()}\n"
            detection_text += f"  Confidence Threshold: {self.confidence_threshold.get():.2f}\n"
            detection_text += f"  NMS Threshold: {self.nms_threshold.get():.2f}\n\n"
            
            detections_by_class = detection_stats.get('detections_by_class', {})
            if detections_by_class:
                detection_text += "Detections by Class:\n"
                for class_name, count in sorted(detections_by_class.items(), key=lambda x: x[1], reverse=True):
                    detection_text += f"  {class_name}: {count}\n"
            
            # Add processing time info
            if hasattr(self.controller, 'detector') and self.controller.detector:
                avg_time = self.controller.detector.get_average_processing_time()
                detection_text += f"\nAverage Processing Time: {avg_time:.3f}s\n"
            
            self.detection_stats_text.delete('1.0', tk.END)
            self.detection_stats_text.insert('1.0', detection_text)
        
        # Schedule next update
        self.root.after(1000, self.update_stats)
    
    def save_config(self):
        """Save current configuration"""
        config = {
            'stream_url': self.stream_url.get(),
            'resolution': self.resolution.get(),
            'weights_path': self.weights_path.get(),
            'config_path': self.config_path.get(),
            'classes_path': self.classes_path.get(),
            'confidence_threshold': self.confidence_threshold.get(),
            'nms_threshold': self.nms_threshold.get(),
            'detection_enabled': self.detection_enabled.get(),
            'save_detections': self.save_detections.get()
        }
        
        try:
            with open('detection_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            self.log_to_console("Configuration saved to detection_config.json")
        except Exception as e:
            self.log_to_console(f"Error saving config: {e}")
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open('detection_config.json', 'r') as f:
                config = json.load(f)
            
            self.stream_url.set(config.get('stream_url', ''))
            self.resolution.set(config.get('resolution', '1280x720'))
            self.weights_path.set(config.get('weights_path', 'yolov4.weights'))
            self.config_path.set(config.get('config_path', 'yolov4.cfg'))
            self.classes_path.set(config.get('classes_path', 'coco.names'))
            self.confidence_threshold.set(config.get('confidence_threshold', 0.5))
            self.nms_threshold.set(config.get('nms_threshold', 0.4))
            self.detection_enabled.set(config.get('detection_enabled', True))
            self.save_detections.set(config.get('save_detections', False))
            
            self.log_to_console("Configuration loaded from detection_config.json")
        except FileNotFoundError:
            self.log_to_console("No config file found, using defaults")
        except Exception as e:
            self.log_to_console(f"Error loading config: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_detection()
        
        self.save_config()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = DetectionGUI(root)
    
    # Load config on startup
    app.load_config()
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()