#!/usr/bin/env python3
"""
Modern Dark GUI using CustomTkinter
Fully black interface with green accents
"""

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
from tkinter import filedialog, messagebox

from simple_camera_detector import SimpleCamera, SimpleDetector

# Set dark theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class ModernDetectionGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("🎯 LETO VISION - OBJECT DETECTION")
        self.root.geometry("1400x900")
        
        # Set completely black background
        self.root.configure(fg_color="#000000")
        
        # State
        self.camera = None
        self.detector = None
        self.running = False
        self.current_frame = None
        
        # Variables
        self.stream_url = "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp"
        self.resolution = "1280x720"
        self.confidence = 0.5
        self.detection_enabled = True
        
        # Model paths
        self.weights_path = "yolov4.weights"
        self.config_path = "yolov4.cfg"
        self.names_path = "coco.names"
        
        self.create_gui()
        self.update_stats()
    
    def create_gui(self):
        """Create the modern GUI"""
        # Main container
        main_frame = ctk.CTkFrame(self.root, fg_color="#000000")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(main_frame, text="⚡ LETO VISION - AI OBJECT DETECTION ⚡", 
                            font=ctk.CTkFont(size=24, weight="bold"),
                            text_color="#00ff00")
        title.pack(pady=20)
        
        # Create main layout
        content_frame = ctk.CTkFrame(main_frame, fg_color="#000000")
        content_frame.pack(fill="both", expand=True)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Left panel - Controls
        self.create_controls(content_frame)
        
        # Right panel - Video and stats
        self.create_display(content_frame)
    
    def create_controls(self, parent):
        """Create control panel"""
        control_frame = ctk.CTkFrame(parent, fg_color="#0a0a0a", width=350)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 20), pady=0)
        control_frame.grid_propagate(False)
        
        # Controls title
        ctk.CTkLabel(control_frame, text="🎮 SYSTEM CONTROLS", 
                    font=ctk.CTkFont(size=16, weight="bold"),
                    text_color="#00ff00").pack(pady=20)
        
        # Stream URL
        ctk.CTkLabel(control_frame, text="Stream URL:", 
                    font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=20)
        self.url_entry = ctk.CTkEntry(control_frame, width=300, height=35)
        self.url_entry.pack(pady=5, padx=20)
        self.url_entry.insert(0, self.stream_url)
        
        # Resolution
        ctk.CTkLabel(control_frame, text="Resolution:", 
                    font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=20, pady=(20,0))
        self.resolution_combo = ctk.CTkComboBox(control_frame, width=200, height=35,
                                               values=["2688x1512", "1920x1080", "1280x720", "640x360"])
        self.resolution_combo.pack(pady=5, padx=20)
        self.resolution_combo.set(self.resolution)
        
        # Confidence threshold
        ctk.CTkLabel(control_frame, text="Confidence Threshold:", 
                    font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=20, pady=(20,0))
        
        conf_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        conf_frame.pack(pady=5, padx=20, fill="x")
        
        self.confidence_slider = ctk.CTkSlider(conf_frame, from_=0.1, to=1.0, 
                                              command=self.update_confidence)
        self.confidence_slider.pack(side="left", fill="x", expand=True)
        self.confidence_slider.set(self.confidence)
        
        self.conf_label = ctk.CTkLabel(conf_frame, text="0.50", width=50,
                                      font=ctk.CTkFont(size=12, weight="bold"),
                                      text_color="#00ff00")
        self.conf_label.pack(side="right", padx=(10, 0))
        
        # Detection toggle
        self.detection_switch = ctk.CTkSwitch(control_frame, text="🔥 Enable Object Detection",
                                             font=ctk.CTkFont(size=12, weight="bold"))
        self.detection_switch.pack(pady=20, padx=20)
        self.detection_switch.select()
        
        # Model files section
        model_frame = ctk.CTkFrame(control_frame, fg_color="#111111")
        model_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(model_frame, text="🤖 YOLO MODEL FILES", 
                    font=ctk.CTkFont(size=14, weight="bold"),
                    text_color="#00ff00").pack(pady=10)
        
        ctk.CTkButton(model_frame, text="📁 Select Model Files", 
                     height=40, command=self.select_files).pack(pady=10)
        
        self.model_status = ctk.CTkLabel(model_frame, text="✅ Default files ready", 
                                        text_color="#00aa00")
        self.model_status.pack(pady=(0, 10))
        
        # Control buttons
        button_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        button_frame.pack(pady=30, padx=20, fill="x")
        
        self.start_btn = ctk.CTkButton(button_frame, text="🚀 START DETECTION", 
                                      height=50, font=ctk.CTkFont(size=14, weight="bold"),
                                      fg_color="#00aa00", hover_color="#00ff00",
                                      command=self.start_detection)
        self.start_btn.pack(fill="x", pady=(0, 10))
        
        self.stop_btn = ctk.CTkButton(button_frame, text="🛑 STOP DETECTION", 
                                     height=50, font=ctk.CTkFont(size=14, weight="bold"),
                                     fg_color="#aa0000", hover_color="#ff0000",
                                     command=self.stop_detection, state="disabled")
        self.stop_btn.pack(fill="x")
    
    def create_display(self, parent):
        """Create display panel"""
        display_frame = ctk.CTkFrame(parent, fg_color="#000000")
        display_frame.grid(row=0, column=1, sticky="nsew")
        display_frame.grid_rowconfigure(0, weight=2)
        display_frame.grid_rowconfigure(1, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        
        # Video display
        video_frame = ctk.CTkFrame(display_frame, fg_color="#0a0a0a")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 20))
        
        video_title = ctk.CTkLabel(video_frame, text="📹 LIVE CAMERA FEED", 
                                  font=ctk.CTkFont(size=16, weight="bold"),
                                  text_color="#00ff00")
        video_title.pack(pady=10)
        
        self.video_label = ctk.CTkLabel(video_frame, text="🎬 CAMERA FEED WILL APPEAR HERE",
                                       font=ctk.CTkFont(size=16),
                                       text_color="#666666")
        self.video_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Stats display
        stats_frame = ctk.CTkFrame(display_frame, fg_color="#0a0a0a")
        stats_frame.grid(row=1, column=0, sticky="nsew")
        
        stats_title = ctk.CTkLabel(stats_frame, text="📊 SYSTEM STATISTICS", 
                                  font=ctk.CTkFont(size=16, weight="bold"),
                                  text_color="#00ff00")
        stats_title.pack(pady=10)
        
        self.stats_text = ctk.CTkTextbox(stats_frame, height=200, 
                                        font=ctk.CTkFont("Consolas", 11),
                                        fg_color="#111111", text_color="#00ff00")
        self.stats_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))
    
    def update_confidence(self, value):
        """Update confidence threshold"""
        self.confidence = float(value)
        self.conf_label.configure(text=f"{self.confidence:.2f}")
        if self.detector:
            self.detector.confidence = self.confidence
    
    def select_files(self):
        """Select YOLO model files"""
        weights = filedialog.askopenfilename(title="Select YOLO weights", 
                                            filetypes=[("Weights", "*.weights")])
        if weights:
            config = filedialog.askopenfilename(title="Select YOLO config", 
                                               filetypes=[("Config", "*.cfg")])
            if config:
                names = filedialog.askopenfilename(title="Select class names", 
                                                  filetypes=[("Names", "*.names")])
                if names:
                    self.weights_path = weights
                    self.config_path = config
                    self.names_path = names
                    self.model_status.configure(text="✅ Custom files selected", 
                                               text_color="#00aa00")
    
    def start_detection(self):
        """Start detection"""
        if self.running:
            return
        
        # Get current values
        self.stream_url = self.url_entry.get()
        self.resolution = self.resolution_combo.get()
        self.detection_enabled = self.detection_switch.get()
        
        # Check model files
        if not all(os.path.exists(f) for f in [self.weights_path, self.config_path, self.names_path]):
            messagebox.showerror("Error", "YOLO model files not found!\nPlease download or select model files.")
            return
        
        # Initialize camera
        self.camera = SimpleCamera(self.stream_url, self.resolution)
        if not self.camera.start():
            messagebox.showerror("Error", "Failed to connect to camera stream")
            return
        
        # Initialize detector
        self.detector = SimpleDetector(self.weights_path, self.config_path, self.names_path)
        if not self.detector.net:
            messagebox.showerror("Error", "Failed to load YOLO model")
            return
        
        # Start processing
        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        # Start video thread
        threading.Thread(target=self.video_loop, daemon=True).start()
        
        self.log("🚀 Detection started successfully!")
    
    def stop_detection(self):
        """Stop detection"""
        self.running = False
        if self.camera:
            self.camera.stop()
        
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.video_label.configure(image=None, text="🎬 CAMERA FEED STOPPED")
        self.log("🛑 Detection stopped")
    
    def video_loop(self):
        """Main video processing loop"""
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                self.current_frame = frame.copy()
                
                # Run detection
                if self.detection_enabled and self.detector:
                    self.detector.confidence = self.confidence
                    frame, detections = self.detector.detect(frame)
                    
                    if detections:
                        detection_names = [f"{d['class']} ({d['confidence']:.2f})" for d in detections]
                        detection_text = f"🎯 Detected: {', '.join(detection_names)}"
                        self.root.after(0, self.log, detection_text)
                
                # Display frame
                display_frame = self.resize_frame(frame)
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(img)
                
                self.root.after(0, self.update_video, photo)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                self.root.after(0, self.log, f"❌ Error: {e}")
                break
    
    def resize_frame(self, frame, max_width=800, max_height=600):
        """Resize frame for display"""
        height, width = frame.shape[:2]
        scale = min(max_width/width, max_height/height)
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame
    
    def update_video(self, photo):
        """Update video display"""
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo
    
    def update_stats(self):
        """Update statistics"""
        if self.running and self.camera and self.detector:
            fps = self.camera.get_fps()
            total = self.detector.total_detections
            
            stats = f"""🎥 FPS: {fps:.1f}
📊 Frames Processed: {self.camera.frame_count:,}
🎯 Total Detections: {total:,}
⚙️ Confidence: {self.confidence:.2f}
🔥 Detection: {'ENABLED' if self.detection_enabled else 'DISABLED'}
📐 Resolution: {self.resolution}
🔴 Status: {'🟢 RUNNING' if self.running else '🔴 STOPPED'}

Recent Activity:
─────────────────"""
            
            self.stats_text.delete("0.0", "end")
            self.stats_text.insert("0.0", stats)
        
        self.root.after(1000, self.update_stats)
    
    def log(self, message):
        """Add log message to stats"""
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"\n[{timestamp}] {message}"
        
        current = self.stats_text.get("0.0", "end")
        # Keep only recent logs
        lines = current.split('\n')
        if len(lines) > 20:
            lines = lines[:15] + lines[-5:]  # Keep first 15 + last 5
        
        new_content = '\n'.join(lines) + log_msg
        self.stats_text.delete("0.0", "end")
        self.stats_text.insert("0.0", new_content)
    
    def run(self):
        """Run the GUI"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            if self.running:
                self.stop_detection()


def main():
    app = ModernDetectionGUI()
    app.run()


if __name__ == "__main__":
    main()