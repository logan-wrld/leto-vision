#!/usr/bin/env python3
"""
Completely Black GUI using regular Tkinter
Stable and simple interface
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import os

from simple_camera_detector import SimpleCamera, SimpleDetector


class BlackDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎯 LETO VISION - OBJECT DETECTION")
        self.root.geometry("1400x900")
        self.root.configure(bg='#000000')
        
        # State
        self.camera = None
        self.detector = None
        self.running = False
        self.current_frame = None
        
        # Variables
        self.stream_url = tk.StringVar(value="rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp")
        self.resolution = tk.StringVar(value="1280x720")
        self.confidence = tk.DoubleVar(value=0.5)
        self.detection_enabled = tk.BooleanVar(value=True)
        
        # Model paths
        self.weights_path = "yolov4.weights"
        self.config_path = "yolov4.cfg"
        self.names_path = "coco.names"
        
        self.setup_black_theme()
        self.create_gui()
        self.update_stats()
    
    def setup_black_theme(self):
        """Setup completely black theme"""
        style = ttk.Style()
        
        # Try different themes
        try:
            style.theme_use('clam')
        except:
            pass
        
        # Configure black theme
        style.configure('Black.TFrame', background='#000000', borderwidth=0)
        style.configure('Black.TLabel', background='#000000', foreground='#00ff00', font=('Consolas', 11))
        style.configure('Black.TLabelFrame', background='#000000', foreground='#00ff00', 
                       font=('Consolas', 12, 'bold'), borderwidth=1, relief='solid')
        style.configure('Black.TEntry', fieldbackground='#111111', background='#111111', 
                       foreground='#cccccc', borderwidth=1, insertcolor='#00ff00')
        style.configure('Black.TButton', background='#111111', foreground='#00ff00', 
                       font=('Consolas', 10, 'bold'), borderwidth=1)
        style.configure('Black.TCombobox', fieldbackground='#111111', background='#111111',
                       foreground='#cccccc', arrowcolor='#00ff00')
        style.configure('Black.TCheckbutton', background='#000000', foreground='#cccccc')
        style.configure('Black.TScale', background='#000000', troughcolor='#111111',
                       sliderlength=20, borderwidth=1)
        
        # Map hover states
        style.map('Black.TButton',
                 background=[('active', '#222222'), ('pressed', '#333333')],
                 foreground=[('active', '#00ff00')])
    
    def create_gui(self):
        """Create the black GUI"""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#000000')
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = tk.Label(main_frame, text="⚡ LETO VISION - AI OBJECT DETECTION ⚡", 
                        font=('Consolas', 20, 'bold'), bg='#000000', fg='#00ff00')
        title.pack(pady=20)
        
        # Content frame
        content_frame = tk.Frame(main_frame, bg='#000000')
        content_frame.pack(fill="both", expand=True)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Left panel - Controls
        self.create_controls(content_frame)
        
        # Right panel - Video and stats
        self.create_display(content_frame)
    
    def create_controls(self, parent):
        """Create control panel"""
        # Control frame with border
        control_outer = tk.Frame(parent, bg='#00ff00', bd=1)
        control_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        
        control_frame = tk.Frame(control_outer, bg='#000000', width=350)
        control_frame.pack(fill="both", expand=True, padx=1, pady=1)
        control_frame.pack_propagate(False)
        
        # Title
        tk.Label(control_frame, text="🎮 SYSTEM CONTROLS", 
                font=('Consolas', 14, 'bold'), bg='#000000', fg='#00ff00').pack(pady=20)
        
        # Stream URL
        tk.Label(control_frame, text="Stream URL:", font=('Consolas', 11, 'bold'), 
                bg='#000000', fg='#cccccc').pack(anchor="w", padx=20)
        self.url_entry = tk.Entry(control_frame, width=40, bg='#111111', fg='#cccccc', 
                                 font=('Consolas', 9), insertbackground='#00ff00', bd=1)
        self.url_entry.pack(pady=5, padx=20)
        self.url_entry.insert(0, self.stream_url.get())
        
        # Resolution
        tk.Label(control_frame, text="Resolution:", font=('Consolas', 11, 'bold'), 
                bg='#000000', fg='#cccccc').pack(anchor="w", padx=20, pady=(20,0))
        
        self.resolution_combo = ttk.Combobox(control_frame, textvariable=self.resolution,
                                           values=["2688x1512", "1920x1080", "1280x720", "640x360"],
                                           state="readonly", style='Black.TCombobox', width=18)
        self.resolution_combo.pack(pady=5, padx=20)
        
        # Confidence threshold
        tk.Label(control_frame, text="Confidence Threshold:", font=('Consolas', 11, 'bold'), 
                bg='#000000', fg='#cccccc').pack(anchor="w", padx=20, pady=(20,0))
        
        conf_frame = tk.Frame(control_frame, bg='#000000')
        conf_frame.pack(pady=5, padx=20, fill="x")
        
        self.confidence_scale = tk.Scale(conf_frame, from_=0.1, to=1.0, resolution=0.01,
                                        orient="horizontal", bg='#000000', fg='#cccccc',
                                        troughcolor='#111111', highlightthickness=0,
                                        activebackground='#00ff00', font=('Consolas', 9),
                                        variable=self.confidence, command=self.update_confidence)
        self.confidence_scale.pack(side="left", fill="x", expand=True)
        
        self.conf_label = tk.Label(conf_frame, text="0.50", bg='#000000', fg='#00ff00', 
                                  font=('Consolas', 11, 'bold'), width=6)
        self.conf_label.pack(side="right", padx=(10, 0))
        
        # Detection toggle
        self.detection_check = tk.Checkbutton(control_frame, text="🔥 Enable Object Detection",
                                            variable=self.detection_enabled, bg='#000000', 
                                            fg='#cccccc', font=('Consolas', 11, 'bold'),
                                            activebackground='#000000', activeforeground='#00ff00',
                                            selectcolor='#111111', highlightthickness=0)
        self.detection_check.pack(pady=20, padx=20)
        
        # Model files section
        model_frame = tk.Frame(control_frame, bg='#111111', bd=1, relief='solid')
        model_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(model_frame, text="🤖 YOLO MODEL FILES", 
                font=('Consolas', 12, 'bold'), bg='#111111', fg='#00ff00').pack(pady=10)
        
        select_btn = tk.Button(model_frame, text="📁 Select Model Files", 
                              bg='#222222', fg='#00ff00', font=('Consolas', 10, 'bold'),
                              activebackground='#333333', activeforeground='#00ff00',
                              bd=1, relief='solid', command=self.select_files)
        select_btn.pack(pady=10)
        
        self.model_status = tk.Label(model_frame, text="✅ Default files ready", 
                                    bg='#111111', fg='#00aa00', font=('Consolas', 9))
        self.model_status.pack(pady=(0, 10))
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#000000')
        button_frame.pack(pady=30, padx=20, fill="x")
        
        self.start_btn = tk.Button(button_frame, text="🚀 START DETECTION", 
                                  bg='#004400', fg='#00ff00', font=('Consolas', 12, 'bold'),
                                  activebackground='#006600', activeforeground='#ffffff',
                                  bd=1, relief='solid', command=self.start_detection)
        self.start_btn.pack(fill="x", pady=(0, 10), ipady=10)
        
        self.stop_btn = tk.Button(button_frame, text="🛑 STOP DETECTION", 
                                 bg='#440000', fg='#ff4444', font=('Consolas', 12, 'bold'),
                                 activebackground='#660000', activeforeground='#ffffff',
                                 bd=1, relief='solid', command=self.stop_detection, 
                                 state="disabled")
        self.stop_btn.pack(fill="x", ipady=10)
    
    def create_display(self, parent):
        """Create display panel"""
        display_frame = tk.Frame(parent, bg='#000000')
        display_frame.grid(row=0, column=1, sticky="nsew")
        display_frame.grid_rowconfigure(0, weight=2)
        display_frame.grid_rowconfigure(1, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        
        # Video display
        video_outer = tk.Frame(display_frame, bg='#00ff00', bd=1)
        video_outer.grid(row=0, column=0, sticky="nsew", pady=(0, 20))
        
        video_frame = tk.Frame(video_outer, bg='#000000')
        video_frame.pack(fill="both", expand=True, padx=1, pady=1)
        
        video_title = tk.Label(video_frame, text="📹 LIVE CAMERA FEED", 
                              font=('Consolas', 14, 'bold'), bg='#000000', fg='#00ff00')
        video_title.pack(pady=10)
        
        self.video_label = tk.Label(video_frame, text="🎬 CAMERA FEED WILL APPEAR HERE",
                                   font=('Consolas', 14), bg='#000000', fg='#666666')
        self.video_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Stats display
        stats_outer = tk.Frame(display_frame, bg='#00ff00', bd=1)
        stats_outer.grid(row=1, column=0, sticky="nsew")
        
        stats_frame = tk.Frame(stats_outer, bg='#000000')
        stats_frame.pack(fill="both", expand=True, padx=1, pady=1)
        
        stats_title = tk.Label(stats_frame, text="📊 SYSTEM STATISTICS", 
                              font=('Consolas', 14, 'bold'), bg='#000000', fg='#00ff00')
        stats_title.pack(pady=10)
        
        self.stats_text = tk.Text(stats_frame, height=12, bg='#111111', fg='#00ff00',
                                 font=('Consolas', 10), insertbackground='#00ff00',
                                 selectbackground='#333333', selectforeground='#ffffff',
                                 bd=0, wrap="word")
        self.stats_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))
    
    def update_confidence(self, value):
        """Update confidence threshold"""
        val = float(value)
        self.conf_label.configure(text=f"{val:.2f}")
        if self.detector:
            self.detector.confidence = val
    
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
                    self.model_status.configure(text="✅ Custom files selected")
    
    def start_detection(self):
        """Start detection"""
        if self.running:
            return
        
        # Get current values
        stream_url = self.url_entry.get()
        resolution = self.resolution.get()
        detection_enabled = self.detection_enabled.get()
        
        # Check model files
        if not all(os.path.exists(f) for f in [self.weights_path, self.config_path, self.names_path]):
            messagebox.showerror("Error", "YOLO model files not found!")
            return
        
        # Initialize camera
        self.camera = SimpleCamera(stream_url, resolution)
        if not self.camera.start():
            messagebox.showerror("Error", "Failed to connect to camera")
            return
        
        # Initialize detector
        self.detector = SimpleDetector(self.weights_path, self.config_path, self.names_path)
        if not self.detector.net:
            messagebox.showerror("Error", "Failed to load YOLO model")
            return
        
        # Start processing
        self.running = True
        self.detection_enabled_val = detection_enabled
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
        self.video_label.configure(image="", text="🎬 CAMERA FEED STOPPED")
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
                if self.detection_enabled.get() and self.detector:
                    self.detector.confidence = self.confidence.get()
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
    
    def resize_frame(self, frame, max_width=800, max_height=500):
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
⚙️ Confidence: {self.confidence.get():.2f}
🔥 Detection: {'ENABLED' if self.detection_enabled.get() else 'DISABLED'}
📐 Resolution: {self.resolution.get()}
🔴 Status: {'🟢 RUNNING' if self.running else '🔴 STOPPED'}

Recent Activity:
─────────────────"""
            
            self.stats_text.delete("1.0", "end")
            self.stats_text.insert("1.0", stats)
        
        # Update confidence display
        self.update_confidence(self.confidence.get())
        
        self.root.after(1000, self.update_stats)
    
    def log(self, message):
        """Add log message to stats"""
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"\n[{timestamp}] {message}"
        
        current = self.stats_text.get("1.0", "end")
        lines = current.split('\n')
        if len(lines) > 25:
            lines = lines[:20] + lines[-5:]
        
        new_content = '\n'.join(lines) + log_msg
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", new_content)
    
    def run(self):
        """Run the GUI"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            if self.running:
                self.stop_detection()


def main():
    app = BlackDetectionGUI()
    app.run()


if __name__ == "__main__":
    main()