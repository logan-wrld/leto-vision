#!/usr/bin/env python3
"""
Terminal-based Object Detection
Clean, stable, no GUI dependencies
"""

import cv2
import threading
import time
import os
import sys
from simple_camera_detector import SimpleCamera, SimpleDetector


class TerminalDetector:
    def __init__(self):
        self.camera = None
        self.detector = None
        self.running = False
        self.detection_enabled = True
        
        # Settings
        self.stream_url = "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp"
        self.resolution = "1280x720"
        self.confidence = 0.5
        
        # Model paths
        self.weights_path = "yolov4.weights"
        self.config_path = "yolov4.cfg"
        self.names_path = "coco.names"
        
        # Stats
        self.total_detections = 0
        self.detection_counts = {}
        self.start_time = None
    
    def print_banner(self):
        """Print cool banner"""
        print("\033[32m")  # Green text
        print("╔══════════════════════════════════════════════════════════╗")
        print("║                 ⚡ LETO VISION ⚡                         ║")
        print("║              AI OBJECT DETECTION SYSTEM                  ║")
        print("║                    Terminal Mode                         ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print("\033[0m")  # Reset color
    
    def print_menu(self):
        """Print main menu"""
        print("\n\033[32m=== CONTROLS ===\033[0m")
        print("1. 🚀 Start Detection")
        print("2. 🛑 Stop Detection") 
        print("3. ⚙️  Change Settings")
        print("4. 📊 Show Stats")
        print("5. 🎯 Toggle Detection (Currently: {})".format("ON" if self.detection_enabled else "OFF"))
        print("6. 💾 Save Current Frame")
        print("7. 📂 Select Model Files")
        print("8. ❌ Exit")
        print("\nStatus: \033[32m{}\033[0m".format("RUNNING" if self.running else "STOPPED"))
    
    def print_settings(self):
        """Print current settings"""
        print("\n\033[32m=== CURRENT SETTINGS ===\033[0m")
        print(f"Stream URL: {self.stream_url}")
        print(f"Resolution: {self.resolution}")
        print(f"Confidence: {self.confidence:.2f}")
        print(f"Detection: {'ENABLED' if self.detection_enabled else 'DISABLED'}")
        print(f"Weights: {os.path.basename(self.weights_path)}")
        print(f"Config: {os.path.basename(self.config_path)}")
        print(f"Classes: {os.path.basename(self.names_path)}")
    
    def print_stats(self):
        """Print detection statistics"""
        if not self.running or not self.camera:
            print("\n❌ Detection not running")
            return
        
        fps = self.camera.get_fps()
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print("\n\033[32m=== LIVE STATISTICS ===\033[0m")
        print(f"🎥 FPS: {fps:.1f}")
        print(f"📊 Frames: {self.camera.frame_count:,}")
        print(f"⏱️  Runtime: {elapsed:.1f}s")
        print(f"🎯 Total Detections: {self.total_detections:,}")
        
        if self.detection_counts:
            print("\n📈 Detections by Class:")
            for class_name, count in sorted(self.detection_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   {class_name}: {count}")
    
    def change_settings(self):
        """Change detection settings"""
        while True:
            print("\n\033[32m=== CHANGE SETTINGS ===\033[0m")
            print("1. Stream URL")
            print("2. Resolution") 
            print("3. Confidence Threshold")
            print("4. Back to Main Menu")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                new_url = input(f"Enter new stream URL (current: {self.stream_url}): ").strip()
                if new_url:
                    self.stream_url = new_url
                    print("✅ Stream URL updated")
            
            elif choice == "2":
                print("Available resolutions:")
                resolutions = ["2688x1512", "1920x1080", "1280x720", "640x360"]
                for i, res in enumerate(resolutions, 1):
                    print(f"  {i}. {res}")
                
                res_choice = input("Select resolution (1-4): ").strip()
                try:
                    idx = int(res_choice) - 1
                    if 0 <= idx < len(resolutions):
                        self.resolution = resolutions[idx]
                        print(f"✅ Resolution set to {self.resolution}")
                except ValueError:
                    print("❌ Invalid choice")
            
            elif choice == "3":
                try:
                    new_conf = float(input(f"Enter confidence threshold (0.1-1.0, current: {self.confidence:.2f}): ").strip())
                    if 0.1 <= new_conf <= 1.0:
                        self.confidence = new_conf
                        if self.detector:
                            self.detector.confidence = new_conf
                        print(f"✅ Confidence set to {new_conf:.2f}")
                    else:
                        print("❌ Value must be between 0.1 and 1.0")
                except ValueError:
                    print("❌ Invalid number")
            
            elif choice == "4":
                break
    
    def select_model_files(self):
        """Select YOLO model files"""
        print("\n\033[32m=== SELECT MODEL FILES ===\033[0m")
        
        weights = input(f"Weights file path (current: {self.weights_path}): ").strip()
        if weights and os.path.exists(weights):
            self.weights_path = weights
            print("✅ Weights file updated")
        elif weights:
            print("❌ File not found")
        
        config = input(f"Config file path (current: {self.config_path}): ").strip()
        if config and os.path.exists(config):
            self.config_path = config
            print("✅ Config file updated")
        elif config:
            print("❌ File not found")
        
        names = input(f"Names file path (current: {self.names_path}): ").strip()
        if names and os.path.exists(names):
            self.names_path = names
            print("✅ Names file updated")
        elif names:
            print("❌ File not found")
    
    def start_detection(self):
        """Start detection process"""
        if self.running:
            print("❌ Detection already running")
            return
        
        print("🔧 Initializing detection system...")
        
        # Check model files
        if not all(os.path.exists(f) for f in [self.weights_path, self.config_path, self.names_path]):
            print("❌ YOLO model files not found! Please check file paths.")
            return
        
        # Initialize camera
        print("📹 Connecting to camera...")
        self.camera = SimpleCamera(self.stream_url, self.resolution)
        if not self.camera.start():
            print("❌ Failed to connect to camera")
            return
        
        # Initialize detector
        print("🤖 Loading YOLO model...")
        self.detector = SimpleDetector(self.weights_path, self.config_path, self.names_path)
        if not self.detector.net:
            print("❌ Failed to load YOLO model")
            return
        
        # Start detection
        self.running = True
        self.start_time = time.time()
        self.total_detections = 0
        self.detection_counts = {}
        
        # Start processing thread
        threading.Thread(target=self.detection_loop, daemon=True).start()
        
        print("✅ Detection started successfully!")
        print("📊 Stats will update automatically. Use menu to control system.")
    
    def stop_detection(self):
        """Stop detection"""
        if not self.running:
            print("❌ Detection not running")
            return
        
        print("🛑 Stopping detection...")
        self.running = False
        
        if self.camera:
            self.camera.stop()
        
        print("✅ Detection stopped")
    
    def detection_loop(self):
        """Main detection processing loop"""
        last_stats_time = time.time()
        
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Run detection if enabled
                if self.detection_enabled and self.detector:
                    self.detector.confidence = self.confidence
                    processed_frame, detections = self.detector.detect(frame)
                    
                    if detections:
                        self.total_detections += len(detections)
                        
                        # Update detection counts
                        for detection in detections:
                            class_name = detection['class']
                            if class_name not in self.detection_counts:
                                self.detection_counts[class_name] = 0
                            self.detection_counts[class_name] += 1
                        
                        # Print detections
                        detection_list = [f"{d['class']}({d['confidence']:.2f})" for d in detections]
                        print(f"\n🎯 [{time.strftime('%H:%M:%S')}] Detected: {', '.join(detection_list)}")
                
                # Show stats every 10 seconds
                if time.time() - last_stats_time > 10:
                    self.print_live_stats()
                    last_stats_time = time.time()
                
                time.sleep(0.1)  # Small delay
                
            except Exception as e:
                print(f"\n❌ Error in detection loop: {e}")
                break
    
    def print_live_stats(self):
        """Print live statistics during detection"""
        if not self.camera:
            return
        
        fps = self.camera.get_fps()
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n📊 Live Stats - FPS: {fps:.1f}, Frames: {self.camera.frame_count:,}, Detections: {self.total_detections:,}")
    
    def save_frame(self):
        """Save current frame"""
        if not self.running or not self.camera:
            print("❌ Detection not running")
            return
        
        ret, frame = self.camera.read()
        if ret:
            try:
                os.makedirs("saved_frames", exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"saved_frames/frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✅ Frame saved: {filename}")
            except Exception as e:
                print(f"❌ Error saving frame: {e}")
        else:
            print("❌ No frame available")
    
    def run(self):
        """Main application loop"""
        self.print_banner()
        
        while True:
            try:
                self.print_menu()
                choice = input("\n\033[32mSelect option (1-8): \033[0m").strip()
                
                if choice == "1":
                    self.start_detection()
                elif choice == "2":
                    self.stop_detection()
                elif choice == "3":
                    self.change_settings()
                elif choice == "4":
                    self.print_stats()
                elif choice == "5":
                    self.detection_enabled = not self.detection_enabled
                    print(f"✅ Detection {'ENABLED' if self.detection_enabled else 'DISABLED'}")
                elif choice == "6":
                    self.save_frame()
                elif choice == "7":
                    self.select_model_files()
                elif choice == "8":
                    if self.running:
                        self.stop_detection()
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid option")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
                if self.running:
                    self.stop_detection()
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    detector = TerminalDetector()
    detector.run()


if __name__ == "__main__":
    main()