#!/usr/bin/env python3
"""
OpenCV Video Player with Detection Overlay + Flashing Light Detection
Detects objects AND flashing/moving lights
"""

import cv2
import threading
import time
import queue
import numpy as np
from collections import deque
from simple_camera_detector import SimpleCamera, SimpleDetector


class FlashingLightDetector:
    def __init__(self):
        self.light_history = {}  # Track light positions and brightness over time
        self.frame_buffer = deque(maxlen=10)  # Keep last 10 frames for comparison
        self.light_threshold = 200  # Brightness threshold for light detection
        self.flash_sensitivity = 30  # How much brightness must change to be "flashing"
        
        # MOVEMENT REQUIREMENTS - Only lights that move enough will be detected
        self.min_movement_threshold = 15  # MINIMUM pixels movement to register (was 10)
        self.high_speed_threshold = 25    # Fast movement threshold
        self.very_high_speed_threshold = 50  # Very fast movement threshold
        
        # Movement validation settings
        self.min_frames_for_movement = 6  # Need at least 6 frames to confirm movement
        self.movement_consistency_required = 0.6  # 60% of movement samples must exceed threshold
        
    def classify_movement_speed(self, avg_movement):
        """Classify movement speed into categories - only for moving lights"""
        if avg_movement >= self.very_high_speed_threshold:
            return "VERY_FAST", "🚁"  # Helicopter, aircraft, very fast vehicle
        elif avg_movement >= self.high_speed_threshold:
            return "FAST", "🚗"       # Fast vehicle, emergency vehicle
        elif avg_movement >= self.min_movement_threshold:
            return "MOVING", "🚶"     # Moving at detectable speed
        else:
            return "TOO_SLOW", "❌"    # Not moving enough - will be filtered out
    
    def classify_flash_intensity(self, flash_intensity):
        """Classify flashing intensity"""
        if flash_intensity >= 80:
            return "INTENSE", "🔴"    # Emergency lights, strobes
        elif flash_intensity >= 50:
            return "BRIGHT", "🟡"     # Vehicle signals, bright flashers
        elif flash_intensity >= 30:
            return "MODERATE", "🟢"   # Regular flashing lights
        else:
            return "DIM", "🔵"        # Weak flashing
    
    def validate_movement(self, history):
        """Validate that a light is actually moving enough to be detected"""
        if len(history) < self.min_frames_for_movement:
            return False, 0, 0
        
        # Get recent positions and timestamps
        recent_data = history[-self.min_frames_for_movement:]
        positions = [data['center'] for data in recent_data]
        timestamps = [data['timestamp'] for data in recent_data]
        
        movement_samples = []
        valid_movements = 0
        
        # Calculate movement between consecutive frames
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            distance = np.sqrt(dx*dx + dy*dy)
            time_diff = timestamps[i] - timestamps[i-1]
            
            # Calculate pixels per second
            if time_diff > 0:
                speed = distance / time_diff
                movement_samples.append(speed)
                
                # Count movements that exceed minimum threshold
                if speed >= self.min_movement_threshold:
                    valid_movements += 1
            else:
                movement_samples.append(distance)
                if distance >= self.min_movement_threshold:
                    valid_movements += 1
        
        if not movement_samples:
            return False, 0, 0
        
        avg_movement = sum(movement_samples) / len(movement_samples)
        max_movement = max(movement_samples)
        
        # Check if enough movement samples exceed the threshold
        consistency_ratio = valid_movements / len(movement_samples)
        is_consistently_moving = consistency_ratio >= self.movement_consistency_required
        
        # Must have average movement above threshold AND consistent movement
        is_moving_enough = (avg_movement >= self.min_movement_threshold and 
                           is_consistently_moving and 
                           max_movement >= self.min_movement_threshold * 1.5)
        
        return is_moving_enough, avg_movement, max_movement
        
    def detect_flashing_lights(self, frame):
        """Detect flashing lights - ONLY if they're moving enough"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray.copy())
        
        flashing_lights = []
        
        if len(self.frame_buffer) < 5:
            return flashing_lights
        
        # Find bright spots (potential lights)
        bright_mask = cv2.threshold(gray, self.light_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Remove noise
        kernel = np.ones((3,3), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of bright spots
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_time = time.time()
        current_lights = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20 or area > 5000:
                continue
            
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            brightness = float(gray[cy, cx])
            
            # Wider tracking for moving lights
            light_id = f"{cx//25}_{cy//25}"
            
            current_lights.append({
                'id': light_id,
                'center': (cx, cy),
                'brightness': brightness,
                'area': area,
                'contour': contour
            })
        
        # Process each light - STRICT MOVEMENT FILTERING
        for light in current_lights:
            light_id = light['id']
            center = light['center']
            brightness = light['brightness']
            
            if light_id in self.light_history:
                history = self.light_history[light_id]
                
                # Update history first
                history.append({
                    'center': center,
                    'brightness': brightness,
                    'timestamp': current_time
                })
                
                # STRICT MOVEMENT VALIDATION
                is_moving_enough, avg_movement, max_movement = self.validate_movement(history)
                
                # ONLY proceed if light is moving enough
                if not is_moving_enough:
                    # Keep tracking but don't register as detection
                    if len(history) > 25:
                        history = history[-20:]
                    self.light_history[light_id] = history
                    continue
                
                # Check for brightness changes (flashing) - only for moving lights
                brightness_changes = [abs(brightness - h['brightness']) for h in history[-6:]]
                avg_brightness_change = sum(brightness_changes) / len(brightness_changes) if brightness_changes else 0
                
                # Classify movement and flashing
                movement_class, movement_icon = self.classify_movement_speed(avg_movement)
                flash_class, flash_icon = self.classify_flash_intensity(avg_brightness_change)
                
                # FINAL CHECK: Must be flashing AND moving enough
                is_flashing = avg_brightness_change > self.flash_sensitivity
                
                if is_flashing and is_moving_enough:
                    # Calculate priority score
                    priority_score = (avg_movement / 100) + (avg_brightness_change / 100)
                    
                    flashing_lights.append({
                        'center': center,
                        'brightness': brightness,
                        'area': light['area'],
                        'contour': light['contour'],
                        'flash_intensity': avg_brightness_change,
                        'movement_speed': avg_movement,
                        'max_movement': max_movement,
                        'movement_class': movement_class,
                        'movement_icon': movement_icon,
                        'flash_class': flash_class,
                        'flash_icon': flash_icon,
                        'priority': priority_score,
                        'confidence': min(1.0, (avg_brightness_change / 100) * (avg_movement / 20)),
                        'is_high_priority': avg_movement >= self.high_speed_threshold and avg_brightness_change >= 50,
                        'movement_validated': True
                    })
                
                # Keep reasonable history length
                if len(history) > 25:
                    history = history[-20:]
                self.light_history[light_id] = history
                
            else:
                # New light - start tracking
                self.light_history[light_id] = [{
                    'center': center,
                    'brightness': brightness,
                    'timestamp': current_time
                }]
        
        # Clean up old lights
        current_light_ids = [light['id'] for light in current_lights]
        old_light_ids = []
        for light_id in self.light_history:
            if light_id not in current_light_ids:
                if current_time - self.light_history[light_id][-1]['timestamp'] > 4.0:
                    old_light_ids.append(light_id)
        
        for old_id in old_light_ids:
            del self.light_history[old_id]
        
        # Sort by priority (fastest moving first)
        flashing_lights.sort(key=lambda x: x['priority'], reverse=True)
        
        return flashing_lights


class OpenCVOverlay:
    def __init__(self):
        self.stream_url = "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp"
        self.resolution = "1280x720"
        
        self.camera = None
        self.detector = None
        self.light_detector = FlashingLightDetector()
        self.running = False
        
        # Stats
        self.fps = 0
        self.frame_count = 0
        self.total_detections = 0
        self.total_flashing_lights = 0
        self.recent_detections = []
        self.recent_lights = []
        self.detection_counts = {}
        
        # Model files
        self.weights_path = "yolov4.weights"
        self.config_path = "yolov4.cfg"
        self.names_path = "coco.names"
    
    def draw_telemetry_overlay(self, frame, flashing_lights):
        """Draw telemetry overlay on frame with flashing light info"""
        overlay = frame.copy()
        
        # Semi-transparent background for telemetry
        cv2.rectangle(overlay, (10, 10), (450, 350), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "LETO VISION - LIVE DETECTION", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw separator line
        cv2.line(frame, (20, 50), (430, 50), (0, 255, 0), 1)
        
        # Stats
        y_pos = 80
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Frames: {self.frame_count:,}",
            f"Object Detections: {self.total_detections:,}",
            f"Flashing Lights: {self.total_flashing_lights:,}",
            f"Status: {'RUNNING' if self.running else 'STOPPED'}"
        ]
        
        for stat in stats:
            cv2.putText(frame, stat, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
            y_pos += 25
        
        # Recent object detections
        y_pos += 10
        cv2.putText(frame, "Recent Objects:", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_pos += 25
        
        for detection in self.recent_detections[-3:]:  # Show last 3
            cv2.putText(frame, f"• {detection}", (25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_pos += 18
        
        # Recent flashing lights with movement classification
        y_pos += 10
        cv2.putText(frame, "Flashing Lights:", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y_pos += 25
        
        for light_info in self.recent_lights[-4:]:  # Show last 4
            cv2.putText(frame, f"• {light_info}", (25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_pos += 16
        
        # Draw flashing light detections on main frame with enhanced visualization
        for light in flashing_lights:
            center = light['center']
            movement_class = light['movement_class']
            movement_icon = light['movement_icon']
            flash_class = light['flash_class']
            flash_icon = light['flash_icon']
            is_high_priority = light['is_high_priority']
            priority = light['priority']
            
            # Choose color based on movement speed and priority
            if is_high_priority:
                circle_color = (0, 0, 255)    # RED for high priority (fast + bright)
                text_color = (0, 0, 255)
                circle_thickness = 4
            elif movement_class == "FAST":
                circle_color = (0, 165, 255)  # ORANGE for fast movement
                text_color = (0, 165, 255)
                circle_thickness = 3
            elif movement_class == "SLOW":
                circle_color = (0, 255, 255)  # YELLOW for slow movement
                text_color = (0, 255, 255)
                circle_thickness = 2
            else:
                circle_color = (255, 255, 0)  # CYAN for stationary (shouldn't happen)
                text_color = (255, 255, 0)
                circle_thickness = 2
            
            # Draw circle around detected flashing light
            cv2.circle(frame, center, 25, circle_color, circle_thickness)
            cv2.circle(frame, center, 5, circle_color, -1)   # Filled center
            
            # Draw movement classification with icon
            main_text = f"{movement_icon} {movement_class} FLASHING {flash_icon}"
            cv2.putText(frame, main_text, (center[0] - 80, center[1] - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Draw detailed info
            detail_text = f"Speed:{light['movement_speed']:.1f} Flash:{light['flash_intensity']:.1f}"
            cv2.putText(frame, detail_text, (center[0] - 80, center[1] + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw priority indicator for high priority lights
            if is_high_priority:
                priority_text = f"HIGH PRIORITY! ({priority:.1f})"
                cv2.putText(frame, priority_text, (center[0] - 80, center[1] + 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame
    
    def detection_worker(self):
        """Background detection processing for both objects and flashing lights"""
        detection_camera = SimpleCamera(self.stream_url, "640x360")  # Lower res for detection
        if not detection_camera.start():
            print("❌ Detection camera failed to start")
            return
        
        detector = SimpleDetector(self.weights_path, self.config_path, self.names_path)
        if not detector.net:
            print("❌ Detector failed to load")
            return
        
        print("✅ Detection worker started (Objects + Flashing Lights)")
        
        while self.running:
            try:
                ret, frame = detection_camera.read()
                if not ret:
                    continue
                
                # Run object detection
                detector.confidence = 0.5
                _, detections = detector.detect(frame)
                
                # Process object detections
                if detections:
                    self.total_detections += len(detections)
                    timestamp = time.strftime("%H:%M:%S")
                    
                    for detection in detections:
                        class_name = detection['class']
                        confidence = detection['confidence']
                        
                        # Add to recent detections
                        detection_text = f"[{timestamp}] {class_name} ({confidence:.2f})"
                        self.recent_detections.append(detection_text)
                        
                        # Update counts
                        if class_name not in self.detection_counts:
                            self.detection_counts[class_name] = 0
                        self.detection_counts[class_name] += 1
                    
                    # Keep only recent detections
                    if len(self.recent_detections) > 15:
                        self.recent_detections = self.recent_detections[-15:]
                
                # Run flashing light detection
                flashing_lights = self.light_detector.detect_flashing_lights(frame)
                
                # Process flashing light detections with enhanced logging
                if flashing_lights:
                    self.total_flashing_lights += len(flashing_lights)
                    timestamp = time.strftime("%H:%M:%S")
                    
                    for light in flashing_lights:
                        center = light['center']
                        flash_intensity = light['flash_intensity']
                        movement_speed = light['movement_speed']
                        movement_class = light['movement_class']
                        movement_icon = light['movement_icon']
                        flash_class = light['flash_class']
                        flash_icon = light['flash_icon']
                        is_high_priority = light['is_high_priority']
                        
                        # Enhanced logging based on movement speed
                        if is_high_priority:
                            priority_marker = "🚨🚨 PRIORITY ALERT 🚨🚨"
                            print(f"{priority_marker} {movement_icon} {movement_class} FLASHING LIGHT {flash_icon}")
                        elif movement_class == "VERY_FAST":
                            print(f"🚁 VERY FAST FLASHING LIGHT: Speed:{movement_speed:.1f} Flash:{flash_intensity:.1f}")
                        elif movement_class == "FAST":
                            print(f"🚗 FAST FLASHING LIGHT: Speed:{movement_speed:.1f} Flash:{flash_intensity:.1f}")
                        
                        # Add to recent lights with enhanced info
                        light_text = f"[{timestamp}] {movement_icon} {movement_class} at ({center[0]},{center[1]}) S:{movement_speed:.1f} F:{flash_intensity:.1f}"
                        self.recent_lights.append(light_text)
                    
                    # Keep only recent lights
                    if len(self.recent_lights) > 10:
                        self.recent_lights = self.recent_lights[-10:]
                
                time.sleep(0.1)  # Detection every 100ms
                
            except Exception as e:
                print(f"Detection worker error: {e}")
                break
        
        detection_camera.stop()
    
    def run(self):
        """Run the overlay video player"""
        # Check model files
        import os
        if not all(os.path.exists(f) for f in [self.weights_path, self.config_path, self.names_path]):
            print("❌ YOLO model files not found!")
            return
        
        # Start main camera
        self.camera = SimpleCamera(self.stream_url, self.resolution)
        if not self.camera.start():
            print("❌ Failed to connect to camera")
            return
        
        self.running = True
        
        # Start detection worker
        detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        detection_thread.start()
        
        print("🚀 Video player with overlay started")
        print("Press 'q' to quit, 'f' for fullscreen, 's' to save frame")
        print("🔦 STRICT MOVEMENT FILTER: Only lights moving 15+ pixels will be detected!")
        print("🎯 Movement thresholds: 15+ pixels (minimum), 25+ (fast), 50+ (very fast)")
        
        # Create window
        cv2.namedWindow('LETO VISION - Moving Flashing Lights Only', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('LETO VISION - Moving Flashing Lights Only', 1280, 720)
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                self.frame_count = self.camera.frame_count
                self.fps = self.camera.get_fps()
                
                # Detect flashing lights in main frame
                flashing_lights = self.light_detector.detect_flashing_lights(frame)
                
                # Process flashing lights for main display with enhanced alerts
                if flashing_lights:
                    timestamp = time.strftime("%H:%M:%S")
                    for light in flashing_lights:
                        center = light['center']
                        flash_intensity = light['flash_intensity']
                        movement_speed = light['movement_speed']
                        movement_class = light['movement_class']
                        movement_icon = light['movement_icon']
                        is_high_priority = light['is_high_priority']
                        
                        light_text = f"[{timestamp}] {movement_icon} {movement_class} at ({center[0]},{center[1]}) S:{movement_speed:.1f}"
                        
                        # Avoid duplicates but allow updates for different speeds
                        similar_found = False
                        for existing in self.recent_lights:
                            if f"({center[0]},{center[1]})" in existing:
                                similar_found = True
                                break
                        
                        if not similar_found:
                            self.recent_lights.append(light_text)
                            
                            # Enhanced console output
                            if is_high_priority:
                                print(f"🚨🚨 HIGH PRIORITY FLASHING LIGHT: {movement_icon} {movement_class} Speed:{movement_speed:.1f} 🚨🚨")
                            elif movement_class in ["VERY_FAST", "FAST"]:
                                print(f"⚡ {movement_icon} {movement_class} FLASHING LIGHT: Speed:{movement_speed:.1f}")
                            else:
                                print(f"🔦 {movement_icon} {movement_class} flashing light: Speed:{movement_speed:.1f}")
                
                # Keep recent lights list manageable
                if len(self.recent_lights) > 12:
                    self.recent_lights = self.recent_lights[-8:]
                
                # Add telemetry overlay
                frame_with_overlay = self.draw_telemetry_overlay(frame, flashing_lights)
                
                # Display frame
                cv2.imshow('LETO VISION - Moving Flashing Lights Only', frame_with_overlay)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    # Toggle fullscreen
                    cv2.setWindowProperty('LETO VISION - Moving Flashing Lights Only', 
                                        cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                elif key == ord('s'):
                    # Save frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame_with_overlay)
                    print(f"📸 Frame saved: {filename}")
                elif key == ord('m'):
                    # Adjust movement threshold
                    print(f"Current minimum movement: {self.light_detector.min_movement_threshold} pixels")
                    print("Use [ and ] keys to adjust movement threshold")
                elif key == ord('['):
                    # Decrease minimum movement (more sensitive)
                    self.light_detector.min_movement_threshold = max(5, self.light_detector.min_movement_threshold - 5)
                    print(f"Movement threshold decreased to {self.light_detector.min_movement_threshold} pixels (more sensitive)")
                elif key == ord(']'):
                    # Increase minimum movement (less sensitive)
                    self.light_detector.min_movement_threshold = min(30, self.light_detector.min_movement_threshold + 5)
                    print(f"Movement threshold increased to {self.light_detector.min_movement_threshold} pixels (less sensitive)")
                elif key == ord('l'):
                    # Adjust light sensitivity
                    print(f"Current light threshold: {self.light_detector.light_threshold}")
                    print("Use +/- keys to adjust light sensitivity")
                elif key == ord('=') or key == ord('+'):
                    # Increase sensitivity (lower threshold)
                    self.light_detector.light_threshold = max(150, self.light_detector.light_threshold - 10)
                    print(f"Light sensitivity increased (threshold: {self.light_detector.light_threshold})")
                elif key == ord('-'):
                    # Decrease sensitivity (higher threshold)
                    self.light_detector.light_threshold = min(250, self.light_detector.light_threshold + 10)
                    print(f"Light sensitivity decreased (threshold: {self.light_detector.light_threshold})")
        
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.camera.stop()
            cv2.destroyAllWindows()
            print("👋 Video player stopped")


def main():
    player = OpenCVOverlay()
    player.run()


if __name__ == "__main__":
    main()