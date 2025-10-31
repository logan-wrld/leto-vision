#!/usr/bin/env python3
"""
Integrated Aerial Object Detection System
Uses SimpleCamera for connection and AerialObjectDetector for processing
Optimized for reducing false positives while detecting genuine objects
"""

import cv2
import numpy as np
import time
from collections import deque
from aerial_object_detector import AerialObjectDetector, TrackedObject
from simple_camera import SimpleCamera
from opencv_flow import OpenCVFlowDetector, OpticalFlowTracker
import argparse
import json
from datetime import datetime


class AerialDetectionSystem:
    """Complete aerial detection system with camera integration"""
    
    def __init__(self, camera_source, resolution="1280x720"):
        """
        Initialize the detection system
        Args:
            camera_source: RTSP URL, file path, or camera index
            resolution: Camera resolution
        """
        self.camera = SimpleCamera(camera_source, resolution)
        self.detector = AerialObjectDetector(min_confidence=0.6)
        
        # OpenCV Dense Optical Flow Integration
        self.flow_detector = OpenCVFlowDetector(flow_threshold=1.5)
        self.flow_tracker = OpticalFlowTracker(max_distance=60)
        self.enable_optical_flow = True
        self.show_optical_flow = False
        
        # AERIAL-ONLY MODE: High-altitude camera detecting airborne objects only
        self.detector.min_consecutive_detections = 15   # Reduced with optical flow assistance
        self.detector.min_tracking_duration = 3.0       # Reduced with better motion detection
        self.detector.brightness_threshold = 240        # Ultra-high threshold - only navigation lights
        self.detector.min_movement_per_frame = 1.0      # Aircraft move smoothly, even distant ones
        self.detector.min_total_distance = 50           # Must travel substantial distance across sky
        self.detector.max_altitude_detection = True     # Enable high-altitude specific filtering
        
        # Display settings
        self.show_trails = True
        self.show_stats = True
        self.show_detection_boxes = True
        self.trail_length = 40
        self.show_debug = False
        
        # Recording
        self.recording = False
        self.video_writer = None
        
        # Detection log
        self.detection_log = []
        self.log_file = None
        
        # Performance stats
        self.process_times = []
        self.max_process_time = 0
        self.avg_process_time = 0
        
    def merge_detections(self, brightness_objects, flow_objects):
        """Merge brightness-based and optical flow detections"""
        if not flow_objects:
            return brightness_objects
        
        merged_objects = list(brightness_objects)
        
        # Convert flow objects to TrackedObject format for compatibility
        for flow_obj in flow_objects:
            if flow_obj['track_length'] >= 3:  # Only consider stable tracks
                # Create a pseudo TrackedObject for visualization
                pseudo_obj = type('FlowObject', (), {})()
                pseudo_obj.positions = deque([flow_obj['center']], maxlen=50)
                pseudo_obj.classification = 'optical_flow_object'
                pseudo_obj.confidence = min(0.8, flow_obj['track_length'] / 10.0)
                pseudo_obj.average_speed = 5.0  # Estimated
                pseudo_obj.area_history = deque([flow_obj['area']], maxlen=50)
                pseudo_obj.is_validated = True
                pseudo_obj.consecutive_detections = flow_obj['track_length']
                
                # Check if this flow object is near any brightness object
                is_duplicate = False
                for bright_obj in brightness_objects:
                    if len(bright_obj.positions) > 0:
                        bright_center = bright_obj.positions[-1]
                        flow_center = flow_obj['center']
                        distance = np.sqrt((bright_center[0] - flow_center[0])**2 + 
                                         (bright_center[1] - flow_center[1])**2)
                        if distance < 30:  # Within 30 pixels
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    merged_objects.append(pseudo_obj)
        
        return merged_objects
        
    def draw_object(self, frame: np.ndarray, obj: TrackedObject):
        """Enhanced object drawing with better visualization"""
        
        if len(obj.positions) == 0:
            return
        
        current_pos = obj.positions[-1]
        
        # Enhanced color scheme for dual detection mode + optical flow
        color_map = {
            # Large isolated objects (10+ pixels)
            'aircraft_body': (0, 255, 0),         # Bright Green - Large aircraft body
            'large_slow_object': (0, 255, 255),   # Cyan - Drone/helicopter 
            'fast_large_object': (0, 128, 255),   # Orange - Fast large aircraft
            
            # Navigation lights (small objects)
            'aircraft_nav_light': (0, 255, 128),  # Light Green - Aircraft nav lights
            'fast_aircraft_light': (0, 200, 255), # Yellow-Orange - Fast nav lights
            'flashing_nav_light': (255, 255, 0),  # Yellow - Flashing strobes
            'supersonic_object': (255, 0, 0),     # Red - Very fast objects
            
            # Optical flow detected objects
            'optical_flow_object': (255, 0, 255), # Magenta - RAFT optical flow detection
            
            # Filtered/noise (shouldn't appear but just in case)
            'static_nav_light': (100, 100, 100),  # Dark gray - filtered
            'moving_light': (128, 128, 255),      # Light blue - unclassified moving
            'high_speed_light': (255, 128, 0),    # Orange - high speed light
            'unknown': (255, 255, 255)            # White - unclassified
        }
        
        color = color_map.get(obj.classification, (255, 255, 255))
        
        # Don't draw stationary objects unless in debug mode
        if not self.show_debug and obj.classification and 'stationary' in obj.classification:
            return
        
        # Draw trail with gradient
        if self.show_trails and len(obj.positions) > 1:
            positions = list(obj.positions)
            trail_start = max(0, len(positions) - self.trail_length)
            
            for i in range(trail_start + 1, len(positions)):
                # Gradient fade
                alpha = (i - trail_start) / (len(positions) - trail_start)
                trail_color = tuple(int(c * alpha) for c in color)
                
                pt1 = positions[i-1]
                pt2 = positions[i]
                thickness = int(1 + alpha * 2)  # Thicker line for recent positions
                cv2.line(frame, pt1, pt2, trail_color, thickness)
        
        # Different highlighting for large vs small objects
        avg_area = np.mean(list(obj.area_history)) if len(obj.area_history) > 0 else 5
        
        if avg_area >= 10:  # Large isolated objects
            # Draw thicker rectangle for large objects
            rect_size = int(15 + obj.confidence * 20)
            cv2.rectangle(frame, 
                         (current_pos[0] - rect_size, current_pos[1] - rect_size),
                         (current_pos[0] + rect_size, current_pos[1] + rect_size),
                         color, 3)
            # Add corner markers for emphasis
            cv2.line(frame, (current_pos[0] - rect_size, current_pos[1] - rect_size),
                    (current_pos[0] - rect_size + 8, current_pos[1] - rect_size), color, 3)
            cv2.line(frame, (current_pos[0] - rect_size, current_pos[1] - rect_size),
                    (current_pos[0] - rect_size, current_pos[1] - rect_size + 8), color, 3)
        else:  # Small navigation lights
            # Draw circle for small objects
            circle_size = int(10 + obj.confidence * 15)
            cv2.circle(frame, current_pos, circle_size, color, 2)
        
        # Center dot
        cv2.circle(frame, current_pos, 3, color, -1)
        
        # Draw information box
        if self.show_detection_boxes:
            # Background for text
            text = f"{obj.classification.replace('_', ' ').title()}"
            conf_text = f"Conf: {obj.confidence:.2f}"
            speed_text = f"Speed: {obj.average_speed:.1f} px/s"
            size_text = f"Size: {avg_area:.0f} px" if avg_area >= 10 else f"Light: {avg_area:.0f} px"
            
            # Calculate text sizes
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw semi-transparent background (larger for size info)
            box_x = current_pos[0] + 20
            box_y = current_pos[1] - 25
            box_w = max(text_w, 140)
            box_h = 65
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Draw text
            cv2.putText(frame, text, (box_x + 5, box_y + 15), font, font_scale, color, thickness)
            cv2.putText(frame, conf_text, (box_x + 5, box_y + 30), font, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, speed_text, (box_x + 5, box_y + 45), font, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, size_text, (box_x + 5, box_y + 60), font, 0.4, (200, 200, 200), 1)
            
            # Draw direction arrow if moving
            if obj.average_speed > 2 and len(obj.positions) >= 3:
                # Calculate direction from last few positions
                recent_pos = list(obj.positions)[-3:]
                dx = recent_pos[-1][0] - recent_pos[0][0]
                dy = recent_pos[-1][1] - recent_pos[0][1]
                
                if abs(dx) > 1 or abs(dy) > 1:
                    angle = np.arctan2(dy, dx)
                    arrow_length = 30
                    arrow_end = (
                        int(current_pos[0] + arrow_length * np.cos(angle)),
                        int(current_pos[1] + arrow_length * np.sin(angle))
                    )
                    cv2.arrowedLine(frame, current_pos, arrow_end, color, 2, tipLength=0.3)
    
    def draw_stats_panel(self, frame: np.ndarray):
        """Draw enhanced statistics panel"""
        
        if not self.show_stats:
            return
        
        height, width = frame.shape[:2]
        
        # Create larger stats panel
        panel_width = 280
        panel_height = 200
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Semi-transparent black background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 255, 0), 1)
        
        # Title
        cv2.putText(frame, "Object in sky?", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Separator
        cv2.line(frame, (panel_x + 10, panel_y + 35), 
                (panel_x + panel_width - 10, panel_y + 35), (0, 255, 0), 1)
        
        # Count active objects by type
        object_counts = {}
        for obj in self.detector.tracked_objects.values():
            if obj.is_validated and not obj.is_noise:
                obj_type = obj.classification
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # Stats
        y_offset = panel_y + 55
        
        # System stats including optical flow
        flow_tracks = len(self.flow_tracker.tracked_objects) if hasattr(self, 'flow_tracker') else 0
        stats = [
            ("Camera FPS", f"{self.camera.get_fps():.1f}"),
            ("Processing", f"{self.avg_process_time:.1f}ms"),
            ("Dense Flow", f"{'ON' if self.enable_optical_flow else 'OFF'}"),
            ("", ""),  # Separator
            ("Active Objects", f"{len([o for o in self.detector.tracked_objects.values() if o.is_validated])}"),
            ("Flow Tracks", f"{flow_tracks}"),
            ("Total Validated", f"{self.detector.validated_objects}"),
            ("Noise Filtered", f"{self.detector.false_positives_filtered}"),
        ]
        
        for label, value in stats:
            if label:  # Skip empty lines
                cv2.putText(frame, f"{label}:", (panel_x + 15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, value, (panel_x + 150, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset += 20
        
        # Object type breakdown
        if object_counts:
            y_offset += 10
            cv2.putText(frame, "Object Types:", (panel_x + 15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 15
            
            for obj_type, count in sorted(object_counts.items()):
                display_name = obj_type.replace('_', ' ').title()
                cv2.putText(frame, f"  {display_name}: {count}", 
                           (panel_x + 20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1)
                y_offset += 15
        
        # Recording indicator
        if self.recording:
            cv2.circle(frame, (panel_x + panel_width - 20, panel_y + 20), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (panel_x + panel_width - 60, panel_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def log_detection(self, obj: TrackedObject):
        """Log detection to file"""
        
        detection_info = {
            'timestamp': datetime.now().isoformat(),
            'id': obj.id,
            'classification': obj.classification,
            'confidence': float(obj.confidence),
            'average_speed': float(obj.average_speed),
            'total_distance': float(obj.total_distance_traveled),
            'tracking_duration': float(obj.last_seen - obj.first_seen),
            'direction_consistency': float(obj.direction_consistency),
            'first_position': obj.positions[0] if len(obj.positions) > 0 else None,
            'last_position': obj.positions[-1] if len(obj.positions) > 0 else None
        }
        
        self.detection_log.append(detection_info)
        
        # Write to log file if open
        if self.log_file:
            self.log_file.write(json.dumps(detection_info) + '\n')
            self.log_file.flush()
    
    def run(self):
        """Main processing loop"""
        
        print("Starting Aerial Detection System...")
        print("=" * 50)
        
        # Start camera
        if not self.camera.start():
            print("ERROR: Failed to start camera!")
            return
        
        print("Camera connected successfully")
        print("\nCONTROLS:")
        print("  q: Quit")
        print("  t: Toggle trails")
        print("  s: Toggle stats panel")
        print("  b: Toggle detection boxes")
        print("  d: Toggle debug mode (show filtered objects)")
        print("  r: Start/stop recording")
        print("  l: Save detection log")
        print("  c: Clear all detections")
        print("  +/-: Adjust brightness threshold")
        print("  [/]: Adjust minimum confidence")
        print("  p: Take screenshot")
        print("  f: Toggle dense optical flow detection")
        print("  o: Toggle optical flow visualization")
        print("  m/n: Increase/decrease flow sensitivity")
        print("=" * 50)
        
        # Create window
        window_name = 'Aerial Object Detection System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        # Open log file
        log_filename = f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.log_file = open(log_filename, 'w')
        print(f"\nLogging detections to: {log_filename}")
        
        frame_count = 0
        
        try:
            while True:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Warning: Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                try:
                    # Process frame with both brightness detection and optical flow
                    start_time = time.time()
                    
                    # Traditional brightness-based detection
                    validated_objects = self.detector.process_frame(frame)
                    
                    # RAFT Optical Flow detection for enhanced motion detection
                    flow_objects = []
                    if self.enable_optical_flow:
                        motion_objects = self.flow_detector.process_frame_pair(frame, min_area=5, max_area=200)
                        flow_objects = self.flow_tracker.update_tracks(motion_objects)
                        
                        # Merge optical flow detections with brightness detections
                        validated_objects = self.merge_detections(validated_objects, flow_objects)
                    
                    process_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Update performance stats
                    self.process_times.append(process_time)
                    if len(self.process_times) > 30:
                        self.process_times.pop(0)
                    self.avg_process_time = np.mean(self.process_times)
                    self.max_process_time = max(self.max_process_time, process_time)
                    
                    # Draw all validated objects
                    for obj in validated_objects:
                        try:
                            self.draw_object(frame, obj)
                            
                            # Log new validated objects
                            if obj.consecutive_detections == self.detector.min_consecutive_detections:
                                self.log_detection(obj)
                                classification_name = obj.classification if obj.classification else "unknown"
                                print(f"NEW DETECTION: {classification_name} (Confidence: {obj.confidence:.2f})")
                        except Exception as e:
                            print(f"Warning: Error drawing object: {e}")
                            continue
                    
                    # Draw debug objects if enabled
                    if self.show_debug:
                        for obj in self.detector.tracked_objects.values():
                            if not obj.is_validated and not obj.is_noise:
                                # Draw unvalidated objects in gray
                                if len(obj.positions) > 0:
                                    try:
                                        cv2.circle(frame, obj.positions[-1], 5, (128, 128, 128), 1)
                                    except:
                                        pass
                    
                    # Draw UI
                    self.draw_stats_panel(frame)
                    
                    # Draw optical flow visualization if enabled
                    if self.show_optical_flow and hasattr(self.flow_detector, 'motion_accumulator'):
                        if self.flow_detector.motion_accumulator is not None:
                            # Create overlay for motion visualization
                            motion_overlay = np.zeros_like(frame)
                            motion_viz = (self.flow_detector.motion_accumulator * 255).astype(np.uint8)
                            motion_colored = cv2.applyColorMap(motion_viz, cv2.COLORMAP_JET)
                            
                            # Blend with original frame
                            cv2.addWeighted(frame, 0.8, motion_colored, 0.2, 0, frame)
                    
                    # Draw crosshair
                    height, width = frame.shape[:2]
                    center_x, center_y = width // 2, height // 2
                    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
                    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)
                    
                    # Record if enabled
                    if self.recording and self.video_writer is not None:
                        self.video_writer.write(frame)
                    
                    # Display
                    cv2.imshow(window_name, frame)
                    
                except Exception as e:
                    print(f"Warning: Error processing frame: {e}")
                    # Still try to display the frame even if processing failed
                    try:
                        cv2.imshow(window_name, frame)
                    except:
                        pass
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                    
                elif key == ord('t'):
                    self.show_trails = not self.show_trails
                    print(f"Trails: {'ON' if self.show_trails else 'OFF'}")
                    
                elif key == ord('s'):
                    self.show_stats = not self.show_stats
                    print(f"Stats panel: {'ON' if self.show_stats else 'OFF'}")
                    
                elif key == ord('b'):
                    self.show_detection_boxes = not self.show_detection_boxes
                    print(f"Detection boxes: {'ON' if self.show_detection_boxes else 'OFF'}")
                    
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")
                    
                elif key == ord('r'):
                    if not self.recording:
                        # Start recording
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"aerial_recording_{timestamp}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fps = 30
                        frame_size = (frame.shape[1], frame.shape[0])
                        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
                        self.recording = True
                        print(f"Recording started: {filename}")
                    else:
                        # Stop recording
                        if self.video_writer:
                            self.video_writer.release()
                            self.video_writer = None
                        self.recording = False
                        print("Recording stopped")
                        
                elif key == ord('l'):
                    # Save detection log
                    log_save_file = f"detection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(log_save_file, 'w') as f:
                        json.dump(self.detection_log, f, indent=2)
                    print(f"Detection log saved to: {log_save_file}")
                    
                elif key == ord('c'):
                    # Clear all detections
                    self.detector.tracked_objects.clear()
                    self.detector.validated_objects = 0
                    self.detector.false_positives_filtered = 0
                    print("All detections cleared")
                    
                elif key == ord('+') or key == ord('='):
                    self.detector.brightness_threshold = max(100, self.detector.brightness_threshold - 10)
                    print(f"Brightness threshold: {self.detector.brightness_threshold} (more sensitive)")
                    
                elif key == ord('-'):
                    self.detector.brightness_threshold = min(250, self.detector.brightness_threshold + 10)
                    print(f"Brightness threshold: {self.detector.brightness_threshold} (less sensitive)")
                    
                elif key == ord('['):
                    self.detector.min_confidence = max(0.3, self.detector.min_confidence - 0.1)
                    print(f"Minimum confidence: {self.detector.min_confidence:.1f}")
                    
                elif key == ord(']'):
                    self.detector.min_confidence = min(0.9, self.detector.min_confidence + 0.1)
                    print(f"Minimum confidence: {self.detector.min_confidence:.1f}")
                    
                elif key == ord('p'):
                    # Take screenshot
                    screenshot_file = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(screenshot_file, frame)
                    print(f"Screenshot saved: {screenshot_file}")
                    
                elif key == ord('f'):
                    # Toggle optical flow
                    self.enable_optical_flow = not self.enable_optical_flow
                    print(f"Dense Optical Flow Detection: {'ON' if self.enable_optical_flow else 'OFF'}")
                    
                elif key == ord('o'):
                    # Toggle optical flow visualization
                    self.show_optical_flow = not self.show_optical_flow
                    print(f"Optical Flow Visualization: {'ON' if self.show_optical_flow else 'OFF'}")
                    
                elif key == ord('m'):
                    # Adjust optical flow sensitivity
                    if self.enable_optical_flow:
                        self.flow_detector.flow_threshold = max(0.5, self.flow_detector.flow_threshold - 0.5)
                        print(f"Flow sensitivity increased (threshold: {self.flow_detector.flow_threshold:.1f})")
                    
                elif key == ord('n'):
                    # Decrease optical flow sensitivity
                    if self.enable_optical_flow:
                        self.flow_detector.flow_threshold = min(5.0, self.flow_detector.flow_threshold + 0.5)
                        print(f"Flow sensitivity decreased (threshold: {self.flow_detector.flow_threshold:.1f})")
                
        except KeyboardInterrupt:
            print("\nShutdown requested...")
            
        finally:
            # Cleanup
            print("\nCleaning up...")
            
            # Stop camera
            self.camera.stop()
            
            # Close video writer
            if self.video_writer:
                self.video_writer.release()
            
            # Close log file
            if self.log_file:
                self.log_file.close()
            
            # Destroy windows
            cv2.destroyAllWindows()
            
            # Print final statistics
            print("\n" + "=" * 50)
            print("FINAL STATISTICS")
            print("=" * 50)
            print(f"Total validated objects: {self.detector.validated_objects}")
            print(f"False positives filtered: {self.detector.false_positives_filtered}")
            
            if self.detector.validated_objects + self.detector.false_positives_filtered > 0:
                filter_rate = self.detector.false_positives_filtered / (self.detector.validated_objects + self.detector.false_positives_filtered) * 100
                print(f"Noise filter effectiveness: {filter_rate:.1f}%")
            
            print(f"Average processing time: {self.avg_process_time:.2f}ms")
            print(f"Max processing time: {self.max_process_time:.2f}ms")
            print(f"Total detections logged: {len(self.detection_log)}")
            
            # Summary by type
            if self.detection_log:
                type_counts = {}
                for detection in self.detection_log:
                    obj_type = detection['classification']
                    type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                
                print("\nDetections by type:")
                for obj_type, count in sorted(type_counts.items()):
                    print(f"  {obj_type.replace('_', ' ').title()}: {count}")
            
            print("\nShutdown complete.")


def main():
    """Main entry point"""
    import signal
    
    parser = argparse.ArgumentParser(description='Aerial Object Detection System')
    parser.add_argument('--source', default='0',
                      help='Camera source (0 for webcam, RTSP URL, or video file)')
    parser.add_argument('--resolution', default='1280x720',
                      help='Camera resolution (e.g., 1920x1080, 1280x720, 640x480)')
    parser.add_argument('--confidence', type=float, default=0.6,
                      help='Minimum confidence threshold (0.0-1.0)')
    parser.add_argument('--brightness', type=int, default=190,
                      help='Brightness threshold (100-250, lower = more sensitive)')
    
    args = parser.parse_args()
    
    # Create detection system
    system = AerialDetectionSystem(args.source, args.resolution)
    
    # Apply settings
    system.detector.min_confidence = args.confidence
    system.detector.brightness_threshold = args.brightness
    
    # For your RTSP camera, you would use:
    # system = AerialDetectionSystem("rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp", "1280x720")
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutdown signal received...")
        system.camera.running = False
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Run the system
        system.run()
    except Exception as e:
        print(f"\nError in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nExiting...")
        try:
            system.camera.stop()
        except:
            pass


if __name__ == "__main__":
    main()