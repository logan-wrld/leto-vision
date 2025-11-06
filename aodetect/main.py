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
        
        # Auto-recording feature
        self.auto_record = True  # Enable auto-recording by default
        self.auto_recording = False
        self.auto_record_start_time = None
        self.auto_record_filename = None
        self.auto_record_buffer_duration = 30.0  # Record for 30 seconds after detection
        self.auto_record_had_detection = False
        self.pending_auto_recordings = []  # Track recordings waiting for validation
        
        # Detection log
        self.detection_log = []
        self.log_file = None
        
        # Performance stats
        self.process_times = []
        self.max_process_time = 0
        self.avg_process_time = 0
        
        # Create output directories for recordings and logs
        self.output_dir = "detections_output"
        self.videos_dir = f"{self.output_dir}/videos"
        self.logs_dir = f"{self.output_dir}/logs"
        self.screenshots_dir = f"{self.output_dir}/screenshots"
        
        # Ensure directories exist
        import os
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
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
        
        # Simple box markers only
        rect_size = 15
        line_thickness = 2 if obj.is_validated else 1
        
        # Draw simple rectangle around object
        cv2.rectangle(frame, 
                     (current_pos[0] - rect_size, current_pos[1] - rect_size),
                     (current_pos[0] + rect_size, current_pos[1] + rect_size),
                     color, line_thickness)
        
        # Clean display - no extra text or badges around objects
        
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
        """Draw military-style HUD statistics panel"""
        
        if not self.show_stats:
            return
        
        height, width = frame.shape[:2]
        
        # Military-style HUD panel - larger and more prominent
        panel_width = 380
        panel_height = 320
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Dark military green background with higher opacity
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 20, 0), -1)  # Dark green
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Military green border with corner brackets
        military_green = (0, 255, 0)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     military_green, 2)
        
        # Corner brackets for military HUD look
        bracket_size = 15
        # Top-left corner
        cv2.line(frame, (panel_x, panel_y), (panel_x + bracket_size, panel_y), military_green, 3)
        cv2.line(frame, (panel_x, panel_y), (panel_x, panel_y + bracket_size), military_green, 3)
        # Top-right corner
        cv2.line(frame, (panel_x + panel_width, panel_y), (panel_x + panel_width - bracket_size, panel_y), military_green, 3)
        cv2.line(frame, (panel_x + panel_width, panel_y), (panel_x + panel_width, panel_y + bracket_size), military_green, 3)
        # Bottom-left corner
        cv2.line(frame, (panel_x, panel_y + panel_height), (panel_x + bracket_size, panel_y + panel_height), military_green, 3)
        cv2.line(frame, (panel_x, panel_y + panel_height), (panel_x, panel_y + panel_height - bracket_size), military_green, 3)
        # Bottom-right corner
        cv2.line(frame, (panel_x + panel_width, panel_y + panel_height), (panel_x + panel_width - bracket_size, panel_y + panel_height), military_green, 3)
        cv2.line(frame, (panel_x + panel_width, panel_y + panel_height), (panel_x + panel_width, panel_y + panel_height - bracket_size), military_green, 3)
        
        # Military HUD header
        cv2.putText(frame, "AERIAL SURVEILLANCE", (panel_x + 20, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, military_green, 2)
        cv2.putText(frame, "SYSTEM ACTIVE", (panel_x + 20, panel_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        
        # Divider line
        cv2.line(frame, (panel_x + 20, panel_y + 55), 
                (panel_x + panel_width - 20, panel_y + 55), military_green, 1)
        
        # Count active objects by type
        object_counts = {}
        for obj in self.detector.tracked_objects.values():
            if obj.is_validated and not obj.is_noise:
                obj_type = obj.classification
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # Start data display
        y_offset = panel_y + 75
        
        # System stats - military style
        flow_tracks = len(self.flow_tracker.tracked_objects) if hasattr(self, 'flow_tracker') else 0
        
        # Recording status with military terminology
        rec_status = "STANDBY"
        rec_color = (100, 100, 100)
        if self.recording:
            rec_status = "MANUAL REC"
            rec_color = (0, 0, 255)
        elif self.auto_recording:
            rec_status = "AUTO REC"
            rec_color = (255, 165, 0)
        
        # Military-style stats with abbreviated labels
        stats = [
            ("SYS", f"FPS {self.camera.get_fps():.0f} | {self.avg_process_time:.0f}MS"),
            ("OPT", f"FLOW {'ACTIVE' if self.enable_optical_flow else 'OFF'}"),
            ("REC", rec_status),
            ("", ""),  # Separator
            ("TGT", f"ACTIVE: {len([o for o in self.detector.tracked_objects.values() if o.is_validated])}"),
            ("TRK", f"TRACKS: {flow_tracks}"),
            ("CNF", f"VALID: {self.detector.validated_objects}"),
            ("FLT", f"NOISE: {self.detector.false_positives_filtered}"),
            ("", ""),  # Separator
            ("MOD", f"AUTO-REC {'ENABLED' if self.auto_record else 'DISABLED'}"),
        ]
        
        for label, value in stats:
            if label:  # Skip empty lines
                cv2.putText(frame, f"{label}:", (panel_x + 25, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, military_green, 1)
                
                # Special color for recording status
                if label == "REC":
                    cv2.putText(frame, value, (panel_x + 80, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 2)
                else:
                    cv2.putText(frame, value, (panel_x + 80, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Add small spacing for separator
                y_offset += 8
            y_offset += 24
        
        # Military-style threat assessment
        if object_counts:
            # Threat level header
            cv2.line(frame, (panel_x + 20, y_offset), 
                    (panel_x + panel_width - 20, y_offset), military_green, 1)
            y_offset += 10
            cv2.putText(frame, "THREAT ASSESSMENT:", (panel_x + 25, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, military_green, 1)
            y_offset += 20
            
            for obj_type, count in sorted(object_counts.items()):
                # Military classifications
                if 'aircraft' in obj_type:
                    threat_name = "AIRCRAFT"
                    threat_color = (0, 255, 255)  # Cyan for aircraft
                elif 'light' in obj_type:
                    threat_name = "NAV-LIGHT"
                    threat_color = (255, 255, 0)  # Yellow for lights
                elif 'flow' in obj_type:
                    threat_name = "MOTION-TGT"
                    threat_color = (255, 0, 255)  # Magenta for optical flow
                else:
                    threat_name = "UNKNOWN"
                    threat_color = (255, 255, 255)  # White for unknown
                
                cv2.putText(frame, f"{threat_name}", (panel_x + 30, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, threat_color, 1)
                cv2.putText(frame, f"x{count}", (panel_x + 280, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, threat_color, 2)
                y_offset += 20
        
        # Military-style status indicators in top-right
        if self.recording:
            cv2.rectangle(frame, (panel_x + panel_width - 90, panel_y + 10), 
                         (panel_x + panel_width - 10, panel_y + 35), (0, 0, 255), 2)
            cv2.putText(frame, "MANUAL", (panel_x + panel_width - 85, panel_y + 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
        elif self.auto_recording:
            cv2.rectangle(frame, (panel_x + panel_width - 90, panel_y + 10), 
                         (panel_x + panel_width - 10, panel_y + 35), (255, 165, 0), 2)
            cv2.putText(frame, "AUTO-REC", (panel_x + panel_width - 88, panel_y + 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 165, 0), 2)

    def draw_military_hud_elements(self, frame: np.ndarray):
        """Draw military-style HUD elements"""
        height, width = frame.shape[:2]
        military_green = (0, 255, 0)
        
        
        # Validation indicator (top center)
        validated_count = len([o for o in self.detector.tracked_objects.values() if o.is_validated])
        tracking_count = len([o for o in self.detector.tracked_objects.values() if not o.is_validated and not o.is_noise])
        
        status_x = width // 2 - 100
        cv2.putText(frame, f"VALIDATED: {validated_count}", (status_x, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"TRACKING: {tracking_count}", (status_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def start_auto_recording(self, frame):
        """Start auto-recording when motion is detected"""
        if self.auto_recording or not self.auto_record:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
        self.auto_record_filename = f"{self.videos_dir}/auto_detection_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        frame_size = (frame.shape[1], frame.shape[0])
        self.video_writer = cv2.VideoWriter(self.auto_record_filename, fourcc, fps, frame_size)
        
        self.auto_recording = True
        self.auto_record_start_time = time.time()
        self.auto_record_had_detection = False
        
        print(f"Auto-recording started: {self.auto_record_filename}")
    
    def stop_auto_recording(self):
        """Stop auto-recording and decide whether to keep or delete the file"""
        if not self.auto_recording or not self.video_writer:
            return
            
        # Close video writer
        self.video_writer.release()
        self.video_writer = None
        
        if self.auto_record_had_detection:
            # Keep the recording - there was a positive detection
            print(f"Auto-recording saved: {self.auto_record_filename} (Positive detection found)")
            # Add to pending list for potential cleanup later
            self.pending_auto_recordings.append({
                'filename': self.auto_record_filename,
                'timestamp': time.time(),
                'had_detection': True
            })
        else:
            # Delete the recording - no positive detection
            try:
                import os
                if os.path.exists(self.auto_record_filename):
                    os.remove(self.auto_record_filename)
                print(f"Auto-recording deleted: {self.auto_record_filename} (No detection found)")
            except Exception as e:
                print(f"Warning: Could not delete auto-recording {self.auto_record_filename}: {e}")
        
        self.auto_recording = False
        self.auto_record_start_time = None
        self.auto_record_filename = None
    
    def check_auto_recording_timeout(self):
        """Check if auto-recording should timeout"""
        if (self.auto_recording and 
            self.auto_record_start_time and 
            time.time() - self.auto_record_start_time > self.auto_record_buffer_duration):
            self.stop_auto_recording()
    
    def cleanup_old_auto_recordings(self):
        """Clean up old auto-recordings to prevent disk space issues"""
        import os
        current_time = time.time()
        cleanup_age = 3600  # 1 hour
        
        recordings_to_remove = []
        for recording in self.pending_auto_recordings:
            if current_time - recording['timestamp'] > cleanup_age:
                # Only delete recordings without detections that are old
                if not recording['had_detection']:
                    try:
                        if os.path.exists(recording['filename']):
                            os.remove(recording['filename'])
                        print(f"Cleaned up old auto-recording: {recording['filename']}")
                    except Exception as e:
                        print(f"Warning: Could not clean up {recording['filename']}: {e}")
                recordings_to_remove.append(recording)
        
        # Remove cleaned recordings from pending list
        for recording in recordings_to_remove:
            self.pending_auto_recordings.remove(recording)

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
        
        # Mark that we had a positive detection for auto-recording
        if self.auto_recording:
            self.auto_record_had_detection = True
        
        # Write to log file if open
        if self.log_file:
            self.log_file.write(json.dumps(detection_info) + '\n')
            self.log_file.flush()
    
    def run(self):
        """Main processing loop"""
        
        print("=" * 60)
        print("AERIAL OBJECT DETECTION SYSTEM v2.0")
        print("=" * 60)
        
        # Start camera
        if not self.camera.start():
            print("ERROR: Failed to start camera!")
            return
        

        print("  [Q] System Shutdown")
        print("  [T] Target Trails")
        print("  [S] HUD Display")
        print("  [B] Target Boxes")
        print("  [D] Debug Mode")
        print("  [R] Manual Recording")
        print("  [A] Auto-Record Mode")
        print("  [L] Export Log")
        print("  [C] Clear Contacts")
        print("  [+/-] Sensitivity")
        print("  [[/]] Confidence")
        print("  [P] Screenshot")
        print("  [F] Optical Flow")
        print("  [O] Flow Overlay")
        print("  [M/N] Flow Tuning")
        print("=" * 60)
        
        # Create window - larger default size with military title
        window_name = 'AERIAL OBJECT DETECTION'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        
        # Open log file
        log_filename = f"{self.logs_dir}/detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
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
                
                # Periodic cleanup of old auto-recordings (every 1000 frames)
                if frame_count % 1000 == 0:
                    self.cleanup_old_auto_recordings()
                
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
                    
                    # Check for motion to trigger auto-recording
                    has_current_motion = (len(validated_objects) > 0 or 
                                        (self.enable_optical_flow and len(flow_objects) > 0))
                    
                    # Auto-recording logic
                    if self.auto_record and not self.auto_recording and has_current_motion:
                        self.start_auto_recording(frame)
                    
                    # Check auto-recording timeout
                    if self.auto_recording:
                        self.check_auto_recording_timeout()
                    
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
                    
                    # Draw military-style UI elements
                    self.draw_stats_panel(frame)
                    self.draw_military_hud_elements(frame)
                    
                    # Draw optical flow visualization if enabled
                    if self.show_optical_flow and hasattr(self.flow_detector, 'motion_accumulator'):
                        if self.flow_detector.motion_accumulator is not None:
                            # Create overlay for motion visualization
                            motion_overlay = np.zeros_like(frame)
                            motion_viz = (self.flow_detector.motion_accumulator * 255).astype(np.uint8)
                            motion_colored = cv2.applyColorMap(motion_viz, cv2.COLORMAP_JET)
                            
                            # Blend with original frame
                            cv2.addWeighted(frame, 0.8, motion_colored, 0.2, 0, frame)
                    
                    # # Draw crosshair
                    # height, width = frame.shape[:2]
                    # center_x, center_y = width // 2, height // 2
                    # cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
                    # cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)
                    
                    # Record if enabled (manual recording or auto-recording)
                    if (self.recording or self.auto_recording) and self.video_writer is not None:
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
                    if not self.recording and not self.auto_recording:
                        # Start manual recording
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"{self.videos_dir}/manual_recording_{timestamp}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fps = 30
                        frame_size = (frame.shape[1], frame.shape[0])
                        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
                        self.recording = True
                        print(f"Manual recording started: {filename}")
                    elif self.recording:
                        # Stop manual recording
                        if self.video_writer:
                            self.video_writer.release()
                            self.video_writer = None
                        self.recording = False
                        print("Manual recording stopped")
                    else:
                        print("Cannot start manual recording - auto-recording is active")
                
                elif key == ord('a'):
                    # Toggle auto-recording feature
                    self.auto_record = not self.auto_record
                    if not self.auto_record and self.auto_recording:
                        # Stop current auto-recording if disabling feature
                        self.stop_auto_recording()
                    print(f"Auto-recording: {'ON' if self.auto_record else 'OFF'}")
                        
                elif key == ord('l'):
                    # Save detection log
                    log_save_file = f"{self.logs_dir}/detection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
                    screenshot_file = f"{self.screenshots_dir}/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
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
            
            # Close video writer and stop any active recordings
            if self.video_writer:
                self.video_writer.release()
            
            # Stop auto-recording if active
            if self.auto_recording:
                self.stop_auto_recording()
            
            # Close log file
            if self.log_file:
                self.log_file.close()
            
            # Destroy windows
            cv2.destroyAllWindows()
            
            # Print final statistics
            print("\n" + "=" * 60)
            print("    MISSION DEBRIEF - SURVEILLANCE COMPLETE")
            print("=" * 60)
            print(f"CONFIRMED TARGETS: {self.detector.validated_objects}")
            print(f"FALSE CONTACTS FILTERED: {self.detector.false_positives_filtered}")
            
            if self.detector.validated_objects + self.detector.false_positives_filtered > 0:
                filter_rate = self.detector.false_positives_filtered / (self.detector.validated_objects + self.detector.false_positives_filtered) * 100
                print(f"FILTER EFFICIENCY: {filter_rate:.1f}%")
            
            print(f"AVG PROCESSING TIME: {self.avg_process_time:.2f}ms")
            print(f"MAX PROCESSING TIME: {self.max_process_time:.2f}ms")
            print(f"INTELLIGENCE RECORDS: {len(self.detection_log)}")
            
            # Threat assessment summary
            if self.detection_log:
                type_counts = {}
                for detection in self.detection_log:
                    obj_type = detection['classification']
                    type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                
                print("\nTHREAT BREAKDOWN:")
                for obj_type, count in sorted(type_counts.items()):
                    threat_type = obj_type.replace('_', ' ').upper()
                    print(f"  {threat_type}: {count} CONTACTS")
            
            print("\nSURVEILLANCE SYSTEM: OFFLINE")


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