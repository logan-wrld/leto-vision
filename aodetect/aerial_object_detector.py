#!/usr/bin/env python3
"""
Aerial Object Detection System
Detects genuine moving objects (aircraft, distant vehicles, UAPs) while filtering noise
"""

import cv2
import numpy as np
import time
from collections import deque, defaultdict
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class TrackedObject:
    """Represents a tracked moving object"""
    id: str
    positions: deque
    timestamps: deque
    brightness_history: deque
    area_history: deque
    first_seen: float
    last_seen: float
    confidence: float = 0.0
    classification: str = "unknown"
    is_validated: bool = False
    consecutive_detections: int = 0
    total_distance_traveled: float = 0.0
    average_speed: float = 0.0
    direction_consistency: float = 0.0
    is_noise: bool = False


class NoiseFilter:
    """Advanced noise filtering to reduce false positives"""
    
    def __init__(self):
        self.noise_patterns = deque(maxlen=100)
        self.static_light_positions = set()
        self.known_noise_regions = []
        
    def is_likely_noise(self, obj: TrackedObject) -> bool:
        """Determine if an object is likely noise"""
        
        # Check for erratic movement (noise tends to jump randomly)
        if len(obj.positions) >= 5:
            positions = list(obj.positions)[-5:]
            
            # Calculate movement vectors
            vectors = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                vectors.append((dx, dy))
            
            # Check for random/erratic movement
            if len(vectors) >= 2:
                # Calculate angle changes
                angle_changes = []
                for i in range(1, len(vectors)):
                    v1 = vectors[i-1]
                    v2 = vectors[i]
                    
                    # Skip if no movement
                    if (v1[0] == 0 and v1[1] == 0) or (v2[0] == 0 and v2[1] == 0):
                        continue
                    
                    # Calculate angle between vectors
                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                    mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
                    
                    if mag1 > 0 and mag2 > 0:
                        cos_angle = dot / (mag1 * mag2)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle) * 180 / np.pi
                        angle_changes.append(angle)
                
                # High angle changes indicate erratic/noise movement
                if angle_changes:
                    avg_angle_change = np.mean(angle_changes)
                    if avg_angle_change > 120:  # Very erratic
                        return True
        
        # AERIAL MODE: Enhanced ground light filtering from high altitude
        if len(obj.positions) >= 8:
            positions = list(obj.positions)
            
            # Check if object barely moves (building lights, stationary ground sources)
            total_movement = 0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                movement = np.sqrt(dx*dx + dy*dy)
                total_movement += movement
            
            avg_movement_per_frame = total_movement / (len(positions) - 1)
            
            # Aircraft from altitude should show consistent movement
            if avg_movement_per_frame < 0.5:  # Even stricter for aerial detection
                return True
            
            # Check for truly stationary objects (building lights, etc.)
            if len(positions) >= 10:
                # Look for any significant movement over time
                first_pos = positions[0]
                last_pos = positions[-1]
                total_displacement = np.sqrt(
                    (last_pos[0] - first_pos[0])**2 + 
                    (last_pos[1] - first_pos[1])**2
                )
                
                # If total displacement is tiny, it's likely a ground light
                if total_displacement < 15:  # Must move at least 15 pixels total
                    return True
            
            # Aircraft should show relatively linear movement patterns
            if len(positions) >= 6:
                # Check for too much oscillation (noise on building lights)
                x_positions = [p[0] for p in positions[-6:]]
                y_positions = [p[1] for p in positions[-6:]]
                
                x_range = max(x_positions) - min(x_positions)
                y_range = max(y_positions) - min(y_positions)
                
                # If movement is confined to tiny area, likely ground noise
                if x_range < 3 and y_range < 3:
                    return True
        
        # Check for static/vibrating position (camera shake, compression artifacts)
        if len(obj.positions) >= 10:
            positions = list(obj.positions)[-10:]
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            
            x_variance = np.var(x_coords)
            y_variance = np.var(y_coords)
            
            # Very low variance with many detections = static noise
            if x_variance < 4 and y_variance < 4 and obj.consecutive_detections > 20:
                return True
        
        # Check brightness consistency (real objects have more stable brightness)
        if len(obj.brightness_history) >= 5:
            recent_brightness = list(obj.brightness_history)[-5:]
            brightness_variance = np.var(recent_brightness)
            
            # Extremely variable brightness = likely noise/artifact
            if brightness_variance > 1000:
                return True
        
        return False
    
    def add_static_light(self, position: Tuple[int, int]):
        """Mark a position as having a static light source"""
        # Use grid cells to group nearby positions
        grid_x = position[0] // 20
        grid_y = position[1] // 20
        self.static_light_positions.add((grid_x, grid_y))
    
    def is_near_static_light(self, position: Tuple[int, int], threshold: int = 30) -> bool:
        """Check if position is near a known static light"""
        grid_x = position[0] // 20
        grid_y = position[1] // 20
        
        # Check surrounding grid cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (grid_x + dx, grid_y + dy) in self.static_light_positions:
                    return True
        return False


class AerialObjectDetector:
    """Main detector for aerial objects with robust filtering"""
    
    def __init__(self, min_confidence: float = 0.6):
        # Detection parameters
        self.min_confidence = min_confidence
        self.min_consecutive_detections = 8  # Need 8 frames to confirm
        self.min_tracking_duration = 1.5  # Seconds
        self.max_tracking_gap = 0.5  # Max seconds between detections
        
        # Movement thresholds - AERIAL OBJECTS from high altitude camera
        self.min_movement_per_frame = 0.8  # Aircraft can appear slow from high altitude
        self.max_movement_per_frame = 20   # Max realistic for distant aircraft
        self.min_total_distance = 40      # Must traverse significant portion of sky
        self.max_altitude_detection = False  # Will be enabled by main.py
        
        # Detection sensitivity - AERIAL NAVIGATION LIGHTS ONLY
        self.brightness_threshold = 240  # Ultra-high - only aircraft navigation lights/strobes
        self.adaptive_threshold_value = 40  # Much less sensitive
        
        # High-altitude camera specific settings
        self.sky_region_only = True       # Focus on sky region only
        self.building_exclusion_zones = []  # Areas to ignore (buildings, static lights)
        self.horizon_line = None         # Will be set to ignore ground level
        
        # Tracking
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.object_id_counter = 0
        self.noise_filter = NoiseFilter()
        
        # Frame buffer for temporal analysis
        self.frame_buffer = deque(maxlen=30)
        self.detection_buffer = deque(maxlen=10)
        
        # Statistics
        self.total_detections = 0
        self.validated_objects = 0
        self.false_positives_filtered = 0
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def detect_bright_points(self, frame: np.ndarray) -> List[Dict]:
        """Detect bright points that could be objects"""
        detections = []
        
        # AERIAL MODE: Ultra-high threshold for navigation lights only
        _, bright_mask = cv2.threshold(frame, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Focus on sky region - exclude bottom portion where buildings/ground lights are
        if self.sky_region_only:
            height, width = frame.shape
            # Mask out bottom 20% of frame (building level from high altitude)
            sky_mask = np.ones_like(bright_mask)
            ground_cutoff = int(height * 0.8)  # Only look at top 80% (sky region)
            sky_mask[ground_cutoff:, :] = 0
            
            # Apply sky region mask
            bright_mask = cv2.bitwise_and(bright_mask, sky_mask)
        
        combined_mask = bright_mask
        
        # AERIAL MODE: Ultra-aggressive filtering for navigation lights only
        kernel = np.ones((2, 2), np.uint8)
        
        # Triple opening to eliminate all but the cleanest point sources
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)  # Triple opening
        
        # Minimal closing to preserve point sources
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))
        
        # Final erosion to ensure only the brightest, cleanest points
        cleaned = cv2.erode(cleaned, np.ones((1, 1), np.uint8), iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # DUAL MODE: Small navigation lights (1-25px) OR larger isolated objects (10-200px)
            if area < 1:  # Too small to be anything
                continue
            
            # Two categories: tiny navigation lights OR larger isolated objects
            is_large_object = area >= 10 and area <= 200  # Larger objects like aircraft bodies
            is_small_light = area >= 1 and area <= 25     # Navigation lights/strobes
            
            if not (is_large_object or is_small_light):
                continue
            
            # Get center and properties
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Get brightness at center
            if 0 <= cy < frame.shape[0] and 0 <= cx < frame.shape[1]:
                brightness = float(frame[cy, cx])
            else:
                continue
            
            # Calculate shape properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # For larger objects, check isolation from other light sources
            is_isolated = True
            if is_large_object:
                is_isolated = self.check_object_isolation(combined_mask, cx, cy, area)
                if not is_isolated:
                    continue  # Skip if connected to other lights (likely ground-based)
            
            # Create detection
            detection = {
                'center': (cx, cy),
                'area': area,
                'brightness': brightness,
                'circularity': circularity,
                'contour': contour,
                'timestamp': time.time(),
                'object_type': 'large_isolated' if is_large_object else 'small_light',
                'is_isolated': is_isolated if is_large_object else True
            }
            
            detections.append(detection)
        
        return detections
    
    def check_object_isolation(self, mask: np.ndarray, cx: int, cy: int, obj_area: float) -> bool:
        """Check if an object is isolated from other light sources"""
        
        # Define isolation radius based on object size (larger objects need more isolation)
        isolation_radius = max(20, int(np.sqrt(obj_area) * 3))  # Minimum 20 pixels
        
        # Create a region around the object
        h, w = mask.shape
        y1 = max(0, cy - isolation_radius)
        y2 = min(h, cy + isolation_radius)
        x1 = max(0, cx - isolation_radius)
        x2 = min(w, cx + isolation_radius)
        
        # Extract the region
        region = mask[y1:y2, x1:x2].copy()
        
        # Create a mask to exclude the object itself
        region_h, region_w = region.shape
        center_y = cy - y1
        center_x = cx - x1
        
        # Create a circular mask around the object to exclude it
        exclude_radius = max(5, int(np.sqrt(obj_area) * 1.5))
        cv2.circle(region, (center_x, center_y), exclude_radius, 0, -1)
        
        # Count remaining white pixels (other light sources)
        other_light_pixels = np.sum(region > 0)
        
        # Object is isolated if there are very few other light pixels nearby
        isolation_threshold = obj_area * 0.3  # Allow some noise but not major light sources
        
        return other_light_pixels < isolation_threshold
        
    def match_detection_to_object(self, detection: Dict, max_distance: float = 30) -> Optional[str]:
        """Match a detection to an existing tracked object"""
        best_match = None
        best_distance = max_distance
        
        for obj_id, obj in self.tracked_objects.items():
            if obj.is_noise:
                continue
                
            # Get last known position
            if len(obj.positions) == 0:
                continue
                
            last_pos = obj.positions[-1]
            
            # Calculate distance
            dx = detection['center'][0] - last_pos[0]
            dy = detection['center'][1] - last_pos[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Check if within reasonable distance based on speed
            time_gap = detection['timestamp'] - obj.last_seen
            if time_gap > 0:
                max_expected_distance = self.max_movement_per_frame * (time_gap * 30)  # Assuming 30 fps
                
                if distance < best_distance and distance < max_expected_distance:
                    best_distance = distance
                    best_match = obj_id
        
        return best_match
    
    def create_new_object(self, detection: Dict) -> str:
        """Create a new tracked object"""
        obj_id = f"obj_{self.object_id_counter}"
        self.object_id_counter += 1
        
        obj = TrackedObject(
            id=obj_id,
            positions=deque([detection['center']], maxlen=100),
            timestamps=deque([detection['timestamp']], maxlen=100),
            brightness_history=deque([detection['brightness']], maxlen=100),
            area_history=deque([detection['area']], maxlen=100),
            first_seen=detection['timestamp'],
            last_seen=detection['timestamp'],
            consecutive_detections=1
        )
        
        self.tracked_objects[obj_id] = obj
        return obj_id
    
    def update_object(self, obj_id: str, detection: Dict):
        """Update a tracked object with new detection"""
        obj = self.tracked_objects[obj_id]
        
        # Update position and properties
        obj.positions.append(detection['center'])
        obj.timestamps.append(detection['timestamp'])
        obj.brightness_history.append(detection['brightness'])
        obj.area_history.append(detection['area'])
        obj.last_seen = detection['timestamp']
        obj.consecutive_detections += 1
        
        # Calculate movement
        if len(obj.positions) >= 2:
            dx = obj.positions[-1][0] - obj.positions[-2][0]
            dy = obj.positions[-1][1] - obj.positions[-2][1]
            distance = np.sqrt(dx*dx + dy*dy)
            obj.total_distance_traveled += distance
            
            # Update average speed
            if len(obj.timestamps) >= 2:
                time_diff = obj.timestamps[-1] - obj.timestamps[0]
                if time_diff > 0:
                    obj.average_speed = obj.total_distance_traveled / time_diff
    
    def validate_object(self, obj: TrackedObject) -> bool:
        """Validate if an object is a genuine AIRCRAFT detection from high altitude"""
        
        # ULTRA-STRICT validation for aircraft from high-altitude camera
        if obj.consecutive_detections < self.min_consecutive_detections:
            return False
        
        tracking_duration = obj.last_seen - obj.first_seen
        if tracking_duration < self.min_tracking_duration:
            return False
        
        # Aircraft must travel significant distance across sky
        if obj.total_distance_traveled < self.min_total_distance:
            return False
        
        # Check for noise patterns (building lights, etc.)
        if self.noise_filter.is_likely_noise(obj):
            obj.is_noise = True
            return False
        
        # AIRCRAFT-SPECIFIC VALIDATION: Must be in sky region
        if hasattr(self, 'sky_region_only') and self.sky_region_only:
            # Check if object has spent time in ground/building region
            if len(obj.positions) > 0:
                # Assume frame height available from first position context
                for pos in obj.positions:
                    # If object spends time in bottom 20% of frame, likely ground-based
                    if len(obj.positions) > 0:
                        # This is a simplified check - in practice you'd pass frame dimensions
                        # For now, we'll trust the sky_region masking in detect_bright_points
                        pass
        
        # Calculate movement consistency
        if len(obj.positions) >= 5:
            positions = list(obj.positions)[-10:]
            
            # Calculate direction vectors
            vectors = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                if dx != 0 or dy != 0:
                    vectors.append((dx, dy))
            
            if len(vectors) >= 3:
                # Check direction consistency
                angles = []
                for dx, dy in vectors:
                    angle = np.arctan2(dy, dx)
                    angles.append(angle)
                
                # Calculate circular variance
                mean_angle = np.arctan2(np.mean([np.sin(a) for a in angles]),
                                       np.mean([np.cos(a) for a in angles]))
                
                deviations = [abs(a - mean_angle) for a in angles]
                deviations = [min(d, 2*np.pi - d) for d in deviations]  # Handle wrap-around
                
                obj.direction_consistency = 1.0 - (np.mean(deviations) / np.pi)
                
                # AIRCRAFT: Require smoother, more consistent movement than ground vehicles
                if obj.direction_consistency < 0.5:  # Higher threshold for aircraft
                    return False
                
        # AIRCRAFT-SPECIFIC: Check for smooth, linear movement patterns
        if len(obj.positions) >= 8:
            positions = list(obj.positions)
            
            # Aircraft should show relatively smooth acceleration/movement
            speeds = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                speed = np.sqrt(dx*dx + dy*dy)
                speeds.append(speed)
            
            if len(speeds) >= 3:
                # Check for consistent speed (aircraft don't suddenly stop/start)
                speed_variance = np.var(speeds)
                avg_speed = np.mean(speeds)
                
                # If speed is too variable, likely noise/ground lights
                if avg_speed > 0 and speed_variance / (avg_speed**2) > 0.8:  # High relative variance
                    return False
        
        # Final check: Object must maintain minimum brightness/visibility
        if len(obj.brightness_history) > 0:
            avg_brightness = np.mean(list(obj.brightness_history))
            # Aircraft navigation lights should be consistently bright
            if avg_brightness < self.brightness_threshold * 0.9:  # Must stay near threshold
                return False
        
        # Calculate confidence score
        obj.confidence = min(1.0, (
            (obj.consecutive_detections / 20) * 0.3 +
            (obj.direction_consistency) * 0.3 +
            (min(obj.total_distance_traveled / 100, 1.0)) * 0.2 +
            (min(tracking_duration / 5, 1.0)) * 0.2
        ))
        
        return obj.confidence >= self.min_confidence
    
    def classify_object(self, obj: TrackedObject) -> str:
        """Classify the type of object based on movement patterns and size"""
        
        # Check if we have size information from detection history
        avg_area = 0
        if hasattr(obj, 'area_history') and len(obj.area_history) > 0:
            avg_area = np.mean(list(obj.area_history))
        
        # Large isolated objects (10+ pixels)
        if avg_area >= 10:
            if obj.average_speed < 5:
                return "large_slow_object"  # Possible drone/helicopter
            elif obj.average_speed < 15:
                return "aircraft_body"      # Aircraft fuselage/body visible
            else:
                return "fast_large_object"  # Fast moving aircraft
        
        # Small navigation lights (1-25 pixels)
        else:
            if obj.average_speed < 3:
                # Very slow or stationary
                if len(obj.brightness_history) > 10:
                    brightness_var = np.var(list(obj.brightness_history))
                    if brightness_var > 100:
                        return "flashing_nav_light"
                    else:
                        return "static_nav_light"
            
            elif obj.average_speed < 12:
                # Moderate speed navigation lights
                if obj.direction_consistency > 0.7:
                    return "aircraft_nav_light"
                else:
                    return "moving_light"
            
            elif obj.average_speed < 25:
                # Fast navigation lights
                if obj.direction_consistency > 0.8:
                    return "fast_aircraft_light"
                else:
                    return "high_speed_light"
            
            else:
                # Very fast navigation lights
                return "supersonic_object"
    
    def process_frame(self, frame: np.ndarray) -> List[TrackedObject]:
        """Process a frame and return validated objects"""
        
        # Preprocess
        processed = self.preprocess_frame(frame)
        
        # Detect bright points
        detections = self.detect_bright_points(processed)
        
        # Update tracking
        current_time = time.time()
        matched_objects = set()
        
        for detection in detections:
            # Try to match to existing object
            obj_id = self.match_detection_to_object(detection)
            
            if obj_id:
                self.update_object(obj_id, detection)
                matched_objects.add(obj_id)
            else:
                # Create new object
                self.create_new_object(detection)
        
        # Update objects that weren't matched
        for obj_id, obj in list(self.tracked_objects.items()):
            if obj_id not in matched_objects:
                # Check if object has been lost for too long
                if current_time - obj.last_seen > self.max_tracking_gap:
                    # Validate before removing
                    if self.validate_object(obj):
                        obj.is_validated = True
                        self.validated_objects += 1
                    else:
                        self.false_positives_filtered += 1
                    
                    # Remove if too old
                    if current_time - obj.last_seen > 2.0:
                        del self.tracked_objects[obj_id]
            else:
                # Object was matched, validate it
                if not obj.is_validated and self.validate_object(obj):
                    obj.is_validated = True
                    obj.classification = self.classify_object(obj)
                    self.validated_objects += 1
        
        # Return only validated objects
        validated = [obj for obj in self.tracked_objects.values() 
                    if obj.is_validated and not obj.is_noise]
        
        return validated


class VideoProcessor:
    """Main video processing and display class"""
    
    def __init__(self, source: str = 0):
        self.source = source
        self.detector = AerialObjectDetector(min_confidence=0.6)
        self.running = False
        
        # Display options
        self.show_trails = True
        self.show_stats = True
        self.show_detection_boxes = True
        self.trail_length = 30
        
        # Recording
        self.recording = False
        self.video_writer = None
        
    def draw_object(self, frame: np.ndarray, obj: TrackedObject):
        """Draw a tracked object on the frame"""
        
        if len(obj.positions) == 0:
            return
        
        current_pos = obj.positions[-1]
        
        # Choose color based on classification
        color_map = {
            'aircraft': (0, 255, 0),           # Green
            'distant_vehicle': (255, 255, 0),   # Cyan
            'high_speed_object': (0, 0, 255),   # Red
            'fast_mover': (0, 165, 255),        # Orange
            'slow_mover': (255, 0, 255),        # Magenta
            'stationary_light': (128, 128, 128), # Gray
            'flashing_stationary': (200, 200, 200), # Light gray
            'unknown': (255, 255, 255)          # White
        }
        
        color = color_map.get(obj.classification, (255, 255, 255))
        
        # Draw trail if enabled
        if self.show_trails and len(obj.positions) > 1:
            positions = list(obj.positions)
            trail_start = max(0, len(positions) - self.trail_length)
            
            for i in range(trail_start + 1, len(positions)):
                # Fade trail
                alpha = (i - trail_start) / (len(positions) - trail_start)
                trail_color = tuple(int(c * alpha) for c in color)
                
                pt1 = positions[i-1]
                pt2 = positions[i]
                cv2.line(frame, pt1, pt2, trail_color, 1)
        
        # Draw current position
        cv2.circle(frame, current_pos, 15, color, 2)
        cv2.circle(frame, current_pos, 2, color, -1)
        
        # Draw classification and confidence
        if self.show_detection_boxes:
            text = f"{obj.classification} ({obj.confidence:.2f})"
            cv2.putText(frame, text, (current_pos[0] + 20, current_pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show speed
            speed_text = f"{obj.average_speed:.1f} px/s"
            cv2.putText(frame, speed_text, (current_pos[0] + 20, current_pos[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_stats(self, frame: np.ndarray):
        """Draw statistics overlay"""
        
        if not self.show_stats:
            return
        
        height, width = frame.shape[:2]
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw stats
        stats = [
            f"Tracked: {len(self.detector.tracked_objects)}",
            f"Validated: {len([o for o in self.detector.tracked_objects.values() if o.is_validated])}",
            f"Total Validated: {self.detector.validated_objects}",
            f"Filtered (Noise): {self.detector.false_positives_filtered}",
            f"Recording: {'ON' if self.recording else 'OFF'}"
        ]
        
        y = 30
        for stat in stats:
            cv2.putText(frame, stat, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 1)
            y += 25
    
    def run(self):
        """Main processing loop"""
        
        # Open video source
        if isinstance(self.source, str) and self.source.startswith(('rtsp', 'http')):
            cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {self.source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30  # Default if can't get FPS
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video source opened: {width}x{height} @ {fps} FPS")
        print("\nControls:")
        print("  q: Quit")
        print("  t: Toggle trails")
        print("  s: Toggle stats")
        print("  b: Toggle detection boxes")
        print("  r: Start/stop recording")
        print("  +/-: Adjust brightness threshold")
        print("  [/]: Adjust minimum confidence")
        print("\n")
        
        # Create window
        cv2.namedWindow('Aerial Object Detector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Aerial Object Detector', 1280, 720)
        
        self.running = True
        frame_count = 0
        last_time = time.time()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("End of video or connection lost")
                break
            
            frame_count += 1
            
            # Process frame
            validated_objects = self.detector.process_frame(frame)
            
            # Draw objects
            for obj in validated_objects:
                self.draw_object(frame, obj)
            
            # Draw stats
            self.draw_stats(frame)
            
            # Calculate and display FPS
            if frame_count % 30 == 0:
                current_time = time.time()
                actual_fps = 30 / (current_time - last_time)
                last_time = current_time
                print(f"FPS: {actual_fps:.1f} | Tracking: {len(validated_objects)} objects")
            
            # Record if enabled
            if self.recording and self.video_writer is not None:
                self.video_writer.write(frame)
            
            # Display
            cv2.imshow('Aerial Object Detector', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.running = False
            elif key == ord('t'):
                self.show_trails = not self.show_trails
                print(f"Trails: {'ON' if self.show_trails else 'OFF'}")
            elif key == ord('s'):
                self.show_stats = not self.show_stats
                print(f"Stats: {'ON' if self.show_stats else 'OFF'}")
            elif key == ord('b'):
                self.show_detection_boxes = not self.show_detection_boxes
                print(f"Detection boxes: {'ON' if self.show_detection_boxes else 'OFF'}")
            elif key == ord('r'):
                if not self.recording:
                    # Start recording
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"aerial_recording_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                    self.recording = True
                    print(f"Recording started: {filename}")
                else:
                    # Stop recording
                    if self.video_writer:
                        self.video_writer.release()
                        self.video_writer = None
                    self.recording = False
                    print("Recording stopped")
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
        
        # Cleanup
        cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n=== Final Statistics ===")
        print(f"Total validated objects: {self.detector.validated_objects}")
        print(f"False positives filtered: {self.detector.false_positives_filtered}")
        print(f"Noise reduction rate: {self.detector.false_positives_filtered / max(1, self.detector.validated_objects + self.detector.false_positives_filtered) * 100:.1f}%")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Aerial Object Detection System')
    parser.add_argument('--source', default='0', 
                      help='Video source (0 for webcam, or RTSP URL, or video file)')
    parser.add_argument('--confidence', type=float, default=0.6,
                      help='Minimum confidence for detection (0.0-1.0)')
    parser.add_argument('--brightness', type=int, default=180,
                      help='Brightness threshold (100-250)')
    
    args = parser.parse_args()
    
    # For your RTSP camera, use:
    # source = "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp"
    
    processor = VideoProcessor(args.source)
    processor.detector.min_confidence = args.confidence
    processor.detector.brightness_threshold = args.brightness
    
    processor.run()


if __name__ == "__main__":
    main()