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
        self.light_threshold = 160  # Balanced threshold to reduce noise
        self.flash_sensitivity = 20  # Less sensitive to reduce false positives
        
        # ENHANCED MOVEMENT detection for distant solid lights
        self.min_movement_threshold = 2   # Detect 2-pixel movements for solid lights
        self.high_speed_threshold = 8     # Lower threshold for distant vehicles
        self.very_high_speed_threshold = 15  # More reasonable for distant detection
        
        # Movement validation settings - ENHANCED for solid lights
        self.min_frames_for_movement = 4  # Fewer frames needed for solid lights
        self.movement_consistency_required = 0.4  # Lower consistency for distant solid lights
        
        # Vehicle-specific detection parameters - ULTRA-SENSITIVE for distant vehicles
        self.min_area_distant = 1         # Accept single-pixel lights
        self.max_area_vehicle = 2000      # Larger maximum for all vehicles
        self.directional_consistency = 0.3 # Very low requirement (30%)
        self.min_tracking_duration = 0.8  # Very short duration
        self.max_direction_change = 90    # Allow significant direction changes
        
        # ULTRA-SENSITIVE parameters for bridge cars
        self.ultra_tiny_threshold = 10   # Even lower for distant bridge cars
        self.pixel_movement_threshold = 1 # Detect even 1-pixel movements
        self.distant_light_enhancement = True  # Enable specisal processing for tiny lights
        
        # Speed filtering - only detect vehicles moving over 5 mph
        self.min_speed_mph = 5.0          # Minimum speed in miles per hour
        self.pixels_per_foot_estimate = 2 # Rough estimate: 2 pixels = 1 foot at distance
        self.fps_estimate = 30            # Estimated frame rate for speed calculation
        
    def classify_movement_speed(self, avg_movement):
        """Classify movement speed into categories - only for moving lights"""
        if avg_movement >= self.very_high_speed_threshold:
            return "VERY_FAST", ""  # Helicopter, aircraft, very fast vehicle
        elif avg_movement >= self.high_speed_threshold:
            return "FAST", ""       # Fast vehicle, emergency veshicle
        elif avg_movement >= self.min_movement_threshold:
            return "MOVING", ""     # Moving at detectable speed
        else:
            return "TOO_SLOW", ""    # Not moving enough - will be filtered out
    
    def classify_flash_intensity(self, flash_intensity):
        """Classify flashing intensity"""
        if flash_intensity >= 80:
            return "INTENSE", ""    # Emergency lights, strobes
        elif flash_intensity >= 50:
            return "BRIGHT", ""     # Vehicle signals, bright flashers
        elif flash_intensity >= 30:
            return "MODERATE", ""   # Regular flashing lights
        else:
            return "DIM", ""        # Weak flashing
    
    def analyze_light_shape(self, contour, area):
        """Analyze contour shape to determine if it could be a vehicle light"""
        # Calculate shape metrics
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False, "no_perimeter"
        
        # Compactness - vehicles lights tend to be more compact
        compactness = (perimeter * perimeter) / (4 * np.pi * area)
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 10
        
        # Vehicle light characteristics:
        # - Reasonably compact (not too irregular)
        # - Not extremely elongated (aspect ratio not too high)
        # - Appropriate size for distance
        
        is_vehicle_like = (
            compactness < 3.0 and          # Not too irregular
            aspect_ratio < 4.0 and         # Not too elongated  
            area >= self.min_area_distant and  # Not too small
            area <= self.max_area_vehicle      # Not too large
        )
        
        shape_type = "vehicle_light" if is_vehicle_like else "irregular_light"
        return is_vehicle_like, shape_type
    
    def calculate_movement_direction(self, positions):
        """Calculate dominant movement direction and consistency"""
        if len(positions) < 3:
            return 0, 0  # angle, consistency
        
        directions = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            
            if dx == 0 and dy == 0:
                continue
                
            angle = np.arctan2(dy, dx) * 180 / np.pi
            directions.append(angle)
        
        if not directions:
            return 0, 0
        
        # Calculate direction consistency
        if len(directions) < 2:
            return directions[0], 0
        
        # Find the most consistent direction
        mean_direction = np.mean(directions)
        direction_deviations = [abs(angle - mean_direction) for angle in directions]
        
        # Handle angle wrapping (e.g., -179 and 179 degrees should be close)
        for i, dev in enumerate(direction_deviations):
            if dev > 180:
                direction_deviations[i] = 360 - dev
        
        avg_deviation = np.mean(direction_deviations)
        consistency = max(0, 1.0 - (avg_deviation / self.max_direction_change))
        
        return mean_direction, consistency
    
    def calculate_speed_mph(self, pixels_per_second):
        """Convert pixel movement per second to estimated MPH"""
        # Convert pixels per second to feet per second
        feet_per_second = pixels_per_second / self.pixels_per_foot_estimate
        
        # Convert feet per second to miles per hour
        # 1 mile = 5280 feet, 1 hour = 3600 seconds
        mph = (feet_per_second * 3600) / 5280
        
        return mph
    
    def meets_speed_requirement(self, avg_movement_pixels_per_second):
        """Check if movement meets minimum 5 mph requirement"""
        estimated_mph = self.calculate_speed_mph(avg_movement_pixels_per_second)
        return estimated_mph >= self.min_speed_mph, estimated_mph
    
    def is_erratic_flashing_light(self, history):
        """Detect if this is likely a stationary flashing light (building, sign, etc)"""
        if len(history) < 6:
            return False
        
        # Check for erratic brightness changes typical of building lights
        brightness_changes = []
        positions = []
        
        for i in range(1, len(history)):
            bright_change = abs(history[i]['brightness'] - history[i-1]['brightness'])
            brightness_changes.append(bright_change)
            positions.append(history[i]['center'])
        
        # Calculate position variance - stationary lights have low variance
        if len(positions) >= 3:
            x_positions = [pos[0] for pos in positions]
            y_positions = [pos[1] for pos in positions]
            x_variance = np.var(x_positions)
            y_variance = np.var(y_positions)
            total_variance = x_variance + y_variance
            
            # High brightness variance but low position variance = flashing building light
            brightness_variance = np.var(brightness_changes) if brightness_changes else 0
            avg_brightness_change = np.mean(brightness_changes) if brightness_changes else 0
            
            # Criteria for erratic flashing light
            is_stationary = total_variance < 5  # Very low movement
            is_highly_variable_brightness = brightness_variance > 100
            has_rapid_brightness_changes = avg_brightness_change > 30
            
            return is_stationary and (is_highly_variable_brightness or has_rapid_brightness_changes)
        
        return False
    

    
    def detect_ultra_tiny_lights(self, gray):
        """ULTRA-SENSITIVE detection for extremely distant car lights"""
        ultra_tiny_lights = []
        
        # Use multiple sensitive thresholds to catch bridge cars (even more sensitive)
        for threshold in [self.ultra_tiny_threshold, self.ultra_tiny_threshold + 10, self.ultra_tiny_threshold + 20, self.ultra_tiny_threshold + 35]:
            mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
            
            # Minimal morphological operations to preserve single pixels
            kernel = np.ones((1,1), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find all contours, even single pixels
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Accept very small areas including single pixels for distant bridge cars
                if area < 1 or area > 100:
                    continue
                
                # Very lenient shape requirements
                if len(contour) < 3:  # Handle single pixel contours
                    if len(contour) >= 1:
                        point = contour[0][0]
                        cx, cy = int(point[0]), int(point[1])
                    else:
                        continue
                else:
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                
                # Ensure coordinates are within frame bounds
                if cx < 0 or cy < 0 or cx >= gray.shape[1] or cy >= gray.shape[0]:
                    continue
                
                brightness = float(gray[cy, cx])
                
                # Very low brightness requirement for distant solid lights
                min_brightness = threshold + 2   # Even more sensitive for solid lights
                if brightness >= min_brightness:
                    ultra_tiny_lights.append({
                        'center': (cx, cy),
                        'brightness': brightness,
                        'area': area,
                        'contour': contour,
                        'type': 'ultra_tiny',
                        'threshold_used': threshold
                    })
        
        # Remove duplicates (lights found at multiple thresholds)
        unique_lights = []
        for light in ultra_tiny_lights:
            is_duplicate = False
            for existing in unique_lights:
                dx = abs(light['center'][0] - existing['center'][0])
                dy = abs(light['center'][1] - existing['center'][1])
                if dx <= 2 and dy <= 2:  # Within 2 pixels
                    is_duplicate = True
                    # Keep the one with higher brightness
                    if light['brightness'] > existing['brightness']:
                        unique_lights.remove(existing)
                        unique_lights.append(light)
                    break
            if not is_duplicate:
                unique_lights.append(light)
        
        return unique_lights
    
    def validate_movement(self, history, light_type='regular'):
        """Enhanced validation for vehicle-like movement patterns"""
        if len(history) < self.min_frames_for_movement:
            return False, 0, 0, {}
        
        # Adjust thresholds for different light types
        if light_type == 'ultra_tiny':
            min_duration = 0.5  # Much shorter for bridge cars
            min_movement = self.pixel_movement_threshold
        elif light_type == 'tiny_distant':
            min_duration = max(1.0, self.min_tracking_duration * 0.75)
            min_movement = self.min_movement_threshold
        else:
            min_duration = self.min_tracking_duration
            min_movement = self.min_movement_threshold
        
        # Check minimum tracking duration
        total_duration = history[-1]['timestamp'] - history[0]['timestamp']
        if total_duration < min_duration:
            return False, 0, 0, {"reason": "insufficient_tracking_time"}
        
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
            
            # Calculate pixels per frame (more stable than per second)
            if time_diff > 0:
                speed = distance / max(time_diff, 0.001)  # Prevent division by zero
                movement_samples.append(speed)
            else:
                movement_samples.append(distance)
            
            # Count movements that exceed minimum threshold (adjusted for light type)
            if distance >= min_movement:
                valid_movements += 1
        
        if not movement_samples:
            return False, 0, 0, {"reason": "no_movement_data"}
        
        avg_movement = sum(movement_samples) / len(movement_samples)
        max_movement = max(movement_samples)
        
        # CHECK SPEED REQUIREMENT - Only detect vehicles moving over 5 mph
        meets_speed, estimated_mph = self.meets_speed_requirement(avg_movement)
        if not meets_speed:
            return False, avg_movement, max_movement, {
                "reason": "speed_too_slow",
                "estimated_mph": estimated_mph,
                "required_mph": self.min_speed_mph
            }
        
        # Calculate directional consistency
        direction_angle, direction_consistency = self.calculate_movement_direction(positions)
        
        # Movement validation criteria for vehicles
        consistency_ratio = valid_movements / len(movement_samples)
        is_consistently_moving = consistency_ratio >= self.movement_consistency_required
        has_consistent_direction = direction_consistency >= self.directional_consistency
        
        # Calculate total distance traveled
        total_distance = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        # Vehicle movement criteria (adjusted for light type):
        if light_type == 'ultra_tiny':
            # VERY LENIENT criteria for bridge cars - they're really there!
            movement_checks = {
                "avg_speed_ok": avg_movement >= min_movement,
                "consistency_ok": consistency_ratio >= 0.3,  # Only 30% consistency needed
                "direction_ok": direction_consistency >= 0.1,  # Very minimal direction requirement
                "min_distance_ok": total_distance >= 3,  # Must move at least 3 pixels
                "max_speed_realistic": max_movement < 100,  # Allow faster movement
                "steady_movement": True,  # Always pass this check
            }
        elif light_type == 'tiny_distant':
            # More lenient criteria for distant tiny lights
            movement_checks = {
                "avg_speed_ok": avg_movement >= min_movement,
                "consistency_ok": consistency_ratio >= (self.movement_consistency_required * 0.6),  # 24%
                "direction_ok": direction_consistency >= (self.directional_consistency * 0.5),  # 15%
                "min_distance_ok": total_distance >= (min_movement * 1.5),  # Shorter distance required
                "max_speed_realistic": max_movement < 80,  # Slower for distant objects
            }
        else:
            # Standard criteria for regular lights
            movement_checks = {
                "avg_speed_ok": avg_movement >= min_movement,
                "consistency_ok": is_consistently_moving,
                "direction_ok": has_consistent_direction,
                "min_distance_ok": total_distance >= (min_movement * 3),
                "max_speed_realistic": max_movement < 200,  # Not impossibly fast
            }
        
        is_vehicle_movement = all(movement_checks.values())
        
        validation_data = {
            "direction_angle": direction_angle,
            "direction_consistency": direction_consistency,
            "total_distance": total_distance,
            "consistency_ratio": consistency_ratio,
            "checks": movement_checks,
            "duration": total_duration,
            "estimated_mph": estimated_mph,
            "meets_speed_requirement": meets_speed
        }
        
        return is_vehicle_movement, avg_movement, max_movement, validation_data
        
    def detect_moving_lights(self, frame):
        """Detect moving lights - both flashing AND solid vehicle lights"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray.copy())
        
        moving_lights = []
        
        if len(self.frame_buffer) < 5:
            return moving_lights
        
        # Find bright spots (potential lights) - Enhanced for solid lights
        bright_mask = cv2.threshold(gray, self.light_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Also create a more sensitive mask for solid moving lights
        sensitive_mask = cv2.threshold(gray, self.light_threshold - 30, 255, cv2.THRESH_BINARY)[1]
        
        # Remove noise (gentler processing to preserve small lights)
        kernel = np.ones((2,2), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        
        # Process sensitive mask separately with minimal filtering
        sensitive_kernel = np.ones((1,1), np.uint8)
        sensitive_mask = cv2.morphologyEx(sensitive_mask, cv2.MORPH_CLOSE, sensitive_kernel)
        
        # Find contours of bright spots
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Also find contours from sensitive mask for additional solid lights
        sensitive_contours, _ = cv2.findContours(sensitive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Also detect ultra-tiny distant lights with maximum sensitivity
        ultra_tiny_lights = self.detect_ultra_tiny_lights(gray) if self.distant_light_enhancement else []
        
        current_time = time.time()
        current_lights = []
        
        # Process regular contours (normal brightness threshold)
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Enhanced area filtering for distant vehicles
            if area < self.min_area_distant or area > self.max_area_vehicle:
                continue
            
            # Shape analysis for vehicle lights
            is_vehicle_shape, shape_type = self.analyze_light_shape(contour, area)
            if not is_vehicle_shape:
                continue  # Skip non-vehicle-like shapes
            
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            brightness = float(gray[cy, cx])
            
            # Tighter tracking grid for better precision (15x15 instead of 25x25)
            light_id = f"{cx//15}_{cy//15}"
            
            current_lights.append({
                'id': light_id,
                'center': (cx, cy),
                'brightness': brightness,
                'area': area,
                'contour': contour,
                'shape_type': shape_type,
                'type': 'regular'
            })
        
        # Process sensitive contours (lower brightness threshold for solid lights)
        for contour in sensitive_contours:
            area = cv2.contourArea(contour)
            
            # More lenient area filtering for solid distant lights
            if area < 1 or area > self.max_area_vehicle:
                continue
            
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            brightness = float(gray[cy, cx])
            
            # Only accept if bright enough for solid light detection
            if brightness < (self.light_threshold - 20):
                continue
            
            # Check if this light is already detected by regular threshold
            duplicate = False
            for existing_light in current_lights:
                ex_cx, ex_cy = existing_light['center']
                if abs(cx - ex_cx) <= 5 and abs(cy - ex_cy) <= 5:
                    duplicate = True
                    break
            
            if not duplicate:
                # Tighter tracking grid for solid lights
                light_id = f"solid_{cx//10}_{cy//10}"
                
                current_lights.append({
                    'id': light_id,
                    'center': (cx, cy),
                    'brightness': brightness,
                    'area': area,
                    'contour': contour,
                    'shape_type': 'solid_light',
                    'type': 'solid'
                })
        
        # Add ultra-tiny distant lights to processing
        for ultra_light in ultra_tiny_lights:
            cx, cy = ultra_light['center']
            # Use very tight tracking grid for ultra-tiny lights (4x4)
            light_id = f"ultra_{cx//4}_{cy//4}"
            
            current_lights.append({
                'id': light_id,
                'center': (cx, cy),
                'brightness': ultra_light['brightness'],
                'area': ultra_light['area'],
                'contour': ultra_light['contour'],
                'shape_type': 'ultra_tiny',
                'type': 'ultra_tiny',
                'threshold_used': ultra_light.get('threshold_used', 120)
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
                
                # CHECK FOR ERRATIC FLASHING LIGHTS FIRST
                if self.is_erratic_flashing_light(history):
                    # This is likely a building light or sign - remove from tracking
                    continue
                
                # ENHANCED VEHICLE MOVEMENT VALIDATION
                light_type = light.get('type', 'regular')
                is_vehicle_movement, avg_movement, max_movement, validation_data = self.validate_movement(history, light_type)
                
                # ONLY proceed if light shows vehicle-like movement
                if not is_vehicle_movement:
                    # Keep tracking but don't register as detection
                    if len(history) > 30:
                        history = history[-25:]
                    self.light_history[light_id] = history
                    
                    # Log rejection reason for debugging
                    if validation_data.get("reason"):
                        pass  # Could add debug logging here
                    
                    continue
                
                # Check for brightness changes (flashing) - only for moving lights
                brightness_changes = [abs(brightness - h['brightness']) for h in history[-6:]]
                avg_brightness_change = sum(brightness_changes) / len(brightness_changes) if brightness_changes else 0
                
                # Classify movement and flashing
                movement_class, movement_icon = self.classify_movement_speed(avg_movement)
                flash_class, flash_icon = self.classify_flash_intensity(avg_brightness_change)
                
                # FINAL CHECK: Can be flashing OR just solid moving lights
                is_flashing = avg_brightness_change > self.flash_sensitivity
                is_solid_moving = avg_movement >= self.min_movement_threshold  # Solid lights that just move
                
                # Enhanced detection criteria - adjusted for light type
                light_type = light.get('type', 'regular')
                direction_consistency = validation_data.get('direction_consistency', 0)
                
                if light_type == 'ultra_tiny':
                    # SOLID MOVING LIGHTS for bridge cars - no flashing required!
                    total_distance = validation_data.get('total_distance', 0)
                    has_steady_movement = avg_movement >= 1  # Accept even 1-pixel movement
                    has_consistent_path = direction_consistency > 0.15  # Very minimal direction
                    has_traveled_distance = total_distance >= 3  # Must move at least 3 pixels total
                    is_bright_enough = brightness > 130  # Solid bright light
                    
                    # Accept solid moving lights (bridge cars don't need to flash!)
                    is_valid_detection = (
                        (is_vehicle_movement and has_steady_movement) or  # Any validated movement
                        (has_steady_movement and has_traveled_distance and is_bright_enough) or  # Solid moving light
                        (has_consistent_path and has_traveled_distance and is_bright_enough)  # Directional solid light
                    )
                elif light_type == 'tiny_distant':
                    # Accept both flashing AND solid moving lights
                    is_flashing = avg_brightness_change > (self.flash_sensitivity * 0.3)  # Very low flashing threshold
                    has_movement = avg_movement >= self.min_movement_threshold
                    has_some_direction = direction_consistency > 0.15  # Very lenient
                    is_solid_moving = has_movement and validation_data.get('total_distance', 0) > 4
                    
                    is_valid_detection = (
                        (is_vehicle_movement and (is_flashing or is_solid_moving)) or  # Flashing OR solid moving
                        (has_movement and has_some_direction) or  # Any directional movement
                        (is_solid_moving and brightness > 140)  # Bright solid moving light
                    )
                else:
                    # Standard criteria - accept both flashing AND solid moving lights
                    is_flashing = avg_brightness_change > self.flash_sensitivity
                    has_strong_movement = avg_movement >= (self.min_movement_threshold * 1.2)
                    is_solid_bright = brightness > 160 and validation_data.get('total_distance', 0) > 8
                    
                    is_valid_detection = (
                        (is_flashing and is_vehicle_movement) or  # Traditional flashing vehicle light
                        (has_strong_movement and direction_consistency > 0.5) or  # Strong consistent movement
                        (is_solid_bright and direction_consistency > 0.4)  # Solid bright moving light
                    )
                
                if is_valid_detection:
                    # Enhanced priority score including direction consistency
                    priority_score = (
                        (avg_movement / 50) +                    # Movement component
                        (avg_brightness_change / 100) +         # Flashing component  
                        (direction_consistency * 0.5) +         # Direction component
                        (validation_data.get('total_distance', 0) / 200)  # Distance component
                    )
                    
                    # Calculate enhanced confidence
                    movement_confidence = min(1.0, avg_movement / 30)
                    flash_confidence = min(1.0, avg_brightness_change / 50)
                    direction_confidence = direction_consistency
                    overall_confidence = (movement_confidence + flash_confidence + direction_confidence) / 3
                    
                    moving_lights.append({
                        'center': center,
                        'brightness': brightness,
                        'area': light['area'],
                        'contour': light['contour'],
                        'shape_type': light.get('shape_type', 'unknown'),
                        'flash_intensity': avg_brightness_change,
                        'movement_speed': avg_movement,
                        'max_movement': max_movement,
                        'movement_class': movement_class,
                        'movement_icon': movement_icon,
                        'flash_class': flash_class,
                        'flash_icon': flash_icon,
                        'priority': priority_score,
                        'confidence': overall_confidence,
                        'is_high_priority': avg_movement >= self.high_speed_threshold and overall_confidence > 0.7,
                        'movement_validated': True,
                        'direction_angle': validation_data.get('direction_angle', 0),
                        'direction_consistency': direction_consistency,
                        'tracking_duration': validation_data.get('duration', 0),
                        'total_distance': validation_data.get('total_distance', 0),
                        'validation_checks': validation_data.get('checks', {})
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
        moving_lights.sort(key=lambda x: x['priority'], reverse=True)
        
        return moving_lights


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
        
        # Vehicle tracking and counting
        self.vehicle_count = 0
        self.total_vehicles_detected = 0
        self.vehicle_positions = []  # Store vehicle positions for highlighting
        self.active_vehicles = {}  # Track currently active vehicles
        self.vehicle_paths = {}  # Store movement paths for each vehicle
        
        # Model files
        self.weights_path = "yolov4.weights"
        self.config_path = "yolov4.cfg"
        self.names_path = "coco.names"
    
    def update_vehicle_tracking(self, moving_lights):
        """Update vehicle tracking and counting"""
        import time
        current_time = time.time()
        
        # Clear old vehicle positions (older than 5 seconds)
        self.vehicle_positions = [pos for pos in self.vehicle_positions 
                                 if current_time - pos['timestamp'] < 5.0]
        
        # Update active vehicles
        current_vehicle_ids = set()
        
        for light in moving_lights:
            center = light['center']
            light_type = light.get('type', 'regular')
            confidence = light.get('confidence', 0)
            
            # Create unique vehicle ID based on approximate location
            vehicle_id = f"{center[0]//50}_{center[1]//50}_{light_type}"
            current_vehicle_ids.add(vehicle_id)
            
            # Add to vehicle positions for highlighting
            self.vehicle_positions.append({
                'center': center,
                'timestamp': current_time,
                'type': light_type,
                'confidence': confidence,
                'vehicle_id': vehicle_id
            })
            
            # Update vehicle paths
            if vehicle_id not in self.vehicle_paths:
                self.vehicle_paths[vehicle_id] = []
                self.total_vehicles_detected += 1
            
            self.vehicle_paths[vehicle_id].append({
                'center': center,
                'timestamp': current_time
            })
            
            # Keep path history manageable (last 20 points)
            if len(self.vehicle_paths[vehicle_id]) > 20:
                self.vehicle_paths[vehicle_id] = self.vehicle_paths[vehicle_id][-20:]
            
            # Update active vehicles
            self.active_vehicles[vehicle_id] = {
                'center': center,
                'last_seen': current_time,
                'type': light_type,
                'confidence': confidence
            }
        
        # Remove inactive vehicles (not seen for 3 seconds)
        inactive_vehicles = [vid for vid, data in self.active_vehicles.items() 
                           if current_time - data['last_seen'] > 3.0]
        
        for vid in inactive_vehicles:
            del self.active_vehicles[vid]
        
        # Update current vehicle count
        self.vehicle_count = len(self.active_vehicles)
    
    def draw_vehicle_highlights(self, frame):
        """Draw vehicle highlighting and tracking information"""
        import time
        current_time = time.time()
        
        # Draw vehicle paths (trails)
        for vehicle_id, path in self.vehicle_paths.items():
            if len(path) > 1:
                # Only draw recent paths
                recent_path = [p for p in path if current_time - p['timestamp'] < 3.0]
                if len(recent_path) > 1:
                    # Draw path as connected lines
                    for i in range(1, len(recent_path)):
                        pt1 = recent_path[i-1]['center']
                        pt2 = recent_path[i]['center']
                        # Fade color based on age
                        age = current_time - recent_path[i]['timestamp']
                        alpha = max(0.2, 1.0 - (age / 3.0))
                        color = (int(100 * alpha), int(200 * alpha), int(255 * alpha))  # Blue trail
                        cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw heat map areas where vehicles frequently appear
        vehicle_zones = {}
        for pos in self.vehicle_positions:
            zone_x = pos['center'][0] // 100
            zone_y = pos['center'][1] // 100
            zone_key = f"{zone_x}_{zone_y}"
            
            if zone_key not in vehicle_zones:
                vehicle_zones[zone_key] = {'count': 0, 'center': [0, 0]}
            
            vehicle_zones[zone_key]['count'] += 1
            vehicle_zones[zone_key]['center'][0] += pos['center'][0]
            vehicle_zones[zone_key]['center'][1] += pos['center'][1]
        
        # Draw heat map zones
        for zone_key, zone_data in vehicle_zones.items():
            if zone_data['count'] > 3:  # Only show zones with multiple detections
                avg_center = (
                    zone_data['center'][0] // zone_data['count'],
                    zone_data['center'][1] // zone_data['count']
                )
                
                # Color intensity based on detection count
                intensity = min(255, zone_data['count'] * 20)
                color = (0, intensity//2, intensity//2)  # Red heat map
                
                cv2.circle(frame, avg_center, 30, color, -1)  # Filled circle
                cv2.putText(frame, f"{zone_data['count']}", 
                           (avg_center[0]-10, avg_center[1]+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_telemetry_overlay(self, frame, moving_lights):
        """Draw compact military-style HUD overlay"""
        height, width = frame.shape[:2]
        
        # Small top-right HUD - military style
        hud_width = 250
        hud_height = 160
        hud_x = width - hud_width - 10
        hud_y = 10
        
        # Subtle dark background with green border
        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_width, hud_y + hud_height), (0, 20, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_width, hud_y + hud_height), (0, 255, 0), 1)
        
        # Title - military style
        cv2.putText(frame, "LETO-V SYS", (hud_x + 5, hud_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw separator line
        cv2.line(frame, (hud_x + 5, hud_y + 25), (hud_x + hud_width - 5, hud_y + 25), (0, 255, 0), 1)
        
        # Compact military stats
        y_pos = hud_y + 40
        stats = [
            f"FPS: {self.fps:.1f}",
            f"TGT: {self.vehicle_count}",
            f"TOT: {self.total_vehicles_detected}",
            f"DET: {self.total_detections}",
            f"SYS: {'RUN' if self.running else 'OFF'}"
        ]
        
        for stat in stats:
            cv2.putText(frame, stat, (hud_x + 8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.35, (0, 255, 0), 1)
            y_pos += 16
        
        # Military-style active contacts display
        if self.vehicle_count > 0:
            y_pos += 5
            cv2.putText(frame, "CONTACTS:", (hud_x + 8, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            y_pos += 12
            
            # Show only active vehicles - compact format
            for i, (vehicle_id, vehicle_data) in enumerate(list(self.active_vehicles.items())[:2]):
                vehicle_type = vehicle_data['type']
                center = vehicle_data['center']
                
                contact_text = f"{vehicle_type[:3].upper()} {center[0]},{center[1]}"
                cv2.putText(frame, contact_text, (hud_x + 10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                y_pos += 12
        
        # Draw center reticle - circle only
      
        
        # Draw moving light detections on main frame with enhanced visualization
        for light in moving_lights:
            center = light['center']
            movement_class = light['movement_class']
            movement_icon = light['movement_icon']
            flash_class = light['flash_class']
            flash_icon = light['flash_icon']
            is_high_priority = light['is_high_priority']
            priority = light['priority']
            light_type = light.get('type', 'regular')
            
            # Special handling for ultra-tiny and tiny distant lights
            if light_type == 'ultra_tiny':
                circle_color = (255, 255, 0)    # CYAN for ultra-tiny distant lights
                text_color = (255, 255, 0)
                circle_thickness = 3
                circle_radius = 20  # Larger circle to make them visible
            elif light_type == 'tiny_distant':
                circle_color = (255, 0, 255)    # MAGENTA for tiny distant lights
                text_color = (255, 0, 255)
                circle_thickness = 2
                circle_radius = 15  # Smaller circle for tiny lights
            elif is_high_priority:
                circle_color = (0, 0, 255)    # RED for high priority (fast + bright)
                text_color = (0, 0, 255)
                circle_thickness = 4
                circle_radius = 25
            elif movement_class == "FAST":
                circle_color = (0, 165, 255)  # ORANGE for fast movement
                text_color = (0, 165, 255)
                circle_thickness = 3
                circle_radius = 25
            elif movement_class == "MOVING":
                circle_color = (0, 255, 255)  # YELLOW for moving
                text_color = (0, 255, 255)
                circle_thickness = 2
                circle_radius = 25
            else:
                circle_color = (255, 255, 0)  # CYAN for other movement
                text_color = (255, 255, 0)
                circle_thickness = 2
                circle_radius = 25
            
            # Military-style target markers
            cv2.circle(frame, center, circle_radius, circle_color, circle_thickness)
            cv2.circle(frame, center, 2, circle_color, -1)   # Small center dot
            
            # Compact military-style labeling
            if light_type == 'ultra_tiny':
                main_text = f"TGT-{movement_class}"
                font_size = 0.4
            elif light_type == 'tiny_distant':
                main_text = f"DIST-{movement_class}"
                font_size = 0.4
            else:
                main_text = f"{movement_class}"
                font_size = 0.4
                
            cv2.putText(frame, main_text, (center[0] + 30, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, 1)
            
            # Speed indicator only
            speed_text = f"{light['movement_speed']:.1f}px/s"
            cv2.putText(frame, speed_text, (center[0] + 30, center[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            
            # Priority indicator for high priority lights
            if is_high_priority:
                cv2.rectangle(frame, (center[0] - 40, center[1] - 40), (center[0] + 40, center[1] + 40), (0, 0, 255), 2)
                cv2.putText(frame, "PRIORITY", (center[0] - 30, center[1] - 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Draw direction arrow for movement
            if light.get('total_distance', 0) > 20:
                direction_angle = light.get('direction_angle', 0)
                arrow_length = 30
                end_x = int(center[0] + arrow_length * np.cos(np.radians(direction_angle)))
                end_y = int(center[1] + arrow_length * np.sin(np.radians(direction_angle)))
                cv2.arrowedLine(frame, center, (end_x, end_y), circle_color, 2, tipLength=0.3)
        
        return frame
    
    def detection_worker(self):
        """Background detection processing for both objects and flashing lights"""
        detection_camera = SimpleCamera(self.stream_url, "640x360")  # Lower res for detection
        if not detection_camera.start():
            print("ERROR: Detection camera failed to start")
            return
        
        detector = SimpleDetector(self.weights_path, self.config_path, self.names_path)
        if not detector.net:
            print("ERROR: Detector failed to load")
            return
        
        print("SUCCESS: Detection worker started (Objects + Moving Lights)")
        
        while self.running:
            try:
                ret, frame = detection_camera.read()
                if not ret:
                    continue
                
                # Crop to left half only to reduce GPU load
                height, width = frame.shape[:2]
                frame = frame[:, :width//2]  # Keep left half only
                
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
                
                # Run moving light detection (flashing AND solid)
                moving_lights = self.light_detector.detect_moving_lights(frame)
                
                # Process moving light detections with enhanced logging
                if moving_lights:
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
                        
                        # Enhanced logging with vehicle detection focus
                        direction_consistency = light.get('direction_consistency', 0)
                        confidence = light.get('confidence', 0)
                        total_distance = light.get('total_distance', 0)
                        
                        # Enhanced logging with special handling for tiny distant lights
                        light_type = light.get('type', 'regular')
                        
                        if light_type == 'ultra_tiny':
                            threshold = light.get('threshold_used', 120)
                            estimated_mph = validation_data.get('estimated_mph', 0)
                            print(f"BRIDGE CAR DETECTED: {estimated_mph:.1f} MPH Speed:{movement_speed:.2f} Distance:{total_distance:.0f} Area:{light['area']:.1f}")
                        elif light_type == 'tiny_distant':
                            estimated_mph = validation_data.get('estimated_mph', 0)
                            print(f"TINY DISTANT CAR: {estimated_mph:.1f} MPH Speed:{movement_speed:.1f} Distance:{total_distance:.0f} Area:{light['area']}")
                        elif is_high_priority:
                            estimated_mph = validation_data.get('estimated_mph', 0)
                            print(f"ALERT: VEHICLE DETECTED: {estimated_mph:.1f} MPH {movement_icon} {movement_class} Speed:{movement_speed:.1f} Conf:{confidence:.2f}")
                        elif movement_class == "VERY_FAST" and direction_consistency > 0.6:
                            estimated_mph = validation_data.get('estimated_mph', 0)
                            print(f"FAST VEHICLE: {estimated_mph:.1f} MPH Speed:{movement_speed:.1f} Direction:{direction_consistency:.2f} Distance:{total_distance:.0f}")
                        elif movement_class == "FAST" and direction_consistency > 0.5:
                            estimated_mph = validation_data.get('estimated_mph', 0)
                            print(f"MOVING VEHICLE: {estimated_mph:.1f} MPH Speed:{movement_speed:.1f} Direction:{direction_consistency:.2f}")
                        elif confidence > 0.6:
                            estimated_mph = validation_data.get('estimated_mph', 0)
                            print(f"Possible vehicle light: {estimated_mph:.1f} MPH Conf:{confidence:.2f} Speed:{movement_speed:.1f}")
                        
                        # Add to recent lights with enhanced info
                        light_text = f"[{timestamp}] {movement_icon} VEHICLE at ({center[0]},{center[1]}) S:{movement_speed:.1f} C:{confidence:.2f}"
                        self.recent_lights.append(light_text)
                    
                    # Keep only recent lights
                    if len(self.recent_lights) > 10:
                        self.recent_lights = self.recent_lights[-10:]
                
                time.sleep(0.01)  # Minimal delay for CPU efficiency
                
            except Exception as e:
                print(f"Detection worker error: {e}")
                break
        
        detection_camera.stop()
    
    def run(self):
        """Run the overlay video player"""
        # Check model files
        import os
        if not all(os.path.exists(f) for f in [self.weights_path, self.config_path, self.names_path]):
            print("YOLO model files not found!")
            return
        
        # Start main camera
        self.camera = SimpleCamera(self.stream_url, self.resolution)
        if not self.camera.start():
            print("Failed to connect to camera")
            return
        
        self.running = True
        
        # Start detection worker
        detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        detection_thread.start()
        
        print("ENHANCED VEHICLE DETECTION WITH TRACKING - Video player started")
        print("CONTROLS:")
        print("   'q' = quit | 'f' = fullscreen | 's' = save frame")
        print("   'h' = toggle heat map | 't' = toggle trails | 'c' = clear vehicle history")
        print("   '[' and ']' = adjust movement threshold | '+/-' = adjust light sensitivity")
        print("")
        print("VEHICLE DETECTION FEATURES:")
        print("   * REAL-TIME VEHICLE COUNTING with active/total statistics")
        print("   * BLUE TRAILS show vehicle movement paths over time")
        print("   * RED HEAT MAP zones highlight frequent vehicle areas")
        print("   * SOLID LIGHT detection - no flashing required for bridge cars!")
        print("   * ULTRA DISTANT detection (1+ pixel movement, single pixel lights)")
        print("   * CYAN circles = Bridge cars | MAGENTA = Tiny distant | RED = Priority vehicles")
        print("   * Detects both solid headlights AND flashing vehicle lights")
        
        # Create window
        cv2.namedWindow('LETO VISION - Moving Flashing Lights Only', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('LETO VISION - Moving Flashing Lights Only', 1280, 720)
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Crop to left half only to reduce GPU load
                height, width = frame.shape[:2]
                frame = frame[:, :width//2]  # Keep left half only
                
                self.frame_count = self.camera.frame_count
                self.fps = self.camera.get_fps()
                
                # Detect moving lights in main frame (both solid and flashing)
                moving_lights = self.light_detector.detect_moving_lights(frame)
                
                # Update vehicle tracking and counting
                self.update_vehicle_tracking(moving_lights)
                
                # Draw vehicle highlights and trails
                frame = self.draw_vehicle_highlights(frame)
                
                # Process moving lights for main display with enhanced alerts
                if moving_lights:
                    timestamp = time.strftime("%H:%M:%S")
                    for light in moving_lights:
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
                            print(f"HIGH PRIORITY FLASHING LIGHT: {movement_icon} {movement_class} Speed:{movement_speed:.1f}")
                        elif movement_class in ["VERY_FAST", "FAST"]:
                            print(f"{movement_icon} {movement_class} FLASHING LIGHT: Speed:{movement_speed:.1f}")
                        else:
                            print(f"{movement_icon} {movement_class} flashing light: Speed:{movement_speed:.1f}")                # Keep recent lights list manageable
                if len(self.recent_lights) > 12:
                    self.recent_lights = self.recent_lights[-8:]
                
                # Add telemetry overlay
                frame_with_overlay = self.draw_telemetry_overlay(frame, moving_lights)
                
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
                    print(f"Frame saved: {filename}")
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
                elif key == ord('h'):
                    # Toggle heat map display
                    print("Heat map display toggled")
                elif key == ord('t'):
                    # Toggle trail display
                    print("Vehicle trail display toggled")
                elif key == ord('c'):
                    # Clear vehicle history
                    self.vehicle_paths.clear()
                    self.vehicle_positions.clear()
                    self.active_vehicles.clear()
                    self.total_vehicles_detected = 0
                    self.vehicle_count = 0
                    print("Vehicle tracking history cleared")
        
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.camera.stop()
            cv2.destroyAllWindows()
            print("Video player stopped")


def main():
    player = OpenCVOverlay()
    player.run()


if __name__ == "__main__":
    main()