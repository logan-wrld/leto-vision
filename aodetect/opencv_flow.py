#!/usr/bin/env python3
"""
Pure OpenCV Optical Flow for Aerial Object Detection
High-quality dense optical flow without PyTorch dependencies
"""

import cv2
import numpy as np
from collections import deque
import time


class OpenCVFlowDetector:
    """Pure OpenCV dense optical flow detector optimized for aerial object detection"""
    
    def __init__(self, flow_threshold=1.0):
        """
        Initialize flow detector
        Args:
            flow_threshold: Minimum flow magnitude to consider as motion
        """
        self.flow_threshold = flow_threshold
        self.frame_buffer = deque(maxlen=2)  # Keep last 2 frames
        
        # Motion accumulation for better detection
        self.motion_accumulator = None
        self.accumulation_decay = 0.95
        
        print("Using OpenCV Farneback dense optical flow for motion detection")
        print(f"Flow threshold: {flow_threshold} pixels")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for optical flow"""
        if frame is None:
            return None
            
        # Convert to grayscale for optical flow
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        return gray
    
    def compute_dense_flow(self, frame1, frame2):
        """Compute dense optical flow using OpenCV Farneback method"""
        try:
            # Use Farneback dense optical flow
            dense_flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 
                pyr_scale=0.5,      # Pyramid scale factor
                levels=3,           # Number of pyramid layers
                winsize=15,         # Window size
                iterations=3,       # Iterations at each pyramid level
                poly_n=5,           # Size of pixel neighborhood
                poly_sigma=1.2,     # Standard deviation of Gaussian
                flags=0
            )
            
            # Convert from HWC to CHW format for consistency
            if dense_flow is not None and len(dense_flow.shape) == 3:
                return np.transpose(dense_flow, (2, 0, 1))  # HWC -> CHW
            else:
                return np.zeros((2, frame1.shape[0], frame1.shape[1]), dtype=np.float32)
                
        except Exception as e:
            print(f"Farneback flow failed: {e}")
            # Return zero flow as last resort
            return np.zeros((2, frame1.shape[0], frame1.shape[1]), dtype=np.float32)
    
    def detect_motion_regions(self, flow, magnitude_threshold=None):
        """
        Detect regions with significant motion from optical flow
        Args:
            flow: Optical flow array (2, H, W)
            magnitude_threshold: Minimum flow magnitude to consider
        Returns:
            motion_mask: Binary mask of motion regions
            motion_magnitude: Flow magnitude array
        """
        if magnitude_threshold is None:
            magnitude_threshold = self.flow_threshold
        
        # Compute flow magnitude
        u = flow[0]  # x component
        v = flow[1]  # y component
        magnitude = np.sqrt(u*u + v*v)
        
        # Create motion mask
        motion_mask = magnitude > magnitude_threshold
        
        # Accumulate motion over time for better detection
        if self.motion_accumulator is None:
            self.motion_accumulator = magnitude.copy()
        else:
            self.motion_accumulator = (self.motion_accumulator * self.accumulation_decay + 
                                     magnitude * (1 - self.accumulation_decay))
        
        # Enhanced motion mask using accumulation
        enhanced_mask = self.motion_accumulator > (magnitude_threshold * 0.7)
        combined_mask = motion_mask | enhanced_mask
        
        return combined_mask.astype(np.uint8) * 255, magnitude
    
    def extract_motion_objects(self, motion_mask, min_area=5, max_area=500):
        """Extract individual moving objects from motion mask"""
        # Morphological operations to clean up motion mask
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of moving regions
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding box and center
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            
            # Calculate motion properties
            mask_region = cleaned_mask[y:y+h, x:x+w]
            motion_pixels = np.sum(mask_region > 0)
            motion_density = motion_pixels / (w * h) if (w * h) > 0 else 0
            
            motion_objects.append({
                'center': center,
                'bbox': (x, y, w, h),
                'area': area,
                'contour': contour,
                'motion_density': motion_density,
                'timestamp': time.time()
            })
        
        return motion_objects
    
    def process_frame_pair(self, current_frame, min_area=5, max_area=500):
        """
        Process a frame pair and return detected motion objects
        Args:
            current_frame: Current frame
            min_area: Minimum object area
            max_area: Maximum object area
        Returns:
            List of motion objects
        """
        if current_frame is None:
            return []
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(current_frame)
        if processed_frame is None:
            return []
        
        # Add to buffer
        self.frame_buffer.append(processed_frame)
        
        # Need at least 2 frames for optical flow
        if len(self.frame_buffer) < 2:
            return []
        
        # Compute optical flow
        frame1 = self.frame_buffer[-2]
        frame2 = self.frame_buffer[-1]
        
        try:
            flow = self.compute_dense_flow(frame1, frame2)
            if flow is None:
                return []
            
            # Detect motion regions
            motion_mask, magnitude = self.detect_motion_regions(flow)
            
            # Extract motion objects
            motion_objects = self.extract_motion_objects(motion_mask, min_area, max_area)
            
            return motion_objects
            
        except Exception as e:
            print(f"Error in optical flow processing: {e}")
            return []
    
    def visualize_flow(self, frame, flow, motion_objects=None):
        """Visualize optical flow and detected objects"""
        if flow is None:
            return frame
        
        vis_frame = frame.copy()
        
        # Draw flow field (subsampled for visibility)
        h, w = flow.shape[1], flow.shape[2]
        step = 20
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                u = flow[0, y, x]
                v = flow[1, y, x]
                magnitude = np.sqrt(u*u + v*v)
                
                if magnitude > self.flow_threshold:
                    # Draw flow vector
                    end_x = int(x + u * 5)  # Scale for visibility
                    end_y = int(y + v * 5)
                    cv2.arrowedLine(vis_frame, (x, y), (end_x, end_y), (0, 255, 0), 1)
        
        # Draw detected motion objects
        if motion_objects:
            for obj in motion_objects:
                center = obj['center']
                bbox = obj['bbox']
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (bbox[0], bbox[1]), 
                             (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
                
                # Draw center point
                cv2.circle(vis_frame, center, 3, (0, 0, 255), -1)
                
                # Draw info
                cv2.putText(vis_frame, f"A:{obj['area']:.0f}", 
                           (center[0] + 10, center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis_frame


class OpticalFlowTracker:
    """Track objects using optical flow information"""
    
    def __init__(self, max_distance=50):
        self.tracked_objects = {}
        self.object_id_counter = 0
        self.max_distance = max_distance
    
    def update_tracks(self, motion_objects):
        """Update object tracks with new detections"""
        # Match detections to existing tracks
        matched_tracks = {}
        unmatched_detections = motion_objects.copy()
        
        for track_id, track in self.tracked_objects.items():
            best_match = None
            best_distance = self.max_distance
            
            for i, detection in enumerate(unmatched_detections):
                distance = np.sqrt(
                    (track['center'][0] - detection['center'][0])**2 + 
                    (track['center'][1] - detection['center'][1])**2
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                # Update existing track
                detection = unmatched_detections.pop(best_match)
                track['center'] = detection['center']
                track['bbox'] = detection['bbox']
                track['last_seen'] = detection['timestamp']
                track['track_length'] += 1
                matched_tracks[track_id] = track
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track_id = self.object_id_counter
            self.object_id_counter += 1
            
            matched_tracks[track_id] = {
                'center': detection['center'],
                'bbox': detection['bbox'],
                'first_seen': detection['timestamp'],
                'last_seen': detection['timestamp'],
                'track_length': 1,
                'area': detection['area']
            }
        
        # Remove old tracks
        current_time = time.time()
        self.tracked_objects = {
            k: v for k, v in matched_tracks.items() 
            if current_time - v['last_seen'] < 5.0  # Keep tracks for 5 seconds
        }
        
        return list(self.tracked_objects.values())