#!/usr/bin/env python3
"""
Camera Detection Module - Core functionality for object detection
"""

import cv2
import numpy as np
import subprocess
import threading
import queue
import time
from typing import Optional, Tuple, List, Dict, Any


class FFmpegCamera:
    def __init__(self, stream_url: str, resolution: str = "1280x720"):
        self.stream_url = stream_url
        self.resolution = resolution
        self.width, self.height = map(int, resolution.split('x'))
        self.frame_queue = queue.Queue(maxsize=5)
        self.process = None
        self.thread = None
        self.running = False
        self.stats = {
            'frames_processed': 0,
            'start_time': None,
            'last_frame_time': None
        }
    
    def start(self) -> bool:
        """Start ffmpeg process and reading thread"""
        try:
            # FFmpeg command to convert RTSPS to raw video frames
            cmd = [
                'ffmpeg',
                '-i', self.stream_url,
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-vf', f'scale={self.width}:{self.height}',
                '-r', '10',  # 10 FPS to reduce CPU load
                '-an',  # No audio
                '-sn',  # No subtitles
                '-loglevel', 'error',
                '-'
            ]
            
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.running = True
            self.stats['start_time'] = time.time()
            self.thread = threading.Thread(target=self._read_frames, daemon=True)
            self.thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def _read_frames(self):
        """Read frames from ffmpeg in separate thread"""
        frame_size = self.width * self.height * 3  # BGR = 3 bytes per pixel
        
        while self.running:
            try:
                raw_frame = self.process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    if len(raw_frame) == 0:
                        print("Stream ended")
                    else:
                        print(f"Warning: Incomplete frame received ({len(raw_frame)}/{frame_size} bytes)")
                    break
                
                # Convert raw bytes to numpy array (make it writable)
                frame = np.frombuffer(raw_frame, dtype=np.uint8).copy()
                frame = frame.reshape((self.height, self.width, 3))
                
                self.stats['frames_processed'] += 1
                self.stats['last_frame_time'] = time.time()
                
                # Add to queue (drop oldest if full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
                
            except Exception as e:
                print(f"Error reading frame: {e}")
                break
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the latest frame"""
        try:
            return True, self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return False, None
    
    def get_fps(self) -> float:
        """Calculate current FPS"""
        if self.stats['start_time'] and self.stats['frames_processed'] > 0:
            elapsed = time.time() - self.stats['start_time']
            return self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        return 0
    
    def is_running(self) -> bool:
        """Check if camera is running"""
        return self.running and (self.thread and self.thread.is_alive())
    
    def stop(self):
        """Stop the camera"""
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()
        if self.thread:
            self.thread.join(timeout=2)


class YOLODetector:
    def __init__(self, weights_path: str = "yolov4.weights", 
                 config_path: str = "yolov4.cfg", 
                 classes_path: str = "coco.names"):
        self.net = None
        self.classes = []
        self.colors = []
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.detection_stats = {
            'total_detections': 0,
            'detections_by_class': {},
            'processing_times': []
        }
        
        self.load_model(weights_path, config_path, classes_path)
    
    def load_model(self, weights_path: str, config_path: str, classes_path: str) -> bool:
        """Load YOLO model and class names"""
        try:
            print("Loading YOLO model...")
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            # Load class names
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Generate colors for each class
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
            # Use GPU if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("Using GPU acceleration")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            print(f"Model loaded with {len(self.classes)} classes")
            return True
            
        except Exception as e:
            print(f"Could not load YOLO model: {e}")
            return False
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for detections"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def set_nms_threshold(self, threshold: float):
        """Set Non-Maximum Suppression threshold"""
        self.nms_threshold = max(0.0, min(1.0, threshold))
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Run object detection on frame"""
        if self.net is None:
            return frame, []
        
        start_time = time.time()
        height, width, _ = frame.shape
        
        # Create blob
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Run inference
        outputs = self.net.forward(output_layers)
        
        # Parse detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 
                                  self.confidence_threshold, self.nms_threshold)
        
        detections = []
        processed_frame = frame.copy()
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_name = self.classes[class_ids[i]]
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                
                detection_info = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2)
                }
                detections.append(detection_info)
                
                # Update stats
                self.detection_stats['total_detections'] += 1
                if class_name not in self.detection_stats['detections_by_class']:
                    self.detection_stats['detections_by_class'][class_name] = 0
                self.detection_stats['detections_by_class'][class_name] += 1
                
                # Draw bounding box and label
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
                
                # Create label with background
                label = f'{class_name}: {confidence:.2f}'
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Draw label background
                cv2.rectangle(processed_frame, (x, y - label_height - baseline), 
                             (x + label_width, y), color, -1)
                
                # Draw label text
                cv2.putText(processed_frame, label, (x, y - baseline), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Record processing time
        processing_time = time.time() - start_time
        self.detection_stats['processing_times'].append(processing_time)
        
        # Keep only last 100 processing times
        if len(self.detection_stats['processing_times']) > 100:
            self.detection_stats['processing_times'] = self.detection_stats['processing_times'][-100:]
        
        return processed_frame, detections
    
    def get_average_processing_time(self) -> float:
        """Get average processing time for detections"""
        if self.detection_stats['processing_times']:
            return sum(self.detection_stats['processing_times']) / len(self.detection_stats['processing_times'])
        return 0
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return self.detection_stats.copy()
    
    def reset_stats(self):
        """Reset detection statistics"""
        self.detection_stats = {
            'total_detections': 0,
            'detections_by_class': {},
            'processing_times': []
        }


class ObjectDetectionController:
    def __init__(self):
        self.camera = None
        self.detector = None
        self.running = False
        self.detection_enabled = True
        
    def initialize_camera(self, stream_url: str, resolution: str = "1280x720") -> bool:
        """Initialize camera with given parameters"""
        if self.camera:
            self.camera.stop()
        
        self.camera = FFmpegCamera(stream_url, resolution)
        return self.camera.start()
    
    def initialize_detector(self, weights_path: str, config_path: str, classes_path: str) -> bool:
        """Initialize YOLO detector"""
        self.detector = YOLODetector(weights_path, config_path, classes_path)
        return self.detector.net is not None
    
    def set_detection_enabled(self, enabled: bool):
        """Enable or disable object detection"""
        self.detection_enabled = enabled
    
    def set_confidence_threshold(self, threshold: float):
        """Set detection confidence threshold"""
        if self.detector:
            self.detector.set_confidence_threshold(threshold)
    
    def set_nms_threshold(self, threshold: float):
        """Set NMS threshold"""
        if self.detector:
            self.detector.set_nms_threshold(threshold)
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], List[Dict[str, Any]]]:
        """Get processed frame with detections"""
        if not self.camera or not self.camera.is_running():
            return False, None, []
        
        ret, frame = self.camera.read()
        if not ret or frame is None:
            return False, None, []
        
        detections = []
        if self.detection_enabled and self.detector:
            frame, detections = self.detector.detect(frame)
        
        return True, frame, detections
    
    def get_camera_stats(self) -> Dict[str, Any]:
        """Get camera statistics"""
        if self.camera:
            return {
                'fps': self.camera.get_fps(),
                'frames_processed': self.camera.stats['frames_processed'],
                'resolution': self.camera.resolution,
                'is_running': self.camera.is_running()
            }
        return {}
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        if self.detector:
            return self.detector.get_detection_stats()
        return {}
    
    def stop(self):
        """Stop all processes"""
        self.running = False
        if self.camera:
            self.camera.stop()