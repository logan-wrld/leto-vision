#!/usr/bin/env python3
"""
Simple Camera Object Detection - Core functionality
Python 3.10.12 compatible
"""

import cv2
import numpy as np
import subprocess
import threading
import queue
import time


class SimpleCamera:
    def __init__(self, stream_url: str, resolution: str = "1280x720"):
        self.stream_url = stream_url
        self.resolution = resolution
        self.width, self.height = map(int, resolution.split('x'))
        self.frame_queue = queue.Queue(maxsize=3)
        self.process = None
        self.thread = None
        self.running = False
        self.frame_count = 0
        self.start_time = None
    
    def start(self) -> bool:
        try:
            cmd = [
                'ffmpeg', '-i', self.stream_url,
                '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-vf', f'scale={self.width}:{self.height}',
                '-r', '15', '-an', '-sn', '-loglevel', 'error', '-'
            ]
            
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.running = True
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._read_frames, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Camera error: {e}")
            return False
    
    def _read_frames(self):
        frame_size = self.width * self.height * 3
        
        while self.running:
            try:
                raw_frame = self.process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    break
                
                frame = np.frombuffer(raw_frame, dtype=np.uint8).copy()
                frame = frame.reshape((self.height, self.width, 3))
                
                self.frame_count += 1
                
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
            except Exception:
                break
    
    def read(self):
        try:
            return True, self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return False, None
    
    def get_fps(self) -> float:
        if self.start_time and self.frame_count > 0:
            elapsed = time.time() - self.start_time
            return self.frame_count / elapsed if elapsed > 0 else 0
        return 0
    
    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()


class SimpleDetector:
    def __init__(self, weights: str = "yolov4.weights", config: str = "yolov4.cfg", names: str = "coco.names"):
        self.net = None
        self.classes = []
        self.colors = []
        self.confidence = 0.5
        self.nms_threshold = 0.4
        self.total_detections = 0
        
        self.load_model(weights, config, names)
    
    def load_model(self, weights: str, config: str, names: str) -> bool:
        try:
            self.net = cv2.dnn.readNet(weights, config)
            
            with open(names, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
            # Use GPU if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            print(f"✅ Model loaded: {len(self.classes)} classes")
            return True
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            return False
    
    def detect(self, frame):
        if self.net is None:
            return frame, []
        
        height, width = frame.shape[:2]
        
        # YOLO detection
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        
        # Parse results
        boxes, confidences, class_ids = [], [], []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_name = self.classes[class_ids[i]]
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                
                detections.append({'class': class_name, 'confidence': confidence})
                self.total_detections += 1
                
                # Draw detection
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, detections