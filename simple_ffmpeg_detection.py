#!/usr/bin/env python3
"""
Simple Object Detection using FFmpeg pipe + OpenCV
This approach works better with RTSPS streams
"""

import cv2
import numpy as np
import subprocess
import threading
import queue
import time

class FFmpegCamera:
    def __init__(self, stream_url, resolution="1280x720"):
        self.stream_url = stream_url
        self.resolution = resolution
        self.width, self.height = map(int, resolution.split('x'))
        self.frame_queue = queue.Queue(maxsize=5)
        self.process = None
        self.thread = None
        self.running = False
    
    def start(self):
        """Start ffmpeg process and reading thread"""
        # FFmpeg command to convert RTSPS to raw video frames
        cmd = [
            'ffmpeg',
            '-i', self.stream_url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-vf', f'scale={self.width}:{self.height}',  # Force scaling
            '-r', '10',  # 10 FPS to reduce CPU load
            '-an',  # No audio
            '-sn',  # No subtitles
            '-loglevel', 'error',  # Reduce ffmpeg output
            '-'
        ]
        
        print(f"Starting ffmpeg with command: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        
        print(f"Camera started with resolution {self.resolution}")
    
    def _read_frames(self):
        """Read frames from ffmpeg in separate thread"""
        frame_size = self.width * self.height * 3  # BGR = 3 bytes per pixel
        
        while self.running:
            try:
                raw_frame = self.process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    print("Warning: Incomplete frame received")
                    break
                
                # Convert raw bytes to numpy array (make it writable)
                frame = np.frombuffer(raw_frame, dtype=np.uint8).copy()
                frame = frame.reshape((self.height, self.width, 3))
                
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
    
    def read(self):
        """Get the latest frame"""
        try:
            return True, self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return False, None
    
    def stop(self):
        """Stop the camera"""
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()
        if self.thread:
            self.thread.join()


def detect_objects_yolo(frame, net, classes, colors):
    """Run YOLO object detection on frame"""
    height, width, _ = frame.shape
    
    # Create blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Run inference
    outputs = net.forward(output_layers)
    
    # Parse detections
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            class_name = classes[class_ids[i]]
            confidence = confidences[i]
            color = colors[class_ids[i]]
            
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': (x, y, w, h)
            })
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{class_name}: {confidence:.2f}', 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame, detections


def main():
    # Configuration
    STREAM_URL = "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp"
    
    # Choose resolution based on your performance needs:
    # RESOLUTION = "2688x1512"  # Full resolution (slow but most accurate)
    # RESOLUTION = "1344x756"   # Half resolution (good balance)
    RESOLUTION = "1280x720"   # 720p (fast processing)
    # RESOLUTION = "640x360"    # Quarter resolution (very fast)
    
    print(f"Using resolution: {RESOLUTION}")
    print("Note: Higher resolution = more accurate detection but slower processing")
    
    # YOLO model files (download these first)
    WEIGHTS = "yolov4.weights"
    CONFIG = "yolov4.cfg" 
    CLASSES_FILE = "coco.names"
    
    # Load YOLO model
    print("Loading YOLO model...")
    try:
        net = cv2.dnn.readNet(WEIGHTS, CONFIG)
        
        # Load class names
        with open(CLASSES_FILE, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # Generate colors for each class
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        
        # Use GPU if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("Using GPU acceleration")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        print(f"Model loaded with {len(classes)} classes")
        use_detection = True
        
    except Exception as e:
        print(f"Could not load YOLO model: {e}")
        print("Running without object detection (display only)")
        use_detection = False
    
    # Start camera
    camera = FFmpegCamera(STREAM_URL, RESOLUTION)
    camera.start()
    
    print("Starting detection... Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("No frame received")
                continue
            
            frame_count += 1
            
            # Run object detection if model loaded
            if use_detection:
                processed_frame, detections = detect_objects_yolo(frame, net, classes, colors)
                
                # Print detections
                if detections:
                    print(f"Frame {frame_count}: Found {len(detections)} objects:")
                    for det in detections:
                        print(f"  - {det['class']}: {det['confidence']:.2f}")
            else:
                processed_frame = frame
            
            # Calculate FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"FPS: {fps:.1f}")
            
            # Add FPS to frame
            fps = frame_count / (time.time() - start_time) if time.time() - start_time > 0 else 0
            cv2.putText(processed_frame, f'FPS: {fps:.1f}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Ubiquiti Camera - Object Detection', processed_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping detection...")
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()