#!/usr/bin/env python3
"""
Simple Camera Module
Handles RTSP and local camera connections without YOLO
"""

import cv2
import threading
import queue
import time
import numpy as np


class SimpleCamera:
    """Simple camera handler for RTSP or local cameras"""
    
    def __init__(self, source, resolution="1280x720", buffer_size=5):
        """
        Initialize camera
        Args:
            source: Camera source (RTSP URL, file path, or camera index)
            resolution: Target resolution as "widthxheight"
            buffer_size: Frame buffer size
        """
        self.source = source
        self.resolution = resolution
        self.buffer_size = buffer_size
        
        # Parse resolution
        if 'x' in resolution:
            width, height = resolution.split('x')
            self.target_width = int(width)
            self.target_height = int(height)
        else:
            self.target_width = 1280
            self.target_height = 720
        
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.capture_thread = None
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.last_fps_frame = 0
    
    def start(self):
        """Start the camera capture"""
        try:
            if self.source.startswith('rtsp'):
                # RTSP stream
                self.cap = cv2.VideoCapture(self.source)
            else:
                # Local camera or file
                self.cap = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
            
            if not self.cap.isOpened():
                print(f"Failed to open camera source: {self.source}")
                return False
            
            # Optimized properties for stream stability
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffering
            
            # Stream stability optimizations
            if self.source.startswith('http'):
                # YouTube/web stream optimizations
                self.cap.set(cv2.CAP_PROP_FPS, 20)  # Limit FPS for stability
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            elif self.source.startswith('rtsp'):
                # RTSP stream optimizations
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # No buffering for RTSP
            
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            print(f"Camera started successfully: {self.source}")
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def _capture_loop(self):
        """Background thread for capturing frames"""
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        
        while self.running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret:
                        self.frame_count += 1
                        reconnect_attempts = 0  # Reset on successful read
                        
                        # Optional: Only resize if requested (keep high res by default)
                        # Comment out these lines if you want to keep original resolution
                        # if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                        #     frame = cv2.resize(frame, (self.target_width, self.target_height))
                        
                        # Aggressive frame dropping for stability
                        # Clear queue if it has more than 1 frame to prevent lag
                        while self.frame_queue.qsize() > 1:
                            try:
                                self.frame_queue.get_nowait()
                            except:
                                break
                        
                        # Only add frame if queue isn't full
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame)
                        
                        # Update FPS
                        current_time = time.time()
                        if current_time - self.last_fps_time >= 1.0:
                            frames_processed = self.frame_count - self.last_fps_frame
                            self.fps = frames_processed / (current_time - self.last_fps_time)
                            self.last_fps_time = current_time
                            self.last_fps_frame = self.frame_count
                    else:
                        # Connection lost or end of file
                        time.sleep(0.1)
                        
                        # Try to reconnect for network streams
                        if isinstance(self.source, str) and self.source.startswith(('rtsp://', 'rtsps://')):
                            reconnect_attempts += 1
                            if reconnect_attempts <= max_reconnect_attempts:
                                print(f"Connection lost, attempting to reconnect... (attempt {reconnect_attempts}/{max_reconnect_attempts})")
                                self.cap.release()
                                time.sleep(2)
                                self.cap = cv2.VideoCapture(self.source)
                                # Set buffer size for network streams
                                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            else:
                                print(f"Failed to reconnect after {max_reconnect_attempts} attempts")
                                self.running = False
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def read(self):
        """Read a frame from the camera"""
        try:
            # Shorter timeout to prevent hanging
            frame = self.frame_queue.get(timeout=2.0)
            return True, frame
        except:
            # Check if still running
            if not self.running:
                return False, None
            # Return False but don't crash
            return False, None
    
    def stop(self):
        """Stop the camera capture"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
    
    def get_fps(self):
        """Get current FPS"""
        return self.fps
    
    def get_frame_count(self):
        """Get total frame count"""
        return self.frame_count


def test_camera():
    """Test the camera connection"""
    
    # Test with your RTSP stream
    # camera = SimpleCamera("rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp", "1280x720")
    
    # Or test with webcam
    camera = SimpleCamera(0, "640x480")
    
    if not camera.start():
        print("Failed to start camera")
        return
    
    print("Camera test started. Press 'q' to quit.")
    
    cv2.namedWindow('Camera Test', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = camera.read()
        
        if ret:
            # Add FPS overlay
            fps_text = f"FPS: {camera.get_fps():.1f} | Frames: {camera.get_frame_count()}"
            cv2.putText(frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.stop()
    cv2.destroyAllWindows()
    print("Camera test completed")


if __name__ == "__main__":
    test_camera()