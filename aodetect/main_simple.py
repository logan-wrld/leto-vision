#!/usr/bin/env python3
"""
Simplified Aerial Object Detection - Optical Flow Only
Stripped down to bare essentials for testing and debugging
"""

import cv2
import numpy as np
import time
import argparse
import subprocess
from simple_camera import SimpleCamera
from opencv_flow import OpenCVFlowDetector, OpticalFlowTracker


class SimpleDetectionSystem:
    """Minimal detection system using optical flow"""
    
    def __init__(self, camera_source, resolution="1280x720"):
        self.camera = SimpleCamera(camera_source, resolution)
        
        # Optical flow detection
        self.flow_detector = OpenCVFlowDetector(flow_threshold=1.5)
        self.flow_tracker = OpticalFlowTracker(max_distance=60)
        
        # Display toggles
        self.show_flow_overlay = False
        self.paused = False
        
    def draw_detections(self, frame, flow_objects):
        """Draw detected objects on frame"""
        for obj in flow_objects:
            if obj['track_length'] >= 3:
                center = obj['center']
                track_len = obj['track_length']
                
                # Color based on track length (more confident = greener)
                confidence = min(1.0, track_len / 10.0)
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                
                # Draw box
                cv2.rectangle(frame, 
                            (center[0] - 15, center[1] - 15),
                            (center[0] + 15, center[1] + 15),
                            color, 2)
                
                # Draw label
                cv2.putText(frame, f"T:{track_len}", 
                           (center[0] - 15, center[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_info(self, frame, fps, num_objects):
        """Draw minimal info overlay"""
        cv2.putText(frame, f"FPS: {fps:.0f} | Objects: {num_objects}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "[Q]uit [O]verlay [SPACE]pause [+/-]sensitivity", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def run(self):
        """Main loop"""
        print("Starting camera...")
        if not self.camera.start():
            print("ERROR: Failed to start camera!")
            return
        
        print("Camera started. Press Q to quit.")
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detection', 1280, 720)
        
        while True:
            if not self.paused:
                ret, frame = self.camera.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Run optical flow detection
                motion_objects = self.flow_detector.process_frame_pair(frame, min_area=5, max_area=200)
                flow_objects = self.flow_tracker.update_tracks(motion_objects)
                
                # Draw detections
                self.draw_detections(frame, flow_objects)
                
                # Draw flow overlay if enabled
                if self.show_flow_overlay and self.flow_detector.motion_accumulator is not None:
                    motion_viz = (self.flow_detector.motion_accumulator * 255).astype(np.uint8)
                    motion_colored = cv2.applyColorMap(motion_viz, cv2.COLORMAP_JET)
                    cv2.addWeighted(frame, 0.7, motion_colored, 0.3, 0, frame)
                
                # Draw info
                self.draw_info(frame, self.camera.get_fps(), len(flow_objects))
                
                cv2.imshow('Detection', frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                self.show_flow_overlay = not self.show_flow_overlay
                print(f"Flow overlay: {'ON' if self.show_flow_overlay else 'OFF'}")
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"{'PAUSED' if self.paused else 'RESUMED'}")
            elif key == ord('+') or key == ord('='):
                self.flow_detector.flow_threshold = max(0.5, self.flow_detector.flow_threshold - 0.5)
                print(f"Sensitivity: {self.flow_detector.flow_threshold:.1f}")
            elif key == ord('-'):
                self.flow_detector.flow_threshold = min(5.0, self.flow_detector.flow_threshold + 0.5)
                print(f"Sensitivity: {self.flow_detector.flow_threshold:.1f}")
        
        # Cleanup
        self.camera.stop()
        cv2.destroyAllWindows()
        print("Done.")


def get_youtube_stream_url(youtube_url):
    """Get direct stream URL from YouTube"""
    cmd = ['yt-dlp', '-f', 'best[ext=mp4]/best', '-g', youtube_url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"yt-dlp error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Simple Optical Flow Detection')
    parser.add_argument('--source', default='0', help='Camera source (webcam index, RTSP URL, YouTube URL, or file)')
    parser.add_argument('--resolution', default='1280x720', help='Resolution')
    args = parser.parse_args()
    
    source = args.source
    
    # Handle YouTube URLs
    if "youtube.com" in source or "youtu.be" in source:
        print(f"Getting YouTube stream: {source}")
        source = get_youtube_stream_url(source)
        if not source:
            print("Failed to get YouTube stream. Exiting.")
            return
        print(f"Stream URL obtained.")
    
    # Run
    system = SimpleDetectionSystem(source, args.resolution)
    system.run()


if __name__ == "__main__":
    main()
