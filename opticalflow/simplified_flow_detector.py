#!/usr/bin/env python3
"""
Simplified Optical Flow Visualization System
FIXED: Better performance, cleaner code, improved playback
"""

import cv2
import numpy as np
import time
import argparse
from collections import deque

# Import the fixed CUDA stream player
from cuda_stream_player import CUDAStreamPlayer, test_cuda_available


class OpticalFlowDetector:
    """Fast optical flow detection using Farneback method"""
    
    def __init__(self, flow_threshold=1.5):
        self.flow_threshold = flow_threshold
        self.prev_gray = None
        self.motion_accumulator = None
        self.decay = 0.85
        
        # Farneback parameters (optimized for speed)
        self.fb_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
    
    def process_frame(self, frame, min_area=10, max_area=500):
        """Process frame and return motion objects"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.motion_accumulator = np.zeros(gray.shape, dtype=np.float32)
            return []
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, **self.fb_params
        )
        
        # Get flow magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Threshold to get motion mask
        motion_mask = (mag > self.flow_threshold).astype(np.uint8) * 255
        
        # Update accumulator with decay
        self.motion_accumulator *= self.decay
        self.motion_accumulator = np.maximum(
            self.motion_accumulator, 
            mag / (mag.max() + 1e-6)
        )
        
        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    objects.append({
                        'center': (cx, cy),
                        'area': area,
                        'contour': cnt
                    })
        
        self.prev_gray = gray
        return objects


class ObjectTracker:
    """Simple object tracker using nearest-neighbor matching"""
    
    def __init__(self, max_distance=60, max_age=10):
        self.max_distance = max_distance
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 0
    
    def update(self, detections):
        """Update tracks with new detections"""
        # Age existing tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]['age'] += 1
            if self.tracks[tid]['age'] > self.max_age:
                del self.tracks[tid]
        
        # Match detections to tracks
        matched = set()
        for det in detections:
            best_tid = None
            best_dist = self.max_distance
            
            for tid, track in self.tracks.items():
                if tid in matched:
                    continue
                dist = np.sqrt(
                    (det['center'][0] - track['center'][0])**2 +
                    (det['center'][1] - track['center'][1])**2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid
            
            if best_tid is not None:
                # Update existing track
                self.tracks[best_tid]['center'] = det['center']
                self.tracks[best_tid]['area'] = det['area']
                self.tracks[best_tid]['age'] = 0
                self.tracks[best_tid]['length'] += 1
                matched.add(best_tid)
            else:
                # Create new track
                self.tracks[self.next_id] = {
                    'center': det['center'],
                    'area': det['area'],
                    'age': 0,
                    'length': 1
                }
                self.next_id += 1
        
        # Return active tracks
        return [
            {
                'id': tid,
                'center': t['center'],
                'area': t['area'],
                'track_length': t['length']
            }
            for tid, t in self.tracks.items()
            if t['age'] == 0
        ]


class FlowVisualizationSystem:
    """Main system for optical flow visualization"""
    
    def __init__(self, source, resolution="1280x720", hwaccel='cuda', use_ffmpeg=True):
        # Parse resolution
        self.width, self.height = map(int, resolution.split('x'))
        
        # Video capture
        self.use_ffmpeg = use_ffmpeg
        self.source = int(source) if str(source).isdigit() else source
        
        if use_ffmpeg:
            self.player = CUDAStreamPlayer(
                self.source,
                width=self.width,
                height=self.height,
                fps=30,
                hwaccel=hwaccel,
                buffer_size=2  # Low buffer for responsiveness
            )
            self.cap = None
        else:
            self.player = None
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Processing components
        self.detector = OpticalFlowDetector(flow_threshold=1.5)
        self.tracker = ObjectTracker(max_distance=60)
        
        # Display options
        self.show_overlay = True
        self.show_tracks = True
        self.show_stats = True
        
        # Performance
        self.fps = 0.0
        self.frame_times = deque(maxlen=30)
        self.overlay_cache = None
        self.overlay_counter = 0
    
    def draw_tracks(self, frame, tracks):
        """Draw tracked objects on frame"""
        for t in tracks:
            cx, cy = t['center']
            length = t['track_length']
            
            # Color by track maturity
            if length < 3:
                color = (128, 128, 128)  # Gray - new
            elif length < 10:
                color = (0, 255, 255)    # Yellow - growing
            else:
                color = (0, 255, 0)      # Green - stable
            
            radius = max(5, min(20, int(np.sqrt(t['area']))))
            cv2.circle(frame, (cx, cy), radius, color, 2)
            cv2.putText(frame, str(length), (cx + 10, cy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_stats(self, frame):
        """Draw stats panel"""
        if not self.show_stats:
            return
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (10, 10), (250, 140), (0, 255, 0), 1)
        
        y = 30
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 22
        
        if self.use_ffmpeg and self.player:
            stats = self.player.get_stats()
            hw_color = (0, 255, 0) if stats['hwaccel'] != 'software' else (0, 200, 200)
            cv2.putText(frame, f"Accel: {stats['hwaccel'].upper()}", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, hw_color, 1)
            y += 22
            cv2.putText(frame, f"Drop: {stats['drop_rate']:.1f}%", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "Backend: OpenCV", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
        y += 22
        
        track_count = len(self.tracker.tracks)
        cv2.putText(frame, f"Tracks: {track_count}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 22
        
        cv2.putText(frame, f"Threshold: {self.detector.flow_threshold:.1f}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Main processing loop"""
        print("=" * 55)
        print("  OPTICAL FLOW VISUALIZATION")
        print("=" * 55)
        
        if self.use_ffmpeg:
            print("\nHardware acceleration:")
            hw = test_cuda_available()
            for k, v in hw.items():
                print(f"  {'✓' if v else '✗'} {k}")
        
        print("\nControls:")
        print("  Q     - Quit")
        print("  O     - Toggle flow overlay")
        print("  T     - Toggle track markers")
        print("  S     - Toggle stats")
        print("  M/N   - Adjust sensitivity")
        print("  SPACE - Pause")
        print("=" * 55)
        
        # Start video
        if self.use_ffmpeg:
            if not self.player.start():
                print("FFmpeg failed, using OpenCV...")
                self.use_ffmpeg = False
                self.cap = cv2.VideoCapture(self.source)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        cv2.namedWindow('Optical Flow', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Optical Flow', self.width, self.height)
        
        paused = False
        frame = None
        last_time = time.time()
        
        try:
            while True:
                if not paused:
                    # Get frame
                    if self.use_ffmpeg:
                        ret, frame = self.player.read()
                    else:
                        ret, frame = self.cap.read()
                    
                    if not ret or frame is None:
                        time.sleep(0.01)
                        continue
                    
                    # Detect motion
                    detections = self.detector.process_frame(frame)
                    tracks = self.tracker.update(detections)
                    
                    # Draw flow overlay
                    if self.show_overlay and self.detector.motion_accumulator is not None:
                        self.overlay_counter += 1
                        if self.overlay_counter >= 3:  # Update every 3 frames
                            self.overlay_counter = 0
                            motion_viz = (self.detector.motion_accumulator * 255).astype(np.uint8)
                            self.overlay_cache = cv2.applyColorMap(motion_viz, cv2.COLORMAP_JET)
                        
                        if self.overlay_cache is not None:
                            cv2.addWeighted(frame, 0.7, self.overlay_cache, 0.3, 0, frame)
                    
                    # Draw tracks
                    if self.show_tracks:
                        self.draw_tracks(frame, tracks)
                    
                    # Draw stats
                    self.draw_stats(frame)
                    
                    # Update FPS
                    now = time.time()
                    self.frame_times.append(now - last_time)
                    last_time = now
                    if self.frame_times:
                        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                
                if frame is not None:
                    cv2.imshow('Optical Flow', frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('o'):
                    self.show_overlay = not self.show_overlay
                    print(f"Overlay: {'ON' if self.show_overlay else 'OFF'}")
                elif key == ord('t'):
                    self.show_tracks = not self.show_tracks
                    print(f"Tracks: {'ON' if self.show_tracks else 'OFF'}")
                elif key == ord('s'):
                    self.show_stats = not self.show_stats
                elif key == ord('m'):
                    self.detector.flow_threshold = max(0.5, self.detector.flow_threshold - 0.3)
                    print(f"Threshold: {self.detector.flow_threshold:.1f}")
                elif key == ord('n'):
                    self.detector.flow_threshold = min(5.0, self.detector.flow_threshold + 0.3)
                    print(f"Threshold: {self.detector.flow_threshold:.1f}")
                elif key == ord(' '):
                    paused = not paused
                    print("PAUSED" if paused else "RUNNING")
        
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        finally:
            if self.use_ffmpeg and self.player:
                self.player.stop()
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Optical Flow Visualization')
    parser.add_argument('--source', default='0', help='Video source (0=webcam, file, rtsp://...)')
    parser.add_argument('--resolution', default='1280x720', help='Resolution WxH')
    parser.add_argument('--threshold', type=float, default=1.5, help='Flow threshold')
    parser.add_argument('--backend', default='ffmpeg', choices=['ffmpeg', 'opencv'])
    parser.add_argument('--hwaccel', default='cuda', help='Hardware accel: cuda, vaapi, none')
    
    args = parser.parse_args()
    
    hwaccel = None if args.hwaccel == 'none' else args.hwaccel
    
    system = FlowVisualizationSystem(
        args.source,
        args.resolution,
        hwaccel=hwaccel,
        use_ffmpeg=(args.backend == 'ffmpeg')
    )
    system.detector.flow_threshold = args.threshold
    system.run()


if __name__ == "__main__":
    main()