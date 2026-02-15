#!/usr/bin/env python3
"""
Simplified Optical Flow Visualization System
FIXED: Better performance, cleaner code, improved playback
"""

import cv2
import numpy as np
import time
import argparse
import os
from datetime import datetime
from collections import deque

# Import the fixed CUDA stream player
from cuda_stream_player import CUDAStreamPlayer, test_cuda_available


class MotionRecorder:
    """Records video clips when sustained motion is detected"""
    
    def __init__(self, output_dir="recordings", pre_buffer_sec=3, post_buffer_sec=5, fps=30):
        self.output_dir = output_dir
        self.pre_buffer_sec = pre_buffer_sec
        self.post_buffer_sec = post_buffer_sec
        self.target_fps = fps
        self.actual_fps = fps  # Will be updated based on real frame timing
        
        # Pre-buffer stores (frame, timestamp) tuples
        self.pre_buffer = deque(maxlen=int(pre_buffer_sec * fps))
        
        # Frame timing for actual FPS calculation
        self.frame_times = deque(maxlen=60)
        self.last_frame_time = None
        
        # Recording state
        self.is_recording = False
        self.recording_writer = None
        self.recording_start_time = None
        self.recording_end_time = None
        self.current_recording_path = None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def _update_fps_estimate(self):
        """Calculate actual FPS from frame timing"""
        now = time.time()
        if self.last_frame_time is not None:
            self.frame_times.append(now - self.last_frame_time)
        self.last_frame_time = now
        
        if len(self.frame_times) >= 10:
            avg_interval = sum(self.frame_times) / len(self.frame_times)
            if avg_interval > 0:
                self.actual_fps = 1.0 / avg_interval
                # Update pre-buffer size based on actual fps
                new_maxlen = int(self.pre_buffer_sec * self.actual_fps)
                if new_maxlen != self.pre_buffer.maxlen and new_maxlen > 0:
                    self.pre_buffer = deque(self.pre_buffer, maxlen=new_maxlen)
    
    @property
    def fps(self):
        """Return the actual measured FPS"""
        return self.actual_fps
    
    def add_frame(self, frame):
        """Add frame to pre-buffer (always called)"""
        self._update_fps_estimate()
        self.pre_buffer.append(frame.copy())
    
    def should_trigger(self, tracks, trigger_seconds=2.0):
        """Check if any track exceeds the trigger duration"""
        frames_threshold = int(trigger_seconds * self.fps)
        for track in tracks:
            if track['track_length'] >= frames_threshold:
                return True, track
        return False, None
    
    def start_recording(self, frame_shape, trigger_track=None):
        """Start a new recording"""
        if self.is_recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_recording_path = os.path.join(
            self.output_dir, f"motion_{timestamp}.mp4"
        )
        
        height, width = frame_shape[:2]
        
        # Use measured FPS, rounded to common values for compatibility
        write_fps = round(self.actual_fps)
        if write_fps < 10:
            write_fps = 15  # Minimum reasonable fps
        elif write_fps > 60:
            write_fps = 30  # Cap at reasonable max
        
        # Use H.264 codec (avc1) for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.recording_writer = cv2.VideoWriter(
            self.current_recording_path, fourcc, write_fps, (width, height)
        )
        
        # Fallback to mp4v if avc1 not available
        if not self.recording_writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.recording_writer = cv2.VideoWriter(
                self.current_recording_path, fourcc, write_fps, (width, height)
            )
        
        if not self.recording_writer.isOpened():
            print(f"ERROR: Could not create recording file: {self.current_recording_path}")
            return
        
        self.is_recording = True
        self.recording_start_time = time.time()
        self.recording_end_time = self.recording_start_time + self.post_buffer_sec
        
        # Write pre-buffer frames first
        pre_buffer_frames = len(self.pre_buffer)
        for buffered_frame in self.pre_buffer:
            self.recording_writer.write(buffered_frame)
        
        track_info = f" (track #{trigger_track['id']}, length={trigger_track['track_length']})" if trigger_track else ""
        print(f"\n🔴 RECORDING STARTED: {self.current_recording_path}")
        print(f"   Trigger{track_info}")
        print(f"   Pre-buffer: {pre_buffer_frames} frames ({pre_buffer_frames/self.actual_fps:.1f}s)")
        print(f"   Recording at: {write_fps} fps (measured: {self.actual_fps:.1f})")
    
    def extend_recording(self):
        """Extend recording duration when motion continues"""
        if self.is_recording:
            self.recording_end_time = time.time() + self.post_buffer_sec
    
    def write_frame(self, frame):
        """Write frame if recording"""
        if self.is_recording and self.recording_writer:
            self.recording_writer.write(frame)
    
    def update(self, frame, tracks, trigger_seconds=2.0):
        """Main update - call every frame"""
        # Always add to pre-buffer
        self.add_frame(frame)
        
        # Check for trigger
        triggered, trigger_track = self.should_trigger(tracks, trigger_seconds)
        
        if triggered:
            if not self.is_recording:
                self.start_recording(frame.shape, trigger_track)
            else:
                self.extend_recording()
        
        # Write frame if recording
        if self.is_recording:
            self.write_frame(frame)
            
            # Check if recording should stop
            if time.time() >= self.recording_end_time:
                self.stop_recording()
        
        return self.is_recording
    
    def stop_recording(self):
        """Stop current recording"""
        if not self.is_recording:
            return
        
        if self.recording_writer:
            self.recording_writer.release()
            self.recording_writer = None
        
        duration = time.time() - self.recording_start_time
        print(f"⬛ RECORDING STOPPED: {self.current_recording_path}")
        print(f"   Duration: {duration:.1f}s")
        
        self.is_recording = False
        self.current_recording_path = None
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_recording()


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
    
    def __init__(self, source, resolution="1280x720", hwaccel='cuda', use_ffmpeg=True,
                 enable_recording=False, trigger_seconds=2.0, output_dir="recordings"):
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
            self.cap = self._create_opencv_capture()
        
        # Keep last valid frame for display continuity
        self.last_frame = None
        self.frame_fail_count = 0
        self.max_frame_fails = 30  # Retry threshold before reconnect
        
        # Processing components
        self.detector = OpticalFlowDetector(flow_threshold=1.5)
        self.tracker = ObjectTracker(max_distance=60)
        
        # Recording
        self.enable_recording = enable_recording
        self.trigger_seconds = trigger_seconds
        self.recorder = None
        if enable_recording:
            self.recorder = MotionRecorder(
                output_dir=output_dir,
                pre_buffer_sec=3,
                post_buffer_sec=5,
                fps=30
            )
        
        # Display options
        self.show_overlay = True
        self.show_tracks = True
        self.show_stats = True
        
        # Performance
        self.fps = 0.0
        self.frame_times = deque(maxlen=30)
        self.overlay_cache = None
        self.overlay_counter = 0
    
    def _create_opencv_capture(self):
        """Create OpenCV VideoCapture with optimal settings for RTSP/RTSPS"""
        source = self.source
        is_rtsp = isinstance(source, str) and source.lower().startswith(('rtsp://', 'rtsps://'))
        
        # Use FFMPEG backend for RTSP streams (better TLS/SRTP support)
        if is_rtsp:
            # Set environment variables for OpenCV's FFMPEG backend
            import os
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|analyzeduration;1000000|fflags;nobuffer'
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(source)
        
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Small buffer, but not 1 (too aggressive)
            if is_rtsp:
                # Additional RTSP optimizations
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        return cap
    
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
        panel_height = 165 if self.enable_recording else 140
        cv2.rectangle(overlay, (10, 10), (250, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (10, 10), (250, panel_height), (0, 255, 0), 1)
        
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
        
        # Recording status
        if self.enable_recording:
            y += 22
            if self.recorder and self.recorder.is_recording:
                cv2.putText(frame, "REC", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Blinking red circle
                if int(time.time() * 2) % 2:
                    cv2.circle(frame, (70, y - 5), 8, (0, 0, 255), -1)
            else:
                cv2.putText(frame, f"Record: trigger>{self.trigger_seconds:.1f}s", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
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
        
        if self.enable_recording:
            print(f"\nRecording: ENABLED")
            print(f"  Trigger: track duration > {self.trigger_seconds}s")
            print(f"  Output dir: {self.recorder.output_dir}")
        
        print("\nControls:")
        print("  Q     - Quit")
        print("  O     - Toggle flow overlay")
        print("  T     - Toggle track markers")
        print("  S     - Toggle stats")
        print("  M/N   - Adjust sensitivity")
        print("  R     - Toggle recording mode" if self.enable_recording else "")
        print("  SPACE - Pause")
        print("=" * 55)
        
        # Start video
        if self.use_ffmpeg:
            if not self.player.start():
                print("FFmpeg failed, using OpenCV...")
                self.use_ffmpeg = False
                self.cap = self._create_opencv_capture()
                if not self.cap.isOpened():
                    print("WARNING: OpenCV VideoCapture failed to open source!")
        
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
                        self.frame_fail_count += 1
                        # Use last valid frame for display continuity
                        if self.last_frame is not None:
                            frame = self.last_frame.copy()
                        else:
                            time.sleep(0.01)
                            continue
                        
                        # Reconnect if too many failures
                        if self.frame_fail_count > self.max_frame_fails:
                            print(f"Too many frame failures ({self.frame_fail_count}), reconnecting...")
                            self.frame_fail_count = 0
                            if not self.use_ffmpeg and self.cap:
                                self.cap.release()
                                time.sleep(0.5)
                                self.cap = self._create_opencv_capture()
                    else:
                        self.frame_fail_count = 0
                        # Resize if needed (some backends ignore size settings)
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = cv2.resize(frame, (self.width, self.height))
                        self.last_frame = frame.copy()
                    
                    # Detect motion
                    detections = self.detector.process_frame(frame)
                    tracks = self.tracker.update(detections)
                    
                    # Update recorder (if enabled)
                    if self.enable_recording and self.recorder:
                        self.recorder.update(frame, tracks, self.trigger_seconds)
                    
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
                elif key == ord('r') and self.enable_recording:
                    if self.recorder and self.recorder.is_recording:
                        self.recorder.stop_recording()
                        print("Recording manually stopped")
                    else:
                        print(f"Recording armed - waiting for {self.trigger_seconds}s motion trigger")
        
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        finally:
            if self.enable_recording and self.recorder:
                self.recorder.cleanup()
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
    
    # Recording options
    parser.add_argument('--record', action='store_true', help='Enable motion-triggered recording')
    parser.add_argument('--trigger', type=float, default=2.0, help='Trigger recording after N seconds of continuous motion')
    parser.add_argument('--output-dir', default='recordings', help='Directory to save recordings')
    
    args = parser.parse_args()
    
    hwaccel = None if args.hwaccel == 'none' else args.hwaccel
    
    system = FlowVisualizationSystem(
        args.source,
        args.resolution,
        hwaccel=hwaccel,
        use_ffmpeg=(args.backend == 'ffmpeg'),
        enable_recording=args.record,
        trigger_seconds=args.trigger,
        output_dir=args.output_dir
    )
    system.detector.flow_threshold = args.threshold
    system.run()


if __name__ == "__main__":
    main()