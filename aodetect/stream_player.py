#!/usr/bin/env python3
"""
Hardware-Accelerated Stream Player
Uses ffmpeg directly for smooth YouTube/RTSP playback on M1 Macs
"""

import cv2
import numpy as np
import subprocess
import threading
import queue
import time
import sys


class HardwareStreamPlayer:
    """
    Hardware-accelerated video stream player using ffmpeg pipe.
    Much smoother than OpenCV's VideoCapture for HTTP/YouTube streams.
    """
    
    def __init__(self, source, width=1280, height=720, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        
        # Larger buffer for smooth playback (prevents jolts)
        self.frame_queue = queue.Queue(maxsize=60)  # ~2 seconds buffer
        self.running = False
        self.process = None
        self.reader_thread = None
        
        self.frame_count = 0
        self.start_time = None
        self.actual_fps = 0
        self.playback_start_time = None
        self.frames_displayed = 0
        self.target_frame_time = 1.0 / fps
        
    def get_youtube_url(self, youtube_url):
        """Get direct stream URL from YouTube using yt-dlp"""
        # Request a lower quality for smoother playback
        cmd = [
            'yt-dlp', 
            '-f', 'best[height<=720][ext=mp4]/best[height<=720]/best',
            '-g', 
            youtube_url
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            return result.stdout.strip()
        except Exception as e:
            print(f"Error getting YouTube URL: {e}")
            return None
    
    def start(self):
        """Start the stream player"""
        stream_url = self.source
        
        # Handle YouTube URLs
        if 'youtube.com' in self.source or 'youtu.be' in self.source:
            print("🎬 Getting YouTube stream URL...")
            stream_url = self.get_youtube_url(self.source)
            if not stream_url:
                print("❌ Failed to get YouTube stream URL")
                return False
            print(f"✅ Got stream URL: {stream_url[:80]}...")
        
        # Build ffmpeg command with real-time pacing
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            # Reconnect options for streams
            '-reconnect', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '5',
            # Input
            '-i', stream_url,
            # Output options - use vsync to pace output
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-vsync', 'cfr',  # Constant frame rate
            '-r', str(self.fps),
            # Output to pipe
            '-'
        ]
        
        print(f"🎬 Starting ffmpeg: {self.width}x{self.height} @ {self.fps}fps")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.width * self.height * 3 * 10  # 10 frames buffer
            )
            
            self.running = True
            self.start_time = time.time()
            
            # Start reader thread
            self.reader_thread = threading.Thread(target=self._read_frames, daemon=True)
            self.reader_thread.start()
            
            print(f"🎥 Stream started: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except FileNotFoundError:
            print("❌ ffmpeg not found. Please install ffmpeg:")
            print("   brew install ffmpeg")
            return False
        except Exception as e:
            print(f"❌ Error starting stream: {e}")
            return False
    
    def _read_frames(self):
        """Background thread to read frames from ffmpeg"""
        frame_size = self.width * self.height * 3
        
        # Wait a moment for ffmpeg to start
        time.sleep(0.5)
        
        consecutive_errors = 0
        max_errors = 10
        
        while self.running and self.process:
            try:
                # Check if process is still running
                if self.process.poll() is not None:
                    print("⚠️ ffmpeg process ended")
                    # Print any error output
                    stderr = self.process.stderr.read()
                    if stderr:
                        print(f"ffmpeg error: {stderr.decode()[:500]}")
                    break
                
                # Read one frame
                raw_frame = self.process.stdout.read(frame_size)
                
                if len(raw_frame) == 0:
                    consecutive_errors += 1
                    if consecutive_errors > max_errors:
                        print("⚠️ Too many empty reads, stopping")
                        break
                    time.sleep(0.01)
                    continue
                
                if len(raw_frame) != frame_size:
                    consecutive_errors += 1
                    if consecutive_errors > max_errors:
                        print(f"⚠️ Frame size mismatch: got {len(raw_frame)}, expected {frame_size}")
                        break
                    continue
                
                consecutive_errors = 0  # Reset on successful read
                
                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8).copy()
                frame = frame.reshape((self.height, self.width, 3))
                
                self.frame_count += 1
                
                # Update FPS calculation
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # Put frame in queue - block if full to maintain sync
                try:
                    self.frame_queue.put(frame, timeout=0.5)
                except queue.Full:
                    # Drop oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
                
            except Exception as e:
                if self.running:
                    print(f"Frame read error: {e}")
                break
        
        self.running = False
    
    def read(self):
        """Read a frame from the stream with proper pacing"""
        if not self.running:
            return False, None
        
        # Initialize playback timing on first read
        if self.playback_start_time is None:
            self.playback_start_time = time.time()
            self.frames_displayed = 0
        
        try:
            # Wait for buffer to fill initially (prevents early jitter)
            if self.frames_displayed == 0:
                while self.frame_queue.qsize() < 30 and self.running:
                    time.sleep(0.05)
                print(f"📦 Buffer ready: {self.frame_queue.qsize()} frames")
            
            frame = self.frame_queue.get(timeout=1.0)
            self.frames_displayed += 1
            
            # Frame pacing - wait if we're ahead of schedule
            expected_time = self.playback_start_time + (self.frames_displayed * self.target_frame_time)
            current_time = time.time()
            sleep_time = expected_time - current_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -0.5:
                # We're way behind, reset timing
                self.playback_start_time = time.time()
                self.frames_displayed = 1
            
            return True, frame
        except queue.Empty:
            return False, None
    
    def get_fps(self):
        """Get current FPS"""
        return self.actual_fps
    
    def stop(self):
        """Stop the stream player"""
        self.running = False
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                self.process.kill()
            self.process = None
        
        print("🛑 Stream stopped")


def main():
    """Test the hardware stream player"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hardware-Accelerated Stream Player')
    parser.add_argument('--source', default='https://www.youtube.com/watch?v=RH5fgOcO0jg',
                       help='Video source (YouTube URL, RTSP, or file)')
    parser.add_argument('--width', type=int, default=960, help='Video width')
    parser.add_argument('--height', type=int, default=540, help='Video height')
    parser.add_argument('--fps', type=int, default=24, help='Target FPS')
    
    args = parser.parse_args()
    
    # Create player
    player = HardwareStreamPlayer(args.source, args.width, args.height, args.fps)
    
    if not player.start():
        print("Failed to start player")
        return
    
    # Create window
    cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stream', args.width, args.height)
    
    print("\n🎮 Controls: Q=Quit")
    
    try:
        while True:
            ret, frame = player.read()
            
            if ret:
                # Add FPS indicator
                fps = player.get_fps()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Stream', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
