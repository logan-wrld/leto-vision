#!/usr/bin/env python3
"""
CUDA-Accelerated FFmpeg Stream Player
High-performance video playback with hardware acceleration
FIXED: Better playback, proper NVIDIA support, reduced latency
"""

import subprocess
import threading
import queue
import numpy as np
import cv2
import time
from collections import deque
import os
import shutil


class CUDAStreamPlayer:
    """
    FFmpeg-based stream player with CUDA hardware acceleration
    Optimized for smooth playback with minimal latency
    """
    
    def __init__(self, source, width=1280, height=720, fps=30, hwaccel='cuda', buffer_size=2):
        """
        Initialize CUDA stream player
        
        Args:
            source: Video source (file, RTSP URL, device index, etc.)
            width: Output width
            height: Output height
            fps: Target FPS
            hwaccel: Hardware acceleration ('cuda', 'nvdec', 'vaapi', 'auto', or None)
            buffer_size: Frame buffer size (keep small for low latency)
        """
        self.source = self._normalize_source(source)
        self.width = width
        self.height = height
        self.fps = fps
        self.hwaccel = hwaccel
        self.buffer_size = max(1, min(buffer_size, 5))  # Clamp to reasonable range
        
        self.process = None
        self.frame_queue = queue.Queue(maxsize=self.buffer_size)
        self.running = False
        self.reader_thread = None
        self._lock = threading.Lock()
        
        # Performance tracking
        self.frames_read = 0
        self.frames_dropped = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.fps_history = deque(maxlen=60)
        self._actual_hwaccel = None
        
        # Detect available hardware acceleration
        self._available_hwaccels = self._detect_hwaccels()
        
    def _normalize_source(self, source):
        """Normalize video source input"""
        # Handle integer camera index
        if isinstance(source, int):
            # Linux video device
            return f"/dev/video{source}"
        # Handle string camera index
        if isinstance(source, str) and source.isdigit():
            return f"/dev/video{source}"
        return source
    
    def _detect_hwaccels(self):
        """Detect available hardware accelerators"""
        available = {
            'cuda': False,
            'nvdec': False,
            'vaapi': False,
            'qsv': False,
        }
        
        try:
            result = subprocess.run(
                ['ffmpeg', '-hwaccels'],
                capture_output=True,
                text=True,
                timeout=5
            )
            output = result.stdout.lower()
            
            for accel in available:
                available[accel] = accel in output
                
            # Also check for NVIDIA decoders
            result = subprocess.run(
                ['ffmpeg', '-decoders'],
                capture_output=True,
                text=True,
                timeout=5
            )
            decoder_output = result.stdout.lower()
            
            # Check for cuvid decoders
            available['h264_cuvid'] = 'h264_cuvid' in decoder_output
            available['hevc_cuvid'] = 'hevc_cuvid' in decoder_output
            available['vp9_cuvid'] = 'vp9_cuvid' in decoder_output
            
        except Exception as e:
            print(f"Warning: Could not detect hardware accelerators: {e}")
        
        return available
    
    def _build_ffmpeg_command(self, use_hwaccel=True):
        """Build optimized ffmpeg command"""
        cmd = ['ffmpeg', '-hide_banner']
        
        is_webcam = self.source.startswith('/dev/video')
        is_rtsp = self.source.startswith('rtsp://') or self.source.startswith('rtsps://')
        is_rtsps = self.source.startswith('rtsps://')
        is_file = os.path.isfile(self.source) if isinstance(self.source, str) else False
        
        # === INPUT CONFIGURATION ===
        
        # Hardware acceleration for decoding (files/streams only, not webcams)
        if use_hwaccel and self.hwaccel and not is_webcam:
            if self.hwaccel in ('cuda', 'nvdec', 'auto'):
                if self._available_hwaccels.get('cuda'):
                    cmd.extend([
                        '-hwaccel', 'cuda',
                        '-hwaccel_device', '0',
                        '-hwaccel_output_format', 'cuda',  # Keep frames on GPU
                    ])
                    self._actual_hwaccel = 'cuda'
            elif self.hwaccel == 'vaapi':
                if self._available_hwaccels.get('vaapi'):
                    cmd.extend([
                        '-hwaccel', 'vaapi',
                        '-hwaccel_device', '/dev/dri/renderD128',
                        '-hwaccel_output_format', 'vaapi',
                    ])
                    self._actual_hwaccel = 'vaapi'
        
        # Webcam-specific options
        if is_webcam:
            # Do not force an input format (e.g. 'mjpeg') here — some devices
            # expose different pixel formats (yuyv422, mjpeg, etc.). Let ffmpeg
            # auto-negotiate the best format for the device to improve
            # compatibility across webcams.
            cmd.extend([
                '-f', 'v4l2',
                '-framerate', str(self.fps),
                '-video_size', f'{self.width}x{self.height}',
            ])
        
        # RTSP/RTSPS-specific options (low latency)
        if is_rtsp:
            cmd.extend([
                '-rtsp_transport', 'tcp',
                '-fflags', '+nobuffer+flush_packets+genpts',
                '-flags', 'low_delay',
                '-analyzeduration', '1000000',  # 1s - give more time for encrypted streams
                '-probesize', '1000000',
                '-max_delay', '500000',
            ])
            # RTSPS (RTSP over TLS) needs additional options
            if is_rtsps:
                cmd.extend([
                    '-rtsp_flags', 'prefer_tcp',
                ])
        
        # File-specific options
        if is_file:
            cmd.extend([
                '-re',  # Read at native frame rate (optional, remove for max speed)
            ])
        
        # Input source
        cmd.extend(['-i', self.source])
        
        # === VIDEO FILTERS ===
        vf_filters = []
        
        # Handle GPU->CPU transfer and scaling
        if self._actual_hwaccel == 'cuda' and not is_webcam:
            # Scale on GPU, download to CPU in nv12, then let -pix_fmt convert to bgr24
            vf_filters.append(f'scale_cuda=w={self.width}:h={self.height}')
            vf_filters.append('hwdownload')
            vf_filters.append('format=nv12')  # hwdownload outputs nv12, -pix_fmt handles bgr24
        elif self._actual_hwaccel == 'vaapi' and not is_webcam:
            vf_filters.append(f'scale_vaapi=w={self.width}:h={self.height}')
            vf_filters.append('hwdownload')
            vf_filters.append('format=nv12')  # hwdownload outputs nv12, -pix_fmt handles bgr24
        else:
            # CPU scaling with fast algorithm
            vf_filters.append(f'scale={self.width}:{self.height}:flags=fast_bilinear')
        
        if vf_filters:
            cmd.extend(['-vf', ','.join(vf_filters)])
        
        # === OUTPUT CONFIGURATION ===
        cmd.extend([
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-an',  # No audio
            '-sn',  # No subtitles
            '-dn',  # No data streams
            '-threads', '2',  # Limit threads to reduce overhead
            'pipe:1',  # Output to stdout
        ])
        
        # Logging level
        cmd.extend(['-loglevel', 'error'])
        
        return cmd
    
    def _reader_loop(self):
        """Background thread for reading frames - optimized for low latency"""
        frame_size = self.width * self.height * 3
        buffer = bytearray(frame_size)
        mv = memoryview(buffer)
        
        while self.running:
            try:
                # Read directly into pre-allocated buffer
                bytes_read = 0
                while bytes_read < frame_size and self.running:
                    chunk = self.process.stdout.read(frame_size - bytes_read)
                    if not chunk:
                        if self.running:
                            print("Stream ended (no data)")
                        self.running = False
                        break
                    mv[bytes_read:bytes_read + len(chunk)] = chunk
                    bytes_read += len(chunk)
                
                if bytes_read != frame_size:
                    break
                
                # Convert to numpy (zero-copy view)
                frame = np.frombuffer(buffer, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
                
                # Non-blocking queue management
                with self._lock:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                            self.frames_dropped += 1
                        except queue.Empty:
                            pass
                    
                    try:
                        self.frame_queue.put_nowait(frame)
                        self.frames_read += 1
                    except queue.Full:
                        self.frames_dropped += 1
                        
            except Exception as e:
                if self.running:
                    print(f"Reader error: {e}")
                break
    
    def start(self):
        """Start the stream player"""
        if self.running:
            return True
        
        is_rtsp = isinstance(self.source, str) and self.source.startswith('rtsp')
        startup_timeout = 5.0 if is_rtsp else 3.0  # Longer timeout for network streams
        
        # Try with hardware acceleration first
        for attempt, use_hw in enumerate([True, False]):
            if attempt == 1:
                print("Falling back to software decoding...")
                self._actual_hwaccel = None
            
            cmd = self._build_ffmpeg_command(use_hwaccel=use_hw)
            print(f"FFmpeg command: {' '.join(cmd)}")
            
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,  # Unbuffered for lowest latency
                )
                
                # Quick check if process started
                time.sleep(0.5)
                if self.process.poll() is not None:
                    stderr = self.process.stderr.read().decode('utf-8', errors='ignore')
                    print(f"FFmpeg stderr: {stderr[:1000]}")
                    raise RuntimeError(f"FFmpeg failed immediately")
                
                self.running = True
                self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
                self.reader_thread.start()
                
                # Wait for first frame
                start_wait = time.time()
                while self.frame_queue.empty() and time.time() - start_wait < startup_timeout:
                    if self.process.poll() is not None:
                        stderr = self.process.stderr.read().decode('utf-8', errors='ignore')
                        print(f"FFmpeg stderr: {stderr[:1000]}")
                        raise RuntimeError("FFmpeg process died")
                    time.sleep(0.1)
                
                if self.frame_queue.empty():
                    # Try to get error info
                    if self.process.poll() is None:
                        self.process.terminate()
                        try:
                            stderr = self.process.stderr.read().decode('utf-8', errors='ignore')
                            print(f"FFmpeg stderr: {stderr[:1000]}")
                        except:
                            pass
                    raise RuntimeError("No frames received")
                
                accel_str = self._actual_hwaccel or 'software'
                print(f"✓ Stream started ({accel_str} decoding)")
                return True
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                self.stop()
                if attempt == 1:  # Both attempts failed
                    return False
        
        return False
    
    def read(self):
        """Read next frame (non-blocking with small timeout)"""
        if not self.running:
            return False, None
        
        try:
            frame = self.frame_queue.get(timeout=0.1)
            
            # Update FPS tracking
            now = time.time()
            self.fps_history.append(now)
            
            # Calculate FPS every second
            if now - self.last_fps_time >= 1.0:
                if len(self.fps_history) > 1:
                    time_span = self.fps_history[-1] - self.fps_history[0]
                    if time_span > 0:
                        self.current_fps = (len(self.fps_history) - 1) / time_span
                self.last_fps_time = now
            
            return True, frame
            
        except queue.Empty:
            return False, None
    
    def stop(self):
        """Stop the stream player"""
        self.running = False
        
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            self.process = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        drop_rate = (self.frames_dropped / max(1, self.frames_read + self.frames_dropped)) * 100
        print(f"Stream stopped. Frames: {self.frames_read} read, {self.frames_dropped} dropped ({drop_rate:.1f}%)")
    
    def get_fps(self):
        """Get current playback FPS"""
        return self.current_fps
    
    def get_stats(self):
        """Get playback statistics"""
        total = self.frames_read + self.frames_dropped
        return {
            'fps': self.current_fps,
            'frames_read': self.frames_read,
            'frames_dropped': self.frames_dropped,
            'drop_rate': (self.frames_dropped / max(1, total)) * 100,
            'queue_size': self.frame_queue.qsize(),
            'hwaccel': self._actual_hwaccel or 'software',
        }


def test_cuda_available():
    """Test available hardware acceleration"""
    result = {'cuda': False, 'nvdec': False, 'vaapi': False, 'h264_cuvid': False}
    
    if not shutil.which('ffmpeg'):
        print("FFmpeg not found!")
        return result
    
    try:
        # Check hwaccels
        proc = subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True, timeout=5)
        output = proc.stdout.lower()
        result['cuda'] = 'cuda' in output
        result['nvdec'] = 'nvdec' in output or 'cuda' in output
        result['vaapi'] = 'vaapi' in output
        
        # Check decoders
        proc = subprocess.run(['ffmpeg', '-decoders'], capture_output=True, text=True, timeout=5)
        output = proc.stdout.lower()
        result['h264_cuvid'] = 'h264_cuvid' in output
        result['hevc_cuvid'] = 'hevc_cuvid' in output
        
    except Exception as e:
        print(f"Error checking CUDA: {e}")
    
    return result


if __name__ == "__main__":
    print("=" * 50)
    print("CUDA Stream Player Test")
    print("=" * 50)
    
    # Check hardware support
    print("\nChecking hardware acceleration...")
    hw_support = test_cuda_available()
    for name, available in hw_support.items():
        print(f"  {'✓' if available else '✗'} {name}")
    
    # Determine best hwaccel
    hwaccel = 'auto'
    if hw_support['cuda']:
        hwaccel = 'cuda'
    elif hw_support['vaapi']:
        hwaccel = 'vaapi'
    else:
        hwaccel = None
    
    print(f"\nUsing: {hwaccel or 'software'}")
    
    # Test with webcam
    player = CUDAStreamPlayer(
        source=0,
        width=1280,
        height=720,
        fps=30,
        hwaccel=hwaccel,
        buffer_size=2
    )
    
    if player.start():
        print("\nPress 'q' to quit")
        cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = player.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                # Overlay stats
                stats = player.get_stats()
                cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"HW: {stats['hwaccel']}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                cv2.putText(frame, f"Drop: {stats['drop_rate']:.1f}%", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.imshow('Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            player.stop()
            cv2.destroyAllWindows()
    else:
        print("Failed to start player")