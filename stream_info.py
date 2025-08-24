#!/usr/bin/env python3
"""
Get information about the RTSP stream to determine proper resolution
"""

import subprocess
import re

def get_stream_info(stream_url):
    """Get stream information using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        stream_url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print("FFprobe output:")
        print(result.stdout)
        if result.stderr:
            print("FFprobe errors:")
            print(result.stderr)
    except Exception as e:
        print(f"FFprobe failed: {e}")
    
    # Alternative: use ffmpeg to get stream info
    print("\n" + "="*50)
    print("Alternative: FFmpeg stream analysis")
    print("="*50)
    
    cmd2 = [
        'ffmpeg',
        '-i', stream_url,
        '-t', '1',  # Just 1 second
        '-f', 'null',
        '-'
    ]
    
    try:
        result = subprocess.run(cmd2, capture_output=True, text=True, timeout=15)
        # Parse the stderr output for resolution info
        stderr = result.stderr
        
        # Look for resolution in the output
        resolution_match = re.search(r'(\d+)x(\d+)', stderr)
        if resolution_match:
            width, height = resolution_match.groups()
            print(f"Detected resolution: {width}x{height}")
        
        # Look for frame rate
        fps_match = re.search(r'(\d+(?:\.\d+)?)\s*fps', stderr)
        if fps_match:
            fps = fps_match.group(1)
            print(f"Detected FPS: {fps}")
        
        print("\nFull FFmpeg output:")
        print(stderr)
        
    except Exception as e:
        print(f"FFmpeg analysis failed: {e}")

if __name__ == "__main__":
    stream_url = "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp"
    get_stream_info(stream_url)