#!/usr/bin/env python3
"""
RTSPS Video Stream Recorder

This script records video from an RTSPS (secure RTSP) stream and saves it to a file.
"""

import cv2
import sys
import time
from datetime import datetime
import argparse
import os

def record_rtsp_stream(rtsp_url, output_file=None, duration=None, codec='mp4v'):
    """
    Record video from an RTSPS stream.
    
    Args:
        rtsp_url: The RTSPS URL to connect to
        output_file: Output filename (default: timestamped filename)
        duration: Recording duration in seconds (default: unlimited, stop with Ctrl+C)
        codec: Video codec to use (default: 'mp4v')
    """
    
    # Generate default output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"recording_{timestamp}.avi"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Connecting to RTSPS stream: {rtsp_url}")
    
    # Configure OpenCV to use TCP transport (more reliable for RTSP)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("Error: Could not open RTSPS stream")
        print("Make sure the URL is correct and the stream is accessible")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 25  # Default to 25 fps if unable to detect
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Stream properties: {width}x{height} @ {fps} FPS")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not open output file for writing")
        cap.release()
        return False
    
    print(f"Recording to: {output_file}")
    if duration:
        print(f"Duration: {duration} seconds")
    else:
        print("Duration: Unlimited (press Ctrl+C to stop)")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("\nWarning: Failed to read frame from stream")
                # Try to reconnect
                time.sleep(1)
                continue
            
            # Write the frame
            out.write(frame)
            frame_count += 1
            
            # Display progress
            elapsed = time.time() - start_time
            if frame_count % (fps * 5) == 0:  # Update every 5 seconds
                print(f"\rRecording... {elapsed:.1f}s - {frame_count} frames", end='', flush=True)
            
            # Check if duration limit reached
            if duration and elapsed >= duration:
                print(f"\n\nRecording complete: {duration} seconds")
                break
                
    except KeyboardInterrupt:
        print("\n\nRecording stopped by user")
    
    finally:
        # Release everything
        elapsed = time.time() - start_time
        print(f"\nTotal recording time: {elapsed:.1f} seconds")
        print(f"Total frames recorded: {frame_count}")
        print(f"Output file: {output_file}")
        
        cap.release()
        out.release()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Record video from an RTSPS stream',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record indefinitely (stop with Ctrl+C)
  python rtsp_recorder.py "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp"
  
  # Record for 60 seconds
  python rtsp_recorder.py "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp" -d 60
  
  # Record to specific file
  python rtsp_recorder.py "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp" -o my_recording.avi
        """
    )
    
    parser.add_argument(
        'rtsp_url',
        help='RTSPS stream URL'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output filename (default: recording_TIMESTAMP.avi)',
        default=None
    )
    parser.add_argument(
        '-d', '--duration',
        type=int,
        help='Recording duration in seconds (default: unlimited)',
        default=None
    )
    parser.add_argument(
        '-c', '--codec',
        help='Video codec (default: mp4v). Options: mp4v, XVID, MJPG, X264',
        default='mp4v'
    )
    
    args = parser.parse_args()
    
    success = record_rtsp_stream(
        args.rtsp_url,
        args.output,
        args.duration,
        args.codec
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
