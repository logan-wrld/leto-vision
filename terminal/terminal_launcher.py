#!/usr/bin/env python3
"""
Terminal Detection Launcher
No GUI, no segfaults, just pure stability
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check required packages"""
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True

def check_ffmpeg():
    """Check ffmpeg"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        print("❌ FFmpeg not found!")
        print("Install: sudo apt install ffmpeg")
        return False

def main():
    print("\033[32m🚀 Starting LETO VISION Terminal Mode...\033[0m")
    
    if not check_dependencies():
        return 1
    
    if not check_ffmpeg():
        return 1
    
    # Check for YOLO files
    yolo_files = ['yolov4.weights', 'yolov4.cfg', 'coco.names']
    missing = [f for f in yolo_files if not os.path.exists(f)]
    
    if missing:
        print(f"\033[33m⚠️  Missing YOLO files: {', '.join(missing)}\033[0m")
        print("\n🔽 Download commands:")
        print("wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights")
        print("wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg")
        print("wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names")
        print("\n✨ App will start anyway - you can specify file paths in settings!")
    
    try:
        from terminal_detector import main as detector_main
        detector_main()
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())