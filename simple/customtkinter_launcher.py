#!/usr/bin/env python3
"""
Launcher for CustomTkinter GUI
"""

import sys
import subprocess
import os

def install_customtkinter():
    """Install customtkinter if missing"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter"])
        print("✅ CustomTkinter installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install CustomTkinter")
        return False

def check_dependencies():
    """Check all dependencies"""
    dependencies = {
        'customtkinter': 'customtkinter',
        'cv2': 'opencv-python', 
        'PIL': 'Pillow',
        'numpy': 'numpy'
    }
    
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        
        # Auto-install CustomTkinter if it's the only missing one
        if missing == ['customtkinter']:
            print("🔧 Installing CustomTkinter...")
            if install_customtkinter():
                return True
        
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
        print("Install: sudo apt install ffmpeg  (or brew install ffmpeg)")
        return False

def main():
    print("🚀 Starting LETO VISION with Modern Dark UI...")
    
    if not check_dependencies():
        return 1
    
    if not check_ffmpeg():
        return 1
    
    # Check for YOLO files
    yolo_files = ['yolov4.weights', 'yolov4.cfg', 'coco.names']
    missing = [f for f in yolo_files if not os.path.exists(f)]
    
    if missing:
        print(f"⚠️  Missing YOLO files: {', '.join(missing)}")
        print("\n🔽 Download commands:")
        print("wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights")
        print("wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg")
        print("wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names")
        print("\n✨ You can still run the app and select files manually!")
    
    try:
        from modern_dark_gui import main as gui_main
        gui_main()
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())