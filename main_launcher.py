#!/usr/bin/env python3
"""
Main Application Launcher for Ubiquiti Camera Object Detection
"""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import messagebox

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        ('cv2', 'opencv-python'),
        ('PIL', 'Pillow'),
        ('numpy', 'numpy')
    ]
    
    missing_packages = []
    
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        message = f"Missing required packages: {', '.join(missing_packages)}\n\n"
        message += "Install them using:\n"
        message += f"pip install {' '.join(missing_packages)}"
        
        messagebox.showerror("Missing Dependencies", message)
        return False
    
    return True

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_model_files():
    """Check if YOLO model files exist"""
    model_files = [
        'yolov4.weights',
        'yolov4.cfg', 
        'coco.names'
    ]
    
    missing_files = []
    for file in model_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def show_setup_dialog():
    """Show setup instructions dialog"""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    message = """Welcome to Ubiquiti Camera Object Detection!

Before you can use this application, you need to set up a few things:

1. FFMPEG Installation:
   - Ubuntu/Debian: sudo apt install ffmpeg
   - macOS: brew install ffmpeg
   - Windows: Download from https://ffmpeg.org/

2. YOLO Model Files (choose one):
   
   YOLOv4 (recommended - good balance):
   wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights
   wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
   wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
   
   YOLOv4-tiny (faster, less accurate):
   wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
   wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
   wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

3. Make sure your camera stream URL is correct in the GUI

Once you have these files, restart the application.
"""
    
    messagebox.showinfo("Setup Required", message)
    root.destroy()

def main():
    """Main application entry point"""
    print("Starting Ubiquiti Camera Object Detection...")
    
    # Check Python dependencies
    if not check_dependencies():
        return 1
    
    # Check ffmpeg
    if not check_ffmpeg():
        show_ffmpeg_error()
        return 1
    
    # Check model files
    missing_files = check_model_files()
    if missing_files:
        show_model_files_warning(missing_files)
        # Continue anyway - user can browse for files in GUI
    
    # Import and start GUI (after dependency checks)
    try:
        from detection_gui import main as gui_main
        gui_main()
        return 0
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start application: {e}")
        return 1

def show_ffmpeg_error():
    """Show ffmpeg installation error"""
    root = tk.Tk()
    root.withdraw()
    
    message = """FFmpeg is required but not found!

Please install FFmpeg:

Ubuntu/Debian:
  sudo apt update
  sudo apt install ffmpeg

macOS (with Homebrew):
  brew install ffmpeg

Windows:
  Download from https://ffmpeg.org/download.html
  Add to your system PATH

After installation, restart this application.
"""
    
    messagebox.showerror("FFmpeg Not Found", message)
    root.destroy()

def show_model_files_warning(missing_files):
    """Show warning about missing model files"""
    root = tk.Tk()
    root.withdraw()
    
    message = f"""YOLO model files not found: {', '.join(missing_files)}

You can either:
1. Download them now using the commands shown in the next dialog
2. Browse for existing files in the GUI

The application will start, but you'll need to specify the model files
in the interface before detection will work.
"""
    
    result = messagebox.askquestion("Model Files Missing", 
                                   message + "\n\nShow download instructions?")
    
    if result == 'yes':
        show_setup_dialog()
    
    root.destroy()

if __name__ == "__main__":
    sys.exit(main())