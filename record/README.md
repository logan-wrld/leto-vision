# RTSPS Video Stream Recorder

A Python script to record video from RTSPS (secure RTSP) streams.

## Requirements

- Python 3.6 or higher
- OpenCV with FFmpeg support

## Installation

### Install OpenCV

```bash
pip install opencv-python --break-system-packages
```

### Install FFmpeg (if not already installed)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

## Usage

### Basic Usage

Record from your RTSPS stream (stop with Ctrl+C):

```bash
python rtsp_recorder.py "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp"
```

### Record for Specific Duration

Record for 60 seconds:

```bash
python rtsp_recorder.py "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp" -d 60
```

### Specify Output Filename

```bash
python rtsp_recorder.py "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp" -o camera_feed.avi
```

### Use Different Codec

```bash
python rtsp_recorder.py "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp" -c XVID -o recording.avi
```

## Command Line Options

- `rtsp_url` (required): The RTSPS stream URL
- `-o, --output`: Output filename (default: recording_TIMESTAMP.avi)
- `-d, --duration`: Recording duration in seconds (default: unlimited)
- `-c, --codec`: Video codec (default: mp4v). Options: mp4v, XVID, MJPG, X264

## Features

- Records RTSPS streams securely
- Automatic reconnection on connection loss
- Timestamped output files
- Real-time recording progress display
- Configurable duration and codecs
- Graceful shutdown with Ctrl+C

## Troubleshooting

### "Could not open RTSPS stream" Error

1. Verify the RTSPS URL is correct and accessible
2. Check network connectivity to the camera/stream
3. Ensure the camera allows connections from your IP
4. Verify credentials if the stream requires authentication

### Poor Video Quality or Lag

- Try using TCP transport (already configured in the script)
- Check your network bandwidth
- Try a different codec (e.g., MJPG for lower latency)

### Installation Issues

If you get codec errors, ensure FFmpeg is properly installed:

```bash
# Test FFmpeg installation
ffmpeg -version

# Test OpenCV FFmpeg support
python -c "import cv2; print(cv2.getBuildInformation())" | grep -i ffmpeg
```

## Notes

- The script uses TCP transport by default for more reliable streaming
- Output files are in AVI format by default
- For long recordings, ensure you have sufficient disk space
- The script will attempt to reconnect if frames are dropped
