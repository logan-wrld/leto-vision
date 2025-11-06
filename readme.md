# leto-vision

## Enhanced Vehicle Detection System
This is an aerial object detection system that captures video/images, uses ML models to detect aerial objects, tracks their movement via optical flow (OpenCV/RAFT), and exports detection results to JSONL logs with visual outputs

### Quick Start
```bash
cd aodetect
python3 main.py --source "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp" --brightness 220 --confidence 0.1
```

### Features
- **60 FPS Native Stream Processing** - No artificial frame rate limits
- **Ultra-Distant Vehicle Detection** - Detects cars as small as 2-pixel areas
- **Smart False Positive Filtering** - Eliminates building lights and erratic flashing
- **Multi-Level Detection**: CYAN (ultra-distant), MAGENTA (tiny), RED (priority vehicles)
- **Real-time Movement Tracking** with direction arrows and confidence scoring