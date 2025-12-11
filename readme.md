# leto-vision

This is an object detection system that captures video/images, uses ML models to detect aerial objects, tracks their movement via optical flow (OpenCV/RAFT), and exports detection results to JSONL logs with visual outputs

Faster R-CNN (deep fully convolutional + region proposal networks)

### Quick Start
```bash
cd aodetect
python3 main.py --source "rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp" --brightness 220 --confidence 0.1
```
```bash
python3 main.py --source "https://www.youtube.com/watch?v=RH5fgOcO0jg" --brightness 220 --confidence 0.9
```
https://www.youtube.com/watch?v=RH5fgOcO0jg

# RCNN Training
