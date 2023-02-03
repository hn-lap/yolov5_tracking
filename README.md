
### YOLOv5 object detection and Sort tracker
1. Run object detection
```
python object_detect.py --weights yolov5s.pt --source 2.mp4 --classes 0
```
2. Run object tracking
```
python object_tracking.py --weights yolov5s.pt --source 2.mp4 --classes 0
```