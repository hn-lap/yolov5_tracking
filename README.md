## Object tracking

1. Yolov5 object detection
2. Simple Online and Realtime Tracking (SORT)
3. Run object detection

```python
# Run object detection
python object_detect.py --weights ./saved_models/yolov5s.pt --source 2.mp4 --classes 0
```

2. Run object tracking

```python
# Run object detection + tracking (yolov5 + sort)
python object_tracking.py --weights ./saved_models/yolov5s.pt --source 2.mp4 --classes 0
```
