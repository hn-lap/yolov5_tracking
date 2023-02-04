## Object tracking

1. Yolov5 object detection
2. Simple Online and Realtime Tracking (SORT)

## Setup

```
conda create -n myenv python=3.9 -y
```
after then
```
pip install -r setup.txt
```

## Run
```python
# Run object detection
python object_detect.py --weights ./saved_models/yolov5s.pt --source 2.mp4 --classes 0
```

2. Run object tracking

```python
# Run object detection + tracking (yolov5 + sort)
python object_tracking.py --weights ./saved_models/yolov5s.pt --source 2.mp4 --classes 0
```

## References
1. [ultralytics/yolov5: YOLOv5 🚀 in PyTorch &gt; ONNX &gt; CoreML &gt; TFLite (github.com)](https://github.com/ultralytics/yolov5)
2. https://github.com/abewley/sort
