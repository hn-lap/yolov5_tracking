## Object tracking

1. Yolov5 object detection
2. Simple Online and Realtime Tracking (SORT)
3. Deep Simple Online and Realtime Tracking (Deep SORT)

## Setup

```
conda create -n myenv python=3.9 -y
```

after then

```
pip install -r setup.txt
```

## Run

1. Run object tracking

```python
# Run object detection
sh scripts/object_detect.sh
```

2. Run object detection + tracking (yolov5 + sort)

```python
# Run object detection + tracking (yolov5 + sort)
sh scripts/object_sort.sh
```
3. Run object detection + tracking (yolov5 + deep sort) 
```python
# Run object detection + tracking (yolov5 + deep_sort)
sh scripts/object_deep_sort.sh
```
## Model library

| **2D Detection** | **Multi Object Tracking** | others |
| ---------------------- | ------------------------------- | ------ |
| Yolov5                 | 1.Sort<br />2. Deep Sort       |        |

## References

1. [ultralytics/yolov5: YOLOv5 🚀 in PyTorch &gt; ONNX &gt; CoreML &gt; TFLite (github.com)](https://github.com/ultralytics/yolov5)
2. https://github.com/abewley/sort
3. https://github.com/ZQPei/deep_sort_pytorch
