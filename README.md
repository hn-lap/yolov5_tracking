# Object tracking
## Description

1. Yolov5 object detection
2. Simple Online and Realtime Tracking (SORT)
3. Deep Simple Online and Realtime Tracking (Deep SORT)
4. ByteTrack

## Model library

| **2D Detection** | **Multi Object Tracking** | **KeyPoint Detection** |
| ---------------------- | ------------------------------- | ------ |
| Yolov5                 | 1.Sort<br />2. Deep Sort<br /> 3. ByteTrack      |     1. TinyPose   |

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
"""
@weigths: file checkpoint
@source:  file input
@classes: filter classes
@type_mot: {null|sort|deep_sort} 
"""
export PYTHONPATH=.

python main.py --weights ./saved_models/yolov5s.pt \
               --source ./test_video/test.mp4 \
               --classes 0 32 --type_mot deep_sort
```
OR 

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

## References

1. [ultralytics/yolov5: YOLOv5 🚀 in PyTorch &gt; ONNX &gt; CoreML &gt; TFLite (github.com)](https://github.com/ultralytics/yolov5)
2. https://github.com/abewley/sort
3. https://github.com/ZQPei/deep_sort_pytorch

## Conclusion

This documentation has demonstrated how to use related module. Before actually start working on anything, please read the whole document first. If you need any clarifications, please contact me. Thanks for reading and good luck on improving the model.
## Happy Coding