export PYTHONPATH=.
python main.py --weights ./saved_models/yolov5s.pt --source ./test_video/test.mp4 --classes 0 32 --type_mot deep_sort
