export PYTHONPATH=.
python tracking.py --weights ./saved_models/yolov5s.pt --source test.mp4 --classes 0 --type_mot sort
