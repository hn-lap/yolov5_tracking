import argparse
from trackers import *
from tracking import YOLOV5_TRACKING


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="./saved_models/yolov5s.pt",
        help="model path(s)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="./test.mp4",
        help="file/dir/URL/glob, 0 for webcam",
    )
    parser.add_argument(
        "--img_size", nargs="+", default=(640, 640), help="inference size h,w"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--type_mot", type=str, default="sort", help="type multi object tracking"
    )

    opt = parser.parse_args()
    return opt


def main(opt):
    opt = vars(opt)
    if opt["type_mot"] == "sort":
        inference = YOLOV5_TRACKING(
            weights=opt["weights"],
            img_size=opt["img_size"],
            classes=opt["classes"],
            type_mot=opt["type_mot"],
        )
        inference.infer_simple_object_recognition_tracking(
            source=opt["source"], view_video=True, save_img=False
        )
    elif opt["type_mot"] == "deep_sort":
        inference = YOLOV5_TRACKING(
            weights=opt["weights"],
            img_size=opt["img_size"],
            classes=opt["classes"],
            type_mot=opt["type_mot"],
        )
        inference.infer_deep_sort(source=opt["source"], view_video=True, save_img=False)
    else:
        inference = YOLOV5_TRACKING(
            weights=opt["weights"],
            img_size=opt["img_size"],
            classes=opt["classes"],
            type_mot=opt["type_mot"],
        )
        inference.infer_object_detect(
            source=opt["source"], view_video=True, save_img=False
        )


if __name__ == "__main__":
    main(parse_opt())
