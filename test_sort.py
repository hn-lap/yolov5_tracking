import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import shapely
import torch
import torch.backends.cudnn as cudnn
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from models.common import DetectMultiBackend
from trackers.sort import *
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (check_img_size, increment_path, non_max_suppression,
                           scale_coords)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def save_frame_alert(img, i):
    cv2.putText(img, "ALARM", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(f"./alert_folder/alert_{i}.png", img)
    return img


def draw_polygon(frame, points):
    for point in points:
        frame = cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)
    frame = cv2.polylines(frame, [np.int32(points)], False, (255, 0, 0), thickness=2)
    return frame


@torch.no_grad()
def infer(
    weights="./saved_models/yolov5s.pt",
    source: str = "video.mp4",
    img_size: Tuple[int] = (640, 640),
    conf_thres: float = 0.25,
    iou_thres: float = 0.35,
    classes: str = 0,
    view_img: bool = True,
    half: bool = False,
    visualize: bool = False,
    agnostic_nms: bool = False,
    max_det: int = 1000,
    points: List[int] = None,
):
    sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)
    device = torch.device("cuda")
    model = DetectMultiBackend(weights, device)
    imgsz = check_img_size(imgsz=img_size, s=model.stride)
    if model.pt or model.jit:
        model.model.half() if half else model.model.float()
    if source.isnumeric():
        cudnn.benchmark = True
        dataset = LoadStreams(
            source, img_size=imgsz, stride=model.stride, auto=model.pt
        )
        batch_size = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=model.stride, auto=model.pt)
        batch_size = 1
    for path, img, img_1, vid_cap, string in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255
        if len(img.shape) == 3:
            img = img[None]

        save_dir = "/"
        visualize = increment_path(save_dir, mkdir=True) if visualize else False
        prediction = model(img, augment=False, visualize=visualize)
        prediction = non_max_suppression(
            prediction, conf_thres, iou_thres, classes, agnostic_nms, max_det
        )

        for i, det in enumerate(prediction):
            if source.isnumeric():
                p, img0, frame = path[i], img_1[i].copy(), dataset.count
                string += f"{i}: "
            else:
                p, img0, frame = path, img_1.copy(), getattr(dataset.count, "frame", 0)

            string += "%gX%g " % img.shape[2:]
            if len(det):
                # rescale bbox img to img0
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # Count object
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    string += f"{n} {model.names[int(c)]}{'s' * (n > 1)}, "

                dets_to_sort = np.empty((0, 6))
                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack(
                        (dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass]))
                    )
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks = sort_tracker.getTrackers()

                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    offset = (0, 0)

                    for i, box in enumerate(bbox_xyxy):
                        x1, y1, x2, y2 = [int(i) for i in box]
                        x1 += offset[0]
                        x2 += offset[0]
                        y1 += offset[1]
                        y2 += offset[1]
                        cat = int(categories[i]) if categories is not None else 0
                        id = int(identities[i]) if identities is not None else 0
                        centroid = (
                            int((box[0] + box[2]) / 2),
                            (int((box[1] + box[3]) / 2)),
                        )
                        label = str(id)

                        (w, h), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                        )
                        cv2.rectangle(img0, (x1, y1), (x2, y2), (255, 191, 0), 2)
                        cv2.rectangle(
                            img0, (x1, y1 - 20), (x1 + w, y1), (255, 191, 0), -1
                        )
                        cv2.putText(
                            img0,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            [255, 255, 255],
                            1,
                        )
                        cv2.circle(img0, centroid, 3, (255, 191, 0), -1)
                        if points is not None:
                            img0 = draw_polygon(img0, points)

                            if (
                                shapely.contains(Polygon(points), Point(centroid))
                            ) == True:
                                print("Save image")
                                img = save_frame_alert(img0, "test")
                                for track in tracks:
                                    for i, _ in enumerate(track.centroidarr):
                                        if i < len(track.centroidarr) - 1:
                                            cv2.line(
                                                img,
                                                (
                                                    int(track.centroidarr[i][0]),
                                                    int(track.centroidarr[i][1]),
                                                ),
                                                (
                                                    int(track.centroidarr[i + 1][0]),
                                                    int(track.centroidarr[i + 1][1]),
                                                ),
                                                (124, 32, 250),
                                                thickness=3,
                                            )

                        # if points == None:
                        #     for track in tracks:
                        #         for i, _ in enumerate(track.centroidarr):
                        #             if i < len(track.centroidarr) - 1:
                        #                 cv2.line(
                        #                     img0,
                        #                     (
                        #                         int(track.centroidarr[i][0]),
                        #                         int(track.centroidarr[i][1]),
                        #                     ),
                        #                     (
                        #                         int(track.centroidarr[i + 1][0]),
                        #                         int(track.centroidarr[i + 1][1]),
                        #                     ),
                        #                     (124, 32, 250),
                        #                     thickness=3,
                        #                 )
            if view_img:
                cv2.imshow(str(p), img0)
                cv2.waitKey(1)
            else:
                return img0


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
        default=ROOT / "data/images",
        help="file/dir/URL/glob, 0 for webcam",
    )
    parser.add_argument(
        "--img_size", nargs="+", default=(640, 640), help="inference size h,w"
    )
    parser.add_argument(
        "--conf_thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou_thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max_det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument("--view_img", action="store_false", help="show results")
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--agnostic_nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )

    opt = parser.parse_args()
    return opt


def main(opt):
    # test
    points = [
        [970, 434],
        [713, 434],
        [702, 183],
        [958, 170],
        [970, 434],
    ]
    print(vars(opt))
    infer(**vars(opt), points=points)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
