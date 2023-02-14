import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import torch
import torch.backends.cudnn as cudnn
from draws import draw_boxes
from tracker.sort import *

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (check_img_size, increment_path, non_max_suppression,
                           scale_coords)
from utils.plots import Annotator, colors


class YOLOV5_SORT:
    def __init__(
        self, weights: str, img_size: Tuple[int], classes, tracking: bool = False
    ):
        self.weights = weights
        self.img_size = img_size
        self.classes = classes
        self.tracking = tracking
        self.hide_labels = False
        self.hide_conf = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DetectMultiBackend(weights=self.weights, device=self.device)
        if self.tracking:
            self.sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

    def _dataloader(self, source):
        img_size = check_img_size(self.img_size, s=self.model.stride)
        dataset = LoadImages(path=source, img_size=img_size)
        return dataset

    def infer(self, source):
        minibatch = self._dataloader(source=source)
        for path, img_1, img_2, vid_cap, string in minibatch:
            img = torch.from_numpy(img_1).to(self.device).float()
            img /= 255
            if len(img.shape) == 3:
                img = img[None]
            pred = self.model(img, augment=False)
            pred = non_max_suppression(pred, classes=self.classes, max_det=1000)
            for i, det in enumerate(pred):
                p, img_3, frame = (
                    path,
                    img_2.copy(),
                    getattr(minibatch.count, "frame", 0),
                )
                if len(det):
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], img_3.shape
                    ).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                    if self.tracking:
                        dets_to_sort = np.empty((0, 6))
                        for x1, y1, x2, y2, conf, detclass in (
                            det.cpu().detach().numpy()
                        ):
                            dets_to_sort = np.vstack(
                                (dets_to_sort, np.array([x1, y1, x2, y2, detclass]))
                            )
                        tracked_dets = self.sort_tracker.update(dets_to_sort)
                        tracks = self.sort_tracker.getTrackers()
                        if len(tracked_dets) > 0:
                            bbox_xyxy = tracked_dets[:, :4]
                            identities = tracked_dets[:, 8]
                            categories = tracked_dets[:, 4]
                            draw_boxes(img_3, bbox_xyxy, identities, categories)
                        for track in tracks:
                            for i, _ in enumerate(track.centroidarr):
                                if i < len(track.centroidarr) - 1:
                                    cv2.line(
                                        img_3,
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
                    else:
                        annotator = Annotator(
                            img_3, line_width=3, example=str(self.model.names)
                        )
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = (
                                None
                                if self.hide_labels
                                else (
                                    self.model.names[c]
                                    if self.hide_conf
                                    else f"{self.model.names[c]} {conf:.2f}"
                                )
                            )
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        img = annotator.result()
                return img_3


if __name__ == "__main__":
    inference = YOLOV5_SORT(
        weights="../saved_models/yolov5s.pt",
        img_size=(640, 640),
        classes=0,
        tracking=False,
    )
    import cv2

    img = cv2.imread("G:/HN_LAP/yolov5_tracking/test_folder/1.jpg")
    inference.infer(source=img)
