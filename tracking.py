from typing import Tuple

import cv2
import torch

from models.common import DetectMultiBackend
from trackers import *
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from graphs import draw_boxes, bbox_rel, draw_bbox


class YOLOV5_TRACKING:
    def __init__(
        self,
        weights: str,
        img_size: Tuple[int],
        classes: str,
        tracking: bool = False,
        type_mot: str = "deep_sort",
        config_file: str = "./trackers/deep_sort_pytorch/configs/deep_sort.yaml",
    ) -> None:
        self.weights = weights
        self.img_size = img_size
        self.classes = classes
        self.tracking = tracking
        self.hide_labels = False
        self.hide_conf = False
        self.type_mot = type_mot
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(tracking=tracking, type_mot=self.type_mot, config_file=config_file)

    def _load_model(
        self,
        tracking: bool = False,
        type_mot: str = "deep_sort",
        config_file: str = None,
    ) -> None:
        self.model = DetectMultiBackend(weights=self.weights, device=self.device)
        if tracking:
            if type_mot == "sort":
                self.sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)
            elif type_mot == "deep_sort":
                cfg = get_config()
                cfg.merge_from_file(config_file=config_file)
                self.deepsort = DeepSort(
                    cfg["DEEPSORT"]["REID_CKPT"],
                    min_confidence=cfg["DEEPSORT"]["MIN_CONFIDENCE"],
                    max_dist=cfg["DEEPSORT"]["MAX_DIST"],
                    nms_max_overlap=cfg["DEEPSORT"]["NMS_MAX_OVERLAP"],
                    max_iou_distance=cfg["DEEPSORT"]["MAX_IOU_DISTANCE"],
                    n_init=cfg["DEEPSORT"]["N_INIT"],
                    nn_budget=cfg["DEEPSORT"]["NN_BUDGET"],
                    use_cuda=True,
                )

    def _dataloader(self, source: os.path) -> Tuple[str, np.ndarray, np.ndarray, str, str]:
        img_size = check_img_size(self.img_size, s=self.model.stride)
        dataset = LoadImages(path=source, img_size=img_size)
        return dataset, img_size

    def infer(self, source: os.path, view_video: bool = True, save_img: bool = False) -> np.ndarray:
        minibatch, img_size = self._dataloader(source=source)
        self.model.warmup(imgsz=(1 if self.model.pt else 1, 3, *img_size))

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
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_3.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()

                    if self.tracking:
                        if self.type_mot == "sort":
                            dets_to_sort = np.empty((0, 6))
                            for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                                dets_to_sort = np.vstack(
                                    (
                                        dets_to_sort,
                                        np.array([x1, y1, x2, y2, conf, detclass]),
                                    )
                                )
                            tracked_dets = self.sort_tracker.update(dets_to_sort)
                            tracks = self.sort_tracker.getTrackers()
                            if len(tracked_dets) > 0:
                                bbox_xyxy = tracked_dets[:, :4]
                                identities = tracked_dets[:, 8]
                                categories = tracked_dets[:, 4]
                                draw_bbox(img_3, bbox_xyxy, categories, identities)
                        elif self.type_mot == "deep_sort":
                            bbox_xywh = []
                            confs = []
                            for *xyxy, conf, cls in det:
                                x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                                obj = [x_c, y_c, bbox_w, bbox_h]
                                bbox_xywh.append(obj)
                                confs.append([conf.item()])

                            xywhs = torch.Tensor(bbox_xywh)
                            confss = torch.Tensor(confs)
                            outputs = self.deepsort.update(xywhs, confss, img_3)
                            if len(outputs) > 0:
                                bbox_xyxy = outputs[:, :4]
                                identities = outputs[:, -1]
                                # draw_boxes(img_3, bbox_xyxy, identities)
                                draw_bbox(img_3, bbox_xyxy=bbox_xyxy, identities=identities)

                            annotator = Annotator(img_3, line_width=3, example=str(self.model.names))
                            # yolo write detection result
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if view_video:
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

                        if view_video:
                            cv2.imshow("test", img_3)
                            cv2.waitKey(1)
                        if save_img:
                            cv2.imwrite("result.png", img_3)
                    else:
                        annotator = Annotator(img_3, line_width=3, example=str(self.model.names))
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = (
                                None
                                if self.hide_labels
                                else (self.model.names[c] if self.hide_conf else f"{self.model.names[c]} {conf:.2f}")
                            )
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        img_3 = annotator.result()

                    if view_video:
                        cv2.imshow("test", img_3)
                        cv2.waitKey(1)
                    if save_img:
                        cv2.imwrite("result.png", img_3)
                else:
                    self.deepsort.increment_ages()
            return img_3


if __name__ == "__main__":
    inference = YOLOV5_TRACKING(
        weights="./saved_models/yolov5s.pt",
        img_size=(640, 640),
        classes=0,
        tracking=True,
        type_mot="deep_sort",
    )
    inference.infer(source="./test.mp4", view_video=True, save_img=False)
