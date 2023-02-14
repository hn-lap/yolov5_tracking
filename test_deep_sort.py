from typing import Tuple

import cv2
import torch

from graphs import bbox_rel, draw_bbox, draw_boxes
from models.common import DetectMultiBackend
from trackers import *
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


@torch.no_grad()
def run(
    weights: os.path = "yolov5s.pt",  # model.pt path(s)
    source: os.path = "data/images",  # file/dir/URL/glob, 0 for webcam
    imgsz: Tuple[int] = (640, 640),  # inference size (height, width)
    conf_thres: float = 0.25,  # confidence threshold
    iou_thres: float = 0.45,  # NMS IOU threshold
    max_det: int = 1000,  # maximum detections per image
    device: str = "",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes: int = None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment: bool = False,  # augmented inference
    line_thickness: int = 3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    config_deepsort="./trackers/deep_sort_pytorch/configs/deep_sort.yaml",  # Deep Sort configuration
):
    source = str(source)
    ## initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    deepsort = DeepSort(
        cfg["DEEPSORT"]["REID_CKPT"],
        min_confidence=cfg["DEEPSORT"]["MIN_CONFIDENCE"],
        max_dist=cfg["DEEPSORT"]["MAX_DIST"],
        nms_max_overlap=cfg["DEEPSORT"]["NMS_MAX_OVERLAP"],
        max_iou_distance=cfg["DEEPSORT"]["MAX_IOU_DISTANCE"],
        n_init=cfg["DEEPSORT"]["N_INIT"],
        nn_budget=cfg["DEEPSORT"]["NN_BUDGET"],
        use_cuda=True,
    )

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        pred = model(im, augment=augment)
        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
            # check detected boxes, process them for deep sort
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                ##variables for boundig boxes
                bbox_xywh = []
                confs = []
                ## Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(
                        im0, bbox_xyxy, identities
                    )  # call function to draw seperate object identity

                annotator = Annotator(
                    im0, line_width=line_thickness, example=str(names)
                )
                # yolo write detection result
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = (
                        None
                        if hide_labels
                        else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    )
                    annotator.box_label(xyxy, label, color=colors(c, True))

            else:
                deepsort.increment_ages()

            cv2.imshow(str(p), im0)
            cv2.waitKey(1)


if __name__ == "__main__":
    run = run(source="./test.mp4", classes=0)
