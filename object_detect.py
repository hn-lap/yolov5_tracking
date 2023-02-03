import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Tuple

import torch

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size,
                           check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                           scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@smart_inference_mode()
class Inference:
    def __init__(
        self,
        weights: str,
        source: str,
        data: str,
        conf_thes: float,
        iou_thres: float,
        max_det: int,
        device: str,
        classes: int,
        dnn: bool,
        half: bool,
        agnostic_nms: bool,
    ):
        self.weights = weights
        self.source = source
        self.data = data
        self.img_size = (640, 640)
        self.conf_thes = conf_thes  # confidence threshold
        self.iou_thres = iou_thres  # NMS IOU threshold
        self.device = device  # cuda device
        self.max_det = max_det  # maximum detections per image
        self.classes = classes  # filter by class
        self.dnn = dnn  # use opencv dnn for onnx inference
        self.half = half  # use fp16 half-precision inference
        self.vid_stride = 1  # video frame-rate stride
        self.agnostic_nms = agnostic_nms  # class-agostic NMS
        self.model = self._load_model()
        self.device = torch.device("cuda")

    def _load_model(self) -> DetectMultiBackend:
        model = DetectMultiBackend(
            self.weights, device=torch.device("cuda"), dnn=self.dnn, data=self.data, fp16=self.half
        )
        return model

    def infer(
        self,
        project: Path = ROOT / "runs/detect",
        name: str = "exp",
        augment: str = False,
        save_txt: bool = False,
        save_crop: bool = False,
        line_thickness: int = 3,
        exist_ok: bool = True,
        view_img: bool = True,
        save_img: bool = False,
        hide_labels=False,
        hide_conf=False,
        visualize=False,
    ):
        source = self.source
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(("rtsp://", "http://", "https://"))
        web_cam = (
            source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
        )
        if is_url and is_file:
            source = check_file(source)

        save_dir = increment_path(
            Path(project) / name, exist_ok=exist_ok
        )  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(
            parents=True, exist_ok=True
        )  # make dir

        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        img_size = check_img_size(self.img_size, s=stride)

        if web_cam:
            view = check_imshow()
            dataset = LoadStreams(
                source,
                img_size=img_size,
                stride=stride,
                auto=pt,
                vid_stride=self.vid_stride,
            )
            batch_size = len(dataset)
        else:
            dataset = LoadImages(
                path=source,
                img_size=img_size,
                stride=stride,
                auto=pt,
                vid_stride=self.vid_stride,
            )
            batch_size = 1

        self.model.warmup(imgsz=(1 if pt else batch_size, 3, *img_size))
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.model.fp16 else im.float()
                im /= 255
                if len(im.shape) == 3:
                    im = im[None]

            with dt[1]:
                visualize = (
                    increment_path(save_dir, exist_ok=exist_ok, mkdir=True)
                    if visualize
                    else False
                )
                pred = self.model(im, augment=augment, visualize=visualize)

            with dt[2]:
                pred = non_max_suppression(
                    pred,
                    conf_thres=self.conf_thes,
                    iou_thres=self.iou_thres,
                    classes=self.classes,
                    agnostic=self.agnostic_nms,
                    max_det=self.max_det,
                )

            # Process prediction
            for i, det in enumerate(pred):
                seen += 1
                if web_cam:
                    p, img0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f"i: "
                else:
                    p, img0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
                s += "%gx%g " % im.shape[2:]
                annotator = Annotator(
                    img0, line_width=line_thickness, example=str(names)
                )
                if len(det):
                    # rescale boxes from img_size to img0 size
                    det[:, :4] = scale_coords(
                        im.shape[2:], det[:, :4], img0.shape
                    ).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detection per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        print(s)

                for *xyxy, conf, cls in reversed(det):
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = (
                            None
                            if hide_labels
                            else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        )
                        annotator.box_label(xyxy, label, color=colors(c, True))

                im0 = annotator.result()
                if view_img:
                    if platform.system() == "Linux" and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(
                            str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
                        )  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolov5s.pt",
        help="model path(s)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT / "data/images",
        help="file/dir/URL/glob, 0 for webcam",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=ROOT / "data/coco128.yaml",
        help="(optional) dataset.yaml path",
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
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
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
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument(
        "--project", default=ROOT / "runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    parser.add_argument(
        "--vid-stride", type=int, default=1, help="video frame-rate stride"
    )
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    weights = opt.weights
    source = opt.source
    data = opt.data
    conf_thes = opt.conf_thres
    iou_thres = opt.iou_thres
    max_det = opt.max_det
    device = opt.device
    classes = opt.classes
    dnn = opt.dnn
    half = opt.half
    agnostic_nms = opt.agnostic_nms
    inference = Inference(
        weights,
        source,
        data,
        conf_thes,
        iou_thres,
        max_det,
        device,
        classes,
        dnn,
        half,
        agnostic_nms,
    )
    inference.infer(name=opt.name,visualize=opt.visualize,view_img=opt.view_img,line_thickness=opt.line_thickness)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
