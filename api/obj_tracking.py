import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from tracker.sort import *
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, increment_path, non_max_suppression, scale_coords


class YOLOV5_SORT:
    def __init__(self, weights, img_size):
        self.weights = weights
        self.img_size = img_size
        pass

    def _load_model(self):
        pass

    def _dataloader(self):
        pass

    def infer(self, source):
        pass
