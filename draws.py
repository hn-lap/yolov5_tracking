import cv2
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def bbox_rel(*xyxy):
    """ " Calculates the relative bounding box from absolute pixel values."""
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs([xyxy[0].item() - xyxy[2].item()])
    bbox_h = abs([xyxy[1].item() - xyxy[3].item()])
    return (bbox_left + bbox_w / 2), (bbox_top + bbox_h / 2), bbox_w, bbox_h


def draw_boxes(
    img,
    bbox,
    identities=None,
    categories=None,
    names=None,
    color_box=None,
    offset=(0, 0),
    points=None,
):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        centroid = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))
        label = str(id)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 191, 0), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 191, 0), -1)
        cv2.putText(
            img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            [255, 255, 255],
            1,
        )
        cv2.circle(img, centroid, 3, (255, 191, 0), -1)
        return
    #     if isInside(points, centroid):
    #         img = alert(img)
    # return isInside(points, centroid)
