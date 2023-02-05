import os
import threading
import time
from queue import Empty, Queue

import cv2
import numpy as np
from flask import Flask, request
from obj_tracking import YOLOV5_SORT

app = Flask(__name__)
requestQueue = Queue()


model = YOLOV5_SORT(weights="../saved_models/yolov5s.pt", img_size=(640, 640))


def byte_to_image(file):
    try:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        return opencv_image
    except Exception:
        return False


def request_handler(data):
    results = model.infer(source=data["image"])

    return results


threading.Thread(target=request_handler).start()


@app.route("/isAlive", methods=["GET"])
def is_alive():
    return {"responses": "Alive"}


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = byte_to_image(file=file)
    data = {"image": img}
    requestQueue.put(data)

    response = {"filename": ""}

    return str(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
