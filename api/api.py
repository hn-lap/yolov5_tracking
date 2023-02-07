import os

import cv2
from flask import Flask, request
from obj_tracking import YOLOV5_SORT

app = Flask(__name__)
app.config["SAVE_IMAGE_UPLOAD"] = "static"
app.config["SAVE_IMAGE_RESULT"] = "results"


model = YOLOV5_SORT(weights="../saved_models/yolov5s.pt", img_size=(640, 640), classes=0, tracking=False)


@app.route("/isAlive", methods=["GET"])
def is_alive():
    return {"responses": "Alive"}


@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"]
    path_to_save = os.path.join(app.config["SAVE_IMAGE_UPLOAD"], image.filename)
    image.save(path_to_save)
    img_results = model.infer(path_to_save)
    cv2.imwrite(os.path.join(app.config["SAVE_IMAGE_RESULT"], image.filename), img_results)
    return str(img_results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
