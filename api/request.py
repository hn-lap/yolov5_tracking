import json
import os
import sys

import requests

url = sys.argv[1]
image_path = sys.argv[2]

with open(image_path, "rb") as img:
    name_img = os.path.basename(image_path)
    files = {"image": (name_img, img)}
    res = requests.post(url, files=files)
