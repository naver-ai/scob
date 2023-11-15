"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def load_img_pillow(img_bytes):
    img_buffer = BytesIO(img_bytes)
    img = Image.open(img_buffer)
    img = img.convert("RGB")
    return img


def load_img_cv2(img_bytes):
    img = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
