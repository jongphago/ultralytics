import os
import sys

sys.path.append(os.getcwd())  # noqa: E402

import json
from io import BytesIO
from typing import Union
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile


app = FastAPI()
model = YOLO("yolov8n.pt")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q.upper()}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    async def load_image_into_numpy_array(data):
        return np.array(Image.open(BytesIO(data)))

    image_bytes = await image.read()
    image_array = await load_image_into_numpy_array(image_bytes)
    results = model.predict(image_array)
    return {
        "xyxy": results[0].boxes.xyxy.cpu().numpy().astype(int).tolist(),
        "cls": results[0].boxes.cls.cpu().numpy().astype(int).tolist(),
    }


@app.post("/image_files/")
def image_files(images: list[UploadFile]):
    def bytes2array(image_bytes: bytes) -> np.ndarray:
        _array = np.frombuffer(image_bytes, np.uint8)
        image_array = cv2.imdecode(
            _array, cv2.IMREAD_COLOR
        )  # 또는 cv2.IMREAD_UNCHANGED
        return image_array

    def gather_frames(images) -> list[np.ndarray, list[str]]:
        frames = []
        names = []
        for image in images:
            names.append(image.filename)
            with image.file as f:
                _frame = f.read()
            frames.append(bytes2array(_frame))
        return frames, names

    frames, _ = gather_frames(images)
    batch_results = model(frames)
    return {"batch": [json.loads(results.tojson()) for results in batch_results]}
