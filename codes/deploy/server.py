import os
import sys

sys.path.append(os.getcwd())  # noqa: E402
import warnings

# FutureWarning을 무시하도록 설정
warnings.filterwarnings("ignore", category=FutureWarning)

import json
from io import BytesIO
import cv2
import torch
import uvicorn
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Form


app = FastAPI()
model_name = "yolov8n.pt"
model = YOLO(model_name)
models = {}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/is_cuda_available")
async def is_available():
    return {"torch.cuda.is_available": f"{torch.cuda.is_available()}"}


@app.get("/items/{item_id}")
def read_item(item_id, q):
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

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    async def load_image_into_numpy_array(data):
        return np.array(Image.open(BytesIO(data)))

    image_bytes = await image.read()
    image_array = await load_image_into_numpy_array(image_bytes)
    results = model.predict(image_array)
    return {
        "xyxy": results[0].boxes.xyxy.cpu().numpy().astype(int).tolist(),
        "cls": results[0].boxes.cls.cpu().numpy().astype(int).tolist(),
    }


@app.post("/track")
async def track(image: UploadFile = File(...), camera_id: str = Form(...)):
    async def load_image_into_numpy_array(data):
        return np.array(Image.open(BytesIO(data)))
    if camera_id not in models:
        models[camera_id] = YOLO(model_name)
    model: YOLO = models[camera_id]

    image_bytes = await image.read()
    image_array = await load_image_into_numpy_array(image_bytes)
    results = model.track(image_array, persist=True)

    # 검출된 객체가 없을 경우 처리
    if results is None or len(results) == 0:
        return {"xyxy": [], "cls": [], "id": []}

    # 결과에서 NoneType 오류 방지
    ids = results[0].boxes.id
    output = {}
    if ids is None:
        id_values = [-1] * len(results[0])
    else:
        id_values = ids.cpu().numpy().astype(int).tolist()

    output =  {
        "xyxy": results[0].boxes.xyxy.cpu().numpy().astype(int).tolist(),
        "cls": results[0].boxes.cls.cpu().numpy().astype(int).tolist(),
        "id": id_values,
    }

    return output


@app.post("/image_files")
def image_files(images: list[UploadFile], camera_id: str = Form(...)):
    def bytes2array(image_bytes: bytes) -> np.ndarray:
        _array = np.frombuffer(image_bytes, np.uint8)
        image_array = cv2.imdecode(
            _array, cv2.IMREAD_COLOR
        )  # 또는 cv2.IMREAD_UNCHANGED
        return image_array

    def gather_frames(images):
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
