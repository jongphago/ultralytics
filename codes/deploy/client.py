import cv2
import requests
import numpy as np


# RTSP 스트림에서 프레임을 읽어옴
cap = cv2.VideoCapture("rtsp://210.99.70.120:1935/live/cctv007.stream")
ret, frame = cap.read()
if not ret:
    print("Failed to read frame")
    exit(1)


# Encode image to send it as bytes
def save_image_into_bytes(frame: np.ndarray) -> bytes:
    _, image_encoded = cv2.imencode(".jpg", frame)
    return image_encoded.tobytes()


image_bytes = save_image_into_bytes(frame)


# Send image to the server
def send(api_url: str, image_bytes: bytes):
    response = requests.post(url=api_url, files={"image": image_bytes})
    return response


api_url = "http://34.64.235.71:8000/predict"
response = send(api_url, image_bytes)

# parse response
xyxys = response.json()["xyxy"]
for xyxy in xyxys:
    print([int(p) for p in xyxy])


# Draw boundingbox on the image
def draw_bbox(frame: np.ndarray, xyxys: list):
    for xyxy in xyxys:
        x1, y1, x2, y2 = [int(p) for p in xyxy]
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame


frame = draw_bbox(frame, xyxys)


# Save image
cv2.imwrite("output.jpg", frame)
