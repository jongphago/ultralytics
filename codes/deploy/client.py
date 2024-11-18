import cv2
import requests
import numpy as np


# Encode image to send it as bytes
def save_image_into_bytes(frame: np.ndarray) -> bytes:
    _, image_encoded = cv2.imencode(".jpg", frame)
    return image_encoded.tobytes()


# Send image to the server
def send_detection(api_url: str, image_bytes: bytes):
    response = requests.post(
        url=api_url,
        files={"image": image_bytes},
    )
    return response


def send_track(api_url: str, image_bytes: bytes, camera_id: str):
    response = requests.post(
        url=api_url,
        files={"image": image_bytes},
        data={"camera_id": camera_id},
    )
    return response


# Draw boundingbox on the image
def draw_bbox(frame: np.ndarray, xyxys: list):
    for xyxy in xyxys:
        x1, y1, x2, y2 = [int(p) for p in xyxy]
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame


if __name__ == "__main__":
    # RTSP 스트림에서 프레임을 읽어옴
    camera_id = "cctv007"
    cap = cv2.VideoCapture(f"rtsp://210.99.70.120:1935/live/{camera_id}.stream")
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        exit(1)
    image_bytes = save_image_into_bytes(frame)

    ip_address = "34.47.64.171"
    api_url = f"http://{ip_address}:8000/predict"
    _response = send_detection(api_url, image_bytes)

    # parse response
    response = _response.json()
    xyxys = response["xyxy"]
    clss = response["cls"]
    print("Detection results")
    for xyxy, cls in zip(xyxys, clss):
        print([cls] + [int(p) for p in xyxy])
    print()

    api_url = f"http://{ip_address}:8000/track"
    _response = send_track(api_url, image_bytes, camera_id)
    response = _response.json()
    xyxys = response["xyxy"]
    clss = response["cls"]
    ids = response["id"]
    print("Tracking results")
    for xyxy, cls, id in zip(xyxys, clss, ids):
        print([cls] + [int(p) for p in xyxy] + [id])

    # Save image
    frame = draw_bbox(frame, xyxys)
    cv2.imwrite("output.jpg", frame)
