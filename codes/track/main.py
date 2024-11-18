import os
import sys

sys.path.append(os.getcwd())  # noqa: E402

import threading
import cv2
from ultralytics import YOLO

# Define model names and video sources
MODEL_NAMES = [
    "yolov8n.pt",
    # "yolo8n-seg.pt",
]
SOURCES = [
    "rtsp://210.99.70.120:1935/live/cctv007.stream",
    # "0",
]  # local video, 0 for webcam


def run_tracker_in_thread(model_name, filename):
    """
    Run YOLO tracker in its own thread for concurrent processing.

    Args:
        model_name (str): The YOLO11 model object.
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
    """
    model = YOLO(model_name)
    results = model.track(filename, save=True)
    for result in results:
        print(result)


# Create and start tracker threads using a for loop
tracker_threads = []
for video_file, model_name in zip(SOURCES, MODEL_NAMES):
    thread = threading.Thread(
        target=run_tracker_in_thread, args=(model_name, video_file), daemon=True
    )
    tracker_threads.append(thread)
    thread.start()

# Wait for all tracker threads to finish
for thread in tracker_threads:
    thread.join()

# Clean up and close windows
cv2.destroyAllWindows()
