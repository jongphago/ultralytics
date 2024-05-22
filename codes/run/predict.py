import os
import sys

sys.path.append(os.getcwd())
from codes.callback import callback
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")
model.add_callback("on_predict_postprocess_end", callback.print_boxes)

# Single stream with batch-size 1 inference
source = "rtsp://210.99.70.120:1935/live/cctv007.stream"  # RTSP, RTMP, TCP or IP streaming address
source = "datasets/rtsp/ch2.streams"

# Run inference on the source
results = model.predict(source, stream=True)  # generator of Results objects
