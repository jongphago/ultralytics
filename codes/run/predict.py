from ultralytics import YOLO
from codes.callback import callback

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")
model.add_callback("on_predict_postprocess_end", callback.print_boxes)

# Single stream with batch-size 1 inference
source = "rtsp://210.99.70.120:1935/live/cctv007.stream"  # RTSP, RTMP, TCP or IP streaming address

# Run inference on the source
results = model.predict(source, stream=True)  # generator of Results objects