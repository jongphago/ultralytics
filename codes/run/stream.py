from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Single stream with batch-size 1 inference
source = "rtsp://example.com/media.mp4"  # RTSP, RTMP, TCP or IP streaming address

# Multiple streams with batched inference (i.e. batch-size 8 for 8 streams)
source = (
    "rtsp://210.99.70.120:1935/live/cctv007.stream",  # *.streams text file with one streaming address per row
)

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects

