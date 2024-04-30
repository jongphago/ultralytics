# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n detection model
model.train(
    data="aihub-subset.yaml",
    epochs=3,
)  # train the model
