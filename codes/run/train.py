# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
import os
import sys

sys.path.append(os.getcwd())
from pathlib import Path
from ultralytics import YOLO

from codes.config import config
from codes.dataset import link


cfg_file = "aihub-subset-train-kaist.yaml"
cfg = config.get_config(Path(cfg_file).stem)
link.link_subset(cfg)

model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n detection model
model.train(
    data=cfg_file,
    epochs=100,
)  # train the model
