import os
import sys

sys.path.append(os.getcwd())
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from codes.callback import callback
from ultralytics import YOLO

# Step 1: Initialize a Weights & Biases run
wandb.init(project="ultralytics", job_type="validation")

# Step 2: Define the YOLOv8 Model and Dataset
model_name = "yolov8n"
dataset_name = "aihub48.yaml"
model = YOLO(f"{model_name}.pt")

# Step 3: Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)
model.add_callback("on_val_end", callback.on_val_end)

# Step 4: Train and Fine-Tune the Model
# model.train(project="ultralytics", data=dataset_name, epochs=5, imgsz=640)

# Step 5: Validate the Model
stats = model.val(
    data=dataset_name,
)

# Step 5.1: Logging Validation Statistics Results
callback.plot_pr_curve(stats)
for i in range(1, 4):  # 3 confidence curves: precision, recall, F1
    callback.plot_confidence_curve(stats, i)

# Step 6: Perform Inference and Log Results
# model(["path/to/image1", "path/to/image2"])

# Step 7: Finalize the W&B Run
wandb.finish()
