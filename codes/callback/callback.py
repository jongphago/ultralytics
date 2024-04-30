from pathlib import Path

import wandb


def on_val_end(Validator):
    print(Validator)
    metrics = dict(zip(Validator.metrics.keys, Validator.metrics.mean_results()))
    wandb.log(metrics)
    return None


def plot_pr_curve(stats):
    px, py, x_title, y_title = stats.curves_results[0]
    data = [[x, y] for (x, y) in zip(px, py[0])]
    table = wandb.Table(data=data, columns=[x_title, y_title])
    wandb.log({"Custom Precision Recall Curve": wandb.plot.line(table, x=x_title, y=y_title)})


def plot_confidence_curve(stats, index):
    px, py, x_title, y_title = stats.curves_results[index]
    data = [[x, y] for (x, y) in zip(px, py[0])]
    table = wandb.Table(data=data, columns=[x_title, y_title])
    wandb.log({f"Custom {y_title}-Confidence Curve": wandb.plot.line(table, x=x_title, y=y_title)})


def print_boxes(Validator):
    """
    _summary_

    Args:
        Validator (_type_): _description_

    Examples:
        from codes.callback import callback

        model.add_callback("on_predict_postprocess_end", callback.print_boxes)
    """

    for camera_index, result in enumerate(Validator.results):
        boxes = result.boxes
        names = result.names
        camera_name = Path(result.path).stem
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()  # Extract coordinates
            confidence = boxes.conf[i].item()  # Extract confidence
            class_index = boxes.cls[i].item()  # Extract class index
            class_name = names[class_index]
            print(
                f"{camera_index:2d}, {camera_name}, {int(x1):4d}, {int(y1):4d}, {int(x2):4d}, {int(y2):4d}, {confidence:.02f}, {class_name}"
            )
