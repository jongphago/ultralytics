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
