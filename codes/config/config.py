import os
from pathlib import Path
import yaml
from box import Box
import pandas as pd


def get_config(config_name):
    """_summary_

    Args:
        config_name (_type_): _description_

    Returns:
        _type_: _description_

    Examples:
        from codes.config import config

        cfg = config.get_config("aihub-sample")
    """
    cfg_path = f"ultralytics/cfg/datasets/{config_name}.yaml"
    with open(cfg_path, "r") as file:
        cfg = Box(yaml.safe_load(file))
        cfg.path = Path(cfg.path)
    return cfg


def write_videos_df(cfg, target_suffix=".avi") -> Path:
    def get_sub_dirs(videos_path: Path) -> list[str]:
        sub_dirs = []
        for _dirpath, _, filenames in os.walk(videos_path):
            dirpath = Path(_dirpath)
            for _filename in filenames:
                if not _filename.endswith(target_suffix):
                    continue
                sub_dir = (Path(dirpath.parts[-1]) / _filename).with_suffix("")
                sub_dirs.append(sub_dir.as_posix())
        return sorted(sub_dirs)

    videos_path: Path = cfg.path / cfg.raw
    sub_dirs: list = get_sub_dirs(videos_path)
    videos_df = pd.DataFrame(sub_dirs, columns=["video_dir"])
    videos_df["task"] = "val"
    videos_df.to_csv(cfg.path / cfg.csv, index=False)
    return cfg.path / cfg.csv


def get_iters(cfg):
    # videos_df
    videos_df = pd.read_csv(cfg.path / cfg.csv)

    # loop
    iters = videos_df.itertuples()

    return iters
