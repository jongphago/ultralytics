import os
import re
import json
import logging
from tqdm import tqdm
from pathlib import Path
import yaml
import pandas as pd
from box import Box


class Label:
    def __init__(self, path):
        self.path = path


class AIHub(Label):
    def __init__(
        self,
        scenario_id,
        camera_id,
        frame_id,
        root="/home/jongphago/project/ultralytics/datasets/aihub/sample/라벨링데이터/",
    ):
        path = (
            root
            + f"시나리오{scenario_id:02d}/카메라{camera_id:02d}/"
            + f"NIA_MTMDC_s{scenario_id:02d}_c{camera_id:02d}_"
            + "pm_sunny_summer_"
            + f"{frame_id:04d}.json"
        )
        super().__init__(path)

    pass


# config file (.yaml)
def load_config_from_yaml(fp: str | Path) -> Box:
    with open(fp) as f:
        cfg = Box(yaml.safe_load(f))
    return cfg


def get_value_dict(cfg: Box) -> dict[str:int]:
    value_dict = {value: key for key, value in cfg.names.items()}
    return value_dict


def get_videos(cfg: Box) -> map:
    """AI is creating summary for get_videos

    Args:
        cfg (Box): directory information

    Returns:
        map: iterable pandas dataframe

    Examples:
    cfg = load_config_from_yaml(cfg_yaml_path)
    rows = get_videos(cfg)
    for row in rows:
        row.task, row.video_dir
    """
    videos_df: pd.DataFrame = pd.read_csv(Path(cfg.path) / cfg.csv)
    rows: map = videos_df.itertuples()
    return rows


def get_json_labels(cfg, sub_dir) -> list[Path]:
    raw_label_path = Path(cfg.path) / cfg.lbl / sub_dir
    label_paths = []
    for dirs, _, file_names in os.walk(raw_label_path):
        parent = Path(dirs)
        for file_name in sorted(file_names):
            label_paths.append(parent / file_name)
    return label_paths


def find(pattern, string):
    return re.findall(pattern, string)[0]


def raw2cvt(raw, cfg):
    cvt_name = f"{raw.stem}.txt"
    cvt = Path(cfg.path) / cfg.cvt / row.video_dir / cvt_name
    return cvt


def cvt2dst(cvt, cfg, row):
    scenario_id_pattern = r"(?<=시나리오)\d{2}"
    camera_id_pattern = r"(?<=카메라)\d{2}"
    frame_id_pattern = r"_(\d{4})."

    scenario_id = find(scenario_id_pattern, str(cvt))
    camera_id = find(camera_id_pattern, str(cvt))
    frame_id = find(frame_id_pattern, str(cvt))
    dst_name = f"s{scenario_id}_c{camera_id}_f{frame_id}.txt"
    dst = Path(cfg.path) / cfg.labels / row.task / dst_name
    return dst


def get_frames(lines: dict) -> pd.DataFrame:
    # Convert to DataFrame [label, info]
    # label = pd.DataFrame(label_dict, index=[0])
    info = pd.DataFrame(lines["info"])

    # Join [label, info] to `frames`
    # frames = label.join(info)

    return info


def get_objects(lines: dict) -> pd.DataFrame | None:
    def _flatten(x):
        return pd.Series(x[0])

    # Define objects dataframe
    objects = pd.DataFrame(lines["objects"])

    # Exception
    if objects.loc[0, "label"] == "void":
        return None

    # Flatten [position, attributes] columns
    position: pd.DataFrame = objects.position.apply(_flatten)
    attributes: pd.DataFrame = objects.attributes.apply(_flatten)

    # Join objects + [position, attributes]
    objects = objects.join(position)
    objects = objects.join(attributes)

    # Drop [position, attributes] from objects dataframe
    objects = objects.drop(["position", "attributes"], axis=1)

    return objects


def get_reorder(
    objects: pd.DataFrame,
    frames: pd.DataFrame,
    value_dict: dict[str:int],
) -> pd.DataFrame | None:
    def xywh2ccwh(objects, shape=(1920, 1080), normalized=True):
        img_width, img_height = shape
        # top-left 수정
        objects.width.where(objects.x > 0, objects.width + objects.x, inplace=True)
        objects.height.where(objects.y > 0, objects.height + objects.y, inplace=True)
        objects.x.where(objects.x > 0, 0, inplace=True)
        objects.y.where(objects.y > 0, 0, inplace=True)
        # bottom-right 수정
        bottom_right_x = objects.x + objects.width
        bottom_right_y = objects.y + objects.height
        bottom_right_x.where(bottom_right_x < img_width, img_width, inplace=True)
        bottom_right_y.where(bottom_right_y < img_height, img_height, inplace=True)
        objects.width = bottom_right_x - objects.x
        objects.height = bottom_right_y - objects.y
        # center xy
        x = objects.x + objects.width / 2
        y = objects.y + objects.height / 2
        # width, height
        width = objects.width
        height = objects.height
        # normalized
        if normalized:
            x = x / img_width
            y = y / img_height
            width = objects.width / img_width
            height = objects.height / img_height
        # concatenate
        x.name, y.name = "x", "y"
        ccwh = pd.concat([x, y, width, height], axis=1)
        return ccwh

    if objects is None:
        return None
    class_col = (
        objects[["label"]]
        .apply(lambda x: value_dict[x[0]], axis=1)
        .to_frame(name="class")
    )
    ccwh = xywh2ccwh(objects)
    joined = class_col.join(ccwh)
    reorder = joined[["class", "x", "y", "width", "height"]]
    return reorder


def convert(reorder, cvt_path) -> None:
    if reorder is not None:
        reorder.to_csv(cvt_path, sep=" ", header=False, index=False)
    else:
        with open(cvt_path, "w") as f:
            f.write("")
    return None


if __name__ == "__main__":
    # Logging
    logging.basicConfig(level=logging.ERROR)
    debug = logging.debug
    info = logging.info
    info(os.getcwd())

    # Config
    cfg_yaml_path = "ultralytics/cfg/datasets/aihub-val.yaml"
    cfg: Box = load_config_from_yaml(cfg_yaml_path)
    debug(cfg)
    value_dict = get_value_dict(cfg)
    debug(value_dict)

    # Videos
    rows = get_videos(cfg)
    debug(rows)
    for row in rows:
        json_labels = get_json_labels(cfg, row.video_dir)
        debug(row)

        os.makedirs(Path(cfg.path) / cfg.cvt / row.video_dir, exist_ok=True)
        os.makedirs(Path(cfg.path) / cfg.labels, exist_ok=True)

        for index, raw in tqdm(enumerate(json_labels)):
            if index % 23 != 0 and index != 7361:
                continue
            cvt = raw2cvt(raw, cfg)
            dst = cvt2dst(cvt, cfg, row)
            if not cvt.exists():  # No overwrite
                with open(raw) as f:
                    lines = json.load(f)
                objects = get_objects(lines)
                frames = get_frames(lines)
                reorder = get_reorder(objects, frames, value_dict)
                convert(reorder, cvt)
            if not os.path.exists(dst):
                if not os.path.exists(dst.parent):
                    os.makedirs(dst.parent)
                os.symlink(cvt, dst)
