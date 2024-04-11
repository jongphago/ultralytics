import os
import sys

sys.path.append("/home/jongphago/project/ultralytics")

import argparse
import subprocess
from pathlib import Path

import yaml
import pandas as pd
from box import Box

from codes.config.pattern import find
from codes.config.pattern import (
    camera_id_pattern,
    scenario_id_pattern,
    frame_id_pattern_f,
)


class Video:
    def __init__(self, video_dir, task):
        self.video_dir = video_dir
        self.task = task


class AIHub(Video):
    FPS = 23

    def __init__(self, video_dir, task, cfg):
        super().__init__(video_dir, task)
        self.camera_id: str = find(camera_id_pattern, self.video_dir)  # 01
        self.scenario_id: str = find(scenario_id_pattern, self.video_dir)  # 11
        self.fps: int = cfg.fps  # 30
        self.dirs(cfg)

    def dirs(self, cfg):
        self.path = Path(cfg.path)
        self.raw = self.path / cfg.raw  # ex. aihub/sample/원천데이터/RGB
        self.video_dir = (
            self.path / self.raw / self.video_dir
        )  # ex. 시나리오01/카메라07.avi
        self.out = self.path / cfg.frames  # ex. aihub/sample/frames
        sub_dir = f"scenario{self.scenario_id}/camera{self.camera_id}"
        self.frames_dir = self.out / sub_dir
        self.img = self.path / cfg.images  # ex. aihub/sample/images
        self.images_dir = self.img / self.task

    def extract(self):
        os.makedirs(self.frames_dir, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                *("-i", self.video_dir),
                *("-start_number", "0"),
                *("-r", str(self.fps)),
                f"{self.frames_dir}/s{self.scenario_id}_c{self.camera_id}_f%04d.png",
            ]
        )

    def link(self):
        # link: Make symbolic link
        os.makedirs(self.images_dir, exist_ok=True)
        srcs = []
        for directory, _, file_names in os.walk(self.frames_dir):
            root = Path(directory)
            for file_name in file_names:
                srcs.append(root / file_name)
        for src in sorted(srcs):
            if self.fps == 1:
                ffmpeg_index = find(frame_id_pattern_f, str(src))
                new_frame_id = min(int(ffmpeg_index) * self.FPS, 7361)
                src = f"s{self.scenario_id}_c{self.camera_id}_f{new_frame_id:04d}.png"
            dst = self.images_dir / (src.stem + ".png")
            if not os.path.exists(dst):
                os.symlink(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and link AIHub data")
    parser.add_argument("config_name", type=str, help="Name of the configuration file")

    args = parser.parse_args()  # 명령행 인자 파싱하기

    root = Path("/home/jongphago/project/ultralytics")
    config_name = args.config_name  # aihub-sample, aihub-val
    cfg_path = root / f"ultralytics/cfg/datasets/{config_name}.yaml"
    with open(cfg_path, "r") as file:
        cfg = Box(yaml.safe_load(file))

    # root directory
    dataset_dir = Path(cfg.path)
    data_df = pd.read_csv(dataset_dir / cfg.csv)
    for rows in data_df.itertuples():
        aihub = AIHub(rows.video_dir, rows.task, cfg)
        aihub.extract()
        aihub.link()
