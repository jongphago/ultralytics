import os
import sys

sys.path.append(os.getcwd())
import argparse
import subprocess
from pathlib import Path
import pandas as pd
from codes.video import extract
from codes.config import config
from codes.config import pattern
from codes.config.pattern import scenario_id_pattern, camera_id_pattern


def get_extract_path(cfg, row, target_suffix=".mp4") -> list[Path, Path]:
    """_summary_

    Args:
        cfg (_type_): _description_
        row (_type_): _description_
        target_suffix (str, optional): _description_. Defaults to ".mp4".

    Returns:
        _type_: _description_

    Examples:
        from codes.video import extract
        
        video, frame = extract.get_extract_path(cfg, row)
    """
    _video = cfg.path / cfg.videos / row.video_dir
    video = _video.with_suffix(target_suffix)
    assert video.exists(), f"FileNotFoundError: {video}"
    frame = cfg.path / cfg.frames / row.video_dir
    return video, frame


def extract_frames(in_video_path:str, output_folder:str) -> None:
    """_summary_

    Args:
        in_video_path (str): _description_
        output_folder (str): _description_
    """
    def get_ids(in_video_path):
        scenario_id = pattern.find(scenario_id_pattern, in_video_path)
        camera_id = pattern.find(camera_id_pattern, in_video_path)
        return scenario_id, camera_id

    scenario_id, camera_id = get_ids(in_video_path)
    # Use ffmpeg to extract frames from the video file
    command = [
        "ffmpeg",
        *("-ss", "00:00:00"),
        *("-i", in_video_path),
        # *("-r", "23"),
        *("-vf", "select=not(mod(n\,23))"),
        *("-start_number", "0"),
        # *("-q", "2"),
        *("-f", "image2"),
        *("-vsync", "vfr"),
        # *("-t", "2"),
        os.path.join(output_folder, f"s{scenario_id}_c{camera_id}_f%04d.jpg"),
    ]
    print(command)
    subprocess.run(command)


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description="extract frames from video")
    parser.add_argument(
        "config",
        type=str,
        default="aihub-sample",
        help="config file name (e.g., 'aihub-val')",
    )
    args = parser.parse_args()

    # path
    cfg = config.get_config(args.config)
    frames: Path = cfg.path / cfg.frames

    # loop
    iters = config.get_iters(cfg)
    for row in iters:
        video, frame = extract.get_extract_path(cfg, row)
        print(f"{frame}")
        if frame.exists():
            continue
        os.makedirs(frame)
        print(f"Make directory: {frame}")
        extract.extract_frames(video.as_posix(), frame.as_posix())
