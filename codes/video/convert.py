import os
import sys

sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import namedtuple
import cv2
import pandas as pd
from codes.config import config


def get_convert_path(
    cfg: dict, row: namedtuple, src_suffix=".avi", dst_suffix=".mp4"
) -> tuple[Path, Path]:
    """
    examples:
        avi_video, mp4_video = config.get_convert_path(cfg, row)
    """
    _avi_video = cfg.path / cfg.raw / row.video_dir
    avi_video = _avi_video.with_suffix(src_suffix)
    _mp4_video = cfg.path / cfg.videos / row.video_dir
    mp4_video = _mp4_video.with_suffix(dst_suffix)
    assert avi_video.exists(), f"File {avi_video} not exist."
    return avi_video, mp4_video


def avi2mp4(video_path, out_video_path=None, fps=23) -> Path:
    # Capture
    cap = cv2.VideoCapture(video_path.as_posix())

    # Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # assert fps == 23
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Writer
    if out_video_path is None:
        out_video_path = video_path.with_suffix(".mp4")
    if not out_video_path.parent.exists():
        os.makedirs(out_video_path.parent)
    writer = cv2.VideoWriter(out_video_path.as_posix(), fourcc, fps, (width, height))

    # Convert
    for _ in tqdm(range(length)):
        _, frame = cap.read()
        writer.write(frame)
    writer.release()
    cap.release()

    return out_video_path


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description="convert video from target suffix")
    parser.add_argument(
        "config",
        type=str,
        default="aihub-sample",
        help="config file name (e.g., 'aihub-val')",
    )
    args = parser.parse_args()

    # path
    cfg = config.get_config(args.config)

    # loop
    iters = config.get_iters(cfg)

    for row in (pbar := tqdm(iters)):
        pbar.set_description(f"Processing: {row.video_dir}")
        avi_video, mp4_video = get_convert_path(cfg, row)
        print(mp4_video)
        if not mp4_video.exists():
            avi2mp4(avi_video, mp4_video)
