import os
import sys
sys.path.append(os.getcwd())
import subprocess
from tqdm import tqdm
from pathlib import Path
import cv2
import yaml
import pandas as pd
from box import Box
from codes.config.pattern import find
from codes.config.pattern import scenario_id_pattern, camera_id_pattern
from codes.config.pattern import s_id_pattern, c_id_pattern, f_id_pattern


def avi2mp4(video_path) -> Path:
    # Capture
    cap = cv2.VideoCapture(video_path.as_posix())

    # Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps == 23
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Writer
    out_video_path = video_path.with_suffix(".mp4")
    writer = cv2.VideoWriter(out_video_path.as_posix(), fourcc, fps, (width, height))

    # Convert
    for _ in tqdm(range(length)):
        _, frame = cap.read()
        writer.write(frame)
    writer.release()
    cap.release()

    return out_video_path


def extract_frames(video_file, output_folder):
    def get_ids(in_video_path):
        scenario_id = find(scenario_id_pattern, in_video_path)
        camera_id = find(camera_id_pattern, in_video_path)
        return scenario_id, camera_id

    scenario_id, camera_id = get_ids(video_file)
    # Use ffmpeg to extract frames from the video file
    command = [
        "ffmpeg",
        *("-ss", "00:00:00"),
        *("-i", video_file),
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


def scale_filename(_filename, fps=23):
    s_id_pattern = r"(?<=s)\d{2}"
    c_id_pattern = r"(?<=c)\d{2}"
    f_id_pattern = r"f(\d{4})."
    scenario_id = find(s_id_pattern, _filename)
    camera_id = find(c_id_pattern, _filename)
    frame_id = find(f_id_pattern, _filename)
    filename = f"s{scenario_id}_c{camera_id}_f{int(frame_id) * fps:04d}.jpg"
    return filename


def get_paths(
    cfg,
    row,
):
    video_dir = row.video_dir
    video = cfg.path / cfg.raw / (video_dir + ".avi")
    frames = cfg.path / cfg.out / video_dir
    return video, frames


if __name__ == "__main__":
    root = Path("/home/jongphago/project/ultralytics")
    config_name = "aihub-val"
    cfg_path = root / f"ultralytics/cfg/datasets/{config_name}.yaml"
    with open(cfg_path, "r") as file:
        cfg = Box(yaml.safe_load(file))
        cfg.path = Path(cfg.path)

    videos_df = pd.read_csv(cfg.path / cfg.csv)
    if not os.path.exists(cfg.path / cfg.val):
        os.makedirs(cfg.path / cfg.val)
    is_link = True

    for row in videos_df.itertuples():
        _video, frames = get_paths(cfg, row)
        assert _video.exists(), "FileExistsError"

        # Convert
        video = _video.with_suffix(".mp4")
        if not video.exists():
            print(f"Convert: {_video}")
            avi2mp4(_video)

        # Extract
        if not frames.exists():
            os.makedirs(frames)
            print(f"Extract: {video}")
            extract_frames(video.as_posix(), frames.as_posix())
        if not is_link:
            continue

        # Link
        print(f"Link: {cfg.path / cfg.val}")
        for _dirpath, dirnames, filenames in os.walk(frames):
            dirpath = Path(_dirpath)
            for _filename in sorted(filenames):
                filename = scale_filename(_filename)
                # os.symlink(root / dirpath / _filename, dst_path / filename)
                os.symlink(
                    cfg.path / dirpath / _filename,
                    cfg.path / cfg.val / filename,
                )
