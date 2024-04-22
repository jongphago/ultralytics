import os
import sys

sys.path.append(os.getcwd())
import argparse
from pathlib import Path

from codes.config import config, pattern
from codes.config.pattern import c_id_pattern, f_id_pattern, s_id_pattern


def get_link_path(cfg: dict, dirpath: Path, _filename: str, scale=False) -> list[Path, Path]:
    """
    _summary_

    Args:
        cfg (dict): _description_
        dirpath (Path): _description_
        _filename (str): _description_

    Examples:
        from codes.video import link

        src, dst = link.get_link_path(cfg, dirpath, _filename)
    """

    def scale_filename(_filename, fps=23) -> str:
        scenario_id = pattern.find(s_id_pattern, _filename)
        camera_id = pattern.find(c_id_pattern, _filename)
        frame_id = pattern.find(f_id_pattern, _filename)
        filename = f"s{scenario_id}_c{camera_id}_f{int(frame_id) * fps:04d}.jpg"
        return filename

    if scale:
        filename = scale_filename(_filename)
    else:
        filename = _filename
    src = dirpath / _filename
    assert src.exists()
    dst = cfg.path / cfg.val / filename
    return src, dst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="link images from frames to images")
    parser.add_argument(
        "config",
        type=str,
        default="aihub-sample",
        help="config file name (e.g., 'aihub-val')",
    )
    args = parser.parse_args()

    # path
    cfg = config.get_config(args.config)
    frames = cfg.path / cfg.frames

    if not (cfg.path / cfg.val).exists():
        os.makedirs(cfg.path / cfg.val)

    for _dirpath, dirnames, filenames in os.walk(frames):
        if dirnames:
            continue
        dirpath = Path(_dirpath)
        print(dirpath)
        for _filename in sorted(filenames):
            src, dst = get_link_path(cfg, dirpath, _filename)
            if not dst.exists():
                os.symlink(src, dst)
