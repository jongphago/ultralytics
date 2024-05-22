import os
import sys

sys.path.append(os.getcwd())
import argparse
import functools
import shutil

from box import Box
from tqdm import tqdm

from codes.config import config


def merge_all_dicts(_pairs: list[list[dict, dict]]) -> list[dict]:
    """
    _summary_

    Args:
        _pairs (list[dict, dict]): _description_

    Returns:
        list[dict]: _description_

    Examples:
        >>> _pairs = [
        ...      [{'image': 'aihub/2.Validation/frames/시나리오19'},
        ...       {'label': 'aihub/2.Validation/converts/시나리오19'}],
        ...      [{'image': 'aihub/2.Validation/frames/시나리오42'},
        ...       {'label': 'aihub/2.Validation/converts/시나리오42'}],
        ... ]
        >>> pairs = merge_all_dicts(_pairs)
        >>> pairs
        [{'image': 'aihub/2.Validation/frames/시나리오19',
          'label': 'aihub/2.Validation/converts/시나리오19'},
         {'image': 'aihub/2.Validation/frames/시나리오42',
          'label': 'aihub/2.Validation/converts/시나리오42'}]
    """

    def merge_dicts(dict1, dict2):
        merged_dict = dict1.copy()
        merged_dict.update(dict2)
        return merged_dict

    pairs = []
    for pair in _pairs:
        _merged = functools.reduce(merge_dicts, pair, {})
        pairs.append(_merged)
    return pairs


def get_source_paths(cfg: dict, pair: dict) -> tuple[str, str]:
    """
    _summary_

    Args:
        cfg (dict): _description_
        pair (dict): _description_

    Returns:
        tuple[str, str]: _description_

    Examples:
        >>> sources = get_paths(cfg, pair, phase)
        >>> source_image_path, source_label_path = sources
    """
    source_image_path = cfg.path / pair.image
    source_label_path = cfg.path / pair.label
    assert source_image_path.exists() and source_label_path.exists()
    return source_image_path, source_label_path


def get_target_paths(cfg: dict, phase: str) -> tuple[str, str]:
    """
    _summary_

    Args:
        cfg (dict): _description_
        phase (str): _description_

    Returns:
        tuple[str, str]: _description_

    Examples:
        >>> targets = get_paths(cfg, pair, phase)
        >>> target_image_path, target_label_path = targets
    """
    target_image_path = cfg.path / cfg[phase]
    target_label_path = cfg.path / cfg[phase].replace("images", "labels")
    return target_image_path, target_label_path


# Remove existing directories
# WARNING: This will remove directories if they exist
def remove_existing_directories(targets: tuple[str, str], is_image=True, is_label=True) -> None:
    target_image_path, target_label_path = targets
    if is_image:
        shutil.rmtree(target_image_path) if target_image_path.exists() else None
    if is_label:
        shutil.rmtree(target_label_path) if target_label_path.exists() else None


def link_files(source_path, target_path: str, suffix: str) -> None:
    pattern = f"**/*{suffix}"
    paths = sorted(source_path.glob(pattern))
    for src in tqdm(paths, desc=f"{source_path.stem} ({target_path.stem})"):
        dst = target_path / src.name
        if not dst.parent.exists():
            dst.parent.mkdir(parents=True)
        if not dst.exists():
            dst.symlink_to(src)


def link_subset(cfg: dict) -> None:
    """
    _summary_

    Args:
        cfg (dict): _description_

    Examples:
        >>> cfg = config.get_config("aihub-subset")
        >>> link_subset(cfg)
    """
    for phase, _pairs in cfg.source.items():
        targets = get_target_paths(cfg, phase)
        remove_existing_directories(targets)
        pairs = merge_all_dicts(_pairs)
        for _pair in iter(pairs):
            pair = Box(_pair)
            sources = get_source_paths(cfg, pair)
            source_image_path, source_label_path = sources
            target_image_path, target_label_path = targets
            link_files(source_image_path, target_image_path, ".jpg")
            link_files(source_label_path, target_label_path, ".txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link files")
    parser.add_argument("--config", type=str, default="aihub-subset", help="config name")
    args = parser.parse_args()
    cfg = config.get_config(args.config)
    link_subset(cfg)
