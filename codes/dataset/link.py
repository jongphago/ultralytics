import functools
from pathlib import Path
import yaml
from box import Box


def merge_all_dicts(_train_pairs: list[list[dict, dict]]) -> list[dict]:
    """_summary_

    Args:
        _train_pairs (list[dict, dict]): _description_

    Returns:
        list[dict]: _description_

    Examples:
        >>> _train_pairs = [
        ...      [{'image': 'aihub/2.Validation/frames/시나리오19'},
        ...       {'label': 'aihub/2.Validation/converts/시나리오19'}],
        ...      [{'image': 'aihub/2.Validation/frames/시나리오42'},
        ...       {'label': 'aihub/2.Validation/converts/시나리오42'}],
        ... ]
        >>> train_pairs = merge_all_dicts(_train_pairs)
        >>> train_pairs
        [{'image': 'aihub/2.Validation/frames/시나리오19',
          'label': 'aihub/2.Validation/converts/시나리오19'},
         {'image': 'aihub/2.Validation/frames/시나리오42',
          'label': 'aihub/2.Validation/converts/시나리오42'}]

    """

    def merge_dicts(dict1, dict2):
        merged_dict = dict1.copy()
        merged_dict.update(dict2)
        return merged_dict

    train_pairs = []
    for pair in _train_pairs:
        _merged = functools.reduce(merge_dicts, pair, {})
        train_pairs.append(_merged)
    return train_pairs


if __name__ == "__main__":
    # Load config yaml
    with open("codes/dataset/config.yaml") as f:
        cfg = yaml.safe_load(f)
        cfg = Box(cfg)
        cfg.root = Path(cfg.root)

    # target_paths
    target_train_path = cfg.root / cfg.name / cfg.target.train
    target_validation_path = cfg.root / cfg.name / cfg.target.validation
    
    # source_paths
    _train_pairs = cfg.source.train
    train_pairs = merge_all_dicts(_train_pairs)
