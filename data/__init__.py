"""
Data loading, splitting, and transforms for Alzheimer MRI classification.
"""
from pathlib import Path

from .dataset import AlzheimerMRIDataset
from .splits import (
    get_stratified_splits,
    discover_classes_and_paths,
    discover_splits,
)
from .transforms import get_train_transforms, get_eval_transforms


def get_class_names_and_splits(data_root, extensions, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Return (class_names, train_pairs, val_pairs, test_pairs).
    If data_root contains train/val/test subdirs, use them; else use stratified split from class subdirs.
    """
    data_root = Path(data_root)
    if (data_root / "train").is_dir() and (data_root / "val").is_dir() and (data_root / "test").is_dir():
        return discover_splits(data_root, extensions)
    class_names, paths_by_class = discover_classes_and_paths(data_root, extensions)
    train_pairs, val_pairs, test_pairs = get_stratified_splits(
        paths_by_class, class_names, train_ratio, val_ratio, test_ratio, random_state
    )
    return class_names, train_pairs, val_pairs, test_pairs


__all__ = [
    "AlzheimerMRIDataset",
    "get_stratified_splits",
    "discover_classes_and_paths",
    "discover_splits",
    "get_class_names_and_splits",
    "get_train_transforms",
    "get_eval_transforms",
]
