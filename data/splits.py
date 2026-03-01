"""
Discover class names from dataset folder structure and create stratified train/val/test splits.
Prevents data leakage by splitting at the file path level with a fixed seed.
"""
from pathlib import Path
from typing import List, Tuple, Dict

from sklearn.model_selection import train_test_split
import numpy as np


def discover_classes_and_paths(
    data_root: Path,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
) -> Tuple[List[str], Dict[str, List[Path]]]:
    """
    Scan data_root for subdirectories; each subdir name is a class.
    Collect all image paths per class. Skips non-directories and empty classes.

    Returns:
        class_names: sorted list of class names (for stable indexing).
        paths_by_class: mapping class_name -> list of Path to image files.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root is not a directory: {data_root}")

    paths_by_class: Dict[str, List[Path]] = {}
    for child in sorted(data_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        class_name = child.name
        paths = [
            p
            for p in child.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]
        if paths:
            paths_by_class[class_name] = paths

    if not paths_by_class:
        raise ValueError(f"No class folders with images found under {data_root}")

    class_names = sorted(paths_by_class.keys())
    return class_names, paths_by_class


def get_stratified_splits(
    paths_by_class: Dict[str, List[Path]],
    class_names: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Stratified split into (train, val, test) by class.
    Each sample is (path, class_index). class_index is w.r.t. class_names.

    Ratios must sum to 1.0. Splits are: first train, then val/test from the remainder.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    all_paths: List[Path] = []
    all_labels: List[int] = []
    for c in class_names:
        for p in paths_by_class[c]:
            all_paths.append(p)
            all_labels.append(class_to_idx[c])

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    # First split: train vs (val+test)
    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=(1.0 - train_ratio),
        stratify=all_labels,
        random_state=random_state,
    )
    # Second split: val vs test
    val_frac = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths,
        rest_labels,
        test_size=(1.0 - val_frac),
        stratify=rest_labels,
        random_state=random_state,
    )

    train_pairs = [(Path(p), int(l)) for p, l in zip(train_paths, train_labels)]
    val_pairs = [(Path(p), int(l)) for p, l in zip(val_paths, val_labels)]
    test_pairs = [(Path(p), int(l)) for p, l in zip(test_paths, test_labels)]
    return train_pairs, val_pairs, test_pairs


def discover_splits(
    data_root: Path,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
) -> Tuple[List[str], List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Discover class names and (path, class_index) pairs from pre-split layout:
        data_root/train/<Class>/...
        data_root/val/<Class>/...
        data_root/test/<Class>/...
    Returns (class_names, train_pairs, val_pairs, test_pairs).
    """
    data_root = Path(data_root)
    split_names = ("train", "val", "test")
    all_class_names = set()
    split_paths: Dict[str, List[Tuple[Path, int]]] = {s: [] for s in split_names}

    for split in split_names:
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        for child in sorted(split_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            class_name = child.name
            all_class_names.add(class_name)
            paths = [
                p
                for p in child.iterdir()
                if p.is_file() and p.suffix.lower() in extensions
            ]
            for p in paths:
                split_paths[split].append((p, -1))  # index set below

    class_names = sorted(all_class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    for split in split_names:
        for i, (p, _) in enumerate(split_paths[split]):
            # infer class from parent dir name
            cls = p.parent.name
            split_paths[split][i] = (p, class_to_idx[cls])

    train_pairs = split_paths["train"]
    val_pairs = split_paths["val"]
    test_pairs = split_paths["test"]
    return class_names, train_pairs, val_pairs, test_pairs
