"""
One-time script: move images from combined_images (class folders) into
alzheimer_classifier/data as stratified train/val/test (70/15/15).
Creates data/train/<Class>, data/val/<Class>, data/test/<Class>.
Uses only stdlib (pathlib, random, shutil) so it runs without extra deps.
"""
import random
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent.parent.parent / "combined_images"
DEST_ROOT = Path(__file__).resolve().parent.parent / "data"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
RANDOM_STATE = 42


def discover_classes_and_paths(data_root):
    data_root = Path(data_root)
    paths_by_class = {}
    for child in sorted(data_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        paths = [p for p in child.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
        if paths:
            paths_by_class[child.name] = paths
    if not paths_by_class:
        raise ValueError(f"No class folders with images under {data_root}")
    return sorted(paths_by_class.keys()), paths_by_class


def stratified_split(paths_by_class, class_names, train_ratio, val_ratio, test_ratio, seed):
    """Return (train_pairs, val_pairs, test_pairs) with (path, class_index)."""
    rng = random.Random(seed)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    train_pairs, val_pairs, test_pairs = [], [], []
    for c in class_names:
        paths = list(paths_by_class[c])
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_train += n_test
            n_test = 0
        idx = class_to_idx[c]
        for i, p in enumerate(paths):
            pair = (Path(p), idx)
            if i < n_train:
                train_pairs.append(pair)
            elif i < n_train + n_val:
                val_pairs.append(pair)
            else:
                test_pairs.append(pair)
    return train_pairs, val_pairs, test_pairs


def main():
    if not SOURCE_ROOT.is_dir():
        print(f"Source not found: {SOURCE_ROOT}")
        sys.exit(1)

    class_names, paths_by_class = discover_classes_and_paths(SOURCE_ROOT)
    train_pairs, val_pairs, test_pairs = stratified_split(
        paths_by_class, class_names, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE
    )

    import shutil
    for split_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        for path, label in pairs:
            class_name = class_names[label]
            dest_dir = DEST_ROOT / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / path.name
            if path.resolve() == dest_file.resolve():
                continue
            if dest_file.exists() and dest_file != path:
                dest_file = dest_dir / f"{path.stem}_{path.suffix}"
            shutil.move(str(path), str(dest_file))
        print(f"Moved {len(pairs)} files to {DEST_ROOT / split_name}")

    print("Done. Dataset is now under", DEST_ROOT, "as train/val/test.")


if __name__ == "__main__":
    main()
