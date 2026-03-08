"""
Test on the archive dataset: runs both AugmentedAlzheimerDataset and OriginalDataset
through one or both models, prints metrics per (dataset, model) and saves CSVs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data import discover_classes_and_paths, AlzheimerMRIDataset, get_eval_transforms
from models import AlzheimerCNN
from utils.metrics import compute_metrics

# Default: archive (1) inside project
ARCHIVE_ROOT = PROJECT_ROOT / "archive (1)"
SUBFOLDERS = ["AugmentedAlzheimerDataset", "OriginalDataset"]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt["class_names"]
    in_channels = int(ckpt["in_channels"])
    num_classes = int(ckpt["num_classes"])
    model = AlzheimerCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        conv_filters=config.CONV_FILTERS,
        fc_sizes=config.FC_SIZES,
        dropout=config.DROPOUT,
        use_se=config.USE_SE_ATTENTION,
        he_init=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names, in_channels


def get_pairs_from_class_folders(data_root: Path, extensions: tuple):
    """Like discover_classes_and_paths but returns (class_names, list of (path, class_index))."""
    class_names, paths_by_class = discover_classes_and_paths(data_root, extensions)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    pairs = []
    for c in class_names:
        for p in paths_by_class[c]:
            pairs.append((Path(p), class_to_idx[c]))
    return class_names, pairs


@torch.no_grad()
def run_bulk_test_simple(model, test_ds, device, batch_size=64):
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    all_preds, all_probs, all_labels = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs, axis=0)
    paths = []
    for i in range(len(test_ds)):
        path, _ = test_ds._get_sample(i)
        paths.append(str(path))
    return y_true, y_pred, y_probs, paths


def run_test_on_subfolder(
    subfolder_path: Path,
    model,
    model_class_names: list,
    in_channels: int,
    device,
    batch_size: int,
    extensions: tuple,
):
    """Run model on one subfolder (Augmented or Original). Returns (y_true, y_pred, y_probs, paths, class_names)."""
    if not subfolder_path.is_dir():
        return None
    class_names, pairs = get_pairs_from_class_folders(subfolder_path, extensions)
    if not pairs:
        return None
    # Align class names with checkpoint (must match order)
    if sorted(class_names) != sorted(model_class_names):
        print(f"  Warning: folder classes {class_names} vs model classes {model_class_names}")
    test_ds = AlzheimerMRIDataset(
        pairs,
        transform=get_eval_transforms(config.IMAGE_SIZE, channels=in_channels),
        skip_corrupted=True,
    )
    return run_bulk_test_simple(model, test_ds, device, batch_size) + (class_names,)


def main():
    parser = argparse.ArgumentParser(description="Test on archive (1) Augmented + Original datasets")
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--model", choices=["model1", "model2", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    args = parser.parse_args()

    if not args.archive_root.is_dir():
        print(f"Archive root not found: {args.archive_root}")
        sys.exit(1)

    device = get_device()
    extensions = config.IMAGE_EXTENSIONS
    in_channels = 3
    first_ckpt = config.CHECKPOINT_DIR / "best_model.pt"
    if first_ckpt.exists():
        ckpt = torch.load(first_ckpt, map_location="cpu")
        in_channels = int(ckpt.get("in_channels", 3))

    checkpoints = []
    if args.model in ("model1", "both"):
        checkpoints.append(("model1", "best_model.pt", config.CHECKPOINT_DIR / "best_model.pt"))
    if args.model in ("model2", "both"):
        checkpoints.append(("model2", "best_model_20260307_141516.pt", config.CHECKPOINT_DIR / "best_model_20260307_141516.pt"))

    out_dir = args.output_dir / "archive_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_key, model_label, ckpt_path in checkpoints:
        if not ckpt_path.exists():
            print(f"Skipping {model_label}: not found at {ckpt_path}\n")
            continue
        model, model_class_names, _ = load_model(ckpt_path, device)
        print(f"\n{'='*70}")
        print(f"Model: {model_label}")
        print("=" * 70)

        for subfolder in SUBFOLDERS:
            sub_path = args.archive_root / subfolder
            if not sub_path.is_dir():
                print(f"\n[ {subfolder} ] not found, skipping.\n")
                continue
            result = run_test_on_subfolder(
                sub_path,
                model,
                model_class_names,
                in_channels,
                device,
                args.batch_size,
                extensions,
            )
            if result is None:
                print(f"\n[ {subfolder} ] no images found.\n")
                continue
            y_true, y_pred, y_probs, paths, class_names = result
            n = len(y_true)
            metrics = compute_metrics(y_true, y_pred, class_names)

            print(f"\n--- {subfolder} (n={n}) ---")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print("  Per-class recall:")
            for name, rec in zip(class_names, metrics["per_class_recall"]):
                print(f"    {name}: {rec:.4f}")
            print("\n  Classification report:")
            print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
            cm = confusion_matrix(y_true, y_pred)
            print("  Confusion matrix (rows=true, cols=pred):")
            print("    " + " ".join(f"{c[:4]:>4}" for c in class_names))
            for i, row in enumerate(cm):
                print("    " + " ".join(f"{v:>4}" for v in row) + f"  {class_names[i]}")

            safe_name = subfolder.replace(" ", "_")
            csv_path = out_dir / f"{safe_name}_{model_key}.csv"
            with open(csv_path, "w") as f:
                header = ["path", "true_class", "predicted_class", "correct", "confidence"] + [f"prob_{c}" for c in class_names]
                f.write(",".join(header) + "\n")
                for i in range(len(paths)):
                    true_idx = int(y_true[i])
                    pred_idx = int(y_pred[i])
                    correct = 1 if true_idx == pred_idx else 0
                    conf = float(y_probs[i, pred_idx])
                    row = [
                        paths[i],
                        class_names[true_idx],
                        class_names[pred_idx],
                        str(correct),
                        f"{conf:.6f}",
                    ] + [f"{y_probs[i, j]:.6f}" for j in range(len(class_names))]
                    f.write(",".join(row) + "\n")
            print(f"\n  CSV: {csv_path}\n")

    print("Archive test done.")


if __name__ == "__main__":
    main()
