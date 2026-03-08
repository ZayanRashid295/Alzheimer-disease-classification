"""
Bulk testing: run all images in data/test/ through one or both models,
save per-image results to CSV and print summary metrics per class.
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
from data import get_class_names_and_splits, AlzheimerMRIDataset, get_eval_transforms
from models import AlzheimerCNN
from utils.metrics import compute_metrics


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


@torch.no_grad()
def run_bulk_test_simple(model, test_ds, device, batch_size=64):
    """Run model on test_ds; return y_true, y_pred, y_probs, paths (same order as dataset)."""
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
    # Get paths in same order as dataset (valid indices only if skip_corrupted)
    paths = []
    for i in range(len(test_ds)):
        path, _ = test_ds._get_sample(i)
        paths.append(str(path))
    return y_true, y_pred, y_probs, paths


def main():
    parser = argparse.ArgumentParser(description="Bulk test on data/test/ for all classes")
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--model", choices=["model1", "model2", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    args = parser.parse_args()

    device = get_device()
    class_names, _, _, test_pairs = get_class_names_and_splits(
        args.data_root,
        config.IMAGE_EXTENSIONS,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.TEST_RATIO,
        config.RANDOM_STATE,
    )
    # Use in_channels from first available checkpoint
    in_channels = 3
    first_ckpt = config.CHECKPOINT_DIR / "best_model.pt"
    if first_ckpt.exists():
        ckpt = torch.load(first_ckpt, map_location="cpu")
        in_channels = int(ckpt.get("in_channels", 3))
    test_ds = AlzheimerMRIDataset(
        test_pairs,
        transform=get_eval_transforms(config.IMAGE_SIZE, channels=in_channels),
        skip_corrupted=True,
    )
    n_test = len(test_ds)
    print(f"Test set: {n_test} images across {len(class_names)} classes: {class_names}\n")

    checkpoints = []
    if args.model in ("model1", "both"):
        checkpoints.append(("model1", "best_model.pt", config.CHECKPOINT_DIR / "best_model.pt"))
    if args.model in ("model2", "both"):
        checkpoints.append(("model2", "best_model_20260307_141516.pt", config.CHECKPOINT_DIR / "best_model_20260307_141516.pt"))

    out_dir = args.output_dir / "bulk_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_key, model_label, ckpt_path in checkpoints:
        if not ckpt_path.exists():
            print(f"Skipping {model_label}: not found at {ckpt_path}")
            continue
        print(f"Running bulk test: {model_label}")
        model, _, _ = load_model(ckpt_path, device)
        y_true, y_pred, y_probs, paths = run_bulk_test_simple(model, test_ds, device, args.batch_size)

        metrics = compute_metrics(y_true, y_pred, class_names)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print("\n  Per-class recall (sensitivity):")
        for name, rec in zip(class_names, metrics["per_class_recall"]):
            print(f"    {name}: {rec:.4f}")
        print("\n  Classification report:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        cm = confusion_matrix(y_true, y_pred)
        print("  Confusion matrix (rows=true, cols=pred):")
        print("    " + " ".join(f"{c[:4]:>4}" for c in class_names))
        for i, row in enumerate(cm):
            print(f"    " + " ".join(f"{v:>4}" for v in row) + f"  {class_names[i]}")

        # CSV: path, true_class, predicted_class, correct, confidence, prob_Class1, prob_Class2, ...
        csv_path = out_dir / f"bulk_test_{model_key}.csv"
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
        print(f"\n  Results written to {csv_path}\n")
        print("=" * 60)

    print("Bulk test done.")


if __name__ == "__main__":
    main()
