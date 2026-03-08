"""
Evaluate both checkpoint models on the test set and print a comparison.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data import (
    get_class_names_and_splits,
    AlzheimerMRIDataset,
    get_eval_transforms,
)
from models import AlzheimerCNN
from utils.metrics import compute_metrics


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_config(ckpt_path: Path, device: torch.device):
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
def evaluate_model(model, test_loader, device):
    all_preds, all_probs, all_labels = [], [], []
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs, axis=0)
    return y_true, y_pred, y_probs


def main():
    ckpt_dir = config.CHECKPOINT_DIR
    model_files = [
        ("Model 1 (best_model.pt)", ckpt_dir / "best_model.pt"),
        ("Model 2 (best_model_20260307_141516.pt)", ckpt_dir / "best_model_20260307_141516.pt"),
    ]
    for name, p in model_files:
        if not p.exists():
            print(f"Skip {name}: not found at {p}")
            model_files = [(n, f) for n, f in model_files if f.exists()]
    if not model_files:
        print("No checkpoint files found.")
        return

    device = get_device()
    class_names_disc, _, _, test_pairs = get_class_names_and_splits(
        config.DATA_ROOT,
        config.IMAGE_EXTENSIONS,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.TEST_RATIO,
        config.RANDOM_STATE,
    )
    test_ds = AlzheimerMRIDataset(
        test_pairs,
        transform=get_eval_transforms(config.IMAGE_SIZE, channels=3),
        skip_corrupted=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    n_test = len(test_ds)
    print(f"Test set size: {n_test}\n")
    print("=" * 70)

    results = []
    for label, ckpt_path in model_files:
        model, class_names, in_channels = load_model_and_config(ckpt_path, device)
        y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred, class_names)
        try:
            auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")
        results.append((label, metrics, classification_report(y_true, y_pred, target_names=class_names, digits=4), auc))

    for i, (label, metrics, report, auc) in enumerate(results):
        print(f"\n{label}")
        print("-" * 50)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:   {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC (OvR): {auc:.4f}")
        print(f"  Per-class recall: {[f'{x:.3f}' for x in metrics['per_class_recall']]}")
        print("\nClassification report:")
        print(report)

    print("\n" + "=" * 70)
    print("COMPARISON (Test set)")
    print("=" * 70)
    print(f"{'Metric':<20} {'Model 1 (best_model.pt)':<28} {'Model 2 (timestamped)':<28}")
    print("-" * 76)
    for key in ["accuracy", "precision", "recall", "f1"]:
        v1, v2 = results[0][1][key], results[1][1][key]
        winner = " (better)" if v2 > v1 else ""
        print(f"{key:<20} {v1:<28.4f} {v2:<28.4f}{winner}")
    print(f"{'ROC-AUC (OvR)':<20} {results[0][3]:<28.4f} {results[1][3]:<28.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
