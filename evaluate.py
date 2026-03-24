"""
Evaluate a trained checkpoint on the test split.

Reports:
- Correct / wrong counts
- Accuracy, macro precision/recall/F1
- Per-class recall
- Full classification report
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data import AlzheimerMRIDataset, get_class_names_and_splits, get_eval_transforms
from models import AlzheimerCNN
from utils.metrics import compute_metrics


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    class_names = ckpt["class_names"]
    in_channels = int(ckpt["in_channels"])
    num_classes = int(ckpt["num_classes"])

    conv_filters = ckpt.get("conv_filters", config.CONV_FILTERS)
    use_se = ckpt.get("use_se", config.USE_SE_ATTENTION)

    # Infer hidden FC sizes from checkpoint if metadata is unavailable.
    fc_weight_keys = sorted(
        (
            k for k, v in state_dict.items()
            if k.startswith("fc.") and k.endswith(".weight") and getattr(v, "ndim", 0) == 2
        ),
        key=lambda k: int(k.split(".")[1]),
    )
    if len(fc_weight_keys) >= 2:
        fc_sizes = [int(state_dict[k].shape[0]) for k in fc_weight_keys[:-1]]
    else:
        fc_sizes = ckpt.get("fc_sizes", config.FC_SIZES)

    model = AlzheimerCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        conv_filters=conv_filters,
        fc_sizes=fc_sizes,
        dropout=(0.0, 0.0),
        use_se=use_se,
        he_init=False,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names, in_channels


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_pred.append(pred)
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on test split")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.CHECKPOINT_DIR / "best_model.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=config.DATA_ROOT,
        help="Dataset root containing train/val/test",
    )
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = get_device()
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print()

    model, class_names_ckpt, in_channels = load_model_from_checkpoint(args.checkpoint, device)
    class_names, _, _, test_pairs = get_class_names_and_splits(
        args.data_root,
        config.IMAGE_EXTENSIONS,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_state=config.RANDOM_STATE,
    )
    if class_names != class_names_ckpt:
        print("Warning: class names in dataset and checkpoint differ.")
        print(f"  Dataset:    {class_names}")
        print(f"  Checkpoint: {class_names_ckpt}")

    test_ds = AlzheimerMRIDataset(
        test_pairs,
        transform=get_eval_transforms(config.IMAGE_SIZE, channels=in_channels),
        skip_corrupted=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    y_true, y_pred = predict(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, class_names_ckpt)

    total = len(y_true)
    correct = int((y_true == y_pred).sum())
    wrong = total - correct

    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total images:  {total}")
    print(f"Correct:       {correct}")
    print(f"Wrong:         {wrong}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1-score:      {metrics['f1']:.4f}")
    print("\nPer-class recall:")
    for i, name in enumerate(class_names_ckpt):
        print(f"  {name:20s}: {metrics['per_class_recall'][i]:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names_ckpt, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)
    print("=" * 70)


if __name__ == "__main__":
    main()
