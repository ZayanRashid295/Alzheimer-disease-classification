"""
Evaluation script: load best checkpoint, run on test set, save confusion matrix,
classification report, ROC-AUC (OvR), per-class sensitivity plots.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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


def get_device() -> torch.device:
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
def get_predictions_and_probs(model, loader, device):
    all_preds = []
    all_probs = []
    all_labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())
    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs, axis=0),
    )


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_ovr(y_true, y_probs, class_names, save_path: Path):
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        if y_binary.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
        auc = roc_auc_score(y_binary, y_probs[:, i])
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_DIR / "best_model.pt")
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=config.EVAL_PLOTS_DIR)
    args = parser.parse_args()

    device = get_device()
    model, class_names, in_channels = load_model_and_config(args.checkpoint, device)

    class_names_disc, _, _, test_pairs = get_class_names_and_splits(
        args.data_root,
        config.IMAGE_EXTENSIONS,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_state=config.RANDOM_STATE,
    )
    assert class_names_disc == class_names, "Checkpoint classes vs dataset classes mismatch"

    test_ds = AlzheimerMRIDataset(
        test_pairs,
        transform=get_eval_transforms(config.IMAGE_SIZE, channels=in_channels),
        skip_corrupted=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
    )

    y_true, y_pred, y_probs = get_predictions_and_probs(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, class_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, class_names, args.output_dir / "confusion_matrix.png")
    if config.ROC_OVR and y_probs.shape[1] == len(class_names):
        plot_roc_ovr(y_true, y_probs, class_names, args.output_dir / "roc_ovr.png")

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Per-class recall (sensitivity):", metrics["per_class_recall"])
    print("Macro F1:", metrics["f1"], "Accuracy:", metrics["accuracy"])

    report_path = args.output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        f.write("\nPer-class recall: " + str(metrics["per_class_recall"]) + "\n")
        f.write(f"Accuracy: {metrics['accuracy']}\nMacro F1: {metrics['f1']}\n")
        try:
            auc_ovr = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
            f.write(f"ROC-AUC (OvR): {auc_ovr}\n")
        except Exception:
            pass
    try:
        auc_ovr = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
        print("ROC-AUC (OvR, macro):", auc_ovr)
    except Exception as e:
        print("ROC-AUC skipped:", e)


if __name__ == "__main__":
    main()
