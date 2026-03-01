"""
Classification metrics: accuracy, precision, recall, F1, per-class recall.
Class weights from label distribution.
"""
from typing import List, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute accuracy, macro precision, recall, F1, and per-class recall.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    out = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }
    n_classes = int(max(y_true.max(), y_pred.max()) + 1)
    per_class_rec = recall_per_class(y_true, y_pred, n_classes)
    out["per_class_recall"] = per_class_rec
    if class_names and len(class_names) == len(per_class_rec):
        for i, name in enumerate(class_names):
            out[f"recall_{name}"] = per_class_rec[i]
    return out


def recall_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> List[float]:
    """Per-class recall (sensitivity)."""
    rec_list = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() == 0:
            rec_list.append(0.0)
        else:
            rec_list.append(float((y_pred[mask] == c).sum() / mask.sum()))
    return rec_list


def compute_class_weights(
    labels: List[int],
    num_classes: int,
) -> torch.Tensor:
    """Inverse frequency class weights for CrossEntropyLoss."""
    labels = np.array(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)
