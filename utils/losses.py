"""
Focal Loss for imbalanced classification.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss: -alpha * (1 - p_t)^gamma * log(p_t).
    alpha: per-class weights (tensor of length num_classes) or None for uniform.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        nll = -(targets_one_hot * log_probs).sum(dim=1)
        loss = focal_weight * nll
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
