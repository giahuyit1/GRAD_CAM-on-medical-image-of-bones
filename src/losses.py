"""
losses.py
---------
Focal Loss cho bài toán phân loại nhị phân mất cân bằng (gãy ~18%).

BCEFocalLoss(alpha, gamma):
  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

  - alpha : trọng số class dương (gãy).  Mặc định = None → tự tính từ dữ liệu.
  - gamma : focusing parameter.          Mặc định = 2 (theo paper gốc).

Sử dụng:
    criterion = BCEFocalLoss(alpha=0.75, gamma=2.0)
    loss = criterion(logits, labels)   # logits: (B,1) hoặc (B,), labels: (B,) float
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 reduction: str = "mean"):
        """
        Parameters
        ----------
        alpha   : Trọng số cho class dương (gãy). Giá trị > 0.5 phạt class âm ít hơn.
        gamma   : Focusing parameter. gamma=0 tương đương BCE thông thường.
        reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, 1) hoặc (B,)  – raw logit chưa qua sigmoid
        targets : (B,)               – 0.0 hoặc 1.0
        """
        logits  = logits.view(-1)
        targets = targets.view(-1)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p_t = sigmoid(logit) nếu target=1, else 1-sigmoid(logit)
        probs = torch.sigmoid(logits)
        p_t   = targets * probs + (1 - targets) * (1 - probs)

        # alpha_t
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_pos_weight(df, label_col: str = "label") -> float:
    """Tính pos_weight = n_negative / n_positive cho BCEWithLogitsLoss."""
    n_pos = (df[label_col] == 1).sum()
    n_neg = (df[label_col] == 0).sum()
    return n_neg / max(n_pos, 1)
