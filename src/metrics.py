"""
metrics.py
----------
Các metric đánh giá cho bài toán phân loại nhị phân mất cân bằng.

Hàm compute_metrics:
  - AUROC
  - AUPRC (Average Precision)
  - F1
  - Sensitivity (Recall)
  - Specificity
  - Optimal threshold (Youden index) từ val, dùng lại trên test

compute_threshold_from_val:
  Tìm ngưỡng tốt nhất trên val theo Youden index (sensitivity + specificity - 1).
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)


def compute_threshold_youden(probs: np.ndarray,
                              labels: np.ndarray) -> float:
    """
    Tìm ngưỡng tối ưu trên tập val bằng Youden index.
    Returns optimal threshold (float).
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    return float(thresholds[best_idx])


def compute_metrics(probs: np.ndarray,
                    labels: np.ndarray,
                    threshold: float | None = None) -> dict:
    """
    Parameters
    ----------
    probs     : predicted probabilities (sigmoid output), shape (N,)
    labels    : ground truth 0/1, shape (N,)
    threshold : nếu None, chọn tự động bằng Youden (dùng cho val).
                Nếu cung cấp, dùng ngưỡng đó (dùng cho test).

    Returns
    -------
    dict: auroc, auprc, f1, sensitivity, specificity, threshold
    """
    metrics = {}

    # ── AUROC & AUPRC ─────────────────────────────────────────────────────
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auroc"] = float("nan")

    try:
        metrics["auprc"] = float(average_precision_score(labels, probs))
    except ValueError:
        metrics["auprc"] = float("nan")

    # ── Ngưỡng ────────────────────────────────────────────────────────────
    if threshold is None:
        try:
            threshold = compute_threshold_youden(probs, labels)
        except Exception:
            threshold = 0.5
    metrics["threshold"] = float(threshold)

    # ── Threshold-based metrics ────────────────────────────────────────────
    preds = (probs >= threshold).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        metrics["sensitivity"] = float(tp / max(tp + fn, 1e-8))   # recall
        metrics["specificity"] = float(tn / max(tn + fp, 1e-8))
        metrics["f1"]          = float(f1_score(labels, preds, zero_division=0))
        metrics["tp"] = int(tp)
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
    except Exception:
        metrics.update({"sensitivity": float("nan"), "specificity": float("nan"),
                        "f1": float("nan"), "tp": 0, "tn": 0, "fp": 0, "fn": 0})

    return metrics
