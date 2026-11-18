"""
eval.py
-------
Script đánh giá model đã huấn luyện trên test set.

Quy trình:
  1. Load checkpoint tốt nhất.
  2. Tìm ngưỡng tối ưu trên val set (Youden index).
  3. Chạy inference trên test set với ngưỡng đó.
  4. In và lưu kết quả: AUROC, AUPRC, F1, Sensitivity, Specificity, ConfMatrix.
  5. Lưu predictions CSV để phân tích sau (TP/TN/FP/FN cho Grad-CAM).

Sử dụng:
    python src/eval.py --config configs/config.yaml
    python src/eval.py --config configs/config.yaml --checkpoint results/best_model.pth
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_dataloaders, build_splits_csv
from src.models  import build_resnet50
from src.metrics import compute_metrics, compute_threshold_youden


# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, loader, device) -> tuple[np.ndarray, np.ndarray, list]:
    """Trả về (probs, labels, file_names)."""
    model.eval()
    all_probs  = []
    all_labels = []
    all_fnames = []

    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].numpy()
            fnames = batch["file_name"]

            logits = model(imgs).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
            all_fnames.extend(fnames)

    return np.array(all_probs), np.array(all_labels), all_fnames


# ─────────────────────────────────────────────────────────────────────────────

def evaluate(cfg: dict, checkpoint_path: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint ────────────────────────────────────────────────────────
    ckpt_path = checkpoint_path or str(results_dir / "best_model.pth")
    print(f"[Eval] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_resnet50(pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[Eval] Loaded from epoch {ckpt.get('epoch', '?')}, "
          f"val AUPRC={ckpt.get('val_auprc', '?'):.4f}")

    # ── splits.csv ────────────────────────────────────────────────────────
    splits_csv = cfg["data"]["splits_csv"]
    if not Path(splits_csv).exists():
        print("[Eval] Tạo splits.csv...")
        build_splits_csv(
            coco_json_path    = cfg["data"]["coco_json"],
            fractured_dir     = cfg["data"]["fractured_dir"],
            non_fractured_dir = cfg["data"]["non_fractured_dir"],
            splits_dir        = cfg["data"].get("splits_dir", None),
            output_csv        = splits_csv,
            seed              = cfg["train"]["seed"],
        )

    # ── DataLoaders ───────────────────────────────────────────────────────
    loaders = get_dataloaders(
        splits_csv            = splits_csv,
        coco_json_path        = cfg["data"]["coco_json"],
        img_dir_fractured     = cfg["data"]["fractured_dir"],
        img_dir_non_fractured = cfg["data"]["non_fractured_dir"],
        img_size              = cfg["data"]["img_size"],
        batch_size            = cfg["train"]["batch_size"],
        num_workers           = cfg["train"].get("num_workers", 4),
        pin_memory            = device.type == "cuda",
    )

    # ── Tìm ngưỡng tối ưu trên VAL ───────────────────────────────────────
    print("\n[Eval] Chạy inference trên val set để tìm ngưỡng...")
    val_probs, val_labels, _ = run_inference(model, loaders["val"], device)
    optimal_threshold = compute_threshold_youden(val_probs, val_labels)
    val_m = compute_metrics(val_probs, val_labels, threshold=optimal_threshold)
    print(f"[Val] Optimal threshold (Youden): {optimal_threshold:.4f}")
    print(f"[Val] AUROC={val_m['auroc']:.4f}  AUPRC={val_m['auprc']:.4f}  "
          f"F1={val_m['f1']:.4f}  Sens={val_m['sensitivity']:.4f}  "
          f"Spec={val_m['specificity']:.4f}")

    # ── Inference trên TEST ───────────────────────────────────────────────
    print("\n[Eval] Chạy inference trên test set...")
    test_probs, test_labels, test_fnames = run_inference(
        model, loaders["test"], device)
    test_m = compute_metrics(test_probs, test_labels, threshold=optimal_threshold)

    # ── In kết quả ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"  AUROC       : {test_m['auroc']:.4f}")
    print(f"  AUPRC       : {test_m['auprc']:.4f}")
    print(f"  F1          : {test_m['f1']:.4f}")
    print(f"  Sensitivity : {test_m['sensitivity']:.4f}")
    print(f"  Specificity : {test_m['specificity']:.4f}")
    print(f"  Threshold   : {test_m['threshold']:.4f}")
    print(f"  TP={test_m['tp']} TN={test_m['tn']} "
          f"FP={test_m['fp']} FN={test_m['fn']}")
    print("="*60)

    # ── Lưu metrics JSON ──────────────────────────────────────────────────
    metrics_out = {
        "val":  {k: v for k, v in val_m.items()},
        "test": {k: v for k, v in test_m.items()},
        "optimal_threshold": optimal_threshold,
        "checkpoint": str(ckpt_path),
    }
    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n[Eval] Metrics saved → {metrics_path}")

    # ── Lưu predictions CSV ───────────────────────────────────────────────
    preds_binary = (test_probs >= optimal_threshold).astype(int)
    outcome = []
    for pred, gt in zip(preds_binary, test_labels.astype(int)):
        if pred == 1 and gt == 1: outcome.append("TP")
        elif pred == 0 and gt == 0: outcome.append("TN")
        elif pred == 1 and gt == 0: outcome.append("FP")
        else: outcome.append("FN")

    df_preds = pd.DataFrame({
        "file_name": test_fnames,
        "label":     test_labels.astype(int).tolist(),
        "prob":      test_probs.tolist(),
        "pred":      preds_binary.tolist(),
        "outcome":   outcome,
    })
    preds_path = results_dir / "test_predictions.csv"
    df_preds.to_csv(preds_path, index=False)
    print(f"[Eval] Predictions saved → {preds_path}")

    # ── Tóm tắt theo outcome ──────────────────────────────────────────────
    print(f"\n[Eval] Phân bổ outcome: "
          + ", ".join(f"{k}={v}" for k, v in
                      df_preds["outcome"].value_counts().items()))

    return test_m


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 on FracAtlas")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint .pth (mặc định: results/best_model.pth)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg, checkpoint_path=args.checkpoint)
