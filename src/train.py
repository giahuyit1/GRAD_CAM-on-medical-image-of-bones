"""
train.py
--------
Script huấn luyện ResNet50 phân loại gãy xương (FracAtlas).

Quy trình 2 bước:
  Bước 1 (warm-up)  : đóng băng backbone, train fc head N_WARMUP epoch.
  Bước 2 (fine-tune): mở toàn bộ, lr nhỏ hơn, cosine schedule.

Early stopping theo val AUPRC.
Lưu checkpoint tốt nhất và log ra results/train_log.csv.

Sử dụng:
    python src/train.py --config configs/config.yaml

Hoặc override:
    python src/train.py --config configs/config.yaml --batch_size 4 --epochs 30
"""

import os
import sys
import time
import argparse
import yaml
import csv
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import build_splits_csv, get_dataloaders
from src.models  import build_resnet50, unfreeze_backbone, count_params
from src.losses  import BCEFocalLoss
from src.metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: dict) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # CLI overrides
    for k, v in overrides.items():
        if v is not None:
            keys = k.split(".")
            d = cfg
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = v
    return cfg


# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device,
              is_train: bool, scaler=None):
    """Chạy một epoch train hoặc eval. Trả về (loss, all_probs, all_labels)."""
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_probs  = []
    all_labels = []

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            imgs   = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            if is_train and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(imgs).squeeze(1)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                logits = model(imgs).squeeze(1)
                loss   = criterion(logits, labels)
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    return avg_loss, np.array(all_probs), np.array(all_labels)


# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    # ── Setup ─────────────────────────────────────────────────────────────
    seed   = cfg["train"]["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Tạo/load splits.csv ───────────────────────────────────────────────
    splits_csv = cfg["data"]["splits_csv"]
    if not Path(splits_csv).exists():
        print("[Train] Tạo splits.csv...")
        build_splits_csv(
            coco_json_path       = cfg["data"]["coco_json"],
            fractured_dir        = cfg["data"]["fractured_dir"],
            non_fractured_dir    = cfg["data"]["non_fractured_dir"],
            splits_dir           = cfg["data"].get("splits_dir", None),
            output_csv           = splits_csv,
            seed                 = seed,
        )
    else:
        print(f"[Train] Dùng splits.csv có sẵn: {splits_csv}")

    # ── DataLoaders ───────────────────────────────────────────────────────
    loaders = get_dataloaders(
        splits_csv           = splits_csv,
        coco_json_path       = cfg["data"]["coco_json"],
        img_dir_fractured    = cfg["data"]["fractured_dir"],
        img_dir_non_fractured= cfg["data"]["non_fractured_dir"],
        img_size             = cfg["data"]["img_size"],
        batch_size           = cfg["train"]["batch_size"],
        num_workers          = cfg["train"].get("num_workers", 4),
        pin_memory           = device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_resnet50(pretrained=cfg["model"]["pretrained"],
                           freeze_backbone=True)
    model = model.to(device)

    params = count_params(model)
    print(f"[Model] Params: total={params['total']:,}, "
          f"trainable={params['trainable']:,}, frozen={params['frozen']:,}")

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = BCEFocalLoss(
        alpha = cfg["train"]["focal_alpha"],
        gamma = cfg["train"]["focal_gamma"],
    )

    # ── Scaler (AMP) ──────────────────────────────────────────────────────
    use_amp  = cfg["train"].get("use_amp", True) and device.type == "cuda"
    scaler   = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Log CSV ───────────────────────────────────────────────────────────
    log_path = results_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=[
        "epoch", "phase", "lr", "loss", "auroc", "auprc", "f1",
        "sensitivity", "specificity",
    ])
    log_writer.writeheader()

    # ─────────────────────────────────────────────────────────────────────
    # Bước 1: Warm-up (đóng băng backbone)
    # ─────────────────────────────────────────────────────────────────────
    n_warmup = cfg["train"]["warmup_epochs"]
    lr_warmup = cfg["train"]["lr_warmup"]
    print(f"\n{'='*60}")
    print(f"[Phase 1] Warm-up {n_warmup} epochs, lr={lr_warmup}")
    print(f"{'='*60}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_warmup,
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_warmup, eta_min=1e-6)

    best_auprc   = -1.0
    best_ckpt    = results_dir / "best_model.pth"
    patience_cnt = 0
    patience     = cfg["train"]["early_stopping_patience"]

    for epoch in range(1, n_warmup + 1):
        t0 = time.time()
        tr_loss, tr_probs, tr_labels = run_epoch(
            model, loaders["train"], criterion, optimizer,
            device, is_train=True, scaler=scaler)
        vl_loss, vl_probs, vl_labels = run_epoch(
            model, loaders["val"], criterion, optimizer,
            device, is_train=False)
        scheduler.step()

        tr_m = compute_metrics(tr_probs, tr_labels, threshold=None)
        vl_m = compute_metrics(vl_probs, vl_labels, threshold=None)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"[W{epoch:02d}/{n_warmup}] "
              f"TrainLoss={tr_loss:.4f} AUPRC={tr_m['auprc']:.4f} | "
              f"ValLoss={vl_loss:.4f} AUPRC={vl_m['auprc']:.4f} | "
              f"Time={time.time()-t0:.1f}s")

        for phase, loss, m in [("train_warmup", tr_loss, tr_m),
                                ("val_warmup",   vl_loss, vl_m)]:
            log_writer.writerow({
                "epoch": epoch, "phase": phase, "lr": f"{lr_now:.2e}",
                "loss": f"{loss:.4f}", **{k: f"{v:.4f}" for k, v in m.items()
                                          if k in ["auroc","auprc","f1",
                                                   "sensitivity","specificity"]},
            })
        log_file.flush()

        # Lưu checkpoint nếu tốt hơn
        if vl_m["auprc"] > best_auprc:
            best_auprc = vl_m["auprc"]
            patience_cnt = 0
            torch.save({
                "epoch": epoch, "phase": "warmup",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auprc": best_auprc,
                "cfg": cfg,
            }, best_ckpt)
            print(f"  ✔ Best checkpoint saved (val AUPRC={best_auprc:.4f})")
        else:
            patience_cnt += 1

    # ─────────────────────────────────────────────────────────────────────
    # Bước 2: Fine-tune (mở toàn bộ backbone)
    # ─────────────────────────────────────────────────────────────────────
    n_finetune = cfg["train"]["finetune_epochs"]
    lr_ft      = cfg["train"]["lr_finetune"]
    print(f"\n{'='*60}")
    print(f"[Phase 2] Fine-tune {n_finetune} epochs, lr={lr_ft}")
    print(f"{'='*60}")

    unfreeze_backbone(model)
    params = count_params(model)
    print(f"[Model] Params sau unfreeze: trainable={params['trainable']:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr_ft,
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_finetune, eta_min=1e-7)
    patience_cnt = 0

    for epoch in range(1, n_finetune + 1):
        t0 = time.time()
        tr_loss, tr_probs, tr_labels = run_epoch(
            model, loaders["train"], criterion, optimizer,
            device, is_train=True, scaler=scaler)
        vl_loss, vl_probs, vl_labels = run_epoch(
            model, loaders["val"], criterion, optimizer,
            device, is_train=False)
        scheduler.step()

        tr_m = compute_metrics(tr_probs, tr_labels, threshold=None)
        vl_m = compute_metrics(vl_probs, vl_labels, threshold=None)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"[FT{epoch:02d}/{n_finetune}] "
              f"TrainLoss={tr_loss:.4f} AUPRC={tr_m['auprc']:.4f} | "
              f"ValLoss={vl_loss:.4f} AUPRC={vl_m['auprc']:.4f} AUROC={vl_m['auroc']:.4f} | "
              f"Time={time.time()-t0:.1f}s")

        for phase, loss, m in [("train_finetune", tr_loss, tr_m),
                                ("val_finetune",   vl_loss, vl_m)]:
            log_writer.writerow({
                "epoch": n_warmup + epoch, "phase": phase, "lr": f"{lr_now:.2e}",
                "loss": f"{loss:.4f}", **{k: f"{v:.4f}" for k, v in m.items()
                                          if k in ["auroc","auprc","f1",
                                                   "sensitivity","specificity"]},
            })
        log_file.flush()

        if vl_m["auprc"] > best_auprc:
            best_auprc = vl_m["auprc"]
            patience_cnt = 0
            torch.save({
                "epoch": n_warmup + epoch, "phase": "finetune",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auprc": best_auprc,
                "cfg": cfg,
            }, best_ckpt)
            print(f"  ✔ Best checkpoint saved (val AUPRC={best_auprc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"[Early Stop] Không cải thiện {patience} epoch liên tiếp.")
                break

    log_file.close()
    print(f"\n[Done] Best val AUPRC = {best_auprc:.4f}")
    print(f"[Done] Checkpoint: {best_ckpt}")
    print(f"[Done] Log: {log_path}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 on FracAtlas")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Đường dẫn file config YAML")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs",    type=int, default=None,
                        help="Override finetune_epochs")
    parser.add_argument("--lr",        type=float, default=None,
                        help="Override lr_finetune")
    parser.add_argument("--seed",      type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {
        "train.batch_size":      args.batch_size,
        "train.finetune_epochs": args.epochs,
        "train.lr_finetune":     args.lr,
        "train.seed":            args.seed,
    }
    cfg = load_config(args.config, overrides)
    train(cfg)
