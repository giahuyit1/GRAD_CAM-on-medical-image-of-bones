"""
dataset.py
----------
FracAtlas Dataset loader dạng COCO JSON cho bài toán phân loại nhị phân:
  - label 1: gãy xương (Fractured)
  - label 0: không gãy (Non_fractured)

Chiến lược split:
  - Dùng file split có sẵn từ FRACATLAS/Utilities/Fracture Split/ (train/valid/test)
    chỉ chứa ảnh gãy. Ảnh không gãy được chia tự động với tỷ lệ tương ứng.
  - Nếu không có file split, tự động chia stratified 70/15/15.
  - Lưu splits.csv tổng hợp (image_path, label, split) để tái lập.
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=0, value=0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                   contrast_limit=0.15, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                           rotate_limit=0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=0, value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Build / load splits.csv
# ─────────────────────────────────────────────────────────────────────────────

def build_splits_csv(
    coco_json_path: str,
    fractured_dir: str,
    non_fractured_dir: str,
    splits_dir: str | None = None,   # thư mục chứa train/valid/test.csv của FracAtlas
    output_csv: str = "data/splits.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tạo splits.csv với các cột: image_path, label, split.

    Ưu tiên dùng file split có sẵn (splits_dir). Nếu không có, chia tự động.
    """
    rng = random.Random(seed)

    # ── Đọc COCO JSON để lấy tập ảnh gãy ──────────────────────────────────
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    annotated_ids = {ann["image_id"] for ann in coco["annotations"]}
    id2filename = {img["id"]: img["file_name"] for img in coco["images"]}
    fractured_files = {id2filename[i] for i in annotated_ids if i in id2filename}

    # ── Lấy tất cả file ảnh thực sự tồn tại ──────────────────────────────
    frac_dir = Path(fractured_dir)
    non_frac_dir = Path(non_fractured_dir)

    frac_all = sorted([f.name for f in frac_dir.iterdir()
                       if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    non_frac_all = sorted([f.name for f in non_frac_dir.iterdir()
                           if f.suffix.lower() in (".jpg", ".jpeg", ".png")])

    # ── Dùng split có sẵn nếu tồn tại ─────────────────────────────────────
    records = []
    if splits_dir and Path(splits_dir).exists():
        split_files = {
            "train": Path(splits_dir) / "train.csv",
            "val":   Path(splits_dir) / "valid.csv",
            "test":  Path(splits_dir) / "test.csv",
        }
        frac_split = {"train": set(), "val": set(), "test": set()}
        for split_name, csv_path in split_files.items():
            if csv_path.exists():
                df_s = pd.read_csv(csv_path)
                col = df_s.columns[0]
                frac_split[split_name] = set(df_s[col].str.strip().tolist())

        # Gán ảnh gãy theo split có sẵn
        assigned_frac = set()
        for split_name, fnames in frac_split.items():
            for fn in fnames:
                full_path = str(frac_dir / fn)
                if (frac_dir / fn).exists():
                    records.append({"image_path": full_path, "label": 1,
                                    "split": split_name, "file_name": fn})
                    assigned_frac.add(fn)

        # Tổng số ảnh gãy mỗi split → tính tỷ lệ cho non-fractured
        n_frac_train = len(frac_split["train"])
        n_frac_val   = len(frac_split["val"])
        n_frac_test  = len(frac_split["test"])
        n_frac_total = n_frac_train + n_frac_val + n_frac_test

        # Chia non-fractured theo cùng tỷ lệ
        non_frac_shuffled = non_frac_all.copy()
        rng.shuffle(non_frac_shuffled)
        n_nf = len(non_frac_shuffled)
        n_nf_train = round(n_nf * n_frac_train / max(n_frac_total, 1))
        n_nf_val   = round(n_nf * n_frac_val   / max(n_frac_total, 1))
        n_nf_test  = n_nf - n_nf_train - n_nf_val

        for i, fn in enumerate(non_frac_shuffled):
            full_path = str(non_frac_dir / fn)
            if i < n_nf_train:
                split = "train"
            elif i < n_nf_train + n_nf_val:
                split = "val"
            else:
                split = "test"
            records.append({"image_path": full_path, "label": 0,
                            "split": split, "file_name": fn})

    else:
        # ── Chia tự động 70/15/15 ──────────────────────────────────────────
        def split_list(lst, ratios=(0.70, 0.15, 0.15)):
            lst = lst.copy()
            rng.shuffle(lst)
            n = len(lst)
            n_train = int(n * ratios[0])
            n_val   = int(n * ratios[1])
            return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

        frac_train, frac_val, frac_test = split_list(frac_all)
        nf_train,   nf_val,   nf_test   = split_list(non_frac_all)

        for split_name, frac_lst, nf_lst in [
            ("train", frac_train, nf_train),
            ("val",   frac_val,   nf_val),
            ("test",  frac_test,  nf_test),
        ]:
            for fn in frac_lst:
                records.append({"image_path": str(frac_dir / fn), "label": 1,
                                "split": split_name, "file_name": fn})
            for fn in nf_lst:
                records.append({"image_path": str(non_frac_dir / fn), "label": 0,
                                "split": split_name, "file_name": fn})

    df = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[Dataset] splits.csv saved → {output_csv}")
    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp]
        print(f"  {sp:5s}: {len(sub):4d} ảnh "
              f"(gãy={sub['label'].sum()}, không_gãy={(sub['label']==0).sum()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FracAtlasDataset(Dataset):
    """
    Dataset phân loại nhị phân gãy / không gãy từ FracAtlas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame với cột image_path, label (0/1).
    transform : albumentations transform
        Pipeline augmentation/normalization.
    coco_json_path : str, optional
        Nếu cung cấp, load bounding box để dùng trong Grad-CAM eval sau.
    """

    def __init__(self, df: pd.DataFrame, transform=None,
                 coco_json_path: str | None = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        # ── Load bbox từ COCO JSON (tùy chọn, dùng khi eval Grad-CAM) ──────
        self.bboxes: dict[str, list] = {}   # file_name -> list of [x,y,w,h]
        if coco_json_path:
            with open(coco_json_path, "r") as f:
                coco = json.load(f)
            id2fn = {img["id"]: img["file_name"] for img in coco["images"]}
            for ann in coco["annotations"]:
                fn = id2fn.get(ann["image_id"], None)
                if fn:
                    self.bboxes.setdefault(fn, []).append(ann["bbox"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label    = int(row["label"])
        fn       = row.get("file_name", Path(img_path).name)

        # ── Load ảnh, chuyển RGB ────────────────────────────────────────────
        if not os.path.isfile(img_path):
            raise FileNotFoundError(
                f"Image not found at index {idx}: {img_path}"
            )

        with Image.open(img_path) as img:
            img = img.convert("RGB")

        img_np = np.array(img)

        # ── Augmentation ────────────────────────────────────────────────────
        if self.transform:
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0

        return {
            "image":      img_tensor,
            "label":      torch.tensor(label, dtype=torch.float32),
            "file_name":  fn,
            "image_path": img_path,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: tạo DataLoader
# ─────────────────────────────────────────────────────────────────────────────


def get_dataloaders(
    splits_csv: str,
    coco_json_path: str,
    img_dir_fractured: str,
    img_dir_non_fractured: str,
    img_size: int = 1024,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """Trả về dict {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}."""
    from torch.utils.data import DataLoader

    df = pd.read_csv(splits_csv)

    # 1. Kiểm tra các đường dẫn ảnh có tồn tại không
    exists_mask = df["image_path"].apply(os.path.isfile)

    missing_df = df[~exists_mask]

    if len(missing_df) > 0:
        print(f"[Dataset] Phát hiện {len(missing_df)} ảnh không tồn tại.")

        for _, row in missing_df.iterrows():
            print(f"  [MISSING] {row['image_path']}")

        # Loại bỏ ảnh không tồn tại
        df = df[exists_mask].reset_index(drop=True)

        # Cập nhật lại splits.csv
        df.to_csv(splits_csv, index=False)

        print(f"[Dataset] Đã cập nhật lại {splits_csv}")

    print(f"[Dataset] Tổng số ảnh hợp lệ: {len(df)}")

    # 2. Tạo DataLoader
    loaders = {}

    for split in ["train", "val", "test"]:
        sub_df = df[df["split"] == split].reset_index(drop=True)

        transform = (
            get_train_transforms(img_size)
            if split == "train"
            else get_val_transforms(img_size)
        )

        ds = FracAtlasDataset(
            df=sub_df,
            transform=transform,
            coco_json_path=coco_json_path if split == "test" else None,
        )

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )

        print(f"[Dataset] {split}: {len(ds)} ảnh")

    return loaders
