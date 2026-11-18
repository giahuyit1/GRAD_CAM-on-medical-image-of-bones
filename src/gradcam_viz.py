"""
gradcam_viz.py
--------------
Sinh ảnh minh họa Grad-CAM / Grad-CAM++ cho tập test của FracAtlas.

Với mỗi ảnh trong test_predictions.csv, tạo 1 file PNG gồm 3 panel:
  [Ảnh gốc + bbox GT] | [GradCAM overlay] | [GradCAM++ overlay]

Output lưu vào Grad_cam_MinhHoa/{TP,TN,FP,FN}/

Sử dụng:
    # Chạy toàn bộ, cả 2 method
    python src/gradcam_viz.py --config configs/config.yaml

    # Lấy 5 ảnh mỗi nhóm để xem nhanh
    python src/gradcam_viz.py --config configs/config.yaml --n_per_group 5

    # Chỉ dùng GradCAM++, nhóm TP và FN
    python src/gradcam_viz.py --config configs/config.yaml --method gradcampp --groups TP FN
"""

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models  import build_resnet50, get_gradcam_target_layer
from src.dataset import get_val_transforms


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_coco_bboxes(coco_json_path: str) -> dict[str, list]:
    """
    Đọc COCO JSON, trả về dict: file_name → list of [x, y, w, h] (tọa độ gốc).
    """
    with open(coco_json_path, "r") as f:
        coco = json.load(f)
    id2fn = {img["id"]: img["file_name"] for img in coco["images"]}
    id2size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    bboxes: dict[str, list] = {}
    sizes:  dict[str, tuple] = {}
    for ann in coco["annotations"]:
        fn = id2fn.get(ann["image_id"])
        if fn:
            bboxes.setdefault(fn, []).append(ann["bbox"])
            sizes[fn] = id2size[ann["image_id"]]
    return bboxes, sizes


def letterbox_resize(img_bgr: np.ndarray,
                     target: int) -> tuple[np.ndarray, float, int, int]:
    """
    Resize ảnh giữ tỷ lệ về target×target (pad đen).
    Trả về (ảnh_mới, scale, pad_left, pad_top).
    """
    h, w = img_bgr.shape[:2]
    scale = target / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    pad_left = (target - new_w) // 2
    pad_top  = (target - new_h) // 2
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    return canvas, scale, pad_left, pad_top


def scale_bboxes(bboxes_orig: list,
                 orig_w: int, orig_h: int,
                 scale: float,
                 pad_left: int, pad_top: int) -> list:
    """
    Chuyển đổi bbox từ tọa độ ảnh gốc sang tọa độ ảnh đã letterbox.
    bbox_orig: [x, y, w, h]
    """
    scaled = []
    for x, y, bw, bh in bboxes_orig:
        x1 = int(x  * scale + pad_left)
        y1 = int(y  * scale + pad_top)
        x2 = int((x + bw) * scale + pad_left)
        y2 = int((y + bh) * scale + pad_top)
        scaled.append((x1, y1, x2, y2))
    return scaled


def draw_bboxes(img_bgr: np.ndarray,
                bboxes_px: list,
                color=(0, 255, 0),
                thickness: int = 2) -> np.ndarray:
    """Vẽ bounding box lên ảnh BGR."""
    out = img_bgr.copy()
    for x1, y1, x2, y2 in bboxes_px:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def add_title_bar(img_bgr: np.ndarray, title: str,
                  bar_height: int = 36,
                  font_scale: float = 0.52) -> np.ndarray:
    """Thêm thanh tiêu đề đen phía trên ảnh."""
    bar = np.zeros((bar_height, img_bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (6, bar_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img_bgr])


def add_panel_label(img_bgr: np.ndarray, label: str) -> np.ndarray:
    """Thêm nhãn nhỏ góc trên trái mỗi panel."""
    out = img_bgr.copy()
    cv2.putText(out, label, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 255), 2, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────────────────────

def process_one(
    img_path: Path,
    file_name: str,
    label: int,
    prob: float,
    outcome: str,
    model: torch.nn.Module,
    cam_gc: GradCAM,
    cam_gcpp: GradCAMPlusPlus | None,
    transform,
    bboxes_dict: dict,
    sizes_dict: dict,
    device: torch.device,
    display_size: int,
    method: str,
) -> np.ndarray:
    """
    Tạo ảnh minh họa 3-panel cho một sample.

    Returns: ảnh BGR (H × W×3) numpy array.
    """
    # ── Load & letterbox ảnh gốc ────────────────────────────────────────
    img_bgr_orig = cv2.imread(str(img_path))
    if img_bgr_orig is None:
        # fallback: PIL → cv2
        img_pil = Image.open(img_path).convert("RGB")
        img_bgr_orig = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    orig_h, orig_w = img_bgr_orig.shape[:2]
    img_lb, scale, pad_left, pad_top = letterbox_resize(img_bgr_orig, display_size)
    img_rgb_float = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # ── Bounding box GT ─────────────────────────────────────────────────
    bboxes_px = []
    if label == 1 and file_name in bboxes_dict:
        bboxes_px = scale_bboxes(
            bboxes_dict[file_name],
            orig_w, orig_h, scale, pad_left, pad_top
        )

    # ── Panel 1: Ảnh gốc (+ bbox GT) ───────────────────────────────────
    panel_orig = draw_bboxes(img_lb, bboxes_px, color=(0, 255, 0), thickness=2)
    panel_orig = add_panel_label(panel_orig, "Original + GT bbox")

    # ── Chuẩn bị tensor ─────────────────────────────────────────────────
    img_pil_full = Image.open(img_path).convert("RGB")
    img_np       = np.array(img_pil_full)
    aug          = transform(image=img_np)
    input_tensor = aug["image"].unsqueeze(0).to(device)

    targets = [BinaryClassifierOutputTarget(1)]

    # ── Panel 2: GradCAM ────────────────────────────────────────────────
    if method in ("gradcam", "both"):
        gc_map = cam_gc(input_tensor=input_tensor, targets=targets)[0]
        # Upsample heatmap về display_size
        gc_map_resized = cv2.resize(gc_map, (display_size, display_size),
                                    interpolation=cv2.INTER_LINEAR)
        overlay_gc = show_cam_on_image(img_rgb_float, gc_map_resized, use_rgb=True)
        overlay_gc_bgr = cv2.cvtColor(overlay_gc, cv2.COLOR_RGB2BGR)
        overlay_gc_bgr = draw_bboxes(overlay_gc_bgr, bboxes_px,
                                     color=(255, 255, 255), thickness=2)
        panel_gc = add_panel_label(overlay_gc_bgr, "Grad-CAM")
    else:
        panel_gc = None

    # ── Panel 3: GradCAM++ ──────────────────────────────────────────────
    if method in ("gradcampp", "both") and cam_gcpp is not None:
        gcpp_map = cam_gcpp(input_tensor=input_tensor, targets=targets)[0]
        gcpp_map_resized = cv2.resize(gcpp_map, (display_size, display_size),
                                      interpolation=cv2.INTER_LINEAR)
        overlay_gcpp = show_cam_on_image(img_rgb_float, gcpp_map_resized, use_rgb=True)
        overlay_gcpp_bgr = cv2.cvtColor(overlay_gcpp, cv2.COLOR_RGB2BGR)
        overlay_gcpp_bgr = draw_bboxes(overlay_gcpp_bgr, bboxes_px,
                                       color=(255, 255, 255), thickness=2)
        panel_gcpp = add_panel_label(overlay_gcpp_bgr, "Grad-CAM++")
    else:
        panel_gcpp = None

    # ── Ghép panels ─────────────────────────────────────────────────────
    if method == "gradcam":
        canvas = np.hstack([panel_orig, panel_gc])
    elif method == "gradcampp":
        canvas = np.hstack([panel_orig, panel_gcpp])
    else:  # both
        canvas = np.hstack([panel_orig, panel_gc, panel_gcpp])

    # ── Thanh tiêu đề ───────────────────────────────────────────────────
    label_str = "Fractured" if label == 1 else "Non-fractured"
    title = (f"{file_name}  |  GT: {label_str}  |  "
             f"Prob(fracture)={prob:.3f}  |  [{outcome}]")
    canvas = add_title_bar(canvas, title, bar_height=36)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GradCAM-Viz] Device: {device}")

    # ── Đọc config gradcam ───────────────────────────────────────────────
    gcfg         = cfg.get("gradcam", {})
    method       = args.method       or gcfg.get("method",       "both")
    n_per_group  = args.n_per_group  or gcfg.get("n_per_group",  None)
    output_dir   = args.output_dir   or gcfg.get("output_dir",   "Grad_cam_MinhHoa")
    display_size = gcfg.get("img_size_display", 512)
    groups       = args.groups       or ["TP", "TN", "FP", "FN"]

    results_dir = Path(cfg["paths"]["results_dir"])
    out_root    = Path(output_dir)

    # ── Load model ───────────────────────────────────────────────────────
    ckpt_path = results_dir / "best_model.pth"
    print(f"[GradCAM-Viz] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_resnet50(pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    # ── Target layer ─────────────────────────────────────────────────────
    target_layers = get_gradcam_target_layer(model)

    # ── Khởi tạo CAM objects ─────────────────────────────────────────────
    # use_cuda=False khi device=cpu để tránh lỗi
    use_cuda = device.type == "cuda"
    cam_gc   = GradCAM(model=model,
                       target_layers=target_layers)
    cam_gcpp = (GradCAMPlusPlus(model=model,
                                target_layers=target_layers)
                if method in ("gradcampp", "both") else None)

    # ── Transform (không augment, chỉ resize + normalize) ────────────────
    transform = get_val_transforms(cfg["data"]["img_size"])

    # ── Load COCO bboxes ─────────────────────────────────────────────────
    bboxes_dict, sizes_dict = load_coco_bboxes(cfg["data"]["coco_json"])
    print(f"[GradCAM-Viz] Loaded GT bboxes for {len(bboxes_dict)} fractured images.")

    # ── Load predictions CSV ─────────────────────────────────────────────
    preds_path = results_dir / "test_predictions.csv"
    df = pd.read_csv(preds_path)
    print(f"[GradCAM-Viz] Loaded {len(df)} predictions từ {preds_path}")

    # ── Tạo thư mục output ───────────────────────────────────────────────
    for grp in ["TP", "TN", "FP", "FN"]:
        (out_root / grp).mkdir(parents=True, exist_ok=True)

    # ── Xử lý từng nhóm ─────────────────────────────────────────────────
    total_saved = 0
    for outcome in groups:
        sub = df[df["outcome"] == outcome].reset_index(drop=True)
        if n_per_group is not None:
            sub = sub.head(int(n_per_group))

        print(f"\n[{outcome}] {len(sub)} ảnh ...")
        for _, row in tqdm(sub.iterrows(), total=len(sub),
                           desc=f"  {outcome}", ncols=80):
            fn      = row["file_name"]
            label   = int(row["label"])
            prob    = float(row["prob"])

            # Tìm đường dẫn ảnh
            if label == 1:
                img_path = Path(cfg["data"]["fractured_dir"]) / fn
            else:
                img_path = Path(cfg["data"]["non_fractured_dir"]) / fn

            if not img_path.exists():
                print(f"  [WARN] Không tìm thấy: {img_path}")
                continue

            try:
                canvas = process_one(
                    img_path    = img_path,
                    file_name   = fn,
                    label       = label,
                    prob        = prob,
                    outcome     = outcome,
                    model       = model,
                    cam_gc      = cam_gc,
                    cam_gcpp    = cam_gcpp,
                    transform   = transform,
                    bboxes_dict = bboxes_dict,
                    sizes_dict  = sizes_dict,
                    device      = device,
                    display_size= display_size,
                    method      = method,
                )

                # Đặt tên file output
                stem     = Path(fn).stem
                out_name = f"{stem}_gradcam.png"
                out_path = out_root / outcome / out_name
                cv2.imwrite(str(out_path), canvas)
                total_saved += 1

            except Exception as e:
                print(f"  [ERROR] {fn}: {e}")
                continue

    print(f"\n[GradCAM-Viz] Hoàn thành! Đã lưu {total_saved} ảnh vào: "
          f"{out_root.resolve()}")
    print(f"  Cấu trúc:")
    for grp in ["TP", "TN", "FP", "FN"]:
        n = len(list((out_root / grp).glob("*.png")))
        print(f"    {grp}/  ({n} ảnh)")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grad-CAM Visualization cho FracAtlas test set"
    )
    parser.add_argument("--config",      default="configs/config.yaml",
                        help="Đường dẫn file config YAML")
    parser.add_argument("--method",      default=None,
                        choices=["gradcam", "gradcampp", "both"],
                        help="Phương pháp CAM (mặc định lấy từ config)")
    parser.add_argument("--n_per_group", type=int, default=None,
                        help="Giới hạn số ảnh mỗi nhóm TP/TN/FP/FN")
    parser.add_argument("--groups",      nargs="+", default=None,
                        choices=["TP", "TN", "FP", "FN"],
                        help="Nhóm cần xử lý (mặc định: tất cả)")
    parser.add_argument("--output_dir",  default=None,
                        help="Thư mục lưu ảnh output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg, args)
