"""
models.py
---------
ResNet50 backbone cho bài toán phân loại nhị phân gãy / không gãy.

Kiến trúc:
  ResNet50 (ImageNet pretrained)
    └─ fc: Linear(2048 → 1)   [output logit, không sigmoid]

Hai chế độ:
  - freeze_backbone=True : chỉ train fc head (warm-up phase)
  - freeze_backbone=False: train toàn bộ (fine-tune phase)

Target layer Grad-CAM: model.layer4[-1]
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def build_resnet50(pretrained: bool = True,
                   freeze_backbone: bool = False) -> nn.Module:
    """
    Xây dựng ResNet50 cho phân loại nhị phân.

    Returns
    -------
    model : nn.Module
        ResNet50 với fc thay thành Linear(2048, 1).
        Gọi model(x) → raw logit shape (B, 1).
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # Thay fc layer
    in_features = model.fc.in_features          # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )

    if freeze_backbone:
        # Đóng băng tất cả trừ fc
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad_(False)
    else:
        for param in model.parameters():
            param.requires_grad_(True)

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Mở toàn bộ tham số để fine-tune."""
    for param in model.parameters():
        param.requires_grad_(True)


def count_params(model: nn.Module) -> dict:
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def get_gradcam_target_layer(model: nn.Module):
    """Trả về target layer cho pytorch-grad-cam (layer4 block cuối)."""
    return [model.layer4[-1]]
