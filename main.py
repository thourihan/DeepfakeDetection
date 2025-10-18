from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from efficientnet_pytorch import EfficientNet
from fastervit import create_model
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------------------------------------------------------------------
# Device & weights (case-sensitive path)
# ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS_DIR = Path("weights")
EN_WEIGHTS = WEIGHTS_DIR / "EfficientNetModel.pth"
FV_WEIGHTS = WEIGHTS_DIR / "FasterVitModel.pth"
EFV2_WEIGHTS = WEIGHTS_DIR / "EfficientFormerV2_S1.pth"

for p in (EN_WEIGHTS, FV_WEIGHTS, EFV2_WEIGHTS):
    if not p.exists():
        raise FileNotFoundError(
            f"Missing weights: {p}. Expected under {WEIGHTS_DIR.resolve()}."
        )

# ---------------------------------------------------------------------
# Labels & preprocessing
# ---------------------------------------------------------------------
CLASS_LABELS: Dict[int, str] = {0: "fake", 1: "real"}

# ImageNet normalization (EfficientNet/FasterViT).
TRANSFORM_IMAGENET = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# Project policy for EfficientFormerV2-S1 (no normalization).
TRANSFORM_NO_NORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


def _prepare_for_cam(
    image: Image.Image, img_size: int, normalize: bool
) -> Tuple[torch.Tensor, np.ndarray]:
    """Return (input_tensor, rgb_float) with aligned spatial transforms."""
    t = TRANSFORM_IMAGENET if normalize else TRANSFORM_NO_NORM
    pil_rc = transforms.Compose(
        [transforms.Resize(img_size), transforms.CenterCrop(img_size)]
    )
    pil_img = pil_rc(image.convert("RGB"))
    rgb = np.asarray(pil_img, dtype=np.float32) / 255.0  # HWC in [0,1]
    x = t(pil_img).unsqueeze(0).to(DEVICE)  # (1,3,H,W)
    return x, rgb


def _find_last_conv_layer(module: nn.Module) -> nn.Module:
    """Pick the last Conv2d for Grad-CAM target."""
    last: Optional[nn.Module] = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return last


def _add_label(img_rgb_uint8: np.ndarray, text: str) -> np.ndarray:
    """Overlay a readable text label onto the top-left of an image."""
    img = Image.fromarray(img_rgb_uint8)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    # Text with stroke for contrast
    draw.text((6, 6), text, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), font=font)
    return np.asarray(img)


# ---------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------
# EfficientNet-B3
efficientnet_model = EfficientNet.from_pretrained("efficientnet-b3")
_en_in = efficientnet_model._fc.in_features
efficientnet_model._fc = nn.Linear(_en_in, 2)
efficientnet_model.load_state_dict(
    torch.load(EN_WEIGHTS, map_location="cpu")
)
efficientnet_model.to(DEVICE).eval()

# FasterViT
faster_vit_model = create_model("faster_vit_2_224", pretrained=False, model_path=None)
_fv_in = faster_vit_model.head.in_features
faster_vit_model.head = nn.Linear(_fv_in, 2)
faster_vit_model.load_state_dict(
    torch.load(FV_WEIGHTS, map_location="cpu")
)
faster_vit_model.to(DEVICE).eval()

# EfficientFormerV2-S1 via timm
efficientformer_model: nn.Module = timm.create_model(
    "efficientformerv2_s1", pretrained=False, num_classes=2
)
efficientformer_model.load_state_dict(
    torch.load(EFV2_WEIGHTS, map_location="cpu"),
    strict=True,
)
efficientformer_model.to(DEVICE).eval()

# ---------------------------------------------------------------------
# Inference + Grad-CAM (show CAM from each model side by side)
# ---------------------------------------------------------------------
def predict_and_visualize(image: Image.Image) -> tuple[np.ndarray, str]:
    # EfficientNet prediction
    with torch.inference_mode():
        x_en = TRANSFORM_IMAGENET(image).unsqueeze(0).to(DEVICE)
        logits_en = efficientnet_model(x_en)
        probs_en = F.softmax(logits_en, dim=1)
        cls_en = int(probs_en.argmax(1))
        conf_en = float(probs_en[0, cls_en] * 100.0)
        label_en = CLASS_LABELS.get(cls_en, f"class_{cls_en}")

    # FasterViT prediction
    with torch.inference_mode():
        x_fv = TRANSFORM_IMAGENET(image).unsqueeze(0).to(DEVICE)
        logits_fv = faster_vit_model(x_fv)
        probs_fv = F.softmax(logits_fv, dim=1)
        cls_fv = int(probs_fv.argmax(1))
        conf_fv = float(probs_fv[0, cls_fv] * 100.0)
        label_fv = CLASS_LABELS.get(cls_fv, f"class_{cls_fv}")

    # EfficientFormerV2-S1 prediction
    with torch.inference_mode():
        x_ef = TRANSFORM_NO_NORM(image).unsqueeze(0).to(DEVICE)
        probs_ef = F.softmax(efficientformer_model(x_ef), dim=1)
        cls_ef = int(probs_ef.argmax(1))
        conf_ef = float(probs_ef[0, cls_ef] * 100.0)
        label_ef = CLASS_LABELS.get(cls_ef, f"class_{cls_ef}")

    # Grad-CAM: one overlay per model
    # EfficientNet CAM
    x_cam_en, rgb_en = _prepare_for_cam(image, img_size=224, normalize=True)
    target_layer_en = efficientnet_model._conv_head
    with GradCAM(model=efficientnet_model, target_layers=[target_layer_en]) as cam_en:
        gray_en = cam_en(input_tensor=x_cam_en, targets=[ClassifierOutputTarget(cls_en)])[0]
    overlay_en = show_cam_on_image(rgb_en, gray_en, use_rgb=True)
    panel_en = _add_label(
        overlay_en, f"EfficientNet-B3 • {label_en} ({conf_en:.1f}%)"
    )

    # FasterViT CAM (last conv found in the model)
    x_cam_fv, rgb_fv = _prepare_for_cam(image, img_size=224, normalize=True)
    target_layer_fv = _find_last_conv_layer(faster_vit_model)
    with GradCAM(model=faster_vit_model, target_layers=[target_layer_fv]) as cam_fv:
        gray_fv = cam_fv(input_tensor=x_cam_fv, targets=[ClassifierOutputTarget(cls_fv)])[0]
    overlay_fv = show_cam_on_image(rgb_fv, gray_fv, use_rgb=True)
    panel_fv = _add_label(
        overlay_fv, f"FasterViT • {label_fv} ({conf_fv:.1f}%)"
    )

    # EfficientFormerV2-S1 CAM
    x_cam_ef, rgb_ef = _prepare_for_cam(image, img_size=224, normalize=False)
    target_layer_ef = _find_last_conv_layer(efficientformer_model)
    with GradCAM(model=efficientformer_model, target_layers=[target_layer_ef]) as cam_ef:
        gray_ef = cam_ef(input_tensor=x_cam_ef, targets=[ClassifierOutputTarget(cls_ef)])[0]
    overlay_ef = show_cam_on_image(rgb_ef, gray_ef, use_rgb=True)
    panel_ef = _add_label(
        overlay_ef, f"EfficientFormerV2-S1 • {label_ef} ({conf_ef:.1f}%)"
    )

    # Concatenate panels horizontally
    side_by_side = np.concatenate([panel_en, panel_fv, panel_ef], axis=1)

    summary = (
        f"EfficientNet-B3: {label_en} ({conf_en:.2f}% confidence)\n"
        f"FasterViT: {label_fv} ({conf_fv:.2f}% confidence)\n"
        f"EfficientFormerV2-S1: {label_ef} ({conf_ef:.2f}% confidence)"
    )
    return side_by_side, summary


# ---------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------
iface = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="numpy"), "text"],
    title="Real vs Fake Face Detection",
    description="Upload an image to determine if the face is real or fake.",
)

if __name__ == "__main__":
    iface.launch()
