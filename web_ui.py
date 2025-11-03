"""Web UI for Real vs Fake face detection with Grad-CAM visualization.

This script loads three trained models (EfficientNet-B3, FasterViT-2-224,
EfficientFormerV2-S1), performs inference on an input image, produces
Grad-CAM overlays for each model, and displays the results side-by-side
via a Gradio interface. It also exports a high-resolution composite image
to disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as f
from PIL import Image, ImageDraw, ImageFont
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torchvision import transforms

from orchestrator import (
    build_eval_transforms,
    load_config,
    load_model,
    resolve_transform_mapping,
)

# ---------------------------------------------------------------------
# Device, config, and export settings
# ---------------------------------------------------------------------
DEFAULT_CONFIG_PATH = Path("config/inference.yaml")

EXPORT_SCALE = 2
EXPORT_DIR = Path("outputs") / "cam_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelBundle:
    """Container for model-specific inference resources."""

    name: str
    display_label: str
    model: nn.Module
    transform: transforms.Compose
    normalize: bool
    device: torch.device
    target_layer: nn.Module

CLASS_LABELS: dict[int, str] = {0: "fake", 1: "real"}
MODEL_CACHE: list[ModelBundle] = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_METADATA: dict[str, Any] = {}


def _resolve_weights_path(path_value: str | None) -> Path | None:
    """Resolve model weights path like orchestrator: relative to CWD."""
    if not path_value:
        return None
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p

def _tensor_to_rgb(tensor: torch.Tensor, *, normalize: bool) -> np.ndarray:
    """Convert a (C,H,W) tensor to an RGB float image in [0, 1]."""

    if tensor.ndim == 4:
        if tensor.size(0) != 1:
            msg = "Expected batch of size 1 for visualization."
            raise ValueError(msg)
        tensor = tensor[0]

    if tensor.ndim != 3:
        msg = "Expected a 3D tensor for visualization."
        raise ValueError(msg)

    arr = tensor.detach().clone()
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=arr.dtype, device=arr.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=arr.dtype, device=arr.device)
        arr = arr * std.view(-1, 1, 1) + mean.view(-1, 1, 1)

    arr = arr.clamp(0.0, 1.0)
    rgb = arr.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return rgb


def _find_last_conv_layer(module: nn.Module) -> nn.Module:
    """Pick the last Conv2d for Grad-CAM target."""
    last: nn.Module | None = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        msg = "No Conv2d layer found for Grad-CAM target."
        raise RuntimeError(msg)
    return last


def _resolve_cam_target(module: nn.Module) -> nn.Module:
    """Return a target layer for Grad-CAM, preferring explicit conv heads."""

    conv_head = getattr(module, "_conv_head", None)
    if isinstance(conv_head, nn.Module):
        return conv_head
    return _find_last_conv_layer(module)


def _add_label(img_rgb_uint8: np.ndarray, text: str) -> np.ndarray:
    """Overlay a readable text label onto the top-left of an image."""
    img = Image.fromarray(img_rgb_uint8)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text(
        (6, 6),
        text,
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
        font=font,
    )
    return np.asarray(img)


def _coerce_device(device_str: str | None) -> torch.device:
    """Return a :class:`torch.device` for the requested string."""

    if device_str:
        requested = torch.device(device_str)
        if requested.type == "cuda" and not torch.cuda.is_available():
            print(
                "[UI] CUDA requested in config but unavailable. Falling back to CPU.",
            )
            return torch.device("cpu")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _detect_normalization(transform: transforms.Compose) -> bool:
    """Check whether a composed transform includes normalization."""

    for op in getattr(transform, "transforms", []):
        if isinstance(op, transforms.Normalize):
            return True
    return False


def initialize_from_config(config_path: Path) -> None:
    """Load orchestrator config and populate inference resources."""

    global CLASS_LABELS, MODEL_CACHE, DEVICE, CONFIG_METADATA

    config = load_config(config_path)

    CONFIG_METADATA = {
        "config_path": config_path,
        "raw": config,
    }

    device_str = config.get("device")
    DEVICE = _coerce_device(device_str)

    data_cfg: dict[str, Any] = config.get("data", {})
    num_classes = int(data_cfg.get("num_classes", 2))
    image_size = int(data_cfg.get("img_size", 224))

    labels_cfg = data_cfg.get("class_labels")
    if isinstance(labels_cfg, dict):
        CLASS_LABELS = {int(k): str(v) for k, v in labels_cfg.items()}

    models_cfg: dict[str, dict[str, Any]] = config.get("models", {})
    selection: list[str] = config.get("selection") or list(models_cfg.keys())

    bundles: list[ModelBundle] = []
    for model_name in selection:
        model_cfg = models_cfg.get(model_name)
        if not isinstance(model_cfg, dict):
            print(f"[UI] Skipping unknown model '{model_name}' in selection.")
            continue

        toggles = resolve_transform_mapping(model_cfg, phase="eval")
        transform = build_eval_transforms(image_size, toggles=toggles)
        normalize = _detect_normalization(transform)

        inference_cfg = model_cfg.get("inference", {})
        weights_path = _resolve_weights_path(inference_cfg.get("weights"))

        model = load_model(model_name, num_classes, weights_path, DEVICE)
        target_layer = _resolve_cam_target(model)

        display_label = (
            str(model_cfg.get("display_name")
                or model_cfg.get("label")
                or model_name)
        )

        bundles.append(
            ModelBundle(
                name=model_name,
                display_label=display_label,
                model=model,
                transform=transform,
                normalize=normalize,
                device=DEVICE,
                target_layer=target_layer,
            ),
        )

    if not bundles:
        msg = "No valid models configured for inference."
        raise RuntimeError(msg)

    MODEL_CACHE = bundles


def build_interface(config_path: Path = DEFAULT_CONFIG_PATH) -> gr.Interface:
    """Create a Gradio interface configured via orchestrator settings."""

    initialize_from_config(config_path)

    return gr.Interface(
        fn=predict_and_visualize,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Image(type="numpy"), "text"],
        title="Real vs Fake Face Detection",
        description="Upload an image to determine if the face is real or fake.",
    )


# ---------------------------------------------------------------------
# Inference + Grad-CAM (show CAM from each model side by side)
# ---------------------------------------------------------------------
def predict_and_visualize(image: Image.Image) -> tuple[np.ndarray, str]:
    """Run inference with three models and visualize Grad-CAM panels side-by-side.

    Returns a high-resolution concatenated image (as a NumPy array) and a
    summary string with predicted labels and confidences for each model.
    """
    panels: list[np.ndarray] = []
    summary_lines: list[str] = []

    for bundle in MODEL_CACHE:
        tensor = bundle.transform(image)
        if not isinstance(tensor, torch.Tensor):
            msg = f"Transform for {bundle.name} must return a tensor."
            raise TypeError(msg)

        if tensor.ndim == 3:
            batch = tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            batch = tensor
        else:
            msg = f"Unexpected tensor rank {tensor.ndim} for model {bundle.name}."
            raise ValueError(msg)

        batch = batch.to(bundle.device)

        with torch.inference_mode():
            logits = bundle.model(batch)
            probs = f.softmax(logits, dim=1)
            cls_idx = int(probs.argmax(1))
            confidence = float(probs[0, cls_idx] * 100.0)

        label = CLASS_LABELS.get(cls_idx, f"class_{cls_idx}")
        summary_lines.append(f"{bundle.display_label}: {label} ({confidence:.2f}% confidence)")

        with GradCAM(model=bundle.model, target_layers=[bundle.target_layer]) as cam:
            grayscale = cam(
                input_tensor=batch,
                targets=[ClassifierOutputTarget(cls_idx)],
            )[0]

        rgb = _tensor_to_rgb(tensor, normalize=bundle.normalize)
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
        panel = _add_label(overlay, f"{bundle.display_label} {label} ({confidence:.1f}%)")
        panels.append(panel)

    if not panels:
        msg = "No models available for inference."
        raise RuntimeError(msg)

    side_by_side = np.concatenate(panels, axis=1)

    # Export high-res image
    h, w, _ = side_by_side.shape
    export_img = Image.fromarray(side_by_side).resize(
        (w * EXPORT_SCALE, h * EXPORT_SCALE),
        resample=Image.BICUBIC,
    )
    out_path = (
        EXPORT_DIR
        / f"cam_triptych_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}.png"
    )
    export_img.save(out_path, format="PNG", optimize=True)

    # Return upscaled image to Gradio as well
    summary = "\n".join(summary_lines + [f"Saved: {out_path.resolve()}"])
    return np.asarray(export_img), summary


# ---------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------
iface = build_interface()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deepfake detection UI")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to an orchestrator inference YAML config.",
    )
    args = parser.parse_args()

    iface = build_interface(args.config)
    iface.launch()
