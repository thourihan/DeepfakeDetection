"""Web UI for Real vs Fake face detection with Grad-CAM visualization.

Loads trained models from an orchestrator inference config, performs inference
on an uploaded image, produces Grad-CAM overlays for each model, and displays
the results side-by-side via a Gradio interface.
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

from orchestration.model_factory import build_model
from orchestration.orchestrator import load_config, merge_model_config
from trainers.imagefolder_classifier import build_transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("config/inference.yaml")
EXPORT_SCALE = 2
EXPORT_DIR = Path("outputs") / "cam_exports"


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


# Module-level cache populated by initialize_from_config().
CLASS_LABELS: dict[int, str] = {0: "fake", 1: "real"}
MODEL_CACHE: list[ModelBundle] = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_weights_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    p = Path(path_value).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _tensor_to_rgb(tensor: torch.Tensor, *, normalize: bool) -> np.ndarray:
    """Convert a (C,H,W) tensor to an RGB float32 image in [0, 1]."""
    if tensor.ndim == 4:
        if tensor.size(0) != 1:
            raise ValueError("Expected batch of size 1 for visualization.")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError("Expected a 3D tensor for visualization.")

    arr = tensor.detach().clone()
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=arr.dtype, device=arr.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=arr.dtype, device=arr.device)
        arr = arr * std.view(-1, 1, 1) + mean.view(-1, 1, 1)

    return arr.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy().astype(np.float32)


def _find_last_conv_layer(module: nn.Module) -> nn.Module:
    last: nn.Module | None = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return last


def _resolve_cam_target(module: nn.Module) -> nn.Module:
    conv_head = getattr(module, "_conv_head", None)
    if isinstance(conv_head, nn.Module):
        return conv_head
    return _find_last_conv_layer(module)


def _add_label(img_rgb_uint8: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(img_rgb_uint8)
    draw = ImageDraw.Draw(img)
    draw.text(
        (6, 6),
        text,
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
        font=ImageFont.load_default(),
    )
    return np.asarray(img)


def _detect_normalization(transform: transforms.Compose) -> bool:
    return any(isinstance(op, transforms.Normalize) for op in getattr(transform, "transforms", []))


def _coerce_device(device_str: str | None) -> torch.device:
    if not device_str:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device_str)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[UI] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return requested


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize_from_config(config_path: Path) -> None:
    """Load orchestrator config and populate the module-level model cache."""
    global CLASS_LABELS, MODEL_CACHE, DEVICE  # noqa: PLW0603

    cfg = load_config(config_path)
    DEVICE = _coerce_device(cfg.device)

    if cfg.data.class_labels:
        CLASS_LABELS = {int(k): str(v) for k, v in cfg.data.class_labels.items()}

    selection = cfg.selection or list(cfg.models.keys())
    bundles: list[ModelBundle] = []

    for model_name in selection:
        model_entry = cfg.models.get(model_name)
        if model_entry is None:
            print(f"[UI] Skipping unknown model '{model_name}' in selection.")
            continue

        resolved = merge_model_config(
            model_name,
            cfg=model_entry,
            defaults=cfg.defaults,
            data_cfg=cfg.data,
        )

        # Build eval transform from the resolved per-model toggles.
        eval_toggles: dict[str, Any] = (resolved.transforms or {}).get("eval") or {}
        image_size = resolved.inference.img_size or cfg.data.img_size
        transform = build_transforms(image_size, eval_toggles, train=False)
        normalize = _detect_normalization(transform)

        weights_path = _resolve_weights_path(resolved.inference.weights)

        model = build_model(resolved.model, resolved.num_classes)
        if weights_path:
            state: Any = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
        model = model.to(DEVICE)
        model.eval()

        display_label = str(model_entry.display_name or model_entry.label or model_name)

        bundles.append(
            ModelBundle(
                name=model_name,
                display_label=display_label,
                model=model,
                transform=transform,
                normalize=normalize,
                device=DEVICE,
                target_layer=_resolve_cam_target(model),
            )
        )

    if not bundles:
        raise RuntimeError("No valid models configured for inference.")

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


# ---------------------------------------------------------------------------
# Inference + Grad-CAM
# ---------------------------------------------------------------------------

def predict_and_visualize(image: Image.Image) -> tuple[np.ndarray, str]:
    """Run inference with all loaded models and return Grad-CAM panels side-by-side."""
    panels: list[np.ndarray] = []
    summary_lines: list[str] = []

    for bundle in MODEL_CACHE:
        tensor = bundle.transform(image)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Transform for {bundle.name} must return a tensor.")

        batch = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
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
        raise RuntimeError("No models available for inference.")

    side_by_side = np.concatenate(panels, axis=1)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    h, w, _ = side_by_side.shape
    export_img = Image.fromarray(side_by_side).resize(
        (w * EXPORT_SCALE, h * EXPORT_SCALE),
        resample=Image.BICUBIC,
    )
    out_path = EXPORT_DIR / f"cam_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}.png"
    export_img.save(out_path, format="PNG", optimize=True)

    summary = "\n".join(summary_lines + [f"Saved: {out_path.resolve()}"])
    return np.asarray(export_img), summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
    build_interface(args.config).launch()
