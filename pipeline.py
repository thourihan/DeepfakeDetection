"""Shared inference pipeline for headless deepfake detection."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from model_registry import ModelSpec, get_model_spec


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


class DeepfakePipeline:
    """Tiny helper around the model registry for inference.

    The pipeline keeps things deliberately light-weight: it knows how to build a
    model architecture, load weights, prepare ImageNet-style transforms, and run
    batched predictions. The default setup targets deepfake frame classification
    but works for general ImageFolder datasets (e.g., MNIST variants converted to
    RGB).
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.spec: ModelSpec | None = None
        self.model: nn.Module | None = None
        self.transforms: transforms.Compose | None = None

    def load_model(
        self,
        model_name: str,
        num_classes: int,
        weights_path: str | Path | None,
    ) -> nn.Module:
        spec = get_model_spec(model_name)
        model = spec.builder(model_name, num_classes)
        model = model.to(self.device)
        model.eval()

        if weights_path is not None:
            state = torch.load(Path(weights_path), map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)

        self.spec = spec
        self.model = model
        return model

    def build_transforms(
        self,
        *,
        eval: bool = True,
        image_size: int | None = None,
    ) -> transforms.Compose:
        size = image_size
        if size is None:
            size = self.spec.default_image_size if self.spec else 224

        if eval:
            ops = [
                transforms.Lambda(_ensure_rgb),
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        else:
            ops = [
                transforms.Lambda(_ensure_rgb),
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]

        self.transforms = transforms.Compose(ops)
        return self.transforms

    def predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            msg = "Model not loaded. Call load_model() first."
            raise RuntimeError(msg)
        self.model.eval()
        with torch.inference_mode():
            inputs = batch.to(self.device, non_blocking=True)
            return self.model(inputs)

    @staticmethod
    def postprocess(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(logits, dim=1)
        top1 = torch.argmax(probs, dim=1)
        return probs, top1


__all__ = ["DeepfakePipeline"]
