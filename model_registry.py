"""Model registry for DeepfakeDetection orchestrator.

This module exposes a tiny registry so orchestration code can look up
metadata about each backbone: where its training script lives, what input
resolution it expects by default, and how to instantiate the architecture
for inference. The goal is to keep the project focused on deepfake frame
classification while making it easy to extend to new backbones.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import timm
from efficientnet_pytorch import EfficientNet
from fastervit import create_model
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    """Metadata required to orchestrate training and inference."""

    name: str
    train_module: str
    weights_key: str
    default_image_size: int
    builder: Callable[[str, int], nn.Module]


def _build_efficientnet(_: str, num_classes: int) -> nn.Module:
    model = EfficientNet.from_name("efficientnet-b3")
    in_features = model._fc.in_features  # type: ignore[attr-defined]
    model._fc = nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]
    return model


def _build_efficientformer(model_name: str, num_classes: int) -> nn.Module:
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def _build_fastervit(model_name: str, num_classes: int) -> nn.Module:
    model = create_model(model_name, pretrained=False)
    in_features = model.head.in_features  # type: ignore[attr-defined]
    model.head = nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]
    return model


_EXACT_SPECS: dict[str, ModelSpec] = {
    "efficientnet_b3": ModelSpec(
        name="efficientnet_b3",
        train_module="trainers.efficientnet",
        weights_key="efficientnet_b3",
        default_image_size=224,
        builder=_build_efficientnet,
    ),
}

_PREFIX_SPECS: dict[str, ModelSpec] = {
    "efficientformer": ModelSpec(
        name="efficientformer_v2_s1",
        train_module="trainers.efficientformer_v2",
        weights_key="efficientformer_v2_s1",
        default_image_size=224,
        builder=_build_efficientformer,
    ),
    "faster_vit": ModelSpec(
        name="faster_vit_2_224",
        train_module="trainers.fastervit",
        weights_key="faster_vit_2_224",
        default_image_size=224,
        builder=_build_fastervit,
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``model_name``.

    The registry keeps the layer thin: it knows where to find the training
    script and how to build the architecture so we can load weights for
    inference. It supports both exact model names (e.g., ``efficientnet_b3``)
    and prefix matches like ``efficientformer_v2_s1`` or ``faster_vit_2_224``.
    """

    if model_name in _EXACT_SPECS:
        return _EXACT_SPECS[model_name]

    for prefix, spec in _PREFIX_SPECS.items():
        if model_name.startswith(prefix):
            return replace(
                spec,
                name=model_name,
                weights_key=model_name,
            )

    raise KeyError(f"Unknown model '{model_name}'. Add it to model_registry.py.")


__all__ = ["ModelSpec", "get_model_spec"]
