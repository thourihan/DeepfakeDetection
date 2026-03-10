"""Model builder functions used via ``kind: import`` in YAML configs.

These are called by :func:`~orchestration.model_factory.build_model` when a
model entry specifies ``kind: import`` and a ``builder`` path like
``orchestration.model_registry:build_fastervit``.
"""

from __future__ import annotations

import timm
from fastervit import create_model
from torch import nn


def build_fastervit(*, num_classes: int, model_name: str = "faster_vit_2_224", pretrained: bool = False) -> nn.Module:
    """Build a FasterViT model with a resized classification head.

    Used as an import-style builder in YAML configs:

    .. code-block:: yaml

        model:
          kind: import
          builder: orchestration.model_registry:build_fastervit
          kwargs:
            model_name: faster_vit_2_224
            pretrained: true
    """
    model = create_model(model_name, pretrained=pretrained)
    in_features = model.head.in_features  # type: ignore[attr-defined]
    model.head = nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]
    return model


def build_timm(*, num_classes: int, model_name: str, pretrained: bool = False, **kwargs: object) -> nn.Module:
    """Generic timm builder for models that need keyword arguments timm supports directly.

    Useful when a model requires extra ``timm.create_model`` kwargs (e.g.
    ``img_size``) that cannot be expressed via the ``kind: timm`` path.
    """
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)


__all__ = ["build_fastervit", "build_timm"]
