"""Model factory used by training and inference.

This module builds models from simple declarative configs so adding a new
backbone is usually a YAML-only change. Both training and inference call the
same functions to avoid drift.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal

import timm
from torch import nn

ModelKind = Literal["timm", "import"]


@dataclass(frozen=True)
class ModelBuildConfig:
    """Configuration describing how to construct a model."""

    kind: ModelKind = "timm"
    id: str | None = None
    builder: str | None = None
    pretrained: bool | None = None
    img_size: int | None = None
    kwargs: dict[str, Any] | None = None


def build_model(model_cfg: ModelBuildConfig, num_classes: int) -> nn.Module:
    """Instantiate a model using a shared codepath for train/inference."""

    kwargs = dict(model_cfg.kwargs or {})

    if model_cfg.kind == "timm":
        model_id = model_cfg.id
        if not model_id:
            msg = "timm models require an 'id' field"
            raise ValueError(msg)
        return timm.create_model(
            model_id,
            pretrained=bool(model_cfg.pretrained),
            num_classes=num_classes,
            **kwargs,
        )

    if model_cfg.kind == "import":
        if not model_cfg.builder:
            msg = "import models require a 'builder' path"
            raise ValueError(msg)
        fn = _import_builder(model_cfg.builder)
        return fn(num_classes=num_classes, **kwargs)

    raise ValueError(f"Unknown model kind: {model_cfg.kind}")


def infer_default_image_size(model_cfg: ModelBuildConfig) -> int:
    """Best-effort guess for the model's expected input resolution."""

    if model_cfg.img_size:
        return int(model_cfg.img_size)

    if model_cfg.kind == "timm" and model_cfg.id:
        try:
            default_cfg = timm.get_pretrained_cfg(model_cfg.id)
            input_size = default_cfg.get("input_size")
            if input_size and len(input_size) >= 2:
                return int(input_size[-1])
        except Exception:
            pass
    return 224


def fingerprint(model_cfg: ModelBuildConfig, num_classes: int) -> dict[str, Any]:
    """Stable identifier for a model topology."""

    fp = {
        "kind": model_cfg.kind,
        "num_classes": int(num_classes),
        "pretrained": bool(model_cfg.pretrained),
        "img_size": model_cfg.img_size,
    }
    if model_cfg.id:
        fp["id"] = model_cfg.id
    if model_cfg.builder:
        fp["builder"] = model_cfg.builder
    if model_cfg.kwargs:
        fp["kwargs"] = model_cfg.kwargs
    return fp


def _import_builder(path: str):
    module_name, _, fn_name = path.partition(":")
    if not module_name or not fn_name:
        msg = "builder must look like 'module.submodule:callable'"
        raise ValueError(msg)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name, None)
    if fn is None:
        msg = f"Callable '{fn_name}' not found in module '{module_name}'"
        raise ImportError(msg)
    return fn


__all__ = ["ModelBuildConfig", "build_model", "infer_default_image_size", "fingerprint"]
