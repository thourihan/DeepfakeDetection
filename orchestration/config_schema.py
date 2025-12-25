from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .model_factory import ModelBuildConfig


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    root: str = Field(..., description="Root directory for the dataset.")
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    num_classes: int = 2
    img_size: int = 224
    class_labels: dict[str, str] | None = None


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    epochs: int = 10
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    img_size: int | None = None
    resume: str | bool | None = None
    accum_steps: int | None = None
    early_stop_patience: int | None = None


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    weights: str | None = None
    split: str | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    img_size: int | None = None


class DefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: ModelBuildConfig | None = None
    training: TrainingConfig | None = None
    inference: InferenceConfig | None = None
    transforms: dict[str, Any] | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: ModelBuildConfig
    output_dir: str | None = None
    transforms: dict[str, Any] | None = None
    training: TrainingConfig | None = None
    inference: InferenceConfig | None = None
    display_name: str | None = None
    label: str | None = None


class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    seed: int | None = None
    device: str | None = None
    data: DataConfig
    defaults: DefaultsConfig | None = None
    models: dict[str, ModelConfig]
    selection: list[str] | None = None

    @field_validator("models")
    @classmethod
    def _ensure_models_not_empty(cls, value: dict[str, ModelConfig]) -> dict[str, ModelConfig]:
        if not value:
            msg = "config.models cannot be empty"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _normalize_selection(self) -> OrchestratorConfig:
        models = self.models or {}
        if self.selection is None:
            self.selection = list(models.keys())
            return self

        missing = [name for name in self.selection if name not in models]
        if missing:
            msg = f"selection references unknown models: {', '.join(missing)}"
            raise ValueError(msg)
        return self
