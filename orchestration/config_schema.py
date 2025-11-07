from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DataConfig(BaseModel):
    # data is fairly stable, so we can ignore extra keys here to stay backward-compatible
    model_config = ConfigDict(extra="ignore")

    root: str = Field(..., description="Root directory for the dataset.")
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    num_classes: int = 2
    img_size: int = 224
    class_labels: dict[str, str] | None = None


class InferenceConfig(BaseModel):
    # inference blocks tend to grow (extra options, transforms), so allow unknown keys
    model_config = ConfigDict(extra="allow")

    weights: str | None = None
    split: str | None = None
    batch_size: int = 64
    num_workers: int = 4
    img_size: int | None = None
    transforms: dict[str, Any] | None = None


class TrainingConfig(BaseModel):
    # same idea as inference: different trainers may read different knobs
    model_config = ConfigDict(extra="allow")

    batch_size: int = 64
    epochs: int = 10
    num_workers: int = 4
    img_size: int | None = None
    transforms: dict[str, Any] | None = None
    resume: str | bool | None = None


class ModelConfig(BaseModel):
    # per-model blocks often have bespoke fields, so don't be strict here
    model_config = ConfigDict(extra="allow")

    output_dir: str | None = None
    transforms: dict[str, Any] | None = None
    training: TrainingConfig | None = None
    inference: InferenceConfig | None = None
    display_name: str | None = None
    label: str | None = None


class OrchestratorConfig(BaseModel):
    # allow extra at the top level so we don't explode if YAML has comments/extra keys
    model_config = ConfigDict(extra="allow")

    seed: int | None = None
    device: str | None = None
    data: DataConfig
    models: dict[str, ModelConfig]
    selection: list[str] | None = None

    @field_validator("models")
    @classmethod
    def _ensure_models_not_empty(cls, value: dict[str, ModelConfig]) -> dict[str, ModelConfig]:
        # we never want to run train/inference with an empty models: block
        if not value:
            msg = "config.models cannot be empty"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _normalize_selection(self) -> OrchestratorConfig:
        models = self.models or {}
        if self.selection is None:
            # default to 'all models' if user didn't specify selection
            self.selection = list(models.keys())
            return self

        # user supplied a selection: make sure every name exists in models:
        missing = [name for name in self.selection if name not in models]
        if missing:
            msg = f"selection references unknown models: {', '.join(missing)}"
            raise ValueError(msg)
        return self
