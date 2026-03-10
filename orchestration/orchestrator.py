"""Unified orchestrator for training and inference using config-driven models."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from rich.console import Console

from .config_schema import DefaultsConfig, InferenceConfig, ModelConfig, OrchestratorConfig, TrainingConfig
from .model_factory import ModelBuildConfig, fingerprint
from .runtime import RunContext
from trainers.imagefolder_classifier import ModelRunConfig, run_inference, train_model

console = Console()


@dataclass
class ResolvedModelConfig:
    name: str
    model: ModelBuildConfig
    training: TrainingConfig
    inference: InferenceConfig
    transforms: dict[str, Any]
    output_dir: Path
    num_classes: int

    @property
    def model_fingerprint(self) -> dict[str, Any]:
        return fingerprint(self.model, self.num_classes)

    def to_runner(self) -> ModelRunConfig:
        return ModelRunConfig(
            name=self.name,
            model=self.model,
            training=self.training,
            inference=self.inference,
            transforms=self.transforms,
            num_classes=self.num_classes,
            model_fingerprint=self.model_fingerprint,
        )


def load_config(path: Path) -> OrchestratorConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return OrchestratorConfig(**raw)


def merge_model_config(
    name: str, *, cfg: ModelConfig, defaults: DefaultsConfig | None, data_cfg: Any
) -> ResolvedModelConfig:
    def _clean_config_dict(config_obj: Any) -> dict[str, Any]:
        raw = asdict(config_obj) if not hasattr(config_obj, "model_dump") else config_obj.model_dump(exclude_unset=True)
        return {key: value for key, value in raw.items() if value is not None}

    model_defaults = _clean_config_dict(defaults.model) if defaults and defaults.model else {}
    merged_model = ModelBuildConfig(**{**model_defaults, **_clean_config_dict(cfg.model)})

    training_defaults = (
        defaults.training.model_dump(exclude_unset=True) if defaults and defaults.training else {}
    )
    training_cfg = TrainingConfig(
        **{**training_defaults, **(cfg.training.model_dump(exclude_unset=True) if cfg.training else {})}
    )

    inference_defaults = (
        defaults.inference.model_dump(exclude_unset=True) if defaults and defaults.inference else {}
    )
    inference_cfg = InferenceConfig(
        **{**inference_defaults, **(cfg.inference.model_dump(exclude_unset=True) if cfg.inference else {})}
    )

    transforms = dict(defaults.transforms or {}) if defaults and defaults.transforms else {}
    if cfg.transforms:
        transforms.update(cfg.transforms)

    output_dir = Path(cfg.output_dir or f"runs/{name}")
    num_classes = data_cfg.num_classes

    return ResolvedModelConfig(
        name=name,
        model=merged_model,
        training=training_cfg,
        inference=inference_cfg,
        transforms=transforms,
        output_dir=output_dir,
        num_classes=num_classes,
    )


def create_run_context(model_cfg: ResolvedModelConfig, device: torch.device, seed: int | None) -> RunContext:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (model_cfg.output_dir / timestamp).expanduser().resolve()
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    plots_dir = run_dir / "plots"
    for path in (run_dir, checkpoints_dir, logs_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "model": model_cfg.name,
        "model_cfg": model_cfg.model.__dict__,
        "training": model_cfg.training.model_dump(),
        "inference": model_cfg.inference.model_dump(),
        "transforms": model_cfg.transforms,
        "num_classes": model_cfg.num_classes,
    }
    with (run_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)
    return RunContext(
        model_name=model_cfg.name,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        plots_dir=plots_dir,
        device=device,
        seed=seed,
    )


def orchestrate(config_path: Path, *, mode: str) -> None:
    cfg = load_config(config_path)

    device_str = cfg.device or "cuda"
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        console.print("[yellow]CUDA requested but unavailable; using CPU[/]")
        device_str = "cpu"
    device = torch.device(device_str)

    for name in cfg.selection:
        resolved = merge_model_config(name, cfg=cfg.models[name], defaults=cfg.defaults, data_cfg=cfg.data)
        run_context = create_run_context(resolved, device, cfg.seed)
        runner_cfg = resolved.to_runner()

        if mode == "training":
            console.rule(f"[bold]Training {name}")
            metrics = train_model(run_context=run_context, data_cfg=cfg.data, model_cfg=runner_cfg)
        elif mode == "inference":
            console.rule(f"[bold]Inference {name}")
            weights = Path(runner_cfg.inference.weights).expanduser() if runner_cfg.inference.weights else None
            metrics = run_inference(
                run_context=run_context,
                data_cfg=cfg.data,
                model_cfg=runner_cfg,
                weights_path=weights,
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        metrics_path = run_context.logs_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        console.print(f"[green]✓ Saved metrics[/] to {metrics_path}")


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="DeepfakeDetection orchestrator")
    parser.add_argument("--mode", choices=["training", "inference"], default="training")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        default = Path("config/train.yaml" if args.mode == "training" else "config/inference.yaml")
        config_path = default

    orchestrate(config_path.resolve(), mode=args.mode)


if __name__ == "__main__":
    run_cli()
