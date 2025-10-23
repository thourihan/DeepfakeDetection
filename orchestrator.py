"""Simple runner to orchestrate training and inference jobs."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from rich.console import Console

from model_registry import get_model_spec

console = Console()
REPO_ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_run_dir(base: Path, timestamp: str) -> Path:
    run_dir = base / timestamp
    for sub in (run_dir, run_dir / "checkpoints", run_dir / "logs", run_dir / "plots"):
        sub.mkdir(parents=True, exist_ok=True)
    return run_dir


def stream_subprocess(cmd: list[str], *, env: dict[str, str], cwd: Path, log_path: Path) -> int:
    console.print(f"[bold blue]→ Running[/] {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd),
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            console.print(line.rstrip())
        return_code = process.wait()
    if return_code != 0:
        console.print(f"[bold red]Command failed with code {return_code}[/]")
    return return_code


def prepare_environment(
    *,
    config: dict[str, Any],
    model_cfg: dict[str, Any],
    run_dir: Path,
    training: bool,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(REPO_ROOT)
    env["DD_OUTPUT_DIR"] = str(run_dir)

    if "seed" in config:
        env["DD_SEED"] = str(config["seed"])

    device = config.get("device")
    if device:
        env["DD_DEVICE"] = str(device)

    data_cfg = config.get("data", {})
    data_root = data_cfg.get("root")
    if data_root:
        env["DD_DATA_ROOT"] = str(Path(data_root).expanduser().resolve())
    if "train_split" in data_cfg:
        env["DD_TRAIN_SPLIT"] = str(data_cfg["train_split"])
    if "val_split" in data_cfg:
        env["DD_VAL_SPLIT"] = str(data_cfg["val_split"])

    num_classes = model_cfg.get("num_classes", data_cfg.get("num_classes"))
    if num_classes is not None:
        env["DD_NUM_CLASSES"] = str(num_classes)
    if "img_size" in data_cfg:
        env["DD_IMG_SIZE"] = str(data_cfg["img_size"])

    if training:
        train_cfg = model_cfg.get("training", {})
        if "batch_size" in train_cfg:
            env["DD_BATCH_SIZE"] = str(train_cfg["batch_size"])
        if "epochs" in train_cfg:
            env["DD_EPOCHS"] = str(train_cfg["epochs"])
        if "num_workers" in train_cfg:
            env["DD_NUM_WORKERS"] = str(train_cfg["num_workers"])
        resume = str(train_cfg.get("resume", "")).lower()
        env["DD_RESUME_AUTO"] = "1" if resume in {"1", "true", "auto"} else "0"
    else:
        infer_cfg = model_cfg.get("inference", {})
        if "batch_size" in infer_cfg:
            env["DD_BATCH_SIZE"] = str(infer_cfg["batch_size"])
        if "num_workers" in infer_cfg:
            env["DD_NUM_WORKERS"] = str(infer_cfg["num_workers"])

    return env


def snapshot_config(run_dir: Path, *, config: dict[str, Any], model_cfg: dict[str, Any]) -> None:
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "global": {
            k: v
            for k, v in config.items()
            if k not in {"models", "selection"}
        },
        "model": model_cfg,
    }
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle)


def run_training(config: dict[str, Any], model_cfg: dict[str, Any], run_dir: Path) -> None:
    spec = get_model_spec(model_cfg["name"])
    env = prepare_environment(config=config, model_cfg=model_cfg, run_dir=run_dir, training=True)
    cmd = [sys.executable, str((REPO_ROOT / spec.train_script).resolve())]
    log_path = run_dir / "logs" / "train.log"
    return_code = stream_subprocess(cmd, env=env, cwd=run_dir, log_path=log_path)
    if return_code != 0:
        raise SystemExit(return_code)


def run_inference(
    config: dict[str, Any],
    model_cfg: dict[str, Any],
    run_dir: Path,
    config_path: Path,
) -> None:
    env = prepare_environment(config=config, model_cfg=model_cfg, run_dir=run_dir, training=False)
    cmd = [
        sys.executable,
        str((REPO_ROOT / "inference_cli.py").resolve()),
        "--config",
        str(config_path),
        "--model-name",
        model_cfg["name"],
        "--run-dir",
        str(run_dir),
    ]
    log_path = run_dir / "logs" / "inference.log"
    return_code = stream_subprocess(cmd, env=env, cwd=run_dir, log_path=log_path)
    if return_code != 0:
        raise SystemExit(return_code)


def orchestrate(config_path: Path, *, mode: str) -> None:
    config = load_config(config_path)

    seed = config.get("seed")
    if seed is not None:
        set_global_seed(int(seed))

    models_cfg = config.get("models", {})
    if not isinstance(models_cfg, dict):
        raise TypeError("models section must be a mapping of name -> config")

    selection = config.get("selection")
    if selection is None:
        selected_models = list(models_cfg.keys())
    else:
        selected_models = [str(name) for name in selection]

    for model_name in selected_models:
        base_cfg = models_cfg.get(model_name)
        if base_cfg is None:
            console.print(f"[bold yellow]Skipping unknown model[/]: {model_name}")
            continue
        model_cfg = {"name": model_name, **base_cfg}
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_output = Path(model_cfg.get("output_dir", f"runs/{model_name}"))
        run_dir = ensure_run_dir(base_output, timestamp)
        snapshot_config(run_dir, config=config, model_cfg=model_cfg)
        console.print(f"[bold]Model[/]: {model_name} → run dir {run_dir}")

        if mode == "training":
            run_training(config, model_cfg, run_dir)
        elif mode == "inference":
            run_inference(config, model_cfg, run_dir, config_path)
        else:
            raise ValueError(f"Unknown mode '{mode}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepfakeDetection orchestrator")
    parser.add_argument("--mode", choices=["training", "inference"], default="training")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    if args.config is not None:
        config_path = args.config
    else:
        default = "config/train.yaml" if args.mode == "training" else "config/inference.yaml"
        config_path = Path(default)

    orchestrate(config_path.resolve(), mode=args.mode)


if __name__ == "__main__":
    main()
