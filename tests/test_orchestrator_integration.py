"""Integration test exercising orchestrator training and inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sys

# Ensure repository root is importable when tests run from nested paths.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")
pytest.importorskip("PIL")
import PIL.Image  # noqa: E402

from orchestration.orchestrator import orchestrate


def _make_dummy_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            dest = root / split / label
            dest.mkdir(parents=True, exist_ok=True)
            for idx in range(2):
                img = PIL.Image.new("RGB", (32, 32), color=(idx * 10, 0, 0))
                img.save(dest / f"{label}_{idx}.png")


def _write_config(path: Path, *, data_root: Path, output_root: Path, weights: Path | None = None) -> None:
    config = {
        "seed": 0,
        "device": "cpu",
        "data": {
            "root": str(data_root),
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "num_classes": 2,
            "img_size": 32,
        },
        "defaults": {
            "model": {"kind": "timm", "id": "resnet18", "pretrained": False, "img_size": 32},
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "num_workers": 0,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "img_size": 32,
            },
            "inference": {"batch_size": 2, "num_workers": 0},
        },
        "models": {
            "tiny_resnet": {
                "model": {},
                "output_dir": str(output_root / "tiny_resnet"),
                "inference": {"weights": str(weights)} if weights else {},
            }
        },
        "selection": ["tiny_resnet"],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle)


def _latest_run_dir(model_run_root: Path) -> Path:
    runs = sorted(model_run_root.glob("*"))
    assert runs, "no run directories created"
    return runs[-1]


def test_orchestrator_train_then_infer(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _make_dummy_dataset(dataset_root)

    config_path = tmp_path / "train_config.yaml"
    output_root = tmp_path / "runs"
    _write_config(config_path, data_root=dataset_root, output_root=output_root)

    orchestrate(config_path, mode="training")

    model_root = output_root / "tiny_resnet"
    training_dir = _latest_run_dir(model_root)
    best_ckpt = training_dir / "checkpoints" / "best.ckpt"
    assert best_ckpt.exists(), "training did not persist a best checkpoint"

    # Run inference using the freshly produced checkpoint.
    _write_config(
        config_path,
        data_root=dataset_root,
        output_root=output_root,
        weights=best_ckpt,
    )
    orchestrate(config_path, mode="inference")

    inference_dir = _latest_run_dir(model_root)
    metrics_path = inference_dir / "logs" / "metrics.json"
    assert metrics_path.exists(), "inference metrics were not written"
    metrics = json.loads(metrics_path.read_text())
    assert "accuracy" in metrics and 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics.get("split") == "test"
