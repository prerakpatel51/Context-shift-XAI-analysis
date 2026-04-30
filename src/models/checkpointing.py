from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from src.utils.io import ensure_dir


def checkpoint_payload(model, optimizer, scheduler, epoch: int, config: dict, train_ds, metrics_so_far, best_values):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "config": config,
        "class_to_idx": train_ds.class_to_idx,
        "idx_to_class": train_ds.idx_to_class,
        "context_to_idx": train_ds.context_to_idx,
        "idx_to_context": train_ds.idx_to_context,
        "metrics_so_far": metrics_so_far,
        "best_metric_values": best_values,
        "seed": config.get("seed", 42),
        "model_arch": config["model_arch"],
        "pretrained": bool(config.get("pretrained", False)),
        "pretrained_cfg": config.get("pretrained_cfg"),
        "init_mode": config.get("init_mode"),
        "resume_from_checkpoint": config.get("resume_from_checkpoint"),
        "transform_settings": {"image_size": config.get("image_size", 224), "augmentation": config.get("augmentation", "strong")},
        "split_paths": {
            "train": str(getattr(train_ds, "csv_path", "")),
            "val": config.get("val_split_path"),
            "test": config.get("test_split_path"),
        },
    }


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def copy_checkpoint(src: str | Path, dst: str | Path) -> None:
    ensure_dir(Path(dst).parent)
    shutil.copy2(src, dst)
