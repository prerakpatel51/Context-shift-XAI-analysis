from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def str2bool(v: str | bool | None) -> bool | None:
    if v is None or isinstance(v, bool):
        return v
    if v.lower() in {"yes", "true", "t", "1", "y"}:
        return True
    if v.lower() in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {v!r}")


def merge_overrides(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    merged = dict(config)
    for key, value in kwargs.items():
        if value is not None:
            merged[key] = value
    return merged
