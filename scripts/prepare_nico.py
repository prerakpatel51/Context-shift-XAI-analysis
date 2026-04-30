#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.nico import discover_samples
from src.datasets.split_utils import create_splits
from src.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--data_root")
    ap.add_argument("--split_dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_root = args.data_root or cfg.get("data_root", "data/raw/nico")
    split_dir = args.split_dir or cfg.get("split_dir", "data/splits")
    df = discover_samples(data_root, cfg.get("use_metadata_if_available", True))
    summary = create_splits(df, split_dir, cfg.get("val_fraction", 0.15), cfg.get("test_fraction", 0.2), cfg.get("seed", 42))
    print(summary)


if __name__ == "__main__":
    main()
