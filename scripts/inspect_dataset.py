#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.nico import IMAGE_EXTS, discover_samples, find_metadata
from src.utils.io import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output", default="outputs/metrics/dataset_inspection.json")
    args = ap.parse_args()
    root = Path(args.data_root)
    top = [p.name for p in root.iterdir() if p.is_dir()] if root.exists() else []
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS] if root.exists() else []
    meta = find_metadata(root) if root.exists() else None
    df = discover_samples(root) if imgs or meta else None
    report = {
        "data_root": str(root),
        "top_level_folders": top,
        "num_images": len(imgs),
        "example_image_paths": [str(p) for p in imgs[:20]],
        "metadata_csv": str(meta) if meta else None,
        "classes": sorted(df.class_name.unique().tolist()) if df is not None else [],
        "contexts": sorted(df.context_name.dropna().unique().tolist()) if df is not None and df.context_name.notna().any() else [],
        "likely_layout": "metadata.csv" if meta else "folder_hierarchy",
    }
    print(report)
    write_json(report, args.output)


if __name__ == "__main__":
    main()
