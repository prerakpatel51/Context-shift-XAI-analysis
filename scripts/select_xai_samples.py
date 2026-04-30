#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate import evaluate_checkpoint
from src.utils.io import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--data_root", default="data/processed/nico")
    ap.add_argument("--split_dir", default="data/splits")
    ap.add_argument("--output_csv", default="outputs/metrics/xai/xai_selected_samples.csv")
    ap.add_argument("--num_samples", type=int, default=100)
    args = ap.parse_args()
    tmp = ensure_dir(Path(args.output_csv).parent / "_predictions")
    merged = None
    for ckpt, name in zip(args.checkpoints, args.names):
        evaluate_checkpoint(ckpt, {}, args.split_dir, "test", tmp / name, args.data_root)
        df = pd.read_csv(tmp / name / "test_predictions.csv")
        keep = df[["sample_id", "image_path", "y_true", "class_name_true", "context_name", "confidence", "correct", "y_pred"]].copy()
        keep = keep.rename(columns={"confidence": f"{name}_confidence", "correct": f"{name}_correct", "y_pred": f"{name}_prediction"})
        merged = keep if merged is None else merged.merge(keep, on=["sample_id", "image_path", "y_true", "class_name_true", "context_name"], how="outer")
    rows = []
    first = args.names[0]
    rows.append(merged[merged[f"{first}_correct"] == True].nlargest(args.num_samples // 5, f"{first}_confidence").assign(selection_reason="correct_high_confidence"))
    rows.append(merged[merged[f"{first}_correct"] == False].nlargest(args.num_samples // 5, f"{first}_confidence").assign(selection_reason="wrong_high_confidence"))
    rows.append(merged.nsmallest(args.num_samples // 5, f"{first}_confidence").assign(selection_reason="low_confidence"))
    rows.append(merged.groupby("class_name_true", group_keys=False).head(2).assign(selection_reason="balanced_by_class"))
    if "context_name" in merged and merged["context_name"].notna().any():
        rows.append(merged.groupby(["class_name_true", "context_name"], group_keys=False).head(1).assign(selection_reason="same_class_different_context"))
    out = pd.concat(rows, ignore_index=True).drop_duplicates("sample_id").head(args.num_samples)
    ensure_dir(Path(args.output_csv).parent)
    out.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
