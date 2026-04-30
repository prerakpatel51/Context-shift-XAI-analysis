from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import ensure_dir, write_json


def create_splits(df: pd.DataFrame, split_dir: str | Path, val_fraction=0.15, test_fraction=0.20, seed=42) -> dict:
    split_dir = ensure_dir(split_dir)
    df = df.copy()
    if "split" in df and df["split"].notna().any():
        df["split"] = df["split"].replace({"valid": "val", "validation": "val"}).fillna("")
    else:
        strat = df["class_name"]
        if df["class_name"].value_counts().min() < 2:
            train_val, test = train_test_split(df, test_size=test_fraction, random_state=seed)
        else:
            train_val, test = train_test_split(df, test_size=test_fraction, stratify=strat, random_state=seed)
        rel_val = val_fraction / max(1e-8, 1 - test_fraction)
        strat2 = train_val["class_name"]
        if train_val["class_name"].value_counts().min() < 2:
            train, val = train_test_split(train_val, test_size=rel_val, random_state=seed)
        else:
            train, val = train_test_split(train_val, test_size=rel_val, stratify=strat2, random_state=seed)
        df.loc[train.index, "split"] = "train"
        df.loc[val.index, "split"] = "val"
        df.loc[test.index, "split"] = "test"
    paths = {}
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split].copy()
        path = split_dir / f"{split}.csv"
        part.to_csv(path, index=False)
        paths[split] = str(path)
    summary = split_summary(df)
    summary["split_paths"] = paths
    write_json(summary, Path("outputs/metrics/split_summary.json"))
    return summary


def split_summary(df: pd.DataFrame) -> dict:
    summary = {
        "samples_per_split": df["split"].value_counts().to_dict(),
        "num_classes": int(df["class_name"].nunique()),
        "class_distribution": df.groupby("split")["class_name"].value_counts().to_dict(),
    }
    if df["context_name"].notna().any():
        group_counts = df.groupby(["class_name", "context_name"]).size().sort_values()
        summary.update({
            "num_contexts": int(df["context_name"].nunique()),
            "context_distribution": df.groupby("split")["context_name"].value_counts().to_dict(),
            "class_context_group_distribution": group_counts.to_dict(),
            "minority_groups": group_counts.head(20).to_dict(),
            "majority_groups": group_counts.tail(20).to_dict(),
        })
    return summary
