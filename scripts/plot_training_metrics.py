#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_file(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pairs = [("loss", "train_loss", "val_loss"), ("accuracy", "train_accuracy", "val_accuracy"),
             ("macro_f1", "train_macro_f1", "val_macro_f1"), ("balanced_accuracy", "train_balanced_accuracy", "val_balanced_accuracy"),
             ("worst_context_accuracy", "train_worst_context_accuracy", "val_worst_context_accuracy")]
    for name, a, b in pairs:
        if a in df and b in df and df[[a, b]].notna().any().any():
            plt.figure()
            plt.plot(df["epoch"], df[a], label=a)
            plt.plot(df["epoch"], df[b], label=b)
            plt.legend(); plt.xlabel("epoch"); plt.ylabel(name); plt.tight_layout()
            plt.savefig(out / f"{name}.png", dpi=160); plt.close()
    if "train_accuracy" in df and "val_accuracy" in df:
        plt.figure(); plt.plot(df["epoch"], df["train_accuracy"] - df["val_accuracy"])
        plt.xlabel("epoch"); plt.ylabel("train-val accuracy gap"); plt.tight_layout()
        plt.savefig(out / "overfitting_gap.png", dpi=160); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", nargs="+", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--output_dir", default="outputs/figures/training")
    args = ap.parse_args()
    files = args.metrics_csv or ([f"outputs/logs/{args.run_name}/metrics_epoch.csv"] if args.run_name else list(Path("outputs/logs").glob("*/metrics_epoch.csv")))
    for f in files:
        run = Path(f).parent.name
        plot_file(f, Path(args.output_dir) / run)


if __name__ == "__main__":
    main()
