#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explanation_dirs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--selected_samples")
    ap.add_argument("--output_dir", default="outputs/figures/xai")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d, name in zip(args.explanation_dirs, args.names):
        d = Path(d)
        for csv in d.rglob("*deletion_curve.csv"):
            df = pd.read_csv(csv)
            if "fraction" in df and "target_prob" in df:
                for r in df[df["fraction"] != "auc"].itertuples():
                    rows.append({"name": name, "sample": csv.name, "fraction": float(r.fraction), "target_prob": r.target_prob, "mode": r.mask_mode})
    if rows:
        df = pd.DataFrame(rows)
        plt.figure()
        for name, g in df[df["mode"] == "top"].groupby("name"):
            curve = g.groupby("fraction")["target_prob"].mean()
            plt.plot(curve.index, curve.values, marker="o", label=name)
        plt.xlabel("masked fraction"); plt.ylabel("target probability"); plt.legend(); plt.tight_layout()
        plt.savefig(out / "deletion_curves_across_checkpoints.png", dpi=160); plt.close()

    swap_rows = []
    token_rows = []
    for d, name in zip(args.explanation_dirs, args.names):
        d = Path(d)
        summary = d / "context_swap" / "context_swap_summary.csv"
        if summary.exists():
            df = pd.read_csv(summary)
            df["name"] = name
            swap_rows.append(df)
        for csv in (d / "token_transformation").glob("*_token_transformation_layers.csv"):
            df = pd.read_csv(csv)
            df["name"] = name
            token_rows.append(df)
    if swap_rows:
        df = pd.concat(swap_rows, ignore_index=True)
        df.to_csv(out / "context_swap_summary.csv", index=False)
        plt.figure()
        df["true_class_probability_drop"].hist(bins=30)
        plt.xlabel("true-class probability drop after context swap")
        plt.ylabel("sample count")
        plt.tight_layout()
        plt.savefig(out / "context_swap_probability_drop_hist.png", dpi=160)
        plt.close()
        by_context = df.groupby("source_context")["true_class_probability_drop"].mean().sort_values()
        if not by_context.empty:
            plt.figure(figsize=(7, 4))
            by_context.plot(kind="barh")
            plt.xlabel("mean true-class probability drop")
            plt.tight_layout()
            plt.savefig(out / "context_swap_drop_by_context.png", dpi=160)
            plt.close()
    if token_rows:
        df = pd.concat(token_rows, ignore_index=True)
        layer_curve = df.groupby(["name", "layer"])["token_transformation_effect"].mean().reset_index()
        layer_curve.to_csv(out / "token_transformation_layer_summary.csv", index=False)
        plt.figure()
        for name, g in layer_curve.groupby("name"):
            plt.plot(g["layer"], g["token_transformation_effect"], marker="o", label=name)
        plt.xlabel("ViT block")
        plt.ylabel("mean token transformation effect")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "token_transformation_by_layer.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
