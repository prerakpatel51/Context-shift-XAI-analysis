#!/usr/bin/env python
"""
Evaluation across three XAI axes for the ViT-NICO++ project.

Axis 1: Stability       - explanation robustness under input perturbations.
Axis 2: Faithfulness    - whether top-attributed patches actually drive predictions.
Axis 4: Representation  - per-layer ViT attention entropy / CLS concentration / head diversity.

Reads existing checkpoint + selected XAI samples and writes:
    outputs/metrics/xai/evaluation/axis1_stability/...
    outputs/metrics/xai/evaluation/axis2_faithfulness/...
    outputs/metrics/xai/evaluation/axis4_representation/...
    outputs/figures/evaluation/...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.nico import NICODataset
from src.datasets.transforms import build_transforms
from src.models.attention_hooks import get_attentions, patch_timm_attention
from src.models.build_model import build_model
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.io import ensure_dir
from src.xai.attention_rollout import rollout_from_attentions
from src.xai.explanation_metrics import topk_overlap


# -------------------------- helpers --------------------------

def cosine(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / n) if n > 0 else 0.0


def normalize01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


def fast_rollout(model, x):
    """Single forward pass + rollout. x is [1,3,H,W] on device."""
    patch_timm_attention(model)
    model.eval()
    with torch.no_grad():
        model(x)
    return rollout_from_attentions(get_attentions(model))


def patch_grad_x_input(model, x, label, patch_size=16):
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    score = logits[0, int(label)]
    grad = torch.autograd.grad(score, x, retain_graph=False)[0]
    g = (grad * x).detach()[0].sum(0)  # [H,W]
    H, W = g.shape
    nr, nc = H // patch_size, W // patch_size
    g = g[: nr * patch_size, : nc * patch_size].reshape(nr, patch_size, nc, patch_size).sum(dim=(1, 3))
    return g.cpu().numpy()


# -------------------------- AXIS 1: STABILITY --------------------------

def axis1_stability(model, ds, sids, out_root, n_perturb=3, sigma=0.05, patch_size=16,
                    methods=("attention_rollout", "patch_attribution")):
    device = next(model.parameters()).device
    rows = []
    out = ensure_dir(out_root / "axis1_stability")
    rng = np.random.default_rng(0)

    for sid in sids:
        if sid not in ds._index:
            continue
        item = ds[ds._index[sid]]
        image, label = item["image"], item["label"]
        x0 = image.unsqueeze(0).to(device)

        # base maps
        base = {}
        if "attention_rollout" in methods:
            base["attention_rollout"] = fast_rollout(model, x0)
        if "patch_attribution" in methods:
            base["patch_attribution"] = patch_grad_x_input(model, x0, label, patch_size)

        # perturbed maps
        for k in range(n_perturb):
            noise = torch.from_numpy(rng.normal(0, sigma, size=tuple(x0.shape)).astype(np.float32)).to(device)
            x_p = (x0 + noise).clamp(-3, 3)
            for m, b in base.items():
                if m == "attention_rollout":
                    h = fast_rollout(model, x_p)
                else:
                    h = patch_grad_x_input(model, x_p, label, patch_size)
                if h.shape != b.shape:
                    continue
                a, c = b.ravel(), h.ravel()
                sp = spearmanr(a, c).correlation if a.std() > 0 and c.std() > 0 else 0.0
                rows.append({
                    "sample_id": sid,
                    "method": m,
                    "perturb_seed": k,
                    "sigma": sigma,
                    "spearman": float(sp) if sp == sp else 0.0,
                    "cosine": cosine(a, c),
                    "topk_overlap@10": topk_overlap(a, c, k=0.1),
                    "context_name": item.get("context", -1),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out / "stability_per_sample.csv", index=False)
    if df.empty:
        print("[Axis1] no samples evaluated.")
        return df

    agg = df.groupby("method").agg(
        spearman_mean=("spearman", "mean"),
        spearman_std=("spearman", "std"),
        cosine_mean=("cosine", "mean"),
        topk10_mean=("topk_overlap@10", "mean"),
        n=("sample_id", "count"),
    ).reset_index()
    agg.to_csv(out / "stability_summary.csv", index=False)

    fig_dir = ensure_dir(out_root.parents[2] / "figures" / "evaluation" / "axis1_stability")
    fig, ax = plt.subplots(figsize=(7, 4))
    methods_list = agg["method"].tolist()
    xs = np.arange(len(methods_list))
    ax.bar(xs - 0.2, agg["spearman_mean"], 0.2, yerr=agg["spearman_std"], label="Spearman", capsize=3)
    ax.bar(xs, agg["cosine_mean"], 0.2, label="Cosine")
    ax.bar(xs + 0.2, agg["topk10_mean"], 0.2, label="Top-10% overlap")
    ax.set_xticks(xs); ax.set_xticklabels(methods_list, rotation=20)
    ax.set_ylabel("Stability score (higher = more stable)")
    ax.set_title(f"Axis 1: Explanation stability under Gaussian noise (σ={sigma})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "stability_summary.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods_list:
        sub = df[df["method"] == m]["spearman"].values
        ax.hist(sub, bins=20, alpha=0.5, label=m)
    ax.set_xlabel("Spearman ρ (perturbed vs original)"); ax.set_ylabel("Count")
    ax.set_title("Axis 1: Stability distribution")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "stability_distribution.png", dpi=150); plt.close(fig)

    print(f"[Axis1] wrote {out/'stability_summary.csv'} ({len(df)} rows)")
    return df


# -------------------------- AXIS 2: FAITHFULNESS --------------------------

def axis2_faithfulness(faith_dir, out_root):
    """Aggregate per-sample deletion curves into faithfulness summary statistics."""
    out = ensure_dir(out_root / "axis2_faithfulness")
    files = sorted(Path(faith_dir).glob("sample_*_deletion_curve.csv"))
    if not files:
        print(f"[Axis2] no deletion curves at {faith_dir}")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception:
            continue
    df = pd.concat(frames, ignore_index=True)

    pts = df[df["fraction"] != "auc"].copy()
    pts["fraction"] = pts["fraction"].astype(float)
    pts["target_prob"] = pts["target_prob"].astype(float)

    base = pts.groupby(["sample_id", "method"])["target_prob"].apply(
        lambda s: s.max()
    ).reset_index().rename(columns={"target_prob": "p_baseline"})
    pts = pts.merge(base, on=["sample_id", "method"], how="left")
    pts["prob_drop"] = pts["p_baseline"] - pts["target_prob"]

    summary = pts.groupby(["method", "mask_mode", "fraction"]).agg(
        target_prob_mean=("target_prob", "mean"),
        target_prob_std=("target_prob", "std"),
        prob_drop_mean=("prob_drop", "mean"),
        n=("sample_id", "count"),
    ).reset_index()
    summary.to_csv(out / "faithfulness_curves.csv", index=False)

    auc_rows = df[df["fraction"] == "auc"].copy()
    auc_rows["target_prob"] = auc_rows["target_prob"].astype(float)
    auc_summary = auc_rows.groupby("method").agg(
        deletion_auc_mean=("target_prob", "mean"),
        deletion_auc_std=("target_prob", "std"),
        n=("sample_id", "count"),
    ).reset_index()

    # comprehensiveness: drop when removing top-k vs random-k at each k
    comp_rows = []
    for (m, k), g in pts.groupby(["method", "fraction"]):
        top = g[g["mask_mode"] == "top"]["prob_drop"].mean()
        rnd = g[g["mask_mode"] == "random"]["prob_drop"].mean()
        bot = g[g["mask_mode"] == "bottom"]["prob_drop"].mean()
        comp_rows.append({"method": m, "fraction": float(k),
                          "drop_top": top, "drop_random": rnd, "drop_bottom": bot,
                          "comprehensiveness": (top - rnd) if pd.notna(top) and pd.notna(rnd) else np.nan})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "faithfulness_comprehensiveness.csv", index=False)

    # flip rate per (method, fraction) for top-mode
    flips = []
    pts_top = pts[pts["mask_mode"] == "top"].copy()
    for (m, k), g in pts_top.groupby(["method", "fraction"]):
        baseline_pred = g.groupby("sample_id")["target_prob"].idxmax()
        flips.append({"method": m, "fraction": float(k),
                      "n": len(g["sample_id"].unique())})
    pd.DataFrame(flips).to_csv(out / "faithfulness_n_samples.csv", index=False)

    auc_summary.to_csv(out / "faithfulness_auc_summary.csv", index=False)

    # plots
    fig_dir = ensure_dir(out_root.parents[2] / "figures" / "evaluation" / "axis2_faithfulness")

    fig, ax = plt.subplots(figsize=(8, 5))
    for (m, mode), g in summary.groupby(["method", "mask_mode"]):
        if mode != "top":
            continue
        g = g.sort_values("fraction")
        ax.plot(g["fraction"], g["target_prob_mean"], marker="o", label=m)
    ax.set_xlabel("Fraction of patches masked (top-attributed)"); ax.set_ylabel("Mean target probability")
    ax.set_title("Axis 2: Deletion curves (lower-faster = more faithful)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "deletion_curves_top.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    methods_list = sorted(summary["method"].unique())
    xs = np.arange(len(methods_list))
    for i, mode in enumerate(["top", "random", "bottom"]):
        ys = []
        for m in methods_list:
            sub = summary[(summary["method"] == m) & (summary["mask_mode"] == mode)]
            if sub.empty:
                ys.append(np.nan)
            else:
                ys.append(np.trapz(sub.sort_values("fraction")["target_prob_mean"].values,
                                   sub.sort_values("fraction")["fraction"].astype(float).values))
        ax.bar(xs + (i - 1) * 0.25, ys, 0.25, label=mode)
    ax.set_xticks(xs); ax.set_xticklabels(methods_list, rotation=25, ha="right")
    ax.set_ylabel("Deletion AUC (lower is better for top, higher for bottom)")
    ax.set_title("Axis 2: Deletion AUC by mask mode")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "deletion_auc_by_mode.png", dpi=150); plt.close(fig)

    if not comp.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in comp["method"].unique():
            g = comp[comp["method"] == m].sort_values("fraction")
            ax.plot(g["fraction"], g["comprehensiveness"], marker="s", label=m)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Fraction"); ax.set_ylabel("Comprehensiveness (drop_top - drop_random)")
        ax.set_title("Axis 2: Comprehensiveness curves")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(fig_dir / "comprehensiveness.png", dpi=150); plt.close(fig)

    print(f"[Axis2] wrote {out/'faithfulness_curves.csv'} and AUC summary ({len(auc_summary)} methods)")
    return summary


# -------------------------- AXIS 4: REPRESENTATION --------------------------

def attention_entropy(attn):
    # attn: [B, H, N, N]. Return entropy per layer-head-token, averaged over tokens.
    p = attn.clamp_min(1e-12)
    ent = -(p * p.log()).sum(-1)
    return ent.mean(-1)  # [B, H]


def axis4_representation(model, ds, sids, out_root, patch_size=16):
    device = next(model.parameters()).device
    out = ensure_dir(out_root / "axis4_representation")
    patch_timm_attention(model)
    model.eval()

    per_layer_rows = []
    sample_summary = []

    for sid in sids:
        if sid not in ds._index:
            continue
        item = ds[ds._index[sid]]
        image = item["image"]
        x = image.unsqueeze(0).to(device)
        with torch.no_grad():
            model(x)
        attns = [b.attn.last_attn.detach().cpu() for b in model.blocks if hasattr(b.attn, "last_attn")]
        if not attns:
            continue

        cls_concentrations = []
        ent_per_layer = []
        head_div_per_layer = []
        for li, a in enumerate(attns):
            # a: [1, H, N, N]
            ent = attention_entropy(a)[0]  # [H]
            ent_per_layer.append(float(ent.mean()))
            cls_to_patch = a[0, :, 0, 1:]  # [H, P]
            # head-averaged CLS attention concentration: top-10% mass
            cls_avg = cls_to_patch.mean(0)
            sorted_v, _ = cls_avg.sort(descending=True)
            k = max(1, int(0.1 * sorted_v.numel()))
            conc = float(sorted_v[:k].sum() / (sorted_v.sum() + 1e-12))
            cls_concentrations.append(conc)
            # head diversity: 1 - mean cosine similarity between heads on CLS-to-patch
            H = cls_to_patch.shape[0]
            cs = []
            for i in range(H):
                for j in range(i + 1, H):
                    a_i, a_j = cls_to_patch[i], cls_to_patch[j]
                    n = (a_i.norm() * a_j.norm()).clamp_min(1e-12)
                    cs.append(float((a_i @ a_j) / n))
            head_div_per_layer.append(1.0 - float(np.mean(cs)) if cs else 0.0)
            per_layer_rows.append({
                "sample_id": sid,
                "layer": li,
                "attn_entropy_mean": float(ent.mean()),
                "attn_entropy_std": float(ent.std()),
                "cls_top10_concentration": conc,
                "head_diversity": head_div_per_layer[-1],
            })

        sample_summary.append({
            "sample_id": sid,
            "mean_entropy": float(np.mean(ent_per_layer)),
            "last_layer_entropy": float(ent_per_layer[-1]),
            "mean_cls_concentration": float(np.mean(cls_concentrations)),
            "last_layer_cls_concentration": float(cls_concentrations[-1]),
            "mean_head_diversity": float(np.mean(head_div_per_layer)),
        })

    if not per_layer_rows:
        print("[Axis4] no samples processed.")
        return pd.DataFrame()

    pl = pd.DataFrame(per_layer_rows)
    pl.to_csv(out / "per_layer_metrics.csv", index=False)
    pd.DataFrame(sample_summary).to_csv(out / "per_sample_summary.csv", index=False)

    layer_agg = pl.groupby("layer").agg(
        attn_entropy=("attn_entropy_mean", "mean"),
        attn_entropy_std=("attn_entropy_mean", "std"),
        cls_concentration=("cls_top10_concentration", "mean"),
        head_diversity=("head_diversity", "mean"),
        n=("sample_id", "count"),
    ).reset_index()
    layer_agg.to_csv(out / "layer_aggregate.csv", index=False)

    fig_dir = ensure_dir(out_root.parents[2] / "figures" / "evaluation" / "axis4_representation")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(layer_agg["layer"], layer_agg["attn_entropy"],
                yerr=layer_agg["attn_entropy_std"], marker="o", capsize=3)
    ax.set_xlabel("Transformer layer"); ax.set_ylabel("Attention entropy (mean over heads/tokens)")
    ax.set_title("Axis 4: Attention entropy by layer")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(fig_dir / "entropy_by_layer.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layer_agg["layer"], layer_agg["cls_concentration"], marker="s", color="C1")
    ax.set_xlabel("Transformer layer"); ax.set_ylabel("CLS top-10% mass")
    ax.set_title("Axis 4: CLS-to-patch concentration by layer")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(fig_dir / "cls_concentration_by_layer.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layer_agg["layer"], layer_agg["head_diversity"], marker="^", color="C2")
    ax.set_xlabel("Transformer layer"); ax.set_ylabel("1 - mean(head cos-sim)")
    ax.set_title("Axis 4: Head diversity by layer")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(fig_dir / "head_diversity_by_layer.png", dpi=150); plt.close(fig)

    print(f"[Axis4] wrote {out/'layer_aggregate.csv'} ({len(layer_agg)} layers)")
    return layer_agg


# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/checkpoints/deit_pretrained/best_val_acc.pt")
    ap.add_argument("--checkpoint_name", default="pretrained_best")
    ap.add_argument("--config", default="configs/xai.yaml")
    ap.add_argument("--selected_samples", default="outputs/metrics/xai/xai_selected_samples.csv")
    ap.add_argument("--split_dir", default="data/splits")
    ap.add_argument("--explanations_dir", default="outputs/explanations")
    ap.add_argument("--metrics_root", default="outputs/metrics/xai/evaluation")
    ap.add_argument("--n_stability_samples", type=int, default=20)
    ap.add_argument("--n_repr_samples", type=int, default=40)
    ap.add_argument("--n_perturb", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--axes", nargs="+", default=["1", "2", "4"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device("auto")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = ckpt.get("config", {}).get("image_size", 224)

    ds = NICODataset(Path(args.split_dir) / "test.csv", transform=build_transforms("eval", image_size))
    ds._index = {str(r.sample_id): i for i, r in ds.df.iterrows()}

    model = build_model(
        ckpt["model_arch"],
        len(ds.class_to_idx),
        bool(ckpt.get("pretrained", ckpt.get("config", {}).get("pretrained", True))),
        ckpt.get("pretrained_cfg", ckpt.get("config", {}).get("pretrained_cfg", "fb_in1k")),
        args.checkpoint,
        str(device),
    )

    selected = pd.read_csv(args.selected_samples)
    all_sids = [s for s in selected["sample_id"].astype(str).tolist() if s in ds._index]

    out_root = ensure_dir(Path(args.metrics_root))
    print(f"Total selected samples available: {len(all_sids)}")

    if "1" in args.axes:
        sids = all_sids[: args.n_stability_samples]
        print(f"--- Axis 1: Stability ({len(sids)} samples, {args.n_perturb} perturbations) ---")
        axis1_stability(model, ds, sids, out_root, n_perturb=args.n_perturb,
                        sigma=args.sigma, patch_size=cfg.get("patch_size", 16))

    if "2" in args.axes:
        print("--- Axis 2: Faithfulness (aggregate from existing curves) ---")
        faith_dir = Path(args.explanations_dir) / args.checkpoint_name / "faithfulness"
        axis2_faithfulness(faith_dir, out_root)

    if "4" in args.axes:
        sids = all_sids[: args.n_repr_samples]
        print(f"--- Axis 4: Representation Behavior ({len(sids)} samples) ---")
        axis4_representation(model, ds, sids, out_root, patch_size=cfg.get("patch_size", 16))

    print("Done.")


if __name__ == "__main__":
    main()
