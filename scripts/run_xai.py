#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.nico import NICODataset
from src.datasets.transforms import build_transforms
from src.models.build_model import build_model
from src.utils.config import load_config
from src.utils.device import get_device
from src.xai.attention_rollout import attention_rollout
from src.xai.attention_visualization import attention_visualization
from src.xai.causal_patch_impact import causal_patch_impact
from src.xai.concept_dissect import concept_dissect
from src.xai.context_swap import context_swap_attribution
from src.xai.counterfactual_perturbation import counterfactual
from src.xai.faithfulness_topk import faithfulness_topk
from src.xai.method_agreement import method_agreement
from src.xai.occlusion_sensitivity import occlusion_sensitivity
from src.xai.patch_attribution import gradient_x_input
from src.xai.patch_interaction import patch_interaction
from src.xai.shap_patch import kernel_shap_patch
from src.xai.token_transformation import token_transformation_attribution
from src.xai.transition_attention_maps import tam_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--checkpoint_name", required=True)
    ap.add_argument("--config", default="configs/xai.yaml")
    ap.add_argument("--selected_samples", required=True)
    ap.add_argument("--data_root", default="data/processed/nico")
    ap.add_argument("--split_dir", default="data/splits")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--methods", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    methods = args.methods or cfg.get("methods", [])
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    device = get_device("auto")
    ds = NICODataset(Path(args.split_dir) / "test.csv", transform=build_transforms("eval", ckpt.get("config", {}).get("image_size", 224)))
    model = build_model(
        ckpt["model_arch"],
        len(ds.class_to_idx),
        bool(ckpt.get("pretrained", ckpt.get("config", {}).get("pretrained", True))),
        ckpt.get("pretrained_cfg", ckpt.get("config", {}).get("pretrained_cfg", "fb_in1k")),
        args.checkpoint,
        str(device),
    )
    selected = pd.read_csv(args.selected_samples).head(int(cfg.get("num_samples", 100)))
    wanted = set(selected["sample_id"].astype(str))
    index = {str(r.sample_id): i for i, r in ds.df.iterrows()}
    context_swap_rows = []
    patch_size = cfg.get("patch_size", 16)
    topk_values = cfg.get("topk_values", [0.1])
    do_faith = "faithfulness_topk" in methods

    for sid in wanted:
        if sid not in index:
            continue
        item = ds[index[sid]]
        image, label = item["image"], item["label"]
        ctx = ds.idx_to_context.get(item["context"], None)
        heats = {}

        if "token_transformation" in methods:
            heats["token_transformation"] = token_transformation_attribution(model, image, sid, args.output_dir)
        if "occlusion" in methods:
            heats["occlusion"] = occlusion_sensitivity(model, image, label, sid, args.output_dir, cfg.get("occlusion_window_size", 16), cfg.get("occlusion_stride", 16), cfg.get("occlusion_baseline", "mean"), ctx)
        if "attention_rollout" in methods:
            heats["attention_rollout"] = attention_rollout(model, image, sid, args.output_dir)
        if "patch_attribution" in methods:
            grid = gradient_x_input(model, image, label, sid, args.output_dir, patch_size, output_mode="both")
            if grid.ndim == 2 and grid.shape == (224 // patch_size, 224 // patch_size):
                heats["patch_attribution"] = grid
            else:
                from src.xai.method_agreement import _resize_to
                heats["patch_attribution"] = _resize_to(grid, (224 // patch_size, 224 // patch_size))
        if "causal_patch_impact" in methods:
            heats["causal_patch_impact"] = causal_patch_impact(model, image, label, sid, args.output_dir, patch_size, ctx)
        if "tam" in methods:
            heats["tam"] = tam_map(model, image, label, sid, args.output_dir)
        if "shap" in methods:
            shap_cfg = cfg.get("shap", {})
            heats["shap"] = kernel_shap_patch(
                model, image, label, sid, args.output_dir,
                patch_size=patch_size,
                n_samples=int(shap_cfg.get("n_samples", 512)),
                baseline=shap_cfg.get("baseline", "mean"),
                seed=int(shap_cfg.get("seed", 0)),
            )

        # Sanity-check / non-heatmap methods
        if "attention_weights" in methods:
            attention_visualization(model, image, sid, args.output_dir)
        if "patch_interaction" in methods:
            patch_interaction(model, image, sid, args.output_dir, cfg.get("patch_interaction", {}).get("top_edges", 30))
        if "concept_dissect" in methods:
            concept_cfg = cfg.get("concept_dissect", {})
            concept_dissect(image, sid, args.output_dir, concept_cfg.get("concept_dict", []), concept_cfg)

        # Per-method faithfulness on each method's OWN heat
        if do_faith:
            for mname, h in heats.items():
                if h is None:
                    continue
                faithfulness_topk(model, image, label, h, sid, args.output_dir, topk_values, patch_size, method_name=mname)

        # Cross-method agreement matrix
        if heats:
            method_agreement(heats, sid, args.output_dir, topk_frac=cfg.get("agreement_topk_frac", 0.1))

        # Pick a heat for context_swap / counterfactual (prefer token_transformation)
        primary_heat = None
        for pref in ("token_transformation", "tam", "attention_rollout", "occlusion", "shap"):
            if heats.get(pref) is not None:
                primary_heat = heats[pref]
                break
        if primary_heat is None and heats:
            primary_heat = next(iter(heats.values()))

        if "context_swap" in methods:
            if primary_heat is None:
                primary_heat = token_transformation_attribution(model, image, sid, args.output_dir)
            row = context_swap_attribution(model, ds, index, sid, image, label, primary_heat, args.output_dir, cfg)
            if row:
                context_swap_rows.append(row)

        if "counterfactual" in methods:
            counterfactual(
                model, image, label, sid, args.output_dir,
                steps=min(cfg.get("counterfactual_steps", 300), 300),
                lr=cfg.get("counterfactual_lr", 0.03),
                lambda_l2=cfg.get("lambda_l2", 0.01),
                lambda_tv=cfg.get("lambda_tv", 0.001),
                lambda_l1=cfg.get("lambda_l1", 0.001),
                lambda_linf=cfg.get("lambda_linf", 0.0),
                linf_budget=cfg.get("linf_budget", None),
                heat=primary_heat,
                patch_size=patch_size,
                semantic_top_fraction=cfg.get("counterfactual_semantic_top_fraction", 0.0),
            )

    if context_swap_rows:
        out = Path(args.output_dir) / "context_swap"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(context_swap_rows).to_csv(out / "context_swap_summary.csv", index=False)


if __name__ == "__main__":
    main()
