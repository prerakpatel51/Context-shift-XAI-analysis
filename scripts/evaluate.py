#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.nico import NICODataset
from src.datasets.transforms import build_transforms
from src.models.build_model import build_model
from src.training.losses import build_loss
from src.training.metrics import predict_epoch
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.io import ensure_dir, write_json


def evaluate_checkpoint(checkpoint, config, split_dir, split, output_dir, data_root=None):
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = {**config, **ckpt.get("config", {})}
    device = get_device(cfg.get("device", "auto"))
    ds = NICODataset(Path(split_dir) / f"{split}.csv", transform=build_transforms("eval", int(cfg.get("image_size", 224))))
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 64)), shuffle=False, num_workers=int(cfg.get("num_workers", 4)))
    model = build_model(
        ckpt.get("model_arch", cfg["model_arch"]),
        len(ds.class_to_idx),
        bool(ckpt.get("pretrained", cfg.get("pretrained", True))),
        ckpt.get("pretrained_cfg", cfg.get("pretrained_cfg", "fb_in1k")),
        checkpoint,
        str(device),
    )
    metrics, pred = predict_epoch(model, loader, device, build_loss())
    out = ensure_dir(output_dir)
    write_json(metrics, out / "test_metrics.json")
    probs = np.asarray(pred["probs"])
    rows = []
    for i, sid in enumerate(pred["sample_ids"]):
        top = probs[i].argsort()[-5:][::-1]
        rows.append({
            "sample_id": sid, "image_path": pred["paths"][i], "y_true": pred["y_true"][i], "y_pred": pred["y_pred"][i],
            "class_name_true": ds.idx_to_class[pred["y_true"][i]], "class_name_pred": ds.idx_to_class[pred["y_pred"][i]],
            "context_id": pred["contexts"][i], "context_name": ds.idx_to_context.get(pred["contexts"][i], None),
            "group_id": pred["groups"][i], "split": split, "confidence": float(probs[i].max()),
            "correct": pred["y_true"][i] == pred["y_pred"][i],
            "top5_predictions": " ".join(map(str, top.tolist())),
            "top5_probabilities": " ".join(map(lambda x: f"{x:.6f}", probs[i][top].tolist())),
        })
    pd.DataFrame(rows).to_csv(out / "test_predictions.csv", index=False)
    (out / "classification_report.txt").write_text(classification_report(pred["y_true"], pred["y_pred"], zero_division=0), encoding="utf-8")
    np.save(out / "confusion_matrix.npy", confusion_matrix(pred["y_true"], pred["y_pred"]))
    if metrics.get("per_context_accuracy"):
        pd.DataFrame([{"context_id": k, "accuracy": v} for k, v in metrics["per_context_accuracy"].items()]).to_csv(out / "context_accuracy.csv", index=False)
    if metrics.get("per_group_accuracy"):
        pd.DataFrame([{"group_id": k, "accuracy": v} for k, v in metrics["per_group_accuracy"].items()]).to_csv(out / "group_accuracy.csv", index=False)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", default="data/processed/nico")
    ap.add_argument("--split_dir", default="data/splits")
    ap.add_argument("--split", default="test")
    ap.add_argument("--output_dir", default="outputs/metrics/eval")
    ap.add_argument("--eval_all_checkpoints")
    ap.add_argument("--device")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.eval_all_checkpoints:
        rows = []
        for ckpt in sorted(Path(args.eval_all_checkpoints).glob("epoch_*.pt")):
            m = evaluate_checkpoint(ckpt, cfg, args.split_dir, args.split, Path(args.output_dir) / ckpt.stem, args.data_root)
            rows.append({"checkpoint": str(ckpt), **{k: v for k, v in m.items() if not isinstance(v, dict)}})
        pd.DataFrame(rows).to_csv(Path(args.output_dir) / f"{Path(args.eval_all_checkpoints).name}_all_checkpoint_metrics.csv", index=False)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --eval_all_checkpoints is used")
        evaluate_checkpoint(args.checkpoint, cfg, args.split_dir, args.split, args.output_dir, args.data_root)


if __name__ == "__main__":
    main()
