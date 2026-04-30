#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trainer import run_training
from src.utils.config import load_config, merge_overrides, str2bool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", default="data/processed/nico")
    ap.add_argument("--split_dir", default="data/splits")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--run_name")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume_from_checkpoint")
    ap.add_argument("--pretrained", type=str2bool)
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr_backbone", type=float)
    ap.add_argument("--lr_head", type=float)
    ap.add_argument("--num_workers", type=int)
    ap.add_argument("--device")
    ap.add_argument("--save_every_epoch", type=str2bool)
    args = ap.parse_args()
    cfg = merge_overrides(load_config(args.config), run_name=args.run_name, resume_from_checkpoint=args.resume_from_checkpoint,
                          pretrained=args.pretrained, epochs=args.epochs, batch_size=args.batch_size,
                          lr_backbone=args.lr_backbone, lr_head=args.lr_head, num_workers=args.num_workers,
                          device=args.device, save_every_epoch=args.save_every_epoch)
    if args.resume and not cfg.get("resume_from_checkpoint"):
        cfg["resume_from_checkpoint"] = str(Path(args.output_dir) / "checkpoints" / cfg.get("run_name", "run") / "last.pt")
    run_training(cfg, args.data_root, args.split_dir, args.output_dir, args.smoke_test)


if __name__ == "__main__":
    main()
