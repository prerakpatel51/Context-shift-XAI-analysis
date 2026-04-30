from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.datasets.nico import NICODataset
from src.datasets.transforms import build_transforms
from src.models.build_model import build_model
from src.models.checkpointing import checkpoint_payload, copy_checkpoint, save_checkpoint
from src.training.logger import MetricLogger
from src.training.losses import build_loss
from src.training.metrics import classification_metrics, predict_epoch
from src.training.seed import seed_everything
from src.utils.device import get_device
from src.utils.io import ensure_dir


def _optimizer(model, cfg):
    head_params, body_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if "head" in name else body_params).append(p)
    return torch.optim.AdamW([
        {"params": body_params, "lr": float(cfg.get("lr_backbone", cfg.get("lr", 3e-4)))},
        {"params": head_params, "lr": float(cfg.get("lr_head", cfg.get("lr", 3e-4)))},
    ], weight_decay=float(cfg.get("weight_decay", 0.05)))


def _scheduler(opt, cfg, steps_per_epoch):
    epochs = int(cfg.get("epochs", 1))
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps_per_epoch))


def run_training(cfg: dict, data_root: str, split_dir: str, output_dir: str = "outputs", smoke_test: bool = False):
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    split_dir = Path(split_dir)
    image_size = int(cfg.get("image_size", 224))
    train_ds = NICODataset(split_dir / "train.csv", transform=build_transforms("train", image_size, cfg.get("augmentation", "strong")))
    val_ds = NICODataset(split_dir / "val.csv", transform=build_transforms("val", image_size, cfg.get("augmentation", "strong")))
    test_path = split_dir / "test.csv"
    cfg["val_split_path"] = str(split_dir / "val.csv")
    cfg["test_split_path"] = str(test_path)
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(16, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(16, len(val_ds)))))
        cfg["epochs"] = min(int(cfg.get("epochs", 2)), 2)
        cfg["batch_size"] = min(int(cfg.get("batch_size", 8)), 8)
        meta_ds = train_ds.dataset
    else:
        meta_ds = train_ds
    num_classes = len(meta_ds.class_to_idx)
    cfg["num_classes"] = num_classes
    model = build_model(cfg["model_arch"], num_classes, bool(cfg.get("pretrained", True)), cfg.get("pretrained_cfg"), cfg.get("resume_from_checkpoint"), str(device))
    criterion = build_loss()
    opt = _optimizer(model, cfg)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 64)), shuffle=True, num_workers=int(cfg.get("num_workers", 4)), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 64)), shuffle=False, num_workers=int(cfg.get("num_workers", 4)), pin_memory=True)
    sched = _scheduler(opt, cfg, max(1, len(train_loader)))
    run_name = cfg.get("run_name", "run")
    ckpt_dir = ensure_dir(Path(output_dir) / "checkpoints" / run_name)
    logger = MetricLogger(Path(output_dir) / "logs" / run_name)
    best = {"val_acc": -1, "val_macro_f1": -1, "val_loss": float("inf"), "val_worst_context_acc": -1}
    metrics_so_far = []
    for epoch in range(1, int(cfg.get("epochs", 1)) + 1):
        start = time.time()
        model.train()
        loss_sum, ys, preds, ctxs, groups = 0.0, [], [], [], []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += float(loss.item()) * len(y)
            p = logits.argmax(1)
            ys.extend(y.detach().cpu().tolist())
            preds.extend(p.detach().cpu().tolist())
            ctxs.extend(batch["context"].cpu().tolist() if torch.is_tensor(batch["context"]) else batch["context"])
            groups.extend(batch["group_id"])
        train_m = classification_metrics(ys, preds, ctxs, groups)
        val_m, _ = predict_epoch(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, len(ys)),
            "val_loss": val_m["loss"],
            "train_accuracy": train_m["accuracy"],
            "val_accuracy": val_m["accuracy"],
            "train_macro_f1": train_m["macro_f1"],
            "val_macro_f1": val_m["macro_f1"],
            "train_balanced_accuracy": train_m["balanced_accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "train_worst_context_accuracy": train_m.get("worst_context_accuracy"),
            "val_worst_context_accuracy": val_m.get("worst_context_accuracy"),
            "train_worst_group_accuracy": train_m.get("worst_group_accuracy"),
            "val_worst_group_accuracy": val_m.get("worst_group_accuracy"),
            "per_class_accuracy": val_m.get("per_class_accuracy"),
            "per_context_accuracy": val_m.get("per_context_accuracy"),
            "learning_rate": opt.param_groups[0]["lr"],
            "epoch_time": time.time() - start,
            "gpu_memory_used": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        }
        logger.log(row)
        metrics_so_far.append(row)
        payload = checkpoint_payload(model, opt, sched, epoch, cfg, meta_ds, metrics_so_far, best)
        epoch_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        if cfg.get("save_every_epoch", True):
            save_checkpoint(payload, epoch_path)
        save_checkpoint(payload, ckpt_dir / "last.pt")
        if row["val_accuracy"] > best["val_acc"]:
            best["val_acc"] = row["val_accuracy"]; copy_checkpoint(ckpt_dir / "last.pt", ckpt_dir / "best_val_acc.pt")
        if row["val_macro_f1"] > best["val_macro_f1"]:
            best["val_macro_f1"] = row["val_macro_f1"]; copy_checkpoint(ckpt_dir / "last.pt", ckpt_dir / "best_val_macro_f1.pt")
        if row["val_loss"] < best["val_loss"]:
            best["val_loss"] = row["val_loss"]; copy_checkpoint(ckpt_dir / "last.pt", ckpt_dir / "best_val_loss.pt")
        if row.get("val_worst_context_accuracy") is not None and row["val_worst_context_accuracy"] > best["val_worst_context_acc"]:
            best["val_worst_context_acc"] = row["val_worst_context_accuracy"]; copy_checkpoint(ckpt_dir / "last.pt", ckpt_dir / "best_val_worst_context_acc.pt")
    copy_checkpoint(ckpt_dir / "last.pt", ckpt_dir / "final.pt")
    return ckpt_dir
