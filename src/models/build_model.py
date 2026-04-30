from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import timm


def _model_name(model_arch: str, pretrained: bool, pretrained_cfg: Optional[str]) -> str:
    if pretrained_cfg and "." not in model_arch:
        return f"{model_arch}.{pretrained_cfg}"
    return model_arch


def build_model(
    model_arch: str,
    num_classes: int,
    pretrained: bool,
    pretrained_cfg: Optional[str] = "fb_in1k",
    resume_from_checkpoint: Optional[str] = None,
    device: str = "cuda",
    allow_head_mismatch: bool = False,
):
    if not pretrained and not resume_from_checkpoint:
        raise ValueError("This project is configured for ImageNet-pretrained DeiT only. Set pretrained: true.")
    model_name = _model_name(model_arch, pretrained, pretrained_cfg)
    if resume_from_checkpoint:
        ckpt = torch.load(resume_from_checkpoint, map_location="cpu")
        ckpt_arch = ckpt.get("model_arch", model_arch)
        if ckpt_arch != model_arch:
            raise ValueError(f"Checkpoint arch {ckpt_arch} does not match requested arch {model_arch}")
        model = timm.create_model(model_arch, pretrained=False, num_classes=num_classes)
        state = ckpt["model_state_dict"]
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            if not allow_head_mismatch:
                raise RuntimeError(f"Checkpoint head shape mismatch. Pass allow_head_mismatch=True to reinit head. {e}") from e
            current = model.state_dict()
            filtered = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
            model.load_state_dict(filtered, strict=False)
        print(f"Loaded checkpoint weights from {resume_from_checkpoint}.")
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        print(f"Loaded ImageNet-pretrained weights for {model_arch}.")
    return model.to(device)
