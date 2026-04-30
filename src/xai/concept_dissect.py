from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.utils.io import ensure_dir

_CLIP_CACHE = {}


def _denorm_np(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().permute(1, 2, 0).numpy()
    arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(arr, 0.0, 1.0)


def _visual_proxy_scores(arr: np.ndarray, concepts: list[str]) -> dict[str, float]:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness = arr.mean(axis=-1)
    saturation = arr.max(axis=-1) - arr.min(axis=-1)
    grayness = 1.0 - saturation
    green = np.clip(g - np.maximum(r, b), 0, 1)
    blue = np.clip(b - np.maximum(r, g), 0, 1)
    yellow = np.clip((r + g) / 2 - b, 0, 1)
    brown = np.clip((r * 0.6 + g * 0.35) - b, 0, 1) * np.clip(0.75 - brightness, 0, 1)
    edge_y = np.abs(np.diff(brightness, axis=0)).mean() if brightness.shape[0] > 1 else 0.0
    edge_x = np.abs(np.diff(brightness, axis=1)).mean() if brightness.shape[1] > 1 else 0.0
    texture = float(edge_x + edge_y)

    proxy = {
        "grass": float(green.mean() + 0.35 * saturation.mean()),
        "forest": float(green.mean() * (1.0 - brightness.mean()) + texture),
        "field": float((green.mean() + yellow.mean()) / 2),
        "water": float(blue.mean() + 0.2 * saturation.mean()),
        "sky": float(blue.mean() + 0.3 * brightness.mean()),
        "snow": float(brightness.mean() * grayness.mean()),
        "road": float(grayness.mean() * (1.0 - brightness.mean()) + texture),
        "street": float(grayness.mean() * (1.0 - brightness.mean()) + texture),
        "rock": float((grayness.mean() + brown.mean()) / 2 + texture),
        "mountain": float((grayness.mean() + brown.mean()) / 2 + 0.5 * texture),
        "beach": float(yellow.mean() + 0.2 * brightness.mean()),
        "outdoor": float(saturation.mean() + brightness.mean()),
        "indoor": float(grayness.mean() * (1.0 - saturation.mean())),
        "vehicle": float(grayness.mean() + texture),
        "animal": float(brown.mean() + texture),
        "bird": float(texture + blue.mean() * 0.25 + brown.mean() * 0.25),
        "dog": float(brown.mean() + texture),
        "cat": float(brown.mean() + grayness.mean() * 0.25 + texture),
        "person": float(texture + saturation.mean() * 0.25),
    }
    vals = {c: max(0.0, float(proxy.get(c, texture))) for c in concepts}
    max_v = max(vals.values()) if vals else 1.0
    if max_v > 0:
        vals = {c: v / max_v for c, v in vals.items()}
    return vals


def _patch_rows(arr: np.ndarray, sample_id, concepts: list[str], patch_size: int = 32, top_k: int = 20):
    rows = []
    h, w = arr.shape[:2]
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = arr[y:min(y + patch_size, h), x:min(x + patch_size, w)]
            scores = _visual_proxy_scores(patch, concepts)
            concept, score = max(scores.items(), key=lambda kv: kv[1])
            rows.append({
                "sample_id": sample_id,
                "patch_idx": len(rows),
                "row": y // patch_size,
                "col": x // patch_size,
                "concept": concept,
                "similarity": score,
                "method": "visual_proxy",
            })
    return sorted(rows, key=lambda r: r["similarity"], reverse=True)[:top_k]


def _pil_from_tensor(image: torch.Tensor) -> Image.Image:
    arr = (_denorm_np(image) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _resolve_clip_path(model_name: str) -> str:
    """Resolve a CLIP model identifier to a local path when possible.

    Accepts:
      - direct local dir containing config.json
      - a HF cache root like ".../models--openai--clip-vit-base-patch32"
        (auto-pick a snapshot containing config.json)
      - a snapshots/<sha> path
      - a HF repo id (returned unchanged)
    """
    p = Path(model_name)
    if p.is_dir() and (p / "config.json").exists():
        return str(p)
    if p.is_dir() and (p / "snapshots").is_dir():
        for snap in sorted((p / "snapshots").iterdir()):
            if (snap / "config.json").exists():
                return str(snap)
    return model_name


def _load_clip(model_name: str, device: str):
    key = (model_name, device)
    if key not in _CLIP_CACHE:
        import os
        from transformers import CLIPModel, CLIPProcessor

        resolved = _resolve_clip_path(model_name)
        is_local = Path(resolved).is_dir() and (Path(resolved) / "config.json").exists()
        if is_local:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            model = CLIPModel.from_pretrained(resolved, local_files_only=is_local).to(device)
            processor = CLIPProcessor.from_pretrained(resolved, local_files_only=is_local)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CLIP from '{model_name}' (resolved='{resolved}', is_local={is_local}). "
                f"If on an offline node, point cfg.clip_model at a snapshot dir containing config.json. {e}"
            ) from e
        model.eval()
        _CLIP_CACHE[key] = (model, processor)
    return _CLIP_CACHE[key]


def _clip_scores(image_pil: Image.Image, concepts: list[str], cfg: dict):
    model_name = cfg.get("clip_model", "openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = _load_clip(model_name, device)
    prompts = [f"a photo of {c}" for c in concepts]
    inputs = processor(text=prompts, images=image_pil, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
        image_features = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_features = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sims = (image_features @ text_features.T)[0].detach().cpu().numpy()
    return {c: float(sims[i]) for i, c in enumerate(concepts)}


def _clip_patch_rows(image_pil: Image.Image, sample_id, concepts: list[str], cfg: dict):
    patch_size = int(cfg.get("patch_size", 32))
    top_k = int(cfg.get("top_patch_concepts", 20))
    model_name = cfg.get("clip_model", "openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = _load_clip(model_name, device)
    prompts = [f"a photo of {c}" for c in concepts]
    patches, meta = [], []
    width, height = image_pil.size
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patches.append(image_pil.crop((x, y, min(x + patch_size, width), min(y + patch_size, height))))
            meta.append((len(meta), y // patch_size, x // patch_size))
    inputs = processor(text=prompts, images=patches, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
        image_features = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_features = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sims = (image_features @ text_features.T).detach().cpu().numpy()
    rows = []
    for patch_idx, row, col in meta:
        best = int(sims[patch_idx].argmax())
        rows.append({
            "sample_id": sample_id,
            "patch_idx": patch_idx,
            "row": row,
            "col": col,
            "concept": concepts[best],
            "similarity": float(sims[patch_idx, best]),
            "method": "clip",
        })
    return sorted(rows, key=lambda r: r["similarity"], reverse=True)[:top_k]


def concept_dissect(image, sample_id, out_dir, concepts, cfg=None):
    out = ensure_dir(Path(out_dir) / "concept_dissect")
    cfg = cfg or {}
    require_clip = bool(cfg.get("require_clip", True))
    image_pil = _pil_from_tensor(image)
    try:
        clip_scores = _clip_scores(image_pil, list(concepts), cfg)
        rows = [{
            "sample_id": sample_id,
            "concept": c,
            "similarity": float(clip_scores[c]),
            "method": "clip",
            "clip_model": cfg.get("clip_model", "openai/clip-vit-base-patch32"),
        } for c in concepts]
        rows = sorted(rows, key=lambda r: r["similarity"], reverse=True)
        patch_rows = _clip_patch_rows(image_pil, sample_id, list(concepts), cfg)
    except Exception as e:
        if require_clip:
            raise RuntimeError(f"CLIP concept dissection failed; refusing to write proxy scores. {e}") from e
        arr = _denorm_np(image)
        proxy_scores = _visual_proxy_scores(arr, list(concepts))
        rows = [{
            "sample_id": sample_id,
            "concept": c,
            "similarity": float(proxy_scores[c]),
            "method": "visual_proxy",
            "clip_model": None,
        } for c in concepts]
        rows = sorted(rows, key=lambda r: r["similarity"], reverse=True)
        patch_rows = _patch_rows(arr, sample_id, list(concepts))
    pd.DataFrame(rows).to_csv(out / f"{sample_id}_concept_scores.csv", index=False)
    pd.DataFrame(patch_rows).to_csv(out / f"{sample_id}_top_patch_concepts.csv", index=False)
