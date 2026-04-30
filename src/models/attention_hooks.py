from __future__ import annotations

import types

import torch


def patch_timm_attention(model):
    """Patch timm ViT attention modules to retain attention and gradients."""
    patched = []
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is None or getattr(attn, "_xai_patched", False):
            continue

        def forward(self, x, attn_mask=None, is_causal=False):
            if attn_mask is not None or is_causal:
                raise NotImplementedError("XAI attention hooks do not support masked or causal ViT attention.")
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            attn_mat = (q @ k.transpose(-2, -1)) * self.scale
            attn_mat = attn_mat.softmax(dim=-1)
            self.last_attn = attn_mat
            if attn_mat.requires_grad:
                attn_mat.retain_grad()
            x = self.attn_drop(attn_mat) @ v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        attn.forward = types.MethodType(forward, attn)
        attn._xai_patched = True
        patched.append(attn)
    return patched


def get_attentions(model):
    return [b.attn.last_attn.detach().cpu() for b in model.blocks if hasattr(b.attn, "last_attn")]


def get_attention_grads(model):
    grads = []
    for b in model.blocks:
        a = getattr(b.attn, "last_attn", None)
        if a is not None and a.grad is not None:
            grads.append(a.grad.detach().cpu())
    return grads
