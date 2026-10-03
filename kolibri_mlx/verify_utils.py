"""Shared helpers for the verification scripts."""

import json
import os

import mlx.core as mx
import numpy as np

from mlx_lm.models.base import create_attention_mask

DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32}


def load_config(path=None, checkpoint=None):
    if path:
        with open(path) as f:
            return json.load(f)
    from kolibri_mlx.checkpoint import Checkpoint

    return Checkpoint(checkpoint) if checkpoint else Checkpoint().config


def require(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("FAIL: missing reference file(s):\n  " + "\n  ".join(missing))


def to_mx(t, dtype):
    """torch tensor or numpy/mx array -> mx array of ``dtype``."""
    if hasattr(t, "detach"):
        t = t.detach().float().cpu().numpy()
    return mx.array(t).astype(dtype)


def hf_layer_to_mlx(weights, layer_idx, num_experts, dtype=mx.bfloat16):
    """HF-named layer tensors (relative to ``model.layers.i.``) -> MLX-named
    mx arrays (relative). Stacks experts into switch_mlp, router bias -> fp32."""
    w = dict(weights)
    out = {}
    bias = w.pop("moe.router.expert_bias", None)
    if bias is not None:
        out["mlp.expert_bias"] = to_mx(bias, mx.float32)
    for n in ("up_proj", "down_proj", "gate_proj"):
        keys = [f"mlp.experts.{e}.{n}.weight" for e in range(num_experts)]
        if keys[0] in w:
            out[f"mlp.switch_mlp.{n}.weight"] = mx.stack(
                [to_mx(w.pop(k), dtype) for k in keys]
            )
    for k, v in w.items():
        out[k] = to_mx(v, dtype)
    return out


def mlx_dir_layer(model_dir, layer_idx, dtype):
    """Layer weights (relative MLX names) from an mlx-lm model directory."""
    import glob

    prefix = f"model.layers.{layer_idx}."
    out = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        for k, v in mx.load(f).items():
            if k.startswith(prefix):
                k = k[len(prefix):]
                out[k] = v if k == "mlp.expert_bias" else v.astype(dtype)
    if not out:
        raise SystemExit(f"FAIL: no weights for layer {layer_idx} in {model_dir}")
    return out


def layer_masks(h, args, layer_idx):
    if args.layer_types[layer_idx] == "sliding_attention":
        return create_attention_mask(h, None, window_size=args.sliding_window)
    return create_attention_mask(h, None)


def route_inds(layer, h, mask):
    """Top-k expert ids [T,k] (sorted) the layer's router picks for input h."""
    r = layer.self_attn(layer.input_layernorm(h), mask, None)
    h2 = h + layer.post_attn_norm(r)
    x = layer.post_attention_layernorm(h2)
    moe = layer.mlp
    logits = moe.gate(x.astype(mx.float32)).astype(mx.float32)
    biased = logits + moe.expert_bias.astype(mx.float32)
    k = moe.top_k
    inds = mx.argpartition(biased, kth=-k, axis=-1)[..., -k:]
    return mx.sort(inds, axis=-1)
