"""Pure-PyTorch fp32 reference of the Kolibri 1 forward pass (ground truth for the MLX port).

layer_w is keyed relative to ``model.layers.N.`` (see Checkpoint.layer_tensors).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rms_norm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos, theta=10000.0):
    """NeoX rotate-half RoPE over the full head dim. x: [T, H, D], pos: [T]."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    freqs = pos.to(torch.float32)[:, None] * inv[None, :]
    cos, sin = freqs.cos()[:, None, :], freqs.sin()[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def attn_mask(T, layer_type, window):
    """Bool [T,T]; True = allowed. Sliding: j <= i and i - j < window."""
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    m = j <= i
    if layer_type == "sliding_attention":
        m = m & (i - j <= window - 1)
    return m


def attention(x, w, layer_type, cfg):
    T = x.shape[0]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    q = (x @ w["self_attn.q_proj.weight"].T).view(T, nh, hd)
    k = (x @ w["self_attn.k_proj.weight"].T).view(T, nkv, hd)
    v = (x @ w["self_attn.v_proj.weight"].T).view(T, nkv, hd)
    q = rms_norm(q, w["self_attn.q_norm.weight"], eps)
    k = rms_norm(k, w["self_attn.k_norm.weight"], eps)
    if layer_type == "sliding_attention":
        pos = torch.arange(T)
        q, k = rope(q, pos, cfg["rope_theta"]), rope(k, pos, cfg["rope_theta"])
    rep = nh // nkv
    k = k.repeat_interleave(rep, dim=1)  # kv head for q head h is h // rep
    v = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("ihd,jhd->hij", q, k) * hd**-0.5
    mask = attn_mask(T, layer_type, cfg["sliding_window"])
    scores = scores.masked_fill(~mask[None], float("-inf"))
    attn = torch.einsum("hij,jhd->ihd", torch.softmax(scores, dim=-1), v)
    return attn.reshape(T, nh * hd) @ w["self_attn.o_proj.weight"].T


def route(logits, bias, k):
    """Top-k on logits+bias; weights = unbiased sigmoid(logits), not renormalised."""
    ids = torch.topk(logits + bias, k, dim=-1, sorted=True).indices
    return ids, torch.sigmoid(logits.gather(1, ids))


def _mlp(x, wg, wu, wd):
    return (F.silu(x @ wg.T) * (x @ wu.T)) @ wd.T


def moe(x, w, cfg):
    k = cfg["num_experts_per_tok"]
    logits = x @ w["mlp.gate.weight"].T
    ids, wts = route(logits, w["moe.router.expert_bias"], k)
    routed = torch.zeros_like(x)
    for e in torch.unique(ids).tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        y = _mlp(
            x[tok],
            w[f"mlp.experts.{e}.gate_proj.weight"],
            w[f"mlp.experts.{e}.up_proj.weight"],
            w[f"mlp.experts.{e}.down_proj.weight"],
        )
        routed.index_add_(0, tok, y * wts[tok, slot][:, None])
    shared = _mlp(
        x,
        w["mlp.shared_experts.gate_proj.weight"],
        w["mlp.shared_experts.up_proj.weight"],
        w["mlp.shared_experts.down_proj.weight"],
    )
    return routed + shared, ids, wts


def decoder_layer(h, w, layer_type, cfg):
    eps = cfg["rms_norm_eps"]
    a = attention(rms_norm(h, w["input_layernorm.weight"], eps), w, layer_type, cfg)
    h = h + rms_norm(a, w["post_attn_norm.weight"], eps)
    y, ids, wts = moe(rms_norm(h, w["post_attention_layernorm.weight"], eps), w, cfg)
    h = h + rms_norm(y, w["post_ffn_norm.weight"], eps)
    return h, ids, wts


def embed(ids, w):
    return w[ids]


def head(h, norm_w, lm_head_w, eps=1e-6):
    return rms_norm(h, norm_w, eps) @ lm_head_w.T


def forward(ids, weights, cfg):
    """Full forward from a flat dict with HF names. ids: [T] int64 -> logits [T, V]."""
    h = embed(ids, weights["model.embed_tokens.weight"])
    for i, lt in enumerate(cfg["layer_types"][: cfg["num_hidden_layers"]]):
        p = f"model.layers.{i}."
        w = {k[len(p):]: v for k, v in weights.items() if k.startswith(p)}
        h, _, _ = decoder_layer(h, w, lt, cfg)
    return head(h, weights["model.norm.weight"], weights["lm_head.weight"], cfg["rms_norm_eps"])
