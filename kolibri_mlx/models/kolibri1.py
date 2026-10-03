# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT
# Derived from mlx-lm: mlx_lm/models/qwen3_moe.py, cohere2.py and deepseek_v3.py.
# Intended for upstream contribution to mlx-lm.

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    sliding_window: int
    layer_types: List[str]
    norm_topk_prob: bool = False
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 262144

    def __post_init__(self):
        if self.layer_types is None:
            raise ValueError("kolibri1 needs layer_types in config.json")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have one entry per layer")


class Attention(nn.Module):
    """GQA with per-head q/k RMSNorm. Sliding-window layers use RoPE; full
    attention layers have no positional encoding."""

    def __init__(self, args: ModelArgs, use_sliding_window: bool):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim**-0.5
        self.use_sliding_window = use_sliding_window

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        if use_sliding_window:
            self.rope = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if self.use_sliding_window:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Kolibri1SparseMoeBlock(nn.Module):
    """Top-k is chosen on fp32 router logits plus a per-expert bias; the
    weights are the unbiased sigmoid of the chosen logits. One ungated shared
    expert is added to the routed sum."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.expert_bias = mx.zeros((num_experts,), dtype=mx.float32)
        self.switch_mlp = SwitchGLU(dim, args.moe_intermediate_size, num_experts)
        self.shared_experts = MLP(dim, args.shared_expert_intermediate_size)

        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        # Router logits in fp32: top-k on logits + bias is sensitive to rounding.
        logits = self.gate(x.astype(mx.float32)).astype(mx.float32)
        biased = logits + self.expert_bias.astype(mx.float32)

        k = self.top_k
        inds = mx.argpartition(biased, kth=-k, axis=-1)[..., -k:]
        inds = mx.stop_gradient(inds)
        scores = mx.sigmoid(mx.take_along_axis(logits, inds, axis=-1))
        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class Kolibri1DecoderLayer(nn.Module):
    """Pre-norm block with an extra norm on each sub-layer output."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        use_sliding_window = args.layer_types[layer_idx] == "sliding_attention"
        self.self_attn = Attention(args, use_sliding_window)
        self.mlp = Kolibri1SparseMoeBlock(args)

        eps = args.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attn_norm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + self.post_attn_norm(r)
        r = self.mlp(self.post_attention_layernorm(h))
        return h + self.post_ffn_norm(r)


class Kolibri1Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Kolibri1DecoderLayer(args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Index of the first layer of each type; its cache builds that type's mask.
        self.layer_types = args.layer_types
        self.sliding_idx = self._first("sliding_attention")
        self.full_idx = self._first("full_attention")

    def _first(self, layer_type):
        return (
            self.layer_types.index(layer_type)
            if layer_type in self.layer_types
            else None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        swa_mask = full_mask = None
        if self.sliding_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.sliding_idx], window_size=self.args.sliding_window
            )
        if self.full_idx is not None:
            full_mask = create_attention_mask(h, cache[self.full_idx])

        for layer, layer_type, c in zip(self.layers, self.layer_types, cache):
            mask = swa_mask if layer_type == "sliding_attention" else full_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Kolibri1Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        # The router bias lives under a different prefix in the checkpoint.
        for k in [k for k in weights if k.endswith(".moe.router.expert_bias")]:
            weights[k.replace(".moe.router.expert_bias", ".mlp.expert_bias")] = (
                weights.pop(k).astype(mx.float32)
            )
        moe_layers = sorted(
            int(k.split(".")[2])
            for k in weights
            if k.startswith("model.layers.")
            and k.endswith(".mlp.experts.0.up_proj.weight")
        )
        for l in moe_layers:
            prefix = f"model.layers.{l}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                to_join = [
                    weights.pop(f"{prefix}.mlp.experts.{e}.{n}.weight")
                    for e in range(self.args.num_experts)
                ]
                weights[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(to_join)
        return weights

    def make_cache(self):
        caches = []
        for layer_type in self.args.layer_types:
            if layer_type == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
        return caches

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.model.layers:
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.n_heads //= N
            layer.self_attn.n_kv_heads //= N

            # The router stays replicated; the MoE aggregates its own result.
            layer.mlp.sharding_group = group
            shard_inplace(layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group)
            shard_inplace(layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group)
            shard_inplace(
                layer.mlp.shared_experts.gate_proj, "all-to-sharded", group=group
            )
            shard_inplace(
                layer.mlp.shared_experts.down_proj, "sharded-to-all", group=group
            )
            shard_inplace(layer.mlp.shared_experts.up_proj, "all-to-sharded", group=group)

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # The router runs in fp32 and is tiny; keep it unquantized.
            if path.endswith("mlp.gate"):
                return False
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers
