import copy

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from kolibri_mlx.models.kolibri1 import Kolibri1SparseMoeBlock, Model, ModelArgs

V = 100
W = 7
NE = 8
TOPK = 2


def make_args(layer_types=None, **kw):
    layer_types = layer_types or ["sliding_attention"] * 4 + ["full_attention"]
    cfg = dict(
        model_type="kolibri1",
        hidden_size=64,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=NE,
        num_experts_per_tok=TOPK,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        rms_norm_eps=1e-6,
        vocab_size=V,
        rope_theta=10000.0,
        sliding_window=W,
        layer_types=layer_types,
    )
    cfg.update(kw)
    return ModelArgs(**cfg)


def make_model(layer_types=None, seed=0):
    mx.random.seed(seed)
    model = Model(make_args(layer_types))
    for l in model.model.layers:
        l.mlp.expert_bias = mx.random.normal((NE,)) * 3
    model.eval()
    mx.eval(model.parameters())
    return model


def rand_ids(T, seed=1):
    return np.random.RandomState(seed).randint(0, V, size=(1, T))


# ---------------------------------------------------------------- reference


def _t(a):
    return torch.from_numpy(np.array(a, copy=False).astype(np.float32))


def rms(x, w, eps=1e-6):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def rope(x, theta):
    # x: [T, H, D], NeoX rotate-half
    T, _, D = x.shape
    inv = 1.0 / (theta ** (torch.arange(0, D, 2).float() / D))
    f = torch.arange(T).float()[:, None] * inv
    cos, sin = f.cos()[:, None, :], f.sin()[:, None, :]
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def reference(params, args, ids):
    """Plain-torch fp32 forward. Returns (logits [T, V], routed ids per layer)."""
    p = {k: _t(v) for k, v in params.items()}
    eps, D = args.rms_norm_eps, args.head_dim
    nh, nkv = args.num_attention_heads, args.num_key_value_heads
    T = len(ids)
    h = p["model.embed_tokens.weight"][torch.as_tensor(ids)]
    route_log = []
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    for n, lt in enumerate(args.layer_types):
        g = lambda s: p[f"model.layers.{n}.{s}"]
        x = rms(h, g("input_layernorm.weight"), eps)
        q = (x @ g("self_attn.q_proj.weight").T).reshape(T, nh, D)
        k = (x @ g("self_attn.k_proj.weight").T).reshape(T, nkv, D)
        v = (x @ g("self_attn.v_proj.weight").T).reshape(T, nkv, D)
        q = rms(q, g("self_attn.q_norm.weight"), eps)
        k = rms(k, g("self_attn.k_norm.weight"), eps)
        sliding = lt == "sliding_attention"
        if sliding:
            q, k = rope(q, args.rope_theta), rope(k, args.rope_theta)
        allowed = j <= i
        if sliding:
            allowed = allowed & (i - j <= args.sliding_window - 1)
        kv_of = torch.arange(nh) // (nh // nkv)
        scores = torch.einsum("ihd,jhd->hij", q, k[:, kv_of]) * D**-0.5
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        attn = torch.softmax(scores.float(), -1)
        a = torch.einsum("hij,jhd->ihd", attn, v[:, kv_of]).reshape(T, -1)
        a = a @ g("self_attn.o_proj.weight").T
        h = h + rms(a, g("post_attn_norm.weight"), eps)

        x = rms(h, g("post_attention_layernorm.weight"), eps)
        logits = x @ g("mlp.gate.weight").T
        ids_k = torch.topk(logits + g("mlp.expert_bias"), TOPK, dim=-1).indices
        wts = torch.sigmoid(logits.gather(-1, ids_k))
        route_log.append(ids_k)
        Wg, Wu, Wd = (g(f"mlp.switch_mlp.{m}.weight") for m in ("gate_proj", "up_proj", "down_proj"))
        routed = torch.zeros_like(h)
        for t in range(T):
            for kk in range(TOPK):
                e = ids_k[t, kk]
                y = (torch.nn.functional.silu(Wg[e] @ x[t]) * (Wu[e] @ x[t])) @ Wd[e].T
                routed[t] += wts[t, kk] * y
        sg, su, sd = (g(f"mlp.shared_experts.{m}.weight") for m in ("gate_proj", "up_proj", "down_proj"))
        shared = (torch.nn.functional.silu(x @ sg.T) * (x @ su.T)) @ sd.T
        h = h + rms(routed + shared, g("post_ffn_norm.weight"), eps)
    x = rms(h, p["model.norm.weight"], eps)
    return x @ p["lm_head.weight"].T, route_log


def params_of(model):
    return {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())}


# -------------------------------------------------------------------- tests


def test_model_runner():
    model = make_model()
    assert len(model.layers) == 5
    assert model.model_type == "kolibri1"

    for t in [mx.float32, mx.float16]:
        model.update(tree_map(lambda p: p.astype(t), model.parameters()))

        inputs = mx.array([[0, 1]])
        outputs = model(inputs)
        assert outputs.shape == (1, 2, V)
        assert outputs.dtype == t

        cache = make_prompt_cache(model)
        outputs = model(inputs, cache=cache)
        assert outputs.shape == (1, 2, V)
        assert outputs.dtype == t

        outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
        assert outputs.shape == (1, 1, V)
        assert outputs.dtype == t

    outputs = model(mx.array([[0, 1], [2, 3]]))
    assert outputs.shape == (2, 2, V)

    copy.deepcopy(model)


@pytest.mark.parametrize("T", [1, 7, 8, 20])
def test_matches_torch_reference(T):
    model = make_model()
    ids = rand_ids(T)
    out = np.array(model(mx.array(ids))[0])
    ref, route = reference(params_of(model), model.args, ids[0])
    assert out.shape == tuple(ref.shape)
    err = np.abs(out - ref.numpy()).max()
    assert err < 1e-3, err


def test_routing_matches_reference():
    # Capture the MLX router's expert ids and compare with the reference.
    model = make_model()
    ids = rand_ids(20)
    seen = []

    class Spy(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __call__(self, x, inds):
            seen.append(np.array(inds)[0])
            return self.inner(x, inds)

    for l in model.model.layers:
        l.mlp.switch_mlp = Spy(l.mlp.switch_mlp)
    model(mx.array(ids))
    params = {k.replace(".inner", ""): v for k, v in params_of(model).items()}
    _, route = reference(params, model.args, ids[0])
    assert len(seen) == len(route) == 5
    for a, b in zip(seen, route):
        assert [set(r) for r in a.tolist()] == [set(r) for r in b.tolist()]


def test_sliding_window_locality():
    ids = rand_ids(20)
    ids2 = ids.copy()
    ids2[0, 0] = (ids[0, 0] + 1) % V

    sw = make_model(["sliding_attention"] * 3)
    a, b = np.array(sw(mx.array(ids))[0]), np.array(sw(mx.array(ids2))[0])
    # Window 7 incl. self: position 6 still sees token 0, position 7 does not.
    # Receptive field grows with depth, so use a single layer for the exact bound.
    sw1 = make_model(["sliding_attention"])
    a1, b1 = np.array(sw1(mx.array(ids))[0]), np.array(sw1(mx.array(ids2))[0])
    assert np.abs(a1[6] - b1[6]).max() > 1e-6
    assert np.abs(a1[7:] - b1[7:]).max() == 0.0
    # With 3 layers token 0 reaches at most 3 * (W - 1) positions ahead.
    reach = 3 * (W - 1)
    assert np.abs(a[6] - b[6]).max() > 1e-6
    assert np.abs(a[reach + 1 :] - b[reach + 1 :]).max() == 0.0


def test_full_attention_sees_token_zero():
    ids = rand_ids(20)
    ids2 = ids.copy()
    ids2[0, 0] = (ids[0, 0] + 1) % V
    full = make_model(["full_attention"] * 2)
    a, b = np.array(full(mx.array(ids))[0]), np.array(full(mx.array(ids2))[0])
    assert np.abs(a[-1] - b[-1]).max() > 1e-6


def test_full_attention_has_no_rope():
    model = make_model(["sliding_attention", "full_attention"])
    assert hasattr(model.model.layers[0].self_attn, "rope")
    assert not hasattr(model.model.layers[1].self_attn, "rope")


def test_full_attention_is_permutation_invariant():
    # No positional info: the last position is invariant to permuting
    # earlier tokens.
    m = make_model(["full_attention"])
    ids = rand_ids(6)
    perm = ids.copy()
    perm[0, :5] = perm[0, :5][::-1]
    a, b = np.array(m(mx.array(ids))[0, -1]), np.array(m(mx.array(perm))[0, -1])
    assert np.abs(a - b).max() < 1e-4


def test_cache_consistency():
    model = make_model()
    T = 20
    ids = mx.array(rand_ids(T))
    full = model(ids)

    cache = make_prompt_cache(model)
    out = model(ids[:, :5], cache=cache)
    assert mx.abs(out - full[:, :5]).max().item() < 1e-3
    for t in range(5, T):
        step = model(ids[:, t : t + 1], cache=cache)
        assert mx.abs(step[0, 0] - full[0, t]).max().item() < 1e-3, t

    cache = make_prompt_cache(model)
    o1 = model(ids[:, :12], cache=cache)
    o2 = model(ids[:, 12:], cache=cache)
    assert mx.abs(o1 - full[:, :12]).max().item() < 1e-3
    assert mx.abs(o2 - full[:, 12:]).max().item() < 1e-3


def to_hf(model):
    hf = {}
    for k, v in tree_flatten(model.parameters()):
        if ".switch_mlp." in k:
            n = k.split(".")[-2]
            for e in range(NE):
                hf[k.replace(f"switch_mlp.{n}", f"experts.{e}.{n}")] = v[e]
        elif k.endswith(".mlp.expert_bias"):
            hf[k.replace(".mlp.expert_bias", ".moe.router.expert_bias")] = v.astype(mx.bfloat16)
        else:
            hf[k] = v
    return hf


def test_sanitize():
    model = make_model()
    ids = mx.array(rand_ids(10))
    # bf16 round trip of the bias is what a checkpoint would carry
    hf = to_hf(model)
    assert any(".moe.router.expert_bias" in k for k in hf)
    clean = model.sanitize(hf)
    assert not any(".experts." in k or ".moe." in k for k in clean)
    for n in range(5):
        assert clean[f"model.layers.{n}.mlp.expert_bias"].dtype == mx.float32
        assert clean[f"model.layers.{n}.mlp.switch_mlp.up_proj.weight"].shape == (NE, 32, 64)
        assert clean[f"model.layers.{n}.mlp.switch_mlp.down_proj.weight"].shape == (NE, 64, 32)

    # Bias values were bf16-rounded; compare against a model with the same bias.
    m2 = Model(model.args)
    m2.load_weights(list(clean.items()))
    m2.eval()
    for a, b in zip(model.model.layers, m2.model.layers):
        b.mlp.expert_bias = a.mlp.expert_bias.astype(mx.bfloat16).astype(mx.float32)
    mx.eval(m2.parameters())
    out = m2(ids)
    for a in model.model.layers:
        a.mlp.expert_bias = a.mlp.expert_bias.astype(mx.bfloat16).astype(mx.float32)
    assert mx.abs(out - model(ids)).max().item() < 1e-5

    # idempotent on already-sanitised weights
    again = m2.sanitize(dict(tree_flatten(m2.parameters())))
    again = m2.sanitize(again)
    assert set(again) == {k for k, _ in tree_flatten(m2.parameters())}

    # tied embeddings drop lm_head
    tied = Model(make_args(tie_word_embeddings=True))
    w = tied.sanitize({"lm_head.weight": mx.zeros((1,)), "model.norm.weight": mx.zeros((1,))})
    assert "lm_head.weight" not in w


def test_quantize():
    model = make_model()
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda p, m: hasattr(m, "to_quantized") and model.quant_predicate(p, m),
    )
    for l in model.model.layers:
        assert type(l.mlp.gate) is nn.Linear
        assert isinstance(l.mlp.switch_mlp.gate_proj, QuantizedSwitchLinear)
        assert isinstance(l.mlp.switch_mlp.down_proj, QuantizedSwitchLinear)
    out = model(mx.array(rand_ids(10)))
    mx.eval(out)
    assert out.shape == (1, 10, V)
    assert mx.all(mx.isfinite(out)).item()


def test_routing_semantics():
    # Real MoE block; gate = identity so router logits equal the input.
    N, E, K = 64, 384, 6
    args = make_args(
        ["sliding_attention"],
        hidden_size=E,
        num_experts=E,
        num_experts_per_tok=K,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
    )
    block = Kolibri1SparseMoeBlock(args)
    block.gate.weight = mx.eye(E)
    for n in ("gate_proj", "up_proj", "down_proj"):
        getattr(block.shared_experts, n).weight = mx.zeros_like(getattr(block.shared_experts, n).weight)

    torch.manual_seed(0)
    logits = torch.randn(N, E) * 3
    bias = torch.randn(E) * 5
    block.expert_bias = mx.array(bias.numpy())

    seen = {}

    class Stub(nn.Module):
        # Records inds; each expert returns ones so the output is sum(weights).
        def __call__(self, x, inds):
            seen["inds"] = np.array(inds)
            return mx.ones((*inds.shape, x.shape[-1]))

    block.switch_mlp = Stub()
    out = np.array(block(mx.array(logits.numpy()))[:, 0])

    scores = torch.sigmoid(logits)
    ref_ids = torch.topk(logits + bias, K, dim=-1).indices
    ref_w = scores.gather(-1, ref_ids)
    got = torch.from_numpy(seen["inds"]).long()
    assert torch.equal(got.sort(-1)[0], ref_ids.sort(-1)[0]), "expert selection mismatch"
    assert np.allclose(out, ref_w.sum(-1).numpy(), atol=1e-5), "routing weights mismatch"

    # Selection must differ from top-k over sigmoid(logits) + bias.
    score_add = torch.topk(scores + bias, K, dim=-1).indices
    assert not torch.equal(ref_ids.sort(-1)[0], score_add.sort(-1)[0])
