# SPDX-License-Identifier: CC0-1.0
"""Smoke test of the kolibri1 model file with a tiny random config.

Checks: forward shape, prefill + decode with cache equals full re-prefill
(window boundary included), fp32 router call works, nn.quantize round trip.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache

from kolibri_mlx.models.kolibri1 import Model, ModelArgs

args = ModelArgs(
    model_type="kolibri1", hidden_size=64, num_hidden_layers=5,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    shared_expert_intermediate_size=32, rms_norm_eps=1e-6, vocab_size=100,
    rope_theta=10000.0, sliding_window=7,
    layer_types=["sliding_attention"] * 4 + ["full_attention"],
)
mx.random.seed(0)
model = Model(args)
# random bias so routing is non-trivial
for l in model.model.layers:
    l.mlp.expert_bias = mx.random.normal((8,)) * 3
model.eval()
mx.eval(model.parameters())

T = 20
ids = mx.random.randint(0, 100, (1, T))
full = model(ids)
mx.eval(full)
print("full logits", full.shape, full.dtype)

# prefill 5 then decode one by one; compare last-position logits at each step
cache = make_prompt_cache(model)
out = model(ids[:, :5], cache=cache)
mx.eval(out)
maxerr = 0.0
for t in range(5, T):
    step = model(ids[:, t:t+1], cache=cache)
    mx.eval(step)
    e = float(mx.abs(step[0, -1] - full[0, t]).max())
    maxerr = max(maxerr, e)
    if t in (6, 7, 8, 12, 19):
        print(f"pos {t}: decode vs full max abs err {e:.2e}")
print("max err over decode", maxerr)
assert maxerr < 1e-3, maxerr

# long prefill (> window) then decode
cache = make_prompt_cache(model)
out = model(ids[:, :15], cache=cache); mx.eval(out)
e = float(mx.abs(out[0, -1] - full[0, 14]).max()); print("prefill15 last vs full", e)
step = model(ids[:, 15:16], cache=cache); mx.eval(step)
e2 = float(mx.abs(step[0, -1] - full[0, 15]).max()); print("decode after long prefill", e2)
assert e < 1e-3 and e2 < 1e-3

# chunked prefill 12 + 8
cache = make_prompt_cache(model)
o1 = model(ids[:, :12], cache=cache); mx.eval(o1)
o2 = model(ids[:, 12:], cache=cache); mx.eval(o2)
e3 = float(mx.abs(o2[0] - full[0, 12:]).max()); print("chunked prefill", e3)
assert e3 < 1e-3

# sanitize round trip + quantize
w = dict(tree_flatten(model.parameters()))
hf = {}
for k, v in w.items():
    if ".switch_mlp." in k:
        n = k.split(".")[-2]
        for e in range(8):
            hf[k.replace(f"switch_mlp.{n}", f"experts.{e}.{n}")] = v[e]
    elif k.endswith(".mlp.expert_bias"):
        hf[k.replace(".mlp.expert_bias", ".moe.router.expert_bias")] = v.astype(mx.bfloat16)
    else:
        hf[k] = v
m2 = Model(args)
m2.load_weights(list(model.sanitize(hf).items()))
m2.eval(); mx.eval(m2.parameters())
print("sanitize roundtrip err", float(mx.abs(m2(ids) - full).max()))

pred = m2.quant_predicate
nn.quantize(m2, group_size=32, bits=4, class_predicate=lambda p, m: hasattr(m, 'to_quantized') and pred(p, m))
print("gate quantized?", type(m2.model.layers[0].mlp.gate).__name__, "experts:", type(m2.model.layers[0].mlp.switch_mlp.gate_proj).__name__)
q = m2(ids); mx.eval(q); print("quantized forward ok", q.shape)

# bf16 model with fp32 router input
m3 = Model(args); m3.load_weights(list(model.sanitize(dict(hf)).items()))
m3.set_dtype(mx.bfloat16); m3.eval()
o = m3(ids); mx.eval(o); print("bf16 forward ok", o.dtype, "router bias dtype", m3.model.layers[0].mlp.expert_bias.dtype)
print("SMOKE OK")
