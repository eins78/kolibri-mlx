# SPDX-License-Identifier: CC0-1.0
"""Compare single-prefill logits with decode-path logits on the converted model.

Both paths use the same weights, so differences come only from kernel/accumulation
order (bf16) and the KV cache path. Reports KL and top-1 per position.
"""

import argparse
import sys, os
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import kolibri_mlx.register  # noqa: F401
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def logsoftmax(x):
    x = x.astype(np.float64); m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl(p_logits, q_logits):
    lp, lq = logsoftmax(p_logits), logsoftmax(q_logits)
    return (np.exp(lp) * (lp - lq)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Kolibri-1-4bit")
    ap.add_argument("--ref-dir", default="verify/out/reference")
    ap.add_argument("--prompts", default="de_short,en_long")
    ap.add_argument("--tail", type=int, default=8)
    a = ap.parse_args()
    model, tok = load(a.model)
    for name in a.prompts.split(","):
        toks = np.load(f"{a.ref_dir}/{name}/tokens.npy")
        T = len(toks)
        x = mx.array(toks)[None]
        full = np.array(model(x).astype(mx.float32))[0]
        # path A: prefill T-tail, then decode tail one by one
        cache = make_prompt_cache(model)
        model(x[:, : T - a.tail], cache=cache)
        rows = []
        for t in range(T - a.tail, T):
            rows.append(np.array(model(x[:, t : t + 1], cache=cache).astype(mx.float32))[0, 0])
        dec = np.stack(rows)
        ref = full[T - a.tail :]
        k = kl(ref, dec); top1 = (ref.argmax(-1) == dec.argmax(-1)).mean()
        print(f"[{name}] T={T} decode-tail vs prefill: KL mean {k.mean():.3e} max {k.max():.3e} top1 {top1:.3f} max|dlogit| {np.abs(ref-dec).max():.2f}")
        # path B: full token-by-token decode from scratch (first 13 positions)
        cache = make_prompt_cache(model)
        n = min(T, 13)
        rows = [np.array(model(x[:, t : t + 1], cache=cache).astype(mx.float32))[0, 0] for t in range(n)]
        dec = np.stack(rows); ref = full[:n]
        k = kl(ref, dec); top1 = (ref.argmax(-1) == dec.argmax(-1)).mean()
        print(f"[{name}] token-by-token first {n} vs prefill: KL mean {k.mean():.3e} max {k.max():.3e} top1 {top1:.3f} max|dlogit| {np.abs(ref-dec).max():.2f}")
        # path C: prefill twice (determinism)
        full2 = np.array(model(x).astype(mx.float32))[0]
        print(f"[{name}] prefill twice max|dlogit| {np.abs(full-full2).max():.2e}")
        # reference comparison for context
        rl = np.load(f"{a.ref_dir}/{name}/logits.npy")
        k = kl(rl, full); print(f"[{name}] prefill vs torch ref: KL mean {k.mean():.3e} top1 {(rl.argmax(-1)==full.argmax(-1)).mean():.3f}; logit scale: ref std {rl.std():.2f} max {np.abs(rl).max():.1f}")
    print(f"peak memory {mx.get_peak_memory()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
