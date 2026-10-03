# SPDX-License-Identifier: CC0-1.0
"""Per-layer check of the MLX decoder layer math against the torch fp32
reference residual stream. Teacher-forced by default; --chain feeds each
layer's own output forward from the real embedding and also checks logits.
Optional on-the-fly quantisation (--bits etc.)."""

import argparse
import gc
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from kolibri_mlx.models.kolibri1 import Kolibri1DecoderLayer, ModelArgs
from kolibri_mlx.verify_utils import (DTYPES, hf_layer_to_mlx, layer_masks,
                                      load_config, logit_metrics,
                                      mlx_dir_layer, require, route_inds)
from verify.prompts import PROMPTS


def make_layer_predicate(group_size, bits, attn_bits, expert_bits):
    """Like convert.make_predicate, with per-module bits (0 = keep unquantised)."""
    def pred(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[-1] % group_size != 0 or path.endswith("mlp.gate"):
            return False
        b = bits
        if path.startswith("self_attn") or path.startswith("mlp.shared_experts"):
            b = bits if attn_bits is None else attn_bits
        elif path.startswith("mlp.switch_mlp"):
            b = bits if expert_bits is None else expert_bits
        if not b:
            return False
        return {"group_size": group_size, "bits": b, "mode": "affine"}
    return pred


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default=os.path.join(ROOT, "verify/out/reference"))
    ap.add_argument("--config", help="config.json (default: HF snapshot)")
    ap.add_argument("--checkpoint", help="HF checkpoint dir/repo for Checkpoint")
    ap.add_argument("--mlx-model", help="load layer weights from this mlx-lm dir")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    ap.add_argument("--layers", help="range a-b inclusive, or single index")
    ap.add_argument("--prompts", help="comma list (default: all PROMPTS)")
    ap.add_argument("--max-rel", type=float)
    ap.add_argument("--out", help="JSON path")
    ap.add_argument("--chain", action="store_true",
                    help="non-teacher-forced: real embedding, chained layers, logits")
    ap.add_argument("--bits", type=int, default=0, help="quantise layers (0 = off)")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--embed-bits", type=int, help="default: --bits; 0 = off")
    ap.add_argument("--lm-head-bits", type=int, help="default: --bits; 0 = off")
    ap.add_argument("--attn-bits", type=int, help="attn + shared experts; 0 = off")
    ap.add_argument("--expert-bits", type=int, help="switch_mlp; 0 = off")
    a = ap.parse_args()

    dtype = DTYPES[a.dtype]
    tdt = torch.bfloat16 if dtype == mx.bfloat16 else torch.float32
    max_rel = a.max_rel or (2e-2 if a.dtype == "bfloat16" else 1e-3)
    if a.chain and a.mlx_model:
        raise SystemExit("--chain needs the HF checkpoint (embed/lm_head), not --mlx-model")
    if a.mlx_model and not a.config:
        a.config = os.path.join(a.mlx_model, "config.json")
    ckpt = None
    if a.mlx_model:
        config = load_config(a.config)
    else:
        from kolibri_mlx.checkpoint import Checkpoint
        ckpt = Checkpoint(a.checkpoint) if a.checkpoint else Checkpoint()
        config = load_config(a.config) if a.config else ckpt.config
    config = dict(config)
    config.setdefault("model_type", "kolibri1")
    args = ModelArgs.from_dict(config)

    lo, hi = 0, args.num_hidden_layers - 1
    if a.layers:
        lo, _, h_ = a.layers.partition("-")
        lo = int(lo)
        hi = int(h_) if h_ else lo
    if a.chain and lo != 0:
        raise SystemExit("--chain must start at layer 0")
    layers = range(lo, hi + 1)
    names = a.prompts.split(",") if a.prompts else list(PROMPTS)

    need = []
    for n in names:
        d = os.path.join(a.ref_dir, n)
        need.append(os.path.join(d, "embed.npy"))
        if a.chain:
            need += [os.path.join(d, "tokens.npy")]
            if hi == args.num_hidden_layers - 1:
                need.append(os.path.join(d, "logits.npy"))
        for i in layers:
            need += [os.path.join(d, f"layer_{i:02d}.npy"),
                     os.path.join(d, f"experts_{i:02d}.npy")]
            if i > 0:
                need.append(os.path.join(d, f"layer_{i-1:02d}.npy"))
    require(*dict.fromkeys(need))

    def maybe_q(mod, bits_override):
        b = a.bits if bits_override is None else bits_override
        if b:
            return mod.to_quantized(group_size=a.group_size, bits=b, mode="affine")
        return mod

    def ref(n, f):
        return np.load(os.path.join(a.ref_dir, n, f))

    # chain: embed tokens up front
    hs, gl = {}, None
    if a.chain:
        gl = ckpt.global_tensors(tdt)
        emb = nn.Embedding(args.vocab_size, args.hidden_size)
        emb.weight = mx.array(gl.pop("model.embed_tokens.weight").float().numpy()).astype(dtype)
        emb = maybe_q(emb, a.embed_bits)
        for n in names:
            hs[n] = emb(mx.array(ref(n, "tokens.npy").astype(np.int64))[None])
            mx.eval(hs[n])
        del emb
        mx.clear_cache()

    pred = make_layer_predicate(a.group_size, a.bits, a.attn_bits, a.expert_bits)
    quant = bool(a.bits or a.attn_bits or a.expert_bits)
    mode = "chain" if a.chain else "teacher-forced"
    rows = []
    print(f"mode={mode} dtype={a.dtype} bits={a.bits} gs={a.group_size} "
          f"embed={a.embed_bits} head={a.lm_head_bits} attn={a.attn_bits} "
          f"expert={a.expert_bits} layers={lo}-{hi} prompts={names}")
    print("| layer | prompt | T | max abs | mean abs | rel | cos(last) | route same | route diff | peak GB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i in layers:
        if a.mlx_model:
            w = mlx_dir_layer(a.mlx_model, i, dtype)
        else:
            w = hf_layer_to_mlx(ckpt.layer_tensors(i, dtype=tdt), i,
                                args.num_experts, dtype)
        layer = Kolibri1DecoderLayer(args, layer_idx=i)
        layer.load_weights(list(w.items()), strict=True)
        del w
        if quant:
            nn.quantize(layer, a.group_size, a.bits, mode="affine", class_predicate=pred)
        layer.eval()
        mx.eval(layer.parameters())
        mx.reset_peak_memory()
        for n in names:
            if a.chain:
                h = hs[n]
            else:
                src = "embed.npy" if i == 0 else f"layer_{i-1:02d}.npy"
                h = mx.array(ref(n, src))[None].astype(dtype)
            rf = ref(n, f"layer_{i:02d}.npy").astype(np.float64)
            ref_exp = np.sort(ref(n, f"experts_{i:02d}.npy"), -1)
            mask = layer_masks(h, args, i)
            out = layer(h, mask, None)
            inds = route_inds(layer, h, mask)
            mx.eval(out, inds)
            if a.chain:
                hs[n] = out
            o = np.array(out.astype(mx.float32))[0].astype(np.float64)
            err = np.abs(o - rf)
            rel = float(np.linalg.norm(o - rf) / np.linalg.norm(rf))
            cos = cosine(o[-1], rf[-1])
            same = (np.array(inds) == ref_exp).all(-1)
            r = dict(layer=i, prompt=n, T=int(rf.shape[0]),
                     max_abs=float(err.max()), mean_abs=float(err.mean()),
                     rel=rel, cos_last=cos, route_same_frac=float(same.mean()),
                     route_diff_tokens=int((~same).sum()),
                     peak_gb=mx.get_peak_memory() / 1e9)
            rows.append(r)
            print(f"| {i} | {n} | {r['T']} | {r['max_abs']:.3e} | {r['mean_abs']:.3e} | "
                  f"{rel:.3e} | {cos:.6f} | {r['route_same_frac']:.3f} | "
                  f"{r['route_diff_tokens']} | {r['peak_gb']:.2f} |", flush=True)
        del layer, out, inds, h
        mx.clear_cache()
        gc.collect()

    logit_rows = []
    if a.chain and hi == args.num_hidden_layers - 1:
        norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        norm.weight = mx.array(gl.pop("model.norm.weight").float().numpy()).astype(dtype)
        head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        head.weight = mx.array(gl.pop("lm_head.weight").float().numpy()).astype(dtype)
        head = maybe_q(head, a.lm_head_bits)
        mx.eval(norm.parameters(), head.parameters())
        print("\n| prompt | T | top1 | top5 | KL mean | KL p95 | KL max | max abs logit diff |")
        print("|---|---|---|---|---|---|---|---|")
        for n in names:
            lg = head(norm(hs[n]))
            mx.eval(lg)
            mine = np.array(lg.astype(mx.float32))[0]
            m, _, _ = logit_metrics(ref(n, "logits.npy").astype(np.float32), mine)
            m.update(prompt=n, T=int(mine.shape[0]))
            logit_rows.append(m)
            print(f"| {n} | {m['T']} | {m['top1']:.3f} | {m['top5']:.3f} | "
                  f"{m['kl_mean']:.3e} | {m['kl_p95']:.3e} | {m['kl_max']:.3e} | "
                  f"{m['max_abs_logit']:.3e} |", flush=True)

    bad = [] if a.chain else [r for r in rows if not r["rel"] <= max_rel]
    default = f"verify/out/{'chain' if a.chain else 'layerwise'}-{a.dtype}.json"
    outp = a.out or os.path.join(ROOT, default)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump(dict(mode=mode, dtype=a.dtype, max_rel=max_rel, opts=vars(a),
                       rows=rows, logits=logit_rows, peak_gb=max(r["peak_gb"] for r in rows),
                       passed=not bad), f, indent=1)
    print(f"wrote {outp}")
    if a.chain:
        print("chain mode: no pass/fail threshold on layers")
        return
    if bad:
        w = max(bad, key=lambda r: r["rel"])
        print(f"FAIL: {len(bad)} layer/prompt rows exceed rel {max_rel:g}; "
              f"worst layer {w['layer']} {w['prompt']} rel={w['rel']:.3e}")
        sys.exit(1)
    print(f"PASS: all rows rel <= {max_rel:g}")


if __name__ == "__main__":
    main()
