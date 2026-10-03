"""Teacher-forced per-layer check of the unquantised MLX decoder layer math
against the torch fp32 reference residual stream."""

import argparse
import gc
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import mlx.core as mx
import numpy as np
import torch

from kolibri_mlx.models.kolibri1 import Kolibri1DecoderLayer, ModelArgs
from kolibri_mlx.verify_utils import (DTYPES, hf_layer_to_mlx, layer_masks,
                                      load_config, mlx_dir_layer, require,
                                      route_inds)
from verify.prompts import PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default=os.path.join(ROOT, "verify/out/reference"))
    ap.add_argument("--config", help="config.json (default: HF snapshot)")
    ap.add_argument("--checkpoint", default="Aleph-Alpha/Kolibri-1", help="HF checkpoint dir/repo for Checkpoint")
    ap.add_argument("--mlx-model", help="load layer weights from this mlx-lm dir")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    ap.add_argument("--layers", help="range a-b inclusive, or single index")
    ap.add_argument("--prompts", help="comma list (default: all PROMPTS)")
    ap.add_argument("--max-rel", type=float)
    ap.add_argument("--out", help="JSON path")
    a = ap.parse_args()

    dtype = DTYPES[a.dtype]
    max_rel = a.max_rel or (2e-2 if a.dtype == "bfloat16" else 1e-3)
    if a.mlx_model and not a.config:
        a.config = os.path.join(a.mlx_model, "config.json")
    ckpt = None
    if a.mlx_model:
        config = load_config(a.config)
    elif a.config:
        from kolibri_mlx.checkpoint import Checkpoint
        config = load_config(a.config)
        ckpt = Checkpoint(a.checkpoint)
    else:
        from kolibri_mlx.checkpoint import Checkpoint
        ckpt = Checkpoint(a.checkpoint)
        config = ckpt.config
    config = dict(config)
    config.setdefault("model_type", "kolibri1")
    args = ModelArgs.from_dict(config)

    lo, hi = 0, args.num_hidden_layers - 1
    if a.layers:
        lo, _, h_ = a.layers.partition("-")
        lo = int(lo)
        hi = int(h_) if h_ else lo
    layers = range(lo, hi + 1)
    names = a.prompts.split(",") if a.prompts else list(PROMPTS)

    need = []
    for n in names:
        d = os.path.join(a.ref_dir, n)
        need.append(os.path.join(d, "embed.npy"))
        for i in layers:
            need += [os.path.join(d, f"layer_{i:02d}.npy"),
                     os.path.join(d, f"experts_{i:02d}.npy")]
            if i > 0:
                need.append(os.path.join(d, f"layer_{i-1:02d}.npy"))
    require(*dict.fromkeys(need))

    rows = []
    print(f"dtype={a.dtype} max_rel={max_rel:g} layers={lo}-{hi} prompts={names}")
    print("| layer | prompt | T | max abs | mean abs | rel | cos(last) | route same | route diff | peak GB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i in layers:
        if a.mlx_model:
            w = mlx_dir_layer(a.mlx_model, i, dtype)
        else:
            tdt = torch.bfloat16 if dtype == mx.bfloat16 else torch.float32
            w = hf_layer_to_mlx(ckpt.layer_tensors(i, dtype=tdt), i,
                                args.num_experts, dtype)
        layer = Kolibri1DecoderLayer(args, layer_idx=i)
        layer.load_weights(list(w.items()), strict=True)
        del w
        layer.eval()
        mx.eval(layer.parameters())
        mx.reset_peak_memory()
        for n in names:
            d = os.path.join(a.ref_dir, n)
            src = "embed.npy" if i == 0 else f"layer_{i-1:02d}.npy"
            h = mx.array(np.load(os.path.join(d, src)))[None].astype(dtype)
            ref = np.load(os.path.join(d, f"layer_{i:02d}.npy")).astype(np.float64)
            ref_exp = np.sort(np.load(os.path.join(d, f"experts_{i:02d}.npy")), -1)
            mask = layer_masks(h, args, i)
            out = layer(h, mask, None)
            inds = route_inds(layer, h, mask)
            mx.eval(out, inds)
            o = np.array(out.astype(mx.float32))[0].astype(np.float64)
            err = np.abs(o - ref)
            rel = float(np.linalg.norm(o - ref) / np.linalg.norm(ref))
            cos = float(o[-1] @ ref[-1] / (np.linalg.norm(o[-1]) * np.linalg.norm(ref[-1])))
            mine = np.array(inds)
            same = (mine == ref_exp).all(-1)
            r = dict(layer=i, prompt=n, T=int(ref.shape[0]),
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

    bad = [r for r in rows if not r["rel"] <= max_rel]
    outp = a.out or os.path.join(ROOT, f"verify/out/layerwise-{a.dtype}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump(dict(dtype=a.dtype, max_rel=max_rel, rows=rows,
                       passed=not bad), f, indent=1)
    print(f"wrote {outp}")
    if bad:
        w = max(bad, key=lambda r: r["rel"])
        print(f"FAIL: {len(bad)} layer/prompt rows exceed rel {max_rel:g}; "
              f"worst layer {w['layer']} {w['prompt']} rel={w['rel']:.3e}")
        sys.exit(1)
    print(f"PASS: all rows rel <= {max_rel:g}")


if __name__ == "__main__":
    main()
