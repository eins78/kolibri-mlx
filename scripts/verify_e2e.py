"""End-to-end check of a converted (quantised) MLX model against the torch
fp32 reference logits."""

import argparse
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import mlx.core as mx
import numpy as np
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

import kolibri_mlx.register  # noqa: F401
from kolibri_mlx.verify_utils import require
from verify.prompts import PROMPTS


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl_rows(ref, mine, chunk=32):
    out = []
    for s in range(0, len(ref), chunk):
        lp, lq = logsoftmax(ref[s:s + chunk]), logsoftmax(mine[s:s + chunk])
        out.append((np.exp(lp) * (lp - lq)).sum(-1))
    return np.concatenate(out)


def topk(x, k):
    idx = np.argpartition(-x, k - 1, axis=-1)[..., :k]
    o = np.take_along_axis(x, idx, -1).argsort(-1)[..., ::-1]
    return np.take_along_axis(idx, o, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models/Kolibri-1-4bit"))
    ap.add_argument("--ref-dir", default=os.path.join(ROOT, "verify/out/reference"))
    ap.add_argument("--config", help="unused hint; the model dir config is used")
    ap.add_argument("--prompts", help="comma list (default: all PROMPTS)")
    ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--no-retokenize-check", action="store_true")
    ap.add_argument("--min-top1", type=float, default=0.9)
    ap.add_argument("--max-kl-mean", type=float, default=None)
    ap.add_argument("--max-decode-diff", type=float, default=None)
    ap.add_argument("--out", help="JSON path")
    a = ap.parse_args()

    names = a.prompts.split(",") if a.prompts else list(PROMPTS)
    need = []
    for n in names:
        d = os.path.join(a.ref_dir, n)
        need += [os.path.join(d, "tokens.npy"), os.path.join(d, "logits.npy")]
        if a.hidden:
            need.append(os.path.join(d, "final_norm.npy"))
    require(*need)

    model, tok = mlx_lm.load(a.model)
    model.eval()
    rows, fails = [], []
    for n in names:
        d = os.path.join(a.ref_dir, n)
        tokens = np.load(os.path.join(d, "tokens.npy")).astype(np.int64)
        ref = np.load(os.path.join(d, "logits.npy")).astype(np.float32)
        T = len(tokens)
        if not a.no_retokenize_check:
            ids = tok.encode(PROMPTS[n])
            if list(ids) != tokens.tolist():
                print(f"WARNING [{n}]: tokenizer ids differ from reference tokens "
                      f"({len(ids)} vs {T})")
        mx.reset_peak_memory()
        t0 = time.time()
        x = mx.array(tokens)[None]
        logits = model(x)
        mx.eval(logits)
        mine = np.array(logits.astype(mx.float32))[0]
        wall = time.time() - t0
        assert mine.shape == ref.shape, (mine.shape, ref.shape)

        top1 = float((mine.argmax(-1) == ref.argmax(-1)).mean())
        tr, tm = topk(ref, 5), topk(mine, 5)
        top5 = float(np.mean([len(set(p) & set(q)) / 5 for p, q in zip(tr, tm)]))
        kl = kl_rows(ref, mine)
        w = int(kl.argmax())
        dec = lambda ids: [tok.decode([int(i)]) for i in ids]
        r = dict(prompt=n, T=T, top1=top1, top5=top5, kl_mean=float(kl.mean()),
                 kl_p95=float(np.percentile(kl, 95)), kl_max=float(kl.max()),
                 max_abs_logit=float(np.abs(mine - ref).max()),
                 worst_kl_pos=w,
                 worst_ref_top3=dec(tr[w][:3]), worst_mlx_top3=dec(tm[w][:3]))

        def top5p(row):
            p = np.exp(logsoftmax(row))
            ids = topk(row, 5)
            return [(tok.decode([int(i)]), float(p[i])) for i in ids]
        r["next_ref"], r["next_mlx"] = top5p(ref[-1]), top5p(mine[-1])

        # decode path: prefill all but last k tokens, then step one at a time
        k = min(8, T - 1)
        cache = make_prompt_cache(model)
        mx.eval(model(x[:, :T - k], cache=cache))
        rows_d = []
        for t in range(T - k, T):
            s = model(x[:, t:t + 1], cache=cache)
            mx.eval(s)
            rows_d.append(np.array(s.astype(mx.float32))[0, -1])
        r["decode_max_diff"] = float(np.abs(np.stack(rows_d) - mine[T - k:]).max())

        if a.hidden:
            hd = np.array(model.model(x).astype(mx.float32))[0].astype(np.float64)
            fn = np.load(os.path.join(d, "final_norm.npy")).astype(np.float64)
            r["hidden_rel"] = float(np.linalg.norm(hd - fn) / np.linalg.norm(fn))
        r["peak_gb"] = mx.get_peak_memory() / 1e9
        r["wall_s"] = wall
        rows.append(r)

        if top1 < a.min_top1:
            fails.append(f"{n}: top1 {top1:.3f} < {a.min_top1}")
        if a.max_kl_mean is not None and r["kl_mean"] > a.max_kl_mean:
            fails.append(f"{n}: kl_mean {r['kl_mean']:.3e} > {a.max_kl_mean}")
        if a.max_decode_diff is not None and r["decode_max_diff"] > a.max_decode_diff:
            fails.append(f"{n}: decode diff {r['decode_max_diff']:.3e} > {a.max_decode_diff}")

    print("\n| prompt | T | top1 | top5 | KL mean | KL p95 | KL max | max abs logit diff | decode-vs-prefill | peak GB | s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['prompt']} | {r['T']} | {r['top1']:.3f} | {r['top5']:.3f} | "
              f"{r['kl_mean']:.3e} | {r['kl_p95']:.3e} | {r['kl_max']:.3e} | "
              f"{r['max_abs_logit']:.3e} | {r['decode_max_diff']:.3e} | "
              f"{r['peak_gb']:.2f} | {r['wall_s']:.1f} |")
    for r in rows:
        print(f"\n[{r['prompt']}] worst KL at pos {r['worst_kl_pos']}: "
              f"ref {r['worst_ref_top3']} mlx {r['worst_mlx_top3']}")
        fmt = lambda l: ", ".join(f"{t!r}:{p:.3f}" for t, p in l)
        print(f"  next ref: {fmt(r['next_ref'])}\n  next mlx: {fmt(r['next_mlx'])}")
        if "hidden_rel" in r:
            print(f"  final_norm rel err {r['hidden_rel']:.3e}")

    outp = a.out or os.path.join(
        ROOT, f"verify/out/e2e-{os.path.basename(a.model.rstrip('/'))}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump(dict(model=a.model, rows=rows, passed=not fails), f, indent=1)
    print(f"wrote {outp}")
    if fails:
        print("FAIL:\n  " + "\n  ".join(fails))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
