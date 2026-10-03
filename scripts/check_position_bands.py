"""Per-position error of the converted model vs the torch reference, by position band."""
import argparse, sys, os
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import kolibri_mlx.register  # noqa: F401
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from scripts.check_decode_consistency import kl

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="models/Kolibri-1-4bit")
ap.add_argument("--ref-dir", default="verify/out/reference"); ap.add_argument("--prompts", default="en_long,de_long")
a = ap.parse_args()
model, tok = load(a.model)
bands = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, 513), (513, 10**6)]
for name in a.prompts.split(","):
    toks = np.load(f"{a.ref_dir}/{name}/tokens.npy"); T = len(toks); x = mx.array(toks)[None]
    ref = np.load(f"{a.ref_dir}/{name}/logits.npy")
    pre = np.array(model(x).astype(mx.float32))[0]
    cache = make_prompt_cache(model)
    dec = np.stack([np.array(model(x[:, t:t+1], cache=cache).astype(mx.float32))[0, 0] for t in range(min(T, 64))])
    kp, kd = kl(ref, pre), kl(ref[:len(dec)], dec)
    t1p = ref.argmax(-1) == pre.argmax(-1); t1d = ref[:len(dec)].argmax(-1) == dec.argmax(-1)
    print(f"[{name}] T={T}  band: prefill KL mean / top1 | decode KL mean / top1")
    for lo, hi in bands:
        hi = min(hi, T)
        if lo >= T: break
        s = f"  [{lo:4d},{hi:4d}) prefill {kp[lo:hi].mean():.3e} / {t1p[lo:hi].mean():.3f}"
        if lo < len(dec): s += f" | decode {kd[lo:min(hi,len(dec))].mean():.3e} / {t1d[lo:min(hi,len(dec))].mean():.3f}"
        print(s)
    print("  first 8 positions prefill KL:", np.array2string(kp[:8], precision=2))
    print("  first 8 positions decode  KL:", np.array2string(kd[:8], precision=2))
    # hidden-state magnitude at early positions (massive activations?)
    h = np.array(model.model.embed_tokens(x).astype(mx.float32))[0]
    print("  embed row norms first 4:", np.linalg.norm(h[:4], axis=-1).round(2))
    fn = np.load(f"{a.ref_dir}/{name}/final_norm.npy"); print("  ref final-norm max|x| first 4 pos:", np.abs(fn[:4]).max(-1).round(1), " later median:", np.median(np.abs(fn[16:]).max(-1)).round(1))
    for i in (0, 1, 24, 35, 49):
        L = np.load(f"{a.ref_dir}/{name}/layer_{i:02d}.npy"); print(f"  ref layer {i:2d} residual max|x| pos0 {np.abs(L[0]).max():.0f} pos1 {np.abs(L[1]).max():.0f} pos2 {np.abs(L[2]).max():.0f} median(pos>=16) {np.median(np.abs(L[16:]).max(-1)):.0f}")
