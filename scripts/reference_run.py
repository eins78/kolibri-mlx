"""Layer-streamed fp32 CPU reference run; writes per-layer activations to verify/out/reference/."""

import argparse, json, os, resource, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch

from kolibri_mlx import reference_torch as R
from kolibri_mlx.checkpoint import Checkpoint
from verify.prompts import PROMPTS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    return r / 2**30


def save(path, arr):
    np.save(path, arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--head-anyway", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "verify", "out", "reference"))
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)

    names = a.prompts.split(",") if a.prompts else list(PROMPTS)
    ck = Checkpoint()
    cfg = ck.config
    n_layers = min(a.layers or cfg["num_hidden_layers"], cfg["num_hidden_layers"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Aleph-Alpha/Kolibri-1")
    enc = {n: tok(PROMPTS[n])["input_ids"] for n in names}
    bos_id = getattr(tok, "bos_token_id", None)
    bos_added = {n: (bos_id is not None and ids[:1] == [bos_id]) for n, ids in enc.items()}
    first_ids = {n: ids[:3] for n, ids in enc.items()}
    print({n: len(v) for n, v in enc.items()}, "bos_added:", bos_added, flush=True)

    os.makedirs(a.out, exist_ok=True)
    dirs = {n: os.path.join(a.out, n) for n in names}
    for n in names:
        os.makedirs(dirs[n], exist_ok=True)
        save(os.path.join(dirs[n], "tokens.npy"), np.array(enc[n], dtype=np.int64))

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip() or None
    except Exception:
        commit = None
    meta = {
        "torch": torch.__version__, "threads": a.threads, "git_commit": commit,
        "tokenizer": {"class": type(tok).__name__, "bos_token_id": bos_id,
                      "bos_added": bos_added, "first_ids": first_ids,
                      "n_tokens": {n: len(v) for n, v in enc.items()}},
        "prompts": names, "layers_requested": n_layers, "layer_seconds": {}, "layer_load_seconds": {},
    }

    def write_meta():
        with open(os.path.join(a.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    g = ck.global_tensors()
    h = {n: R.embed(torch.tensor(enc[n]), g["model.embed_tokens.weight"]) for n in names}
    for n in names:
        save(os.path.join(dirs[n], "embed.npy"), h[n].numpy())

    done = 0
    for i in range(n_layers):
        t0 = time.time()
        w = ck.layer_tensors(i)
        t1 = time.time()
        lt = cfg["layer_types"][i]
        for n in names:
            h[n], ids, wts = R.decoder_layer(h[n], w, lt, cfg)
            ids, order = ids.sort(dim=-1)
            wts = wts.gather(1, order)
            save(os.path.join(dirs[n], f"layer_{i:02d}.npy"), h[n].numpy().astype(np.float32))
            save(os.path.join(dirs[n], f"experts_{i:02d}.npy"), ids.numpy().astype(np.int64))
            save(os.path.join(dirs[n], f"expert_weights_{i:02d}.npy"), wts.numpy().astype(np.float32))
        del w
        t2 = time.time()
        meta["layer_load_seconds"][i] = round(t1 - t0, 2)
        meta["layer_seconds"][i] = round(t2 - t0, 2)
        meta["peak_rss_gb"] = round(rss_gb(), 2)
        write_meta()
        done = i + 1
        print(f"layer {i:02d} ({lt[:4]}) load {t1-t0:5.1f}s compute {t2-t1:5.1f}s "
              f"total {t2-t0:5.1f}s  peak RSS {rss_gb():.1f} GiB", flush=True)

    meta["layers_done"] = done
    if done == cfg["num_hidden_layers"] or a.head_anyway:
        t0 = time.time()
        for n in names:
            x = R.rms_norm(h[n], g["model.norm.weight"], cfg["rms_norm_eps"])
            save(os.path.join(dirs[n], "final_norm.npy"), x.numpy())
            save(os.path.join(dirs[n], "logits.npy"), (x @ g["lm_head.weight"].T).numpy())
        meta["head_seconds"] = round(time.time() - t0, 2)
        print(f"head {meta['head_seconds']}s", flush=True)
    write_meta()


if __name__ == "__main__":
    main()
