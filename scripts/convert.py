"""Convert the Kolibri-1 FP8 checkpoint to an mlx-lm directory, one layer at a time.

Each decoder layer is dequantised, stacked into the MLX layout, quantised and
appended to a shard writer, so peak memory stays at a few GB instead of 78 GB.
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import mlx.nn as nn
import torch
from mlx.utils import tree_flatten

from kolibri_mlx.checkpoint import Checkpoint
from kolibri_mlx.models.kolibri1 import Kolibri1DecoderLayer, Model, ModelArgs

MAX_SHARD_BYTES = 5 * 1024**3  # same limit as mlx-lm's MAX_FILE_SIZE_GB
COPY_FILES = ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "LICENSE"]
REPO_ROOT = Path(__file__).resolve().parent.parent
T0 = time.time()
_log_fh = None


def log(msg=""):
    print(msg, flush=True)
    if _log_fh:
        _log_fh.write(msg + "\n")
        _log_fh.flush()


def rss_gb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
    return int(out.stdout.strip() or 0) / 1e6


def to_mx(t: torch.Tensor) -> mx.array:
    """torch -> mx without an fp32 detour for bf16 (numpy has no bf16)."""
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        return mx.array(t.view(torch.int16).numpy()).view(mx.bfloat16)
    return mx.array(t.numpy())


class ShardWriter:
    def __init__(self, out_dir: Path):
        self.dir = out_dir
        self.cur: dict[str, mx.array] = {}
        self.cur_bytes = 0
        self.shards: list[tuple[str, list[str]]] = []  # (tmp name, keys)
        self.total_size = 0
        self.total_params = 0

    def add(self, name: str, arr: mx.array, params: int):
        if self.cur and self.cur_bytes + arr.nbytes > MAX_SHARD_BYTES:
            self.flush()
        self.cur[name] = arr
        self.cur_bytes += arr.nbytes
        self.total_size += arr.nbytes
        self.total_params += params

    def flush(self):
        if not self.cur:
            return
        tmp = f"tmp-shard-{len(self.shards):05d}.safetensors"
        mx.save_safetensors(str(self.dir / tmp), self.cur, metadata={"format": "mlx"})
        self.shards.append((tmp, list(self.cur)))
        self.cur, self.cur_bytes = {}, 0
        mx.clear_cache()

    def finish(self):
        self.flush()
        n = len(self.shards)
        weight_map = {}
        for i, (tmp, keys) in enumerate(self.shards, 1):
            final = f"model-{i:05d}-of-{n:05d}.safetensors"
            os.rename(self.dir / tmp, self.dir / final)
            weight_map.update({k: final for k in keys})
        index = {
            "metadata": {"total_size": self.total_size, "total_parameters": self.total_params},
            "weight_map": dict(sorted(weight_map.items())),
        }
        with open(self.dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=4)


def make_predicate(args_q: dict, prefix: str):
    """Same rules as mlx-lm's quantize_model plus the model's quant_predicate.

    Per-group bits (`attn`, `expert`, `embed`, `lm_head`) override the default;
    0 keeps a group unquantised. Returns a dict so nn.quantize uses those bits.
    """
    model_pred = Model.quant_predicate.fget(None)
    group_size = args_q["group_size"]

    def predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[-1] % group_size != 0:
            return False
        if not model_pred(prefix + path, module):
            return False
        bits = args_q["bits"]
        if path.startswith(("self_attn.", "mlp.shared_experts.")):
            bits = args_q.get("attn_bits", bits)
        elif path.startswith("mlp.switch_mlp."):
            bits = args_q.get("expert_bits", bits)
        elif path == "embed_tokens":
            bits = args_q.get("embed_bits", bits)
        elif path == "lm_head":
            bits = args_q.get("lm_head_bits", bits)
        if not bits:
            return False
        return {"group_size": group_size, "bits": bits, "mode": "affine"}

    return predicate


OVERRIDES: dict = {}


def quantize_and_collect(module, prefix, writer, quant, args_q):
    """Quantise `module` in place (if requested), eval, hand tensors to the writer."""
    if quant:
        nn.quantize(module, class_predicate=make_predicate(args_q, prefix))
        # Record modules whose bits differ from the default, as mlx-lm does.
        for path, m in module.named_modules():
            if hasattr(m, "bits") and (m.bits != args_q["bits"] or m.group_size != args_q["group_size"]):
                OVERRIDES[prefix + path] = {"group_size": m.group_size, "bits": m.bits, "mode": "affine"}
    mx.eval(module.parameters())
    flat = tree_flatten(module.parameters())
    quantized = {k[: -len(".scales")] for k, _ in flat if k.endswith(".scales")}
    for k, v in flat:
        base = k.rsplit(".", 1)[0]
        if k.endswith((".scales", ".biases")) and base in quantized:
            params = 0  # quantisation side data is not a model parameter
        elif k.endswith(".weight") and base in quantized:
            params = v.size * 32 // args_q["bits"]
        else:
            params = v.size
        writer.add(prefix + k, v, params)


def build_layer_weights(ck, i, args, dtype):
    """Layer weights in MLX layout, relative to `model.layers.i.`.

    Experts are stacked one projection at a time (instead of through
    Model.sanitize for the whole layer) to avoid holding sources plus stacks
    of all three projections at once; sanitize still handles the renaming.
    """
    p = f"model.layers.{i}."
    keys = [k for k in ck.layer_keys(i) if not k.endswith(".weight_scale_inv")]
    w = {k: to_mx(ck.tensor(k, dtype)) for k in keys if ".mlp.experts." not in k}
    w = Model.sanitize(SimpleNamespace(args=args), w)
    out = [(k[len(p):], v) for k, v in w.items()]
    for n in ("up_proj", "down_proj", "gate_proj"):
        stacked = mx.stack([
            to_mx(ck.tensor(f"{p}mlp.experts.{e}.{n}.weight", dtype))
            for e in range(args.num_experts)
        ])
        mx.eval(stacked)
        out.append((f"mlp.switch_mlp.{n}.weight", stacked))
    return out


def main():
    global _log_fh
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-path", default="Aleph-Alpha/Kolibri-1")
    ap.add_argument("--mlx-path", default="models/Kolibri-1-4bit")
    ap.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--no-quant", action="store_true", help="write bf16, no quantisation")
    for name, what in [("attn", "self_attn.* and mlp.shared_experts.*"), ("expert", "mlp.switch_mlp.*"),
                       ("embed", "embed_tokens"), ("lm-head", "lm_head")]:
        ap.add_argument(f"--{name}-bits", type=int, help=f"bits for {what} (default --bits; 0 = bf16)")
    ap.add_argument("--layers", type=int, help="debug: convert only the first N layers")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", action="store_true", help="also log to run/convert-<timestamp>.out")
    a = ap.parse_args()

    if a.log:
        (REPO_ROOT / "run").mkdir(exist_ok=True)
        _log_fh = open(REPO_ROOT / "run" / f"convert-{datetime.now():%Y%m%d-%H%M%S}.out", "w")

    quant = not a.no_quant
    q = {"group_size": a.group_size, "bits": a.bits, "mode": "affine"}
    q_cfg = dict(q)
    for k in ("attn_bits", "expert_bits", "embed_bits", "lm_head_bits"):
        if getattr(a, k) is not None:
            q[k] = getattr(a, k)
    ck = Checkpoint(a.hf_path)
    config = dict(ck.config)
    n_layers = a.layers if a.layers is not None else config["num_hidden_layers"]
    if not 1 <= n_layers <= config["num_hidden_layers"]:
        sys.exit(f"--layers must be in 1..{config['num_hidden_layers']}")
    config["num_hidden_layers"] = n_layers
    config["layer_types"] = config["layer_types"][:n_layers]
    margs = ModelArgs.from_dict(config)
    out_dir = Path(a.mlx_path)

    log(f"source:   {ck.dir}")
    log(f"target:   {out_dir}")
    log(f"layers:   {n_layers}   quant: {q if quant else 'none (bf16)'}")

    if a.dry_run:
        log("\nlayer 0 tensors (checkpoint -> MLX layout):")
        for k, v in build_layer_weights(ck, 0, margs, torch.bfloat16):
            log(f"  {k:55s} {str(tuple(v.shape)):22s} {v.dtype}")
        for k, v in ck.global_tensors(torch.bfloat16).items():
            log(f"  {k:55s} {str(tuple(v.shape)):22s} {v.dtype}")
        log("\nplan: embed_tokens, layers 0..%d one at a time, norm, lm_head; shards <= 5 GB" % (n_layers - 1))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.safetensors")):
        sys.exit(f"{out_dir} already has shards; move it to the Trash first")

    writer = ShardWriter(out_dir)
    mx.reset_peak_memory()

    def step(label, fn):
        t = time.time()
        fn()
        ck.close()  # drop mmaps of source shards so RSS doesn't grow with file pages
        gc.collect()
        mx.clear_cache()
        log(f"{label:10s} {time.time() - t:6.1f}s  total {time.time() - T0:7.1f}s  "
            f"mlx peak {mx.get_peak_memory() / 1e9:5.2f} GB  rss {rss_gb():5.2f} GB")

    def do_global(prefix, name, module, weight):
        # nn.quantize swaps children only, so the module needs a parent
        holder = nn.Module()
        holder[name] = module
        holder.load_weights([(f"{name}.weight", weight)], strict=True)
        quantize_and_collect(holder, prefix, writer, quant, q)

    glob = ck.global_tensors(torch.bfloat16)
    if margs.tie_word_embeddings:
        glob.pop("lm_head.weight", None)

    def embed():
        do_global("model.", "embed_tokens", nn.Embedding(margs.vocab_size, margs.hidden_size),
                  to_mx(glob.pop("model.embed_tokens.weight")))
    step("embed", embed)

    for i in range(n_layers):
        def layer():
            weights = build_layer_weights(ck, i, margs, torch.bfloat16)
            lyr = Kolibri1DecoderLayer(margs, layer_idx=i)
            lyr.load_weights(weights, strict=True)
            del weights
            quantize_and_collect(lyr, f"model.layers.{i}.", writer, quant, q)
        step(f"layer {i}", layer)

    def norm():
        do_global("model.", "norm", nn.RMSNorm(margs.hidden_size, eps=margs.rms_norm_eps),
                  to_mx(glob.pop("model.norm.weight")))
    step("norm", norm)

    if not margs.tie_word_embeddings:
        def head():
            do_global("", "lm_head", nn.Linear(margs.hidden_size, margs.vocab_size, bias=False),
                      to_mx(glob.pop("lm_head.weight")))
        step("lm_head", head)

    writer.finish()

    config.pop("quantization_config", None)  # drop the HF fp8 block
    config.pop("quantization", None)
    if quant:
        config["quantization"] = {**q_cfg, **OVERRIDES}
        config["quantization_config"] = {**q_cfg, **OVERRIDES}
        log(f"per-module overrides: {len(OVERRIDES)}")
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    for name in COPY_FILES:
        src = Path(ck.dir) / name
        if src.exists():
            shutil.copy(src, out_dir / name)
        else:
            log(f"warning: {name} not found in snapshot")

    disk = sum(p.stat().st_size for p in out_dir.iterdir())
    log(f"\ndone: {len(writer.shards)} shards, {disk / 1e9:.2f} GB on disk, "
        f"{time.time() - T0:.1f}s, mlx peak {mx.get_peak_memory() / 1e9:.2f} GB, rss {rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
