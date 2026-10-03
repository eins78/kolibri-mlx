# SPDX-License-Identifier: CC0-1.0
"""Validate FP8 block dequantisation against the BF16 checkpoint.

Compares every tensor of the given shard of ``Aleph-Alpha/Kolibri-1`` (FP8) with
the same tensor in ``Aleph-Alpha/Kolibri-1-BF16``. Requires both shard files in
the Hugging Face cache (``hf download``).

Usage: uv run python scripts/check_dequant.py [--shard 1]
"""

import argparse
import glob
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kolibri_mlx.fp8 import dequantize_fp8_block  # noqa: E402

HUB = os.path.expanduser("~/.cache/huggingface/hub")


def shard_path(repo: str, idx: int) -> str:
    pat = f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*/model-{idx:05d}-of-00032.safetensors"
    hits = glob.glob(pat)
    if not hits:
        sys.exit(f"missing {pat}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=1)
    args = ap.parse_args()
    fp8 = shard_path("Aleph-Alpha/Kolibri-1", args.shard)
    bf16 = shard_path("Aleph-Alpha/Kolibri-1-BF16", args.shard)
    worst = 0.0
    n_fp8 = n_exact = n_other = 0
    with safe_open(fp8, "pt") as a, safe_open(bf16, "pt") as b:
        bkeys = set(b.keys())
        for k in a.keys():
            if k.endswith(".weight_scale_inv"):
                continue
            if k not in bkeys:
                print("not in bf16 shard:", k)
                continue
            ta = a.get_tensor(k)
            tb = b.get_tensor(k).float()
            if ta.dtype == torch.float8_e4m3fn:
                s = a.get_tensor(k[: -len(".weight")] + ".weight_scale_inv")
                td = dequantize_fp8_block(ta, s)
                # BF16 is the source; FP8 is a lossy copy. Error should be bounded
                # by fp8 e4m3 step (relative 2^-3) times the per-block max.
                rel = ((td - tb).abs().amax() / (tb.abs().amax() + 1e-12)).item()
                worst = max(worst, rel)
                n_fp8 += 1
                if rel > 0.07:
                    print(f"LARGE {k}: rel max err {rel:.4f}")
            else:
                if torch.equal(ta.float(), tb):
                    n_exact += 1
                else:
                    n_other += 1
                    print(f"MISMATCH non-fp8 {k}: max abs {(ta.float()-tb).abs().max().item()}")
    print(f"fp8 tensors: {n_fp8}, worst rel max err: {worst:.5f}")
    print(f"non-fp8 tensors exact: {n_exact}, mismatched: {n_other}")


if __name__ == "__main__":
    main()
