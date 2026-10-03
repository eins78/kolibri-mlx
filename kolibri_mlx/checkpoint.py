# SPDX-License-Identifier: CC0-1.0
"""Streamed access to the Aleph-Alpha/Kolibri-1 FP8 checkpoint.

Tensors are read shard by shard through ``safetensors.safe_open`` and FP8 weights
are dequantised on the fly, so one layer at a time fits in memory.
"""

from __future__ import annotations

import glob
import json
import os
import re
from functools import lru_cache

import torch
from safetensors import safe_open

from kolibri_mlx.fp8 import dequantize_fp8_block

HUB = os.path.expanduser("~/.cache/huggingface/hub")
DEFAULT_REPO = "Aleph-Alpha/Kolibri-1"


def snapshot_dir(repo: str = DEFAULT_REPO) -> str:
    """Local snapshot directory of ``repo`` in the Hugging Face cache."""
    if os.path.isdir(repo):
        return repo
    hits = glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*/config.json")
    if not hits:
        raise FileNotFoundError(f"{repo} not in {HUB}; run `hf download {repo}`")
    return os.path.dirname(hits[0])


class Checkpoint:
    """Lazy reader over a sharded safetensors checkpoint."""

    def __init__(self, path: str = DEFAULT_REPO):
        self.dir = snapshot_dir(path)
        with open(os.path.join(self.dir, "config.json")) as f:
            self.config = json.load(f)
        with open(os.path.join(self.dir, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._handles: dict[str, object] = {}

    @property
    def num_layers(self) -> int:
        return self.config["num_hidden_layers"]

    def keys(self):
        return self.weight_map.keys()

    def layer_keys(self, layer: int) -> list[str]:
        prefix = f"model.layers.{layer}."
        return sorted(k for k in self.weight_map if k.startswith(prefix))

    def _handle(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(self.dir, shard), "pt")
            self._handles[shard] = h
        return h

    def raw(self, key: str) -> torch.Tensor:
        return self._handle(self.weight_map[key]).get_tensor(key)

    def tensor(self, key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return ``key`` dequantised (if FP8) and cast to ``dtype``."""
        t = self.raw(key)
        if t.dtype == torch.float8_e4m3fn:
            scale = self.raw(key[: -len(".weight")] + ".weight_scale_inv")
            return dequantize_fp8_block(t, scale, dtype=dtype)
        return t.to(dtype)

    def layer_tensors(self, layer: int, dtype: torch.dtype = torch.float32) -> dict:
        """All tensors of one decoder layer, keyed relative to ``model.layers.N.``.

        ``weight_scale_inv`` entries are consumed by dequantisation and omitted.
        """
        prefix = f"model.layers.{layer}."
        out = {}
        for k in self.layer_keys(layer):
            if k.endswith(".weight_scale_inv"):
                continue
            out[k[len(prefix):]] = self.tensor(k, dtype)
        return out

    def global_tensors(self, dtype: torch.dtype = torch.float32) -> dict:
        """Embedding, final norm and lm_head."""
        return {
            k: self.tensor(k, dtype)
            for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
        }

    def close(self) -> None:
        self._handles.clear()


def expert_index(key: str) -> int | None:
    m = re.search(r"\.experts\.(\d+)\.", key)
    return int(m.group(1)) if m else None
