# SPDX-License-Identifier: CC0-1.0
"""OpenAI-compatible mlx-lm server with the kolibri1 model registered.

Usage: uv run python serve.py --model models/Kolibri-1-4bit-mixed --port 8080 [mlx_lm.server flags]

Set MLX_CACHE_LIMIT_GB (for example 1) to cap MLX's buffer cache; with a 45 GB model and
long prompts the default cache can push Metal into out-of-memory.
"""

import os

import mlx.core as mx

import kolibri_mlx.register  # noqa: F401  (must run before the model is loaded)
from mlx_lm.server import main


def _apply_cache_limit() -> None:
    gb = os.environ.get("MLX_CACHE_LIMIT_GB")
    if gb:
        mx.set_cache_limit(int(float(gb) * 1024**3))


if __name__ == "__main__":
    _apply_cache_limit()
    main()
