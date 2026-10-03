"""OpenAI-compatible mlx-lm server with the kolibri1 model registered.

Usage: uv run python serve.py --model models/Kolibri-1-4bit --port 8080 [mlx_lm.server flags]
"""

import kolibri_mlx.register  # noqa: F401  (must run before the model is loaded)
from mlx_lm.server import main

if __name__ == "__main__":
    main()
