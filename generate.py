"""mlx_lm.generate with the kolibri1 model registered.

Usage: uv run python generate.py --model models/Kolibri-1-4bit -p "..." -m 200
"""

import kolibri_mlx.register  # noqa: F401  (must run before the model is loaded)
from mlx_lm.generate import main

if __name__ == "__main__":
    main()
