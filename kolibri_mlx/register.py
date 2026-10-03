"""Register the out-of-tree ``kolibri1`` model with mlx-lm.

mlx-lm resolves ``model_type`` with ``importlib.import_module("mlx_lm.models.<type>")``,
which returns an existing ``sys.modules`` entry first. Inserting our module under
that name makes ``mlx_lm.load`` and the CLIs find it without patching the install.
"""

import sys

from kolibri_mlx.models import kolibri1

MODULE_NAME = "mlx_lm.models.kolibri1"


def register() -> None:
    sys.modules.setdefault(MODULE_NAME, kolibri1)


register()
