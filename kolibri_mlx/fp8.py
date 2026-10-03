"""FP8 block dequantisation for the Aleph-Alpha/Kolibri-1 checkpoint.

The checkpoint stores linear weights as ``float8_e4m3fn`` with one fp32
``weight_scale_inv`` per 128x128 block (``quantization_config.weight_block_size``).
Dequantised value = ``float(w) * scale[i // 128, j // 128]`` for weight element
``(i, j)`` of the ``[out, in]`` matrix. This is the single dequant used by the
converter, the torch reference and the MLX checks.
"""

from __future__ import annotations

import torch

BLOCK = 128


def dequantize_fp8_block(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block: int = BLOCK,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``weight`` (fp8, ``[out, in]``) dequantised to ``dtype``.

    ``scale_inv`` has shape ``[ceil(out/block), ceil(in/block)]`` in fp32.
    Partial edge blocks use the last scale row/column.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected 2-D weight, got {tuple(weight.shape)}")
    out_dim, in_dim = weight.shape
    rows = -(-out_dim // block)
    cols = -(-in_dim // block)
    if tuple(scale_inv.shape) != (rows, cols):
        raise ValueError(
            f"scale shape {tuple(scale_inv.shape)} does not match "
            f"weight {tuple(weight.shape)} with block {block}"
        )
    w = weight.to(torch.float32)
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, dim=0)[:out_dim, :]
    s = s.repeat_interleave(block, dim=1)[:, :in_dim]
    return (w * s).to(dtype)


def is_fp8_weight(name: str, tensors: dict | None = None) -> bool:
    """True if ``name`` is a weight with a companion ``weight_scale_inv``."""
    return name.endswith(".weight") and (
        tensors is None or name[: -len(".weight")] + ".weight_scale_inv" in tensors
    )
