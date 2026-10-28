"""Linear layers whose weights stay in fp8 (weight-only quantization).

A weight is stored as float8_e4m3fn together with an fp32 scale, either one per
tensor or one per output row, so that weight ≈ fp8 × scale. Before each matmul the
layer widens it back to the activations' dtype (bf16), so the arithmetic is the
same as for a bf16 model; only the storage, and the copies over PCIe when a model
is streamed, are halved.

Imported only where torch is installed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

FP8 = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8).max)  # 448


class ScaledFP8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, device: torch.device | str | None = None) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=FP8, device=device), requires_grad=False)
        # () for one scale per tensor, (out_features,) for one per output row.
        self.register_buffer("weight_scale", torch.ones((), dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device), requires_grad=False) if bias else None

    def dequantized(self, dtype: torch.dtype) -> torch.Tensor:
        scale = self.weight_scale
        if scale.ndim:
            scale = scale[:, None]
        return self.weight.to(dtype) * scale.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.dequantized(x.dtype), bias)

    def extra_repr(self) -> str:
        per = "row" if self.weight_scale.ndim else "tensor"
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, scale per {per}"


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8 weight and fp32 per-row scale for a float weight, so that weight ≈ fp8 × scale."""
    w = weight.float()
    scale = (w.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
    return (w / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale


def swap_in(model: nn.Module, names: set[str]) -> None:
    """Replace the nn.Linear modules at `names` (module paths) with empty ScaledFP8Linear ones on the meta device."""
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child) if not child.isdigit() else parent[int(child)]
        if not isinstance(old, nn.Linear):
            raise TypeError(f"{name} is a {type(old).__name__}, not a linear layer")
        new = ScaledFP8Linear(old.in_features, old.out_features, old.bias is not None, device="meta")
        if child.isdigit():
            parent[int(child)] = new
        else:
            setattr(parent, child, new)
