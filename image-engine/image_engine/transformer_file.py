"""Building a diffusers transformer from a single checked .safetensors file.

single_file.inspect has already matched every tensor of the file to a parameter of the
model (diffusers name, the rows of a fused weight it comes from, its fp8 scale). This
builds the model with empty weights and fills each parameter exactly once from the
file, then checks that none was left empty. Linear weights stored in fp8 stay in fp8
(fp8.ScaledFP8Linear); others can be quantized to fp8 per output row on the way in.

Imported only where torch is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import fp8
from .single_file import (Inspection, LargeConfig, SixBConfig, SingleFileError, VideoConfig, check_six_b_config,
                          check_video_config)

GIB = 2**30
PIN_BLOCK = 256 * 2**20  # a power of two, so PyTorch locks exactly this much for each block
PIN_ALIGN = 64
PIN_SHARE_OF_RAM = 0.5   # page-locked memory cannot be paged out; leave the rest to the system


def pin_budget() -> int:
    """How much RAM the transformer may lock: half of it, or 8 GiB when that is unknown."""
    try:
        import psutil

        return int(psutil.virtual_memory().total * PIN_SHARE_OF_RAM)
    except Exception:
        return 8 * GIB


def clear_refused_lock() -> None:
    """Consume the error a refused page-locked allocation leaves behind.

    PyTorch raises the refusal, but CUDA also keeps it as its last error, and the next GPU
    operation, whatever it is, reports it again as "CUDA error: out of memory". One small
    operation run here takes that report instead; the ones after it run normally.
    """
    if not torch.cuda.is_available():
        return
    try:
        torch.zeros(1, device="cuda").add_(1)
    except RuntimeError:
        pass


class PinnedArena:
    """Page-locked memory handed out as slices of a few large blocks.

    PyTorch rounds every page-locked allocation up to a power of two, so pinning a model
    tensor by tensor locks up to twice its size, enough on a 32 GB machine for Windows to
    refuse partway (as "CUDA error: out of memory", though no GPU memory is involved, and
    again at the next GPU operation unless cleared: clear_refused_lock).
    Slices of fixed-size blocks lock what they hold. Past the budget, or once the system
    refuses a block, tensors stay pageable: streaming then pins them only while they are
    copied to the GPU.
    """

    def __init__(self, budget: int, block: int = PIN_BLOCK) -> None:
        self.budget = budget
        self.block_size = block
        self.locked = 0
        self.complete = True  # every tensor kept so far is pinned
        self._block: torch.Tensor | None = None
        self._offset = 0

    def keep(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.contiguous()
        size = tensor.numel() * tensor.element_size()
        if size == 0:
            return tensor
        if size > self.block_size:
            self.complete = False
            return tensor
        start = -(-self._offset // PIN_ALIGN) * PIN_ALIGN
        if self._block is None or start + size > self.block_size:
            if not self._new_block():
                self.complete = False
                return tensor
            start = 0
        out = self._block[start:start + size].view(tensor.dtype).view(tensor.shape)
        out.copy_(tensor)
        self._offset = start + size
        return out

    def _new_block(self) -> bool:
        if self.locked + self.block_size > self.budget:
            return False
        try:
            self._block = torch.empty(self.block_size, dtype=torch.uint8, pin_memory=True)
        except RuntimeError:
            clear_refused_lock()
            self.budget = self.locked  # the system refused; ask no more
            return False
        self.locked += self.block_size
        self._offset = 0
        return True


def fill(model: nn.Module, path: Path, inspection: Inspection, quantize: set[str], pin: bool,
         pin_budget_bytes: int | None = None, widen: set[str] = frozenset()) -> bool:
    """Load the file into `model` (built on the meta device). `quantize`: module names of the
    linear layers to keep in fp8: those stored in fp8 must be among them or in `widen`, whose
    fp8 weights are multiplied out to an ordinary float weight as they load; float ones in
    `quantize` are quantized per output row. With `pin`, tensors go to page-locked memory as
    they are read (PinnedArena), up to `pin_budget_bytes` (default pin_budget()); the rest stay
    pageable. Returns whether all were pinned."""
    from safetensors import safe_open

    stored_fp8 = {name[: -len(".weight")] for name in inspection.quantized}
    if not stored_fp8 <= quantize | widen or quantize & widen:
        raise SingleFileError("Every fp8 weight must load into an fp8 layer or be widened")
    fp8.swap_in(model, quantize)
    keep_fp32 = set(getattr(type(model), "_keep_in_fp32_modules", None) or ())
    expected = inspection.expected
    arena = PinnedArena(pin_budget() if pin_budget_bytes is None else pin_budget_bytes) if pin else None

    def keep(tensor: torch.Tensor) -> torch.Tensor:
        return arena.keep(tensor) if arena is not None else tensor.contiguous()

    def put(module: nn.Module, name: str, tensor: torch.Tensor, buffer: bool = False) -> None:
        if buffer:
            module._buffers[name] = keep(tensor)
        else:
            module._parameters[name] = nn.Parameter(keep(tensor), requires_grad=False)

    cached: tuple[str, torch.Tensor] | None = None  # a fused weight is read once for its three parts
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        for new, old in inspection.names.items():
            module_name, _, attr = new.rpartition(".")
            module = model.get_submodule(module_name) if module_name else model
            if cached is None or cached[0] != old:
                cached = (old, fh.get_tensor(old))
            tensor = cached[1]
            rows = inspection.rows.get(new)
            if rows is not None:
                tensor = tensor[rows[0]:rows[1]]
            if module_name in quantize and attr == "weight":
                quant = inspection.quantized.get(new)
                if quant is not None:
                    if tensor.dtype != fp8.FP8:
                        raise SingleFileError(f"{old} should be fp8 but reads as {tensor.dtype}")
                    scale = fh.get_tensor(quant.scale).float()
                    if quant.rows is not None:
                        scale = scale[quant.rows[0]:quant.rows[1]]
                    weight, scale = tensor, scale.reshape(quant.scale_shape)
                else:
                    weight, scale = fp8.quantize_rows(tensor)
                put(module, "weight", weight)
                put(module, "weight_scale", scale, buffer=True)
                continue
            if module_name in widen and attr == "weight":
                quant = inspection.quantized[new]
                scale = fh.get_tensor(quant.scale).float()
                if quant.rows is not None:
                    scale = scale[quant.rows[0]:quant.rows[1]]
                scale = scale.reshape(quant.scale_shape)
                tensor = tensor.float() * (scale[:, None] if scale.ndim else scale)
            tensor = tensor.reshape(expected[new])  # e.g. modulation tables stored flat
            fp32 = any(part in keep_fp32 for part in new.split("."))
            put(module, attr, tensor.to(torch.float32 if fp32 else torch.bfloat16))

    left = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if left:
        raise SingleFileError(f"The transformer file did not fill {len(left)} weights (first: {left[0]})")
    model.eval()
    return arena is not None and arena.complete


# Linear layers quantized when a bf16 file of the 12B model loads: the transformer blocks and
# the text fusion stage (all but ~3% of the weights). The small input and output layers stay in bf16.
LARGE_QUANTIZE_PREFIXES = ("transformer_blocks.", "text_fusion.")


def build_large(path: Path, inspection: Inspection, pin: bool) -> tuple[Any, bool]:
    """The 12B transformer on the CPU, its linear weights in fp8 whatever the file stores."""
    from accelerate import init_empty_weights
    from diffusers import Krea2Transformer2DModel

    if not isinstance(inspection.config, LargeConfig):
        raise SingleFileError("Not the 12B image model")
    with init_empty_weights():
        model = Krea2Transformer2DModel(**inspection.config.to_kwargs())
    if inspection.quantized:
        quantize = {name[: -len(".weight")] for name in inspection.quantized}
    else:
        quantize = {name[: -len(".weight")] for name in inspection.names
                    if name.endswith(".weight") and name.startswith(LARGE_QUANTIZE_PREFIXES)
                    and isinstance(model.get_submodule(name[: -len(".weight")]), nn.Linear)}
    pinned = fill(model, path, inspection, quantize, pin)
    return model, pinned


def build_six_b(path: Path, inspection: Inspection, config: dict[str, Any]) -> Any:
    """The 6B transformer on the CPU. fp8 weights stay fp8; a bf16 file stays bf16, so the memory
    plan can treat it like the downloaded model."""
    from accelerate import init_empty_weights
    from diffusers import ZImageTransformer2DModel

    with init_empty_weights():
        if not isinstance(inspection.config, SixBConfig):
            raise SingleFileError("Not the 6B image model")
        model = ZImageTransformer2DModel.from_config(check_six_b_config(config, inspection.config))
    fill(model, path, inspection, {name[: -len(".weight")] for name in inspection.quantized}, pin=False)
    return model


# The video transformer's layers kept in fp8: the blocks' linear layers (all but ~1% of the
# weights). Outside them an fp8 weight is widened as it loads: diffusers casts the timestep
# embedding's input to that layer's own dtype, which must not be fp8.
VIDEO_QUANTIZE_PREFIX = "blocks."


def build_video(path: Path, inspection: Inspection, config: dict[str, Any], quantize: bool = False) -> Any:
    """The video transformer on the CPU. The blocks' fp8 weights stay fp8, with `quantize` (the
    14B experts) their float weights become fp8 too; otherwise a bf16 (or fp16, fp32) file loads
    in bf16 like the downloaded model, its norms and modulation tables in fp32 as diffusers keeps them."""
    from accelerate import init_empty_weights
    from diffusers import WanTransformer3DModel

    if not isinstance(inspection.config, VideoConfig):
        raise SingleFileError("Not the video model")
    with init_empty_weights():
        model = WanTransformer3DModel.from_config(check_video_config(config, inspection.config))
    stored = {name[: -len(".weight")] for name in inspection.quantized}
    in_blocks = {name for name in stored if name.startswith(VIDEO_QUANTIZE_PREFIX)}
    keep = set(in_blocks)
    if quantize:
        keep |= {name[: -len(".weight")] for name in inspection.names
                 if name.endswith(".weight") and name.startswith(VIDEO_QUANTIZE_PREFIX)
                 and isinstance(model.get_submodule(name[: -len(".weight")]), nn.Linear)}
    fill(model, path, inspection, keep, pin=False, widen=stored - in_blocks)
    return model
