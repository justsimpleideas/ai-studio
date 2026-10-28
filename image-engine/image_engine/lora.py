"""LoRA, implemented here instead of through a LoRA library (docs/training.md, "LoRA").

A LoRA adds a low-rank correction to selected linear layers:

    y = W x + strength * (alpha / rank) * B (A x)

W stays frozen; only A (rank x in) and B (out x rank) are trained or loaded.
Training needs two LoRAs on the same layers at once (a frozen training adapter
plus the one being trained), the adapter is removed before generating, and W
may be stored in fp8 by the backend's layerwise casting. Doing it here keeps
all of that visible in a hundred lines, with no extra dependency.

How it hooks in: a targeted nn.Linear (or fp8.ScaledFP8Linear, the fp8 layer of a
model imported from a single file) is replaced in its parent by a LoRALinear
that holds the original layer as `base` and calls it first. Any hooks on the
original layer (such as fp8 upcasting) keep working. A and B are plain
Parameters rather than nn.Linear modules, so layerwise casting, which acts on
module types, never stores them in fp8.

Files use the diffusers layout so other tools can load them:
`transformer.<module>.lora_A.weight` / `lora_B.weight`, with rank and alpha in
the safetensors metadata under "lora_adapter_metadata". Loading also accepts
the other layouts in common use (see `_parse_key`), and fails loudly on any key
it cannot place rather than silently ignoring part of a file.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .fp8 import ScaledFP8Linear

# Layers a LoRA can sit on: ordinary linear layers, and those a single-file model keeps in fp8.
LINEAR = (nn.Linear, ScaledFP8Linear)

#: Attention and feed-forward projections of the main transformer blocks.
DEFAULT_TARGETS = r"^layers\.\d+\.(attention\.(to_q|to_k|to_v|to_out\.0)|feed_forward\.(w1|w2|w3))$"
METADATA_KEY = "lora_adapter_metadata"
FILE_PREFIX = "transformer."
_NAME = re.compile(r"^[a-z0-9_]{1,40}$")
_PREFIXES = ("transformer.", "diffusion_model.", "model.diffusion_model.", "base_model.model.", "unet.")
_KOHYA_PREFIXES = ("lora_transformer_", "lora_unet_")


class LoRAError(Exception):
    """A LoRA cannot be applied. The message is safe to show to the user."""


@dataclass
class _Slot:
    rank: int
    alpha: float
    strength: float = 1.0


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.lora = nn.ParameterDict()
        self.slots: dict[str, _Slot] = {}

    def add(self, name: str, down: torch.Tensor, up: torch.Tensor, alpha: float, trainable: bool) -> None:
        if down.shape != (up.shape[1], self.base.in_features) or up.shape[0] != self.base.out_features:
            raise LoRAError(f"LoRA '{name}' has shapes {tuple(down.shape)}/{tuple(up.shape)}, "
                            f"which do not fit a {self.base.in_features}->{self.base.out_features} layer")
        self.lora[f"{name}_A"] = nn.Parameter(down, requires_grad=trainable)
        self.lora[f"{name}_B"] = nn.Parameter(up, requires_grad=trainable)
        self.slots[name] = _Slot(rank=down.shape[0], alpha=alpha)

    def remove(self, name: str) -> None:
        del self.lora[f"{name}_A"], self.lora[f"{name}_B"]
        del self.slots[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        for name, slot in self.slots.items():
            if slot.strength == 0:
                continue
            down, up = self.lora[f"{name}_A"], self.lora[f"{name}_B"]
            delta = F.linear(F.linear(x.to(down.dtype), down), up)
            out = out + (delta * (slot.strength * slot.alpha / slot.rank)).to(out.dtype)
        return out


def _check_name(name: str) -> None:
    if not _NAME.match(name):
        raise LoRAError(f"Invalid LoRA name {name!r}: use 1-40 lowercase letters, digits or underscores")


def _wrap(model: nn.Module, module_name: str) -> LoRALinear:
    try:
        module = model.get_submodule(module_name)
    except AttributeError as exc:
        raise LoRAError(f"The model has no layer named {module_name}") from exc
    if isinstance(module, LoRALinear):
        return module
    if not isinstance(module, LINEAR):
        raise LoRAError(f"{module_name} is a {type(module).__name__}, not a linear layer")
    parent_name, _, child = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    wrapped = LoRALinear(module)
    setattr(parent, child, wrapped)  # also works for ModuleList / Sequential children ("0")
    return wrapped


def _wrapped(model: nn.Module) -> dict[str, LoRALinear]:
    return {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}


def target_names(model: nn.Module, targets: str = DEFAULT_TARGETS) -> list[str]:
    pattern = re.compile(targets)
    names = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module = module.base
        elif name.endswith(".base") and isinstance(model.get_submodule(name[: -len(".base")]), LoRALinear):
            continue  # the original layer inside a LoRALinear: already counted under the wrapper's name
        if isinstance(module, LINEAR) and pattern.search(name):
            names.append(name)
    return names


def adapters(model: nn.Module) -> dict[str, dict]:
    """Every LoRA on the model: rank, alpha, strength, trainable, layer count."""
    found: dict[str, dict] = {}
    for module in _wrapped(model).values():
        for name, slot in module.slots.items():
            info = found.setdefault(name, {"rank": slot.rank, "alpha": slot.alpha, "strength": slot.strength,
                                           "trainable": module.lora[f"{name}_A"].requires_grad, "layers": 0})
            info["layers"] += 1
    return found


def add_trainable(model: nn.Module, name: str, rank: int, *, alpha: float | None = None,
                  targets: str = DEFAULT_TARGETS, dtype: torch.dtype = torch.float32,
                  generator: torch.Generator | None = None) -> list[nn.Parameter]:
    """A new LoRA that starts as a no-op (B = 0), on every layer matching `targets`.

    Returns its parameters, for the optimiser."""
    _check_name(name)
    if name in adapters(model):
        raise LoRAError(f"The model already has a LoRA named '{name}'")
    if not 1 <= rank <= 256:
        raise LoRAError("rank must be between 1 and 256")
    names = target_names(model, targets)
    if not names:
        raise LoRAError(f"No layer matches {targets}")
    params = []
    for module_name in names:
        layer = _wrap(model, module_name)
        device = layer.base.weight.device
        down = torch.empty(rank, layer.base.in_features, dtype=dtype, device=device)
        # Same initialisation as nn.Linear: B A starts at exactly zero, A does not.
        bound = 1 / math.sqrt(layer.base.in_features)
        down.uniform_(-bound, bound, generator=generator)
        up = torch.zeros(layer.base.out_features, rank, dtype=dtype, device=device)
        layer.add(name, down, up, float(alpha if alpha is not None else rank), trainable=True)
        params += [layer.lora[f"{name}_A"], layer.lora[f"{name}_B"]]
    return params


def set_strength(model: nn.Module, name: str, strength: float) -> None:
    found = False
    for module in _wrapped(model).values():
        if name in module.slots:
            module.slots[name].strength = float(strength)
            found = True
    if not found:
        raise LoRAError(f"No LoRA named '{name}'")


def remove(model: nn.Module, name: str) -> None:
    """Take a LoRA off. Layers left with no LoRA get their original nn.Linear back."""
    found = False
    for module_name, module in _wrapped(model).items():
        if name in module.slots:
            module.remove(name)
            found = True
        if not module.slots:
            parent_name, _, child = module_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, module.base)
    if not found:
        raise LoRAError(f"No LoRA named '{name}'")


def state_dict(model: nn.Module, name: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Tensors and safetensors metadata for one LoRA, in the diffusers layout."""
    tensors: dict[str, torch.Tensor] = {}
    slot = None
    for module_name, module in _wrapped(model).items():
        if name in module.slots:
            slot = module.slots[name]
            tensors[f"{FILE_PREFIX}{module_name}.lora_A.weight"] = module.lora[f"{name}_A"].detach().cpu().contiguous()
            tensors[f"{FILE_PREFIX}{module_name}.lora_B.weight"] = module.lora[f"{name}_B"].detach().cpu().contiguous()
    if slot is None:
        raise LoRAError(f"No LoRA named '{name}'")
    targets = sorted({k[len(FILE_PREFIX):].rsplit(".lora_", 1)[0] for k in tensors})
    meta = {f"{FILE_PREFIX}r": slot.rank, f"{FILE_PREFIX}lora_alpha": slot.alpha,
            f"{FILE_PREFIX}target_modules": targets}
    return tensors, {"format": "pt", METADATA_KEY: json.dumps(meta, indent=2, sort_keys=True)}


def load(model: nn.Module, name: str, tensors: dict[str, torch.Tensor], metadata: dict[str, str] | None = None,
         *, strength: float = 1.0, dtype: torch.dtype = torch.bfloat16) -> int:
    """Attach a frozen LoRA read from a file. Returns the number of layers it touches.

    All-or-nothing: if any key cannot be placed, nothing is attached."""
    _check_name(name)
    if name in adapters(model):
        raise LoRAError(f"The model already has a LoRA named '{name}'")
    kohya = {n.replace(".", "_"): n for n in target_names(model, r".")}
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    unplaced = []
    for key, tensor in tensors.items():
        parsed = _parse_key(key, kohya)
        if parsed is None:
            unplaced.append(key)
        else:
            module_name, part = parsed
            pairs.setdefault(module_name, {})[part] = tensor
    if unplaced:
        shown = ", ".join(unplaced[:5]) + (f" and {len(unplaced) - 5} more" if len(unplaced) > 5 else "")
        raise LoRAError(f"This LoRA has {len(unplaced)} entries this model has no place for: {shown}")
    if not pairs:
        raise LoRAError("The file contains no LoRA weights")
    default_alpha = _metadata_alpha(metadata)
    plan = []
    for module_name, parts in sorted(pairs.items()):
        if "A" not in parts or "B" not in parts:
            raise LoRAError(f"{module_name}: the file has only half of the LoRA (A or B is missing)")
        try:
            layer = model.get_submodule(module_name)
        except AttributeError as exc:
            raise LoRAError(f"The model has no layer named {module_name}") from exc
        base = layer.base if isinstance(layer, LoRALinear) else layer
        if not isinstance(base, LINEAR):
            raise LoRAError(f"{module_name} is a {type(base).__name__}, not a linear layer")
        down, up = parts["A"], parts["B"]
        if down.ndim != 2 or up.ndim != 2 or down.shape != (up.shape[1], base.in_features) \
                or up.shape[0] != base.out_features:
            raise LoRAError(f"{module_name}: LoRA shapes {tuple(down.shape)}/{tuple(up.shape)} do not fit "
                            f"a {base.in_features}->{base.out_features} layer (made for another model?)")
        alpha = float(parts["alpha"].item()) if "alpha" in parts else (default_alpha or float(down.shape[0]))
        plan.append((module_name, down, up, alpha))
    for module_name, down, up, alpha in plan:
        layer = _wrap(model, module_name)
        device = layer.base.weight.device
        layer.add(name, down.to(device=device, dtype=dtype), up.to(device=device, dtype=dtype), alpha,
                  trainable=False)
        layer.slots[name].strength = float(strength)
    return len(plan)


def _parse_key(key: str, kohya: dict[str, str]) -> tuple[str, str] | None:
    """(module name, "A" | "B" | "alpha") for one file key, or None."""
    for suffix, part in ((".lora_A.weight", "A"), (".lora_B.weight", "B"), (".lora_A.default.weight", "A"),
                         (".lora_B.default.weight", "B"), (".lora_down.weight", "A"), (".lora_up.weight", "B"),
                         (".alpha", "alpha")):
        if key.endswith(suffix):
            stem = key[: -len(suffix)]
            break
    else:
        return None
    for prefix in _KOHYA_PREFIXES:
        if stem.startswith(prefix):
            module_name = kohya.get(stem[len(prefix):])
            return (module_name, part) if module_name else None
    for prefix in _PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem, part


def _metadata_alpha(metadata: dict[str, str] | None) -> float | None:
    if not metadata or METADATA_KEY not in metadata:
        return None
    try:
        data: Any = json.loads(metadata[METADATA_KEY])
    except ValueError:
        return None
    for key in (f"{FILE_PREFIX}lora_alpha", "lora_alpha"):
        if isinstance(data, dict) and isinstance(data.get(key), (int, float)):
            return float(data[key])
    return None
