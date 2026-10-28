"""Choosing where the models' parts live, given the GPU's memory (docs/image-engine-decisions.md, D4 and D8).

Approximate resident sizes for the image model:
  transformer  12.3 GB in bf16, ~6.2 GB stored as fp8
  text encoder  8.0 GB in bf16 (a 4B language model), needed once per new prompt
  VAE           0.2 GB
plus working memory for activations, which grows with image size.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 2**30


@dataclass(frozen=True)
class MemoryPlan:
    name: str
    label: str
    transformer_fp8: bool    # store transformer weights in fp8, compute in bf16
    swap_text_encoder: bool  # keep the text encoder in RAM, move it to the GPU only to encode
    cpu_offload: bool        # move every component to the GPU only while it runs


PLANS = {
    "full": MemoryPlan("full", "everything on GPU (bf16)", False, False, False),
    "quality": MemoryPlan("quality", "bf16, text encoder swapped", False, True, False),
    "balanced": MemoryPlan("balanced", "fp8 weights, text encoder swapped", True, True, False),
    "low": MemoryPlan("low", "fp8 weights, full CPU offload", True, False, True),
}

# Minimum total VRAM for each automatic choice, leaving room for activations and the desktop.
AUTO_THRESHOLDS = (("full", 23 * GIB), ("quality", 15.5 * GIB), ("balanced", 9.5 * GIB))


def choose_plan(mode: str, total_vram_bytes: int) -> MemoryPlan:
    if mode != "auto":
        return PLANS[mode]
    for name, minimum in AUTO_THRESHOLDS:
        if total_vram_bytes >= minimum:
            return PLANS[name]
    return PLANS["low"]


# ---------------------------------------------------------------- 12B image model
#
# Approximate resident sizes for the 12B image model (backends/large.py):
#   transformer  ~13 GB with its linear weights in fp8 (always: a bf16 file is quantized as it loads)
#   text encoder  8.1 GB in bf16 (a 4B language model; its vision tower is dropped), once per new prompt
#   VAE           0.25 GB in bf16
# plus activations: at 1024x1024 the transformer attends over ~4,600 tokens.
#
# The transformer alone does not fit a 12 GB card. Streamed, its blocks wait in pinned RAM
# and visit the GPU one at a time, the next one copied while the current one computes;
# a block's matmuls take longer than its copy over PCIe, so streaming costs little.


@dataclass(frozen=True)
class LargeMemoryPlan:
    name: str
    label: str
    stream_transformer: bool  # blocks wait in RAM and visit the GPU one at a time while sampling
    text_encoder: str         # "gpu": stays there; "swap": moved there to encode; "cpu": encodes on the CPU


LARGE_PLANS = {
    "full": LargeMemoryPlan("full", "everything on GPU (fp8 weights)", False, "gpu"),
    "quality": LargeMemoryPlan("quality", "fp8 weights on GPU, text encoder on CPU", False, "cpu"),
    "balanced": LargeMemoryPlan("balanced", "fp8 weights streamed from RAM, text encoder swapped", True, "swap"),
    "low": LargeMemoryPlan("low", "fp8 weights streamed from RAM, text encoder on CPU", True, "cpu"),
}

LARGE_AUTO_THRESHOLDS = (("full", 26 * GIB), ("quality", 17 * GIB), ("balanced", 10.5 * GIB))


def choose_large_plan(mode: str, total_vram_bytes: int) -> LargeMemoryPlan:
    if mode != "auto":
        return LARGE_PLANS[mode]
    for name, minimum in LARGE_AUTO_THRESHOLDS:
        if total_vram_bytes >= minimum:
            return LARGE_PLANS[name]
    return LARGE_PLANS["low"]


# ---------------------------------------------------------------- video
#
# Approximate resident sizes for the video model:
#   transformer  10 GB in bf16, ~5 GB stored as fp8 (published in fp32: 20 GB on disk)
#   text encoder 11.4 GB in bf16 (a 5.7B encoder), needed once per new prompt
#   VAE           2.8 GB in fp32
# plus activations: at 1280x704 and 121 frames the transformer attends over ~27,000 tokens.
#
# The text encoder alone does not fit next to anything on a 12 GB card, and swapping
# 11 GB over PCIe for every new prompt costs more than running it on the CPU, where a
# prompt takes seconds. So below 30 GB it stays in system RAM and runs there.


@dataclass(frozen=True)
class VideoMemoryPlan:
    name: str
    label: str
    transformer_fp8: bool      # store transformer weights in fp8, compute in bf16
    text_encoder_on_gpu: bool  # otherwise it stays in RAM and encodes on the CPU
    vae_on_gpu: bool           # otherwise it moves to the GPU only to decode
    park_transformer: bool     # move the transformer to RAM while the VAE decodes
    group_offload: bool        # stream transformer blocks from RAM while sampling


VIDEO_PLANS = {
    "full": VideoMemoryPlan("full", "everything on GPU (bf16)", False, True, True, False, False),
    "quality": VideoMemoryPlan("quality", "bf16, text encoder on CPU", False, False, True, False, False),
    "balanced": VideoMemoryPlan("balanced", "fp8 weights, text encoder on CPU", True, False, False, True, False),
    "low": VideoMemoryPlan("low", "streamed from RAM, text encoder on CPU", False, False, False, False, True),
}

VIDEO_AUTO_THRESHOLDS = (("full", 30 * GIB), ("quality", 18 * GIB), ("balanced", 9.5 * GIB))


def choose_video_plan(mode: str, total_vram_bytes: int) -> VideoMemoryPlan:
    if mode != "auto":
        return VIDEO_PLANS[mode]
    for name, minimum in VIDEO_AUTO_THRESHOLDS:
        if total_vram_bytes >= minimum:
            return VIDEO_PLANS[name]
    return VIDEO_PLANS["low"]


# ---------------------------------------------------------------- 14B video models
#
# Approximate resident sizes for a 14B video model (two experts, backends/video.py):
#   each expert   ~14.3 GB with its linear weights in fp8 (always: a bf16 file is quantized as it loads)
#   text encoder  11.4 GB in bf16 (the same encoder as the 5B model's), once per new prompt
#   VAE            0.5 GB in fp32
# plus activations: at 832x480 and 81 frames the transformer attends over ~33,000 tokens.
#
# Only one expert runs at a time: the first denoises down to the boundary, the second the
# rest. An expert alone does not fit a 12 GB card, so there its blocks wait in RAM and visit
# the GPU one at a time. Both experts together (~29 GB) and the text encoder do not fit
# 32 GB of RAM either, so unless there is room for both, the first is dropped at the
# boundary and the second read from its file, and the other way round for the next clip:
# about 14 GB read from disk per expert switch, seconds from an internal SSD.


@dataclass(frozen=True)
class TwoExpertPlan:
    name: str
    label: str
    stream_experts: bool       # blocks wait in RAM and visit the GPU one at a time while sampling
    text_encoder_on_gpu: bool  # otherwise it stays in RAM and encodes on the CPU
    # The VAE (0.5 GB) stays on the GPU in every plan: the image model encodes the start picture
    # together with the whole clip's empty frames, too slow on the CPU. Nothing is parked to decode.
    vae_on_gpu: bool = True
    park_transformer: bool = False


TWO_EXPERT_PLANS = {
    "full": TwoExpertPlan("full", "fp8 experts on GPU, everything resident", False, True),
    "quality": TwoExpertPlan("quality", "fp8 experts on GPU, text encoder on CPU", False, False),
    "balanced": TwoExpertPlan("balanced", "fp8 experts streamed from RAM, text encoder on CPU", True, False),
    "low": TwoExpertPlan("low", "fp8 experts streamed from RAM, one at a time, text encoder on CPU", True, False),
}

TWO_EXPERT_AUTO_THRESHOLDS = (("full", 40 * GIB), ("quality", 22 * GIB))

# RAM left to the system and the rest of the engine when deciding whether both experts stay loaded.
RAM_HEADROOM = 8 * GIB


def choose_two_expert_plan(mode: str, total_vram_bytes: int) -> TwoExpertPlan:
    if mode != "auto":
        return TWO_EXPERT_PLANS[mode]
    for name, minimum in TWO_EXPERT_AUTO_THRESHOLDS:
        if total_vram_bytes >= minimum:
            return TWO_EXPERT_PLANS[name]
    return TWO_EXPERT_PLANS["balanced"]


def keep_both_experts(total_ram_bytes: int, expert_bytes: int, text_encoder_bytes: int, plan: TwoExpertPlan) -> bool:
    """Whether both experts fit in RAM with the text encoder (when it lives there) and some headroom.
    The Low VRAM plan never keeps both: it is the one for machines short of everything."""
    if plan.name == "low":
        return False
    text = 0 if plan.text_encoder_on_gpu else text_encoder_bytes
    return total_ram_bytes >= 2 * expert_bytes + text + RAM_HEADROOM
