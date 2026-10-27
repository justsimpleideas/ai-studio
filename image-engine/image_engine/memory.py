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
