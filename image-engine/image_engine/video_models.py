"""The video models the engine can run, told apart by their model folder.

Two variants of the same 5B text/image-to-video model are supported, with the same
architecture, text encoder, VAE and tokenizer (fetch_model.py `--model video` and
`--model video-fast`):

* standard: the original weights, sampled in 50 steps with classifier-free guidance;
* fast: a distillation (DMD) of the transformer that makes a clip in 3 steps without
  guidance. Its steps are fixed noise levels, and after each step the model's clean
  estimate is noised again to the next level rather than stepped along an ODE.

A folder names its pipeline in model_index.json. That name is used only to choose a
profile here; the engine always builds the pipeline from its own built-in classes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoProfile:
    name: str
    label: str
    # Defaults for a request; with `fixed` they are the only values the model takes.
    steps: int
    guidance: float
    fixed: bool
    # The noise levels of a fixed few-step schedule, highest first (None: the folder's own scheduler).
    sigmas: tuple[float, ...] | None = None
    # The clip length the model was trained on; other lengths work but may degrade toward the end.
    trained_frames: int = 121

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("sigmas")
        return out


STANDARD = VideoProfile("standard", "Standard · 50 steps", steps=50, guidance=5.0, fixed=False)
FAST = VideoProfile("fast", "Fast · 3 steps (distilled)", steps=3, guidance=1.0, fixed=True,
                    sigmas=(1.0, 0.757, 0.522))

# model_index.json `_class_name` -> profile. The distilled folder names a pipeline class that
# diffusers does not have; it is a label only and never imported.
PIPELINE_PROFILES = {"WanPipeline": STANDARD, "WanDMDPipeline": FAST}


class UnknownVideoModel(Exception):
    pass


def profile_of(folder: Path) -> VideoProfile:
    """The profile for a model folder. Raises UnknownVideoModel if the folder names no
    supported pipeline, or has no readable model_index.json."""
    try:
        index = json.loads((folder / "model_index.json").read_text(encoding="utf-8"))
        name = index.get("_class_name") if isinstance(index, dict) else None
    except (OSError, ValueError) as exc:
        raise UnknownVideoModel(f"{folder} has no readable model_index.json") from exc
    profile = PIPELINE_PROFILES.get(name) if isinstance(name, str) else None
    if profile is None:
        supported = ", ".join(sorted(PIPELINE_PROFILES))
        raise UnknownVideoModel(f"{folder} holds a '{name}' model; the video engine supports {supported}")
    return profile


def try_profile_of(folder: str | Path) -> VideoProfile | None:
    """profile_of, or None for an unset, missing or unsupported folder (for status display)."""
    if not folder:
        return None
    try:
        return profile_of(Path(folder))
    except UnknownVideoModel:
        return None
