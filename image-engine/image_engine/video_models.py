"""The video models the engine can run, told apart by their model folder.

Two variants of the same 5B text/image-to-video model are supported, with the same
architecture, text encoder, VAE and tokenizer (fetch_model.py `--model video` and
`--model video-fast`):

* standard: the original weights, sampled in 50 steps with classifier-free guidance;
* fast: a distillation (DMD) of the transformer that makes a clip in 3 steps without
  guidance. Its steps are fixed noise levels, and after each step the model's clean
  estimate is noised again to the next level rather than stepped along an ODE.

And the 14B models of the same family, one for text to video and one for image to video
(the image model takes only a start picture, the text model never does). Each is two
transformers, "experts": the first denoises from pure noise down to a boundary noise
level, the second from there to the end. They come only as single transformer files
(scripts/import_model.py), either the original weights (40 steps with guidance, as the
models' authors publish them) or a few-step distillation sampled by plain Euler steps on
a shifted schedule, with the steps and shift its publisher gives.

A folder names its pipeline in model_index.json. That name, the boundary between the
experts and a few-step setting the import writes are used only to choose a profile
here; the engine always builds the pipeline from its own built-in classes. A folder
imported from a single transformer file (scripts/import_model.py) carries the same
names: the import writes the distilled label when told the file is distilled.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
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
    # Frames per second the model makes, and the sizes it takes: sides in multiples of
    # `side_multiple` (the VAE's downscale times the transformer's 2x2 patches).
    fps: int = 24
    side_multiple: int = 32
    max_pixels: int = 1280 * 704
    width: int = 1280
    height: int = 704
    # What a clip starts from: "any" (text, or a start picture too), "text" or "image".
    start: str = "any"
    # Two experts: the second takes over below this fraction of the noise schedule.
    boundary: float | None = None
    # The second expert's guidance (None: one expert, or the same as `guidance`).
    guidance_2: float | None = None
    # A few-step distillation: plain Euler steps on a schedule shifted by this much.
    shift: float | None = None

    @property
    def experts(self) -> int:
        return 2 if self.boundary is not None else 1

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("sigmas")
        out["experts"] = self.experts
        return out


STANDARD = VideoProfile("standard", "Standard · 50 steps", steps=50, guidance=5.0, fixed=False)
FAST = VideoProfile("fast", "Fast · 3 steps (distilled)", steps=3, guidance=1.0, fixed=True,
                    sigmas=(1.0, 0.757, 0.522))

# The 14B models: 81 frames at 16 fps (5 seconds), 480p or 720p (sides in multiples of 16:
# the VAE shrinks 8x, the transformer patches 2x2). 480p is the default, the size a 12 GB
# card can make in a reasonable time. Steps, guidance and boundaries are the models' own.
_FOURTEEN_B = {"trained_frames": 81, "fps": 16, "side_multiple": 16, "max_pixels": 1280 * 720,
               "width": 832, "height": 480}
TEXT_14B = VideoProfile("text-14b", "14B text to video · 40 steps", steps=40, guidance=4.0, guidance_2=3.0,
                        fixed=False, start="text", boundary=0.875, **_FOURTEEN_B)
IMAGE_14B = VideoProfile("image-14b", "14B image to video · 40 steps", steps=40, guidance=3.5, guidance_2=3.5,
                         fixed=False, start="image", boundary=0.9, **_FOURTEEN_B)

# model_index.json `_class_name` -> profile. The distilled folder names a pipeline class that
# diffusers does not have; it is a label only and never imported.
PIPELINE_PROFILES = {"WanPipeline": STANDARD, "WanDMDPipeline": FAST}
# The same, for folders with two experts (a `boundary_ratio`).
TWO_EXPERT_PROFILES = {"WanPipeline": TEXT_14B, "WanImageToVideoPipeline": IMAGE_14B}

# What a few-step setting in model_index.json may hold.
FEW_STEP_KEY = "few_step"
MAX_FEW_STEPS = 16
MAX_SHIFT = 20.0


class UnknownVideoModel(Exception):
    pass


def few_step(profile: VideoProfile, steps: int, shift: float) -> VideoProfile:
    """A two-expert profile as a few-step distillation: fixed steps, no guidance, Euler on a shifted schedule."""
    what = "text" if profile.start == "text" else "image"
    return replace(profile, name=f"{profile.name}-few", label=f"14B {what} to video · {steps} steps (distilled)",
                   steps=steps, guidance=1.0, guidance_2=1.0, fixed=True, shift=shift)


def check_few_step(value: object) -> tuple[int, float]:
    """The (steps, shift) of a few-step setting; raises ValueError for anything else."""
    if not isinstance(value, dict) or set(value) != {"steps", "shift"}:
        raise ValueError(f"{FEW_STEP_KEY} must be {{\"steps\": n, \"shift\": x}}")
    steps, shift = value["steps"], value["shift"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_FEW_STEPS:
        raise ValueError(f"few-step steps must be 1 to {MAX_FEW_STEPS}")
    if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not math.isfinite(shift) or not 0 < shift <= MAX_SHIFT:
        raise ValueError(f"few-step shift must be above 0 and at most {MAX_SHIFT:g}")
    return steps, float(shift)


def profile_of(folder: Path) -> VideoProfile:
    """The profile for a model folder. Raises UnknownVideoModel if the folder names no
    supported pipeline, or has no readable model_index.json."""
    try:
        index = json.loads((folder / "model_index.json").read_text(encoding="utf-8"))
        name = index.get("_class_name") if isinstance(index, dict) else None
    except (OSError, ValueError) as exc:
        raise UnknownVideoModel(f"{folder} has no readable model_index.json") from exc
    two_experts = index.get("boundary_ratio") is not None
    table = TWO_EXPERT_PROFILES if two_experts else PIPELINE_PROFILES
    profile = table.get(name) if isinstance(name, str) else None
    if profile is None:
        supported = ", ".join(sorted({*PIPELINE_PROFILES, *TWO_EXPERT_PROFILES}))
        raise UnknownVideoModel(f"{folder} holds a '{name}' model; the video engine supports {supported}")
    if two_experts:
        if index.get("boundary_ratio") != profile.boundary:
            raise UnknownVideoModel(f"{folder}: the experts' boundary is {index.get('boundary_ratio')}, "
                                    f"this model's is {profile.boundary}")
        if FEW_STEP_KEY in index:
            try:
                profile = few_step(profile, *check_few_step(index[FEW_STEP_KEY]))
            except ValueError as exc:
                raise UnknownVideoModel(f"{folder}: {exc}") from exc
    return profile


def try_profile_of(folder: str | Path) -> VideoProfile | None:
    """profile_of, or None for an unset, missing or unsupported folder (for status display)."""
    if not folder:
        return None
    try:
        return profile_of(Path(folder))
    except UnknownVideoModel:
        return None
