"""The image models the engine can run, told apart by their model folder.

Two families are supported:

* `diffusion`: the 6B model (backends/diffusion.py), as the diffusers folder
  fetch_model.py downloads, or as a single transformer file (fp8 or bf16) that
  scripts/import_model.py puts together with the rest of that folder. The distilled
  (few-step) model takes LoRAs and can be trained; a base one takes LoRAs.
* `large`: a 12B single-stream model with a text-fusion stage. Its transformer is a
  single file (fp8 or bf16) that scripts/import_model.py puts together with the text
  encoder and VAE it needs (backends/large.py). No LoRAs or training yet.

Each comes as a distilled model (8 steps, no guidance) or a base one (more steps, with
guidance); an imported folder says which in model_index.json (`is_distilled`).

A folder names its pipeline in model_index.json. That name only chooses a profile;
the engine always builds the pipeline from its own built-in classes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageProfile:
    name: str
    family: str  # which backend runs it
    label: str
    model_id: str  # stored with every image
    # The model's own defaults for a request.
    steps: int
    guidance: float
    distilled: bool
    loras: bool
    training: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TURBO = ImageProfile("turbo", "diffusion", "6B · 8 steps (distilled)", "turbo-6b",
                     steps=8, guidance=0.0, distilled=True, loras=True, training=True)
BASE_6B = ImageProfile("base", "diffusion", "6B · 30 steps", "base-6b",
                       steps=30, guidance=4.0, distilled=False, loras=True, training=False)
LARGE = ImageProfile("large", "large", "12B · 28 steps", "dit-12b",
                     steps=28, guidance=4.5, distilled=False, loras=False, training=False)
LARGE_FAST = ImageProfile("large-fast", "large", "12B · 8 steps (distilled)", "dit-12b-fast",
                          steps=8, guidance=0.0, distilled=True, loras=False, training=False)

# model_index.json `_class_name` -> family. Only these names are recognised; none is imported by name.
PIPELINE_FAMILIES = {"ZImagePipeline": "diffusion", "Krea2Pipeline": "large"}


class UnknownImageModel(Exception):
    pass


def read_index(folder: Path) -> dict[str, Any]:
    try:
        index = json.loads((folder / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UnknownImageModel(f"{folder} has no readable model_index.json") from exc
    if not isinstance(index, dict):
        raise UnknownImageModel(f"{folder}: model_index.json is not a JSON object")
    return index


def profile_of(folder: Path) -> ImageProfile:
    """The profile for a model folder. Raises UnknownImageModel if the folder names no
    supported pipeline, or has no readable model_index.json."""
    index = read_index(folder)
    name = index.get("_class_name")
    family = PIPELINE_FAMILIES.get(name) if isinstance(name, str) else None
    if family is None:
        raise UnknownImageModel(f"{folder} holds a '{name}' model; the image engine supports "
                                f"{', '.join(sorted(PIPELINE_FAMILIES))}")
    # The downloaded 6B folder is the distilled model and does not say so; imported folders do.
    distilled = index.get("is_distilled", family == "diffusion")
    if not isinstance(distilled, bool):
        raise UnknownImageModel(f"{folder}: is_distilled in model_index.json must be true or false")
    if family == "diffusion":
        return TURBO if distilled else BASE_6B
    return LARGE_FAST if distilled else LARGE


def try_profile_of(folder: str | Path) -> ImageProfile | None:
    """profile_of, or None for an unset, missing or unsupported folder (for status display)."""
    if not folder:
        return None
    try:
        return profile_of(Path(folder))
    except UnknownImageModel:
        return None
