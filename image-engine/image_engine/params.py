"""Validation of generation requests.

A request is plain JSON so that the same shape can later be produced by the
node-graph editor (see docs/decisions.md, D1). Everything coming from the HTTP
API goes through `parse_request` before it reaches the GPU.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import asdict, dataclass
from typing import ClassVar

from .video_models import STANDARD, VideoProfile

MAX_PROMPT_CHARS = 4000
MIN_SIDE, MAX_SIDE, SIDE_MULTIPLE = 256, 2048, 16
MAX_PIXELS = 2048 * 2048
MAX_STEPS = 50
MAX_GUIDANCE = 10.0
MAX_COUNT = 16
MAX_SEED = 2**32 - 1
# LoRAs from the library applied to one image request, and how strongly.
MAX_LORAS = 3
MAX_LORA_STRENGTH = 2.0

# Defaults recommended for the image model: 8 steps (8 transformer passes in diffusers >= 0.42), no CFG.
DEFAULTS = {"width": 1024, "height": 1024, "steps": 8, "guidance": 0.0, "count": 1}

# The 5B video model (text or image to video). Sides are multiples of 32: the VAE
# shrinks each side 16x and the transformer patches 2x2. Frame counts are 4k + 1,
# because the VAE compresses time 4x after the first frame. It was trained at
# 1280x704 and 121 frames, which is 5 seconds at its 24 fps. The 14B models differ
# (video_models.py); a request is checked against the configured model's profile.
VIDEO_SIDE_MULTIPLE = 32
VIDEO_MIN_SIDE, VIDEO_MAX_SIDE = 256, 1280
VIDEO_MAX_PIXELS = 1280 * 704
VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES, VIDEO_FRAME_STEP = 5, 121, 4
VIDEO_FPS = 24
VIDEO_MAX_COUNT = 4
VIDEO_DEFAULTS = {"width": 1280, "height": 704, "frames": 121, "steps": 50, "guidance": 5.0, "count": 1}
INPUT_ID = re.compile(r"^[0-9a-f]{16}$")


class RequestError(ValueError):
    """The request is malformed. The message is safe to show to the user."""


@dataclass(frozen=True)
class LoraUse:
    id: str
    strength: float


@dataclass(frozen=True)
class GenerationRequest:
    kind: ClassVar[str] = "image"

    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    count: int
    loras: tuple[LoraUse, ...] = ()

    def seed_for(self, index: int) -> int:
        """Images in a batch use consecutive seeds, so any one of them can be reproduced alone."""
        return (self.seed + index) % (MAX_SEED + 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["loras"] = list(d["loras"])
        if not d["loras"]:
            del d["loras"]  # requests without LoRAs keep the shape they always had
        return d


@dataclass(frozen=True)
class VideoRequest:
    kind: ClassVar[str] = "video"

    prompt: str
    negative_prompt: str
    width: int
    height: int
    frames: int
    fps: int
    steps: int
    guidance: float
    seed: int
    count: int
    # An engine input id (see Gallery.save_input): the video starts from this picture.
    start_image: str | None = None

    def seed_for(self, index: int) -> int:
        return (self.seed + index) % (MAX_SEED + 1)

    @property
    def seconds(self) -> float:
        return self.frames / self.fps

    def to_dict(self) -> dict:
        return {"kind": self.kind, **asdict(self)}


def _int(data: dict, key: str, lo: int, hi: int, defaults: dict = DEFAULTS) -> int:
    value = data.get(key, defaults.get(key))
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(f"{key} must be an integer")
    if not lo <= value <= hi:
        raise RequestError(f"{key} must be between {lo} and {hi}")
    return value


def _float(data: dict, key: str, lo: float, hi: float, defaults: dict = DEFAULTS) -> float:
    value = data.get(key, defaults.get(key))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(f"{key} must be a number")
    if not lo <= value <= hi:
        raise RequestError(f"{key} must be between {lo:g} and {hi:g}")
    return float(value)


def _text(data: dict, key: str, required: bool) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise RequestError(f"{key} must be a string")
    value = value.strip()
    if required and not value:
        raise RequestError(f"{key} is required")
    if len(value) > MAX_PROMPT_CHARS:
        raise RequestError(f"{key} is longer than {MAX_PROMPT_CHARS} characters")
    return value


def _seed(data: dict) -> int:
    seed = data.get("seed")
    if seed is None or seed == -1:
        return secrets.randbelow(MAX_SEED + 1)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise RequestError(f"seed must be an integer between 0 and {MAX_SEED}, or -1 for random")
    return seed


def _loras(data: dict) -> tuple[LoraUse, ...]:
    value = data.get("loras", [])
    usage = f"loras must be a list of up to {MAX_LORAS} {{\"id\": <library id>, \"strength\": 0..{MAX_LORA_STRENGTH:g}}}"
    if not isinstance(value, list) or len(value) > MAX_LORAS:
        raise RequestError(usage)
    out = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"id", "strength"}:
            raise RequestError(usage)
        lora_id, strength = entry["id"], entry["strength"]
        if not isinstance(lora_id, str) or not INPUT_ID.match(lora_id):
            raise RequestError(usage)
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not 0 <= strength <= MAX_LORA_STRENGTH:
            raise RequestError(usage)
        if any(u.id == lora_id for u in out):
            raise RequestError("each LoRA can be used once per request")
        out.append(LoraUse(lora_id, float(strength)))
    return tuple(out)


def parse_job(data: object, video_profile: VideoProfile | None = None) -> GenerationRequest | VideoRequest:
    """A job of either kind. Requests without a `kind` are images, as they always were.
    `video_profile`: the configured video model, whose sizes and inputs a video request must fit."""
    if isinstance(data, dict) and "kind" in data:
        kind = data["kind"]
        if kind == "video":
            return parse_video_request(data, video_profile)
        if kind != "image":
            raise RequestError("kind must be \"image\" or \"video\"")
        data = {k: v for k, v in data.items() if k != "kind"}
    return parse_request(data)


def video_defaults(profile: VideoProfile | None) -> dict:
    """A video request's defaults for this model (the 5B model's when none is configured)."""
    if profile is None:
        return dict(VIDEO_DEFAULTS)
    return {**VIDEO_DEFAULTS, "width": profile.width, "height": profile.height, "frames": profile.trained_frames,
            "steps": profile.steps, "guidance": profile.guidance}


def parse_video_request(data: object, profile: VideoProfile | None = None) -> VideoRequest:
    if not isinstance(data, dict):
        raise RequestError("request body must be a JSON object")
    model = profile or STANDARD
    d = video_defaults(profile)
    width = _int(data, "width", VIDEO_MIN_SIDE, VIDEO_MAX_SIDE, d)
    height = _int(data, "height", VIDEO_MIN_SIDE, VIDEO_MAX_SIDE, d)
    for name, side in (("width", width), ("height", height)):
        if side % model.side_multiple:
            raise RequestError(f"{name} must be a multiple of {model.side_multiple}")
    if width * height > model.max_pixels:
        raise RequestError(f"video frames are limited to {model.max_pixels:,} pixels for this model")
    frames = _int(data, "frames", VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES, d)
    if (frames - 1) % VIDEO_FRAME_STEP:
        raise RequestError(f"frames must be one more than a multiple of {VIDEO_FRAME_STEP} (for example 49, 81 or 121)")

    start = data.get("start_image")
    if start is not None and not (isinstance(start, str) and INPUT_ID.match(start)):
        raise RequestError("start_image must be an input id")
    if start is None and model.start == "image":
        raise RequestError("this video model starts from a picture: choose a start picture")
    if start is not None and model.start == "text":
        raise RequestError("this video model starts from text only: remove the start picture")

    return VideoRequest(
        prompt=_text(data, "prompt", required=True),
        negative_prompt=_text(data, "negative_prompt", required=False),
        width=width,
        height=height,
        frames=frames,
        fps=model.fps,  # fixed by the model; accepted in no request
        steps=_int(data, "steps", 1, MAX_STEPS, d),
        guidance=_float(data, "guidance", 0.0, MAX_GUIDANCE, d),
        seed=_seed(data),
        count=_int(data, "count", 1, VIDEO_MAX_COUNT, d),
        start_image=start,
    )


def parse_request(data: object) -> GenerationRequest:
    if not isinstance(data, dict):
        raise RequestError("request body must be a JSON object")
    width = _int(data, "width", MIN_SIDE, MAX_SIDE)
    height = _int(data, "height", MIN_SIDE, MAX_SIDE)
    for name, side in (("width", width), ("height", height)):
        if side % SIDE_MULTIPLE:
            raise RequestError(f"{name} must be a multiple of {SIDE_MULTIPLE}")
    if width * height > MAX_PIXELS:
        raise RequestError(f"image is larger than {MAX_PIXELS // 1_000_000} megapixels")
    seed = _seed(data)

    return GenerationRequest(
        prompt=_text(data, "prompt", required=True),
        negative_prompt=_text(data, "negative_prompt", required=False),
        width=width,
        height=height,
        steps=_int(data, "steps", 1, MAX_STEPS),
        guidance=_float(data, "guidance", 0.0, MAX_GUIDANCE),
        seed=seed,
        count=_int(data, "count", 1, MAX_COUNT),
        loras=_loras(data),
    )
