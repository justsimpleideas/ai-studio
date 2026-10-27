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

MAX_PROMPT_CHARS = 4000
MIN_SIDE, MAX_SIDE, SIDE_MULTIPLE = 256, 2048, 16
MAX_PIXELS = 2048 * 2048
MAX_STEPS = 50
MAX_GUIDANCE = 10.0
MAX_COUNT = 16
MAX_SEED = 2**32 - 1

# Defaults recommended for the image model: 8 steps (8 transformer passes in diffusers >= 0.42), no CFG.
DEFAULTS = {"width": 1024, "height": 1024, "steps": 8, "guidance": 0.0, "count": 1}

# The video model (5B, text or image to video). Sides are multiples of 32: the VAE
# shrinks each side 16x and the transformer patches 2x2. Frame counts are 4k + 1,
# because the VAE compresses time 4x after the first frame. It was trained at
# 1280x704 and 121 frames, which is 5 seconds at its 24 fps.
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

    def seed_for(self, index: int) -> int:
        """Images in a batch use consecutive seeds, so any one of them can be reproduced alone."""
        return (self.seed + index) % (MAX_SEED + 1)

    def to_dict(self) -> dict:
        return asdict(self)


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


def parse_job(data: object) -> GenerationRequest | VideoRequest:
    """A job of either kind. Requests without a `kind` are images, as they always were."""
    if isinstance(data, dict) and "kind" in data:
        kind = data["kind"]
        if kind == "video":
            return parse_video_request(data)
        if kind != "image":
            raise RequestError("kind must be \"image\" or \"video\"")
        data = {k: v for k, v in data.items() if k != "kind"}
    return parse_request(data)


def parse_video_request(data: object) -> VideoRequest:
    if not isinstance(data, dict):
        raise RequestError("request body must be a JSON object")
    d = VIDEO_DEFAULTS
    width = _int(data, "width", VIDEO_MIN_SIDE, VIDEO_MAX_SIDE, d)
    height = _int(data, "height", VIDEO_MIN_SIDE, VIDEO_MAX_SIDE, d)
    for name, side in (("width", width), ("height", height)):
        if side % VIDEO_SIDE_MULTIPLE:
            raise RequestError(f"{name} must be a multiple of {VIDEO_SIDE_MULTIPLE}")
    if width * height > VIDEO_MAX_PIXELS:
        raise RequestError(f"video frames are limited to {VIDEO_MAX_PIXELS:,} pixels (1280×704)")
    frames = _int(data, "frames", VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES, d)
    if (frames - 1) % VIDEO_FRAME_STEP:
        raise RequestError(f"frames must be one more than a multiple of {VIDEO_FRAME_STEP} (for example 49, 97 or 121)")

    start = data.get("start_image")
    if start is not None and not (isinstance(start, str) and INPUT_ID.match(start)):
        raise RequestError("start_image must be an input id")

    return VideoRequest(
        prompt=_text(data, "prompt", required=True),
        negative_prompt=_text(data, "negative_prompt", required=False),
        width=width,
        height=height,
        frames=frames,
        fps=VIDEO_FPS,  # fixed by the model; accepted in no request
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
    )
