"""The engine interface every model backend implements.

Generation is split into three stages instead of one opaque call:

    encode(prompt)            -> conditioning   (text encoder)
    sample(cond, request, s)  -> latents        (transformer, the slow part)
    decode(latents)           -> PIL image, or a list of frames for video (VAE)

The v1 UI always runs them in this order, but keeping them separate lets the
runner cache conditioning across a batch, and lets a future node graph wire the
stages together freely (docs/image-engine-decisions.md, D1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from ..params import GenerationRequest, VideoRequest

StepCallback = Callable[[int, int], None]
"""Called after each sampling step with (steps_done, steps_total). May raise to abort."""

StatusCallback = Callable[[str], None]
"""Called with a short human-readable message while a slow operation runs."""


class Cancelled(Exception):
    """Raised from a step callback to stop sampling."""


class BackendError(Exception):
    """A failure the user can act on (missing model, out of memory, ...)."""


class Backend(ABC):
    #: Stable identifier stored with every generated image.
    model_id: str = "unknown"
    #: What decode() produces: "image" (one PIL image) or "video" (a list of PIL frames).
    kind: str = "image"
    #: Guidance above this runs classifier-free guidance (a negative prompt, two passes per step).
    #: Pipelines disagree: the image pipeline uses any value above 0, the video pipeline above 1.
    cfg_above: float = 0.0

    def uses_cfg(self, guidance: float) -> bool:
        return guidance > self.cfg_above

    @abstractmethod
    def device_info(self) -> dict[str, Any]:
        """GPU name, total and used memory. Must not require the model to be loaded."""

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    @abstractmethod
    def load(self, on_status: StatusCallback) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def encode(self, prompt: str, negative_prompt: str, guidance: float) -> Any: ...

    @abstractmethod
    def sample(self, cond: Any, request: GenerationRequest | VideoRequest, seed: int, on_step: StepCallback,
               start_frame: Any = None) -> Any:
        """`start_frame` is a PIL image the video must open with, already at the request's size.
        Only video backends receive one; image backends always get None."""

    @abstractmethod
    def decode(self, latents: Any) -> Any: ...

    def stats(self) -> dict[str, Any]:
        """Optional per-image measurements (for example peak memory) gathered during the last sample."""
        return {}

    def provenance(self) -> dict[str, Any]:
        """Where the loaded model and the code running it came from, recorded with every image
        so it can be reproduced: `model_revision` ("repo@commit") and `libraries` (name -> version)."""
        return {}
