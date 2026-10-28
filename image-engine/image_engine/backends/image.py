"""The engine's image model: whichever supported model the configured folder holds.

The job queue and the server see one image backend. Underneath, each model family has
its own (image_models.py): the folder's model_index.json picks one when a model loads,
and while nothing is loaded, the folder in Settings decides what the status reports.
Only one is ever loaded; switching folders unloads the other first.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from ..config import Settings
from ..image_models import TURBO, ImageProfile, UnknownImageModel, profile_of
from ..params import GenerationRequest
from .base import Backend, BackendError, StatusCallback, StepCallback


def _default_factories() -> dict[str, Callable[[Callable[[], Settings]], Backend]]:
    from .diffusion import DiffusionBackend
    from .large import LargeImageBackend

    return {"diffusion": DiffusionBackend, "large": LargeImageBackend}


class ImageBackend(Backend):
    def __init__(self, get_settings: Callable[[], Settings],
                 factories: dict[str, Callable[[Callable[[], Settings]], Backend]] | None = None) -> None:
        self._get_settings = get_settings
        self._factories = factories or _default_factories()
        self._backends: dict[str, Backend] = {}
        self._active: Backend | None = None

    # ------------------------------------------------------------ which model

    def folder_profile(self) -> ImageProfile | None:
        """The profile of the folder in Settings; None if it holds no model the engine knows.
        A folder that does not exist yet counts as the default model, the one it would download."""
        folder = Path(self._get_settings().model_dir)
        if not (folder / "model_index.json").exists():
            return TURBO
        try:
            return profile_of(folder)
        except UnknownImageModel:
            return None

    @property
    def profile(self) -> ImageProfile | None:
        """The loaded model's profile, else the configured folder's."""
        if self._active is not None:
            return getattr(self._active, "profile", None) or TURBO
        return self.folder_profile()

    def _backend(self, family: str) -> Backend:
        if family not in self._backends:
            self._backends[family] = self._factories[family](self._get_settings)
        return self._backends[family]

    def _current(self) -> Backend:
        """The backend of the loaded model, else the one the folder would use."""
        if self._active is not None:
            return self._active
        profile = self.folder_profile()
        return self._backend(profile.family if profile is not None else TURBO.family)

    @property
    def model_id(self) -> str:  # type: ignore[override]
        profile = self.profile
        return profile.model_id if profile is not None else self._current().model_id

    @property
    def cfg_above(self) -> float:  # type: ignore[override]
        return self._current().cfg_above

    # ------------------------------------------------------------ lifecycle

    def device_info(self) -> dict[str, Any]:
        return self._current().device_info()

    @property
    def loaded(self) -> bool:
        return self._active is not None and self._active.loaded

    def load(self, on_status: StatusCallback) -> None:
        folder = Path(self._get_settings().model_dir)
        family = TURBO.family
        if not (folder / "model_index.json").exists() and any(folder.glob("*.safetensors")):
            raise BackendError(f"{folder} holds a model file, not a model folder. Use Settings → Image model → "
                               "Import a model file… to put it together with the parts it needs.")
        if (folder / "model_index.json").exists():
            try:
                family = profile_of(folder).family
            except UnknownImageModel as exc:
                raise BackendError(str(exc)) from exc
        backend = self._backend(family)
        if self._active is not None and self._active is not backend:
            self._active.unload()
        self._active = None
        backend.load(on_status)
        self._active = backend

    def unload(self) -> None:
        if self._active is not None:
            self._active.unload()
        self._active = None

    # ------------------------------------------------------------ stages

    def _loaded(self) -> Backend:
        if self._active is None:
            raise BackendError("Model is not loaded")
        return self._active

    def encode(self, prompt: str, negative_prompt: str, guidance: float) -> Any:
        return self._loaded().encode(prompt, negative_prompt, guidance)

    def sample(self, cond: Any, request: GenerationRequest, seed: int, on_step: StepCallback,
               start_frame: Any = None) -> Any:
        return self._loaded().sample(cond, request, seed, on_step, start_frame=start_frame)

    def decode(self, latents: Any) -> Any:
        return self._loaded().decode(latents)

    def stats(self) -> dict[str, Any]:
        return self._active.stats() if self._active is not None else {}

    def set_loras(self, loras: list[tuple[str, Any, float]]) -> None:
        if not loras and self._active is None:
            return
        backend = self._loaded()
        if loras and not getattr(backend, "profile", TURBO).loras:
            raise BackendError("This image model does not take LoRAs. They work with the 6B model.")
        backend.set_loras(loras)

    def provenance(self) -> dict[str, Any]:
        return self._active.provenance() if self._active is not None else {}

    # ------------------------------------------------------------ training

    @contextmanager
    def training_session(self) -> Iterator[Any]:
        backend = self._loaded()
        session = getattr(backend, "training_session", None)
        if session is None:
            raise BackendError("LoRA training needs the 6B image model. Choose its folder in Settings.")
        with session() as parts:
            yield parts
