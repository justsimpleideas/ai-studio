"""Test double for the GPU backend. Used by the test suite only, never by the app."""

from __future__ import annotations

import threading
import time

from PIL import Image, ImageOps

from image_engine.backends import Backend, BackendError


class FakeBackend(Backend):
    model_id = "fake-model"

    def __init__(self, step_delay: float = 0.0, fail_load: str | None = None,
                 fail_sample: str | None = None) -> None:
        self.step_delay = step_delay
        self.fail_load = fail_load
        self.fail_sample = fail_sample
        self._loaded = False
        self.calls: list[tuple] = []
        self.step_gate: threading.Event | None = None  # when set, sampling waits for it before each step

    def device_info(self) -> dict:
        return {"name": "Fake GPU", "total_bytes": 12 * 2**30, "used_bytes": 3 * 2**30, "cuda": True}

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, on_status) -> None:
        self.calls.append(("load",))
        on_status("Loading fake weights")
        if self.fail_load:
            raise BackendError(self.fail_load)
        self._loaded = True

    def unload(self) -> None:
        self.calls.append(("unload",))
        self._loaded = False

    def encode(self, prompt, negative_prompt, guidance):
        self.calls.append(("encode", prompt))
        return {"prompt": prompt}

    def sample(self, cond, request, seed, on_step, start_frame=None):
        self.calls.append(("sample", seed))
        if self.fail_sample:
            raise BackendError(self.fail_sample)
        for i in range(request.steps):
            if self.step_gate is not None:
                self.step_gate.wait(5)
            if self.step_delay:
                time.sleep(self.step_delay)
            on_step(i + 1, request.steps)
        return (request.width, request.height, seed)

    def decode(self, latents):
        width, height, seed = latents
        self.calls.append(("decode", seed))
        # A seed-dependent gradient, so placeholder images are distinguishable in the UI.
        a = ((seed * 67) % 200 + 30, (seed * 31) % 160 + 40, (seed * 13) % 200 + 40)
        b = (a[2] // 3, a[0] // 3, a[1] // 2)
        mask = Image.radial_gradient("L").resize((width, height))
        return ImageOps.colorize(mask, black=a, white=b)

    def stats(self) -> dict:
        return {"peak_vram_bytes": 1234}

    def provenance(self) -> dict:
        return {"model_revision": f"example/fake-model@{'0' * 40}", "libraries": {"pillow": "test"}}


class FakeVideoBackend(FakeBackend):
    """Frames of a gradient that drifts across the picture, so playback is visibly moving.
    With a start frame, the video opens on it and blends into the gradient."""

    model_id = "fake-video"
    kind = "video"
    cfg_above = 1.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_frames: list = []

    def sample(self, cond, request, seed, on_step, start_frame=None):
        self.start_frames.append(start_frame)
        width, height, seed = super().sample(cond, request, seed, on_step)
        return (width, height, seed, request.frames, start_frame)

    def decode(self, latents):
        width, height, seed, count, start_frame = latents
        self.calls.append(("decode", seed))
        base = super().decode((width * 2, height, seed))
        frames = []
        for i in range(count):
            offset = int(width * i / max(1, count - 1))
            frame = base.crop((offset, 0, offset + width, height))
            if start_frame is not None:
                frame = Image.blend(start_frame.convert("RGB"), frame, i / max(1, count - 1))
            frames.append(frame)
        return frames

    def provenance(self) -> dict:
        return {"model_revision": f"example/fake-video@{'0' * 40}", "libraries": {"pillow": "test"}}
