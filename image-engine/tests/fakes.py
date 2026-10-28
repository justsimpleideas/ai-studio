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

    def set_loras(self, loras) -> None:
        if self.kind == "video":
            return super().set_loras(loras)
        self.calls.append(("set_loras", [(lora_id, round(strength, 3)) for lora_id, _, strength in loras]))


class FakeTrainingBackend(FakeBackend):
    """A FakeBackend that can also be trained: a tiny random-weight Z-Image transformer on the CPU,
    a VAE stand-in that shrinks the photo 8x into 4 channels, and a text encoder stand-in."""

    def training_session(self):
        import contextlib

        @contextlib.contextmanager
        def session():
            self.calls.append(("training_session",))
            yield FakeTrainingParts()
            self.calls.append(("training_session_end",))

        return session()


class FakeTrainingParts:
    def __init__(self) -> None:
        import torch
        from diffusers import ZImageTransformer2DModel

        torch.manual_seed(0)
        self.transformer = ZImageTransformer2DModel(
            all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4, dim=64, n_layers=2, n_refiner_layers=1,
            n_heads=2, n_kv_heads=2, cap_feat_dim=32, axes_dims=[8, 12, 12], axes_lens=[256, 128, 128])
        self.device, self.dtype, self.shift, self.model_key = "cpu", torch.float32, 3.0, "example/fake-model@0"

    def vae_encoder(self):
        import torch

        def encode(image):
            small = image.convert("RGBA").resize((image.width // 8, image.height // 8))
            data = torch.frombuffer(bytearray(small.tobytes()), dtype=torch.uint8).float() / 127.5 - 1
            return data.view(small.height, small.width, 4).permute(2, 0, 1).contiguous()

        return encode, f"{self.model_key}|vae"

    def text_encoder(self):
        import contextlib

        import torch

        @contextlib.contextmanager
        def session():
            def encode(text):
                g = torch.Generator().manual_seed(sum(map(ord, text)) % 10_000)
                return torch.randn(max(1, len(text.split())), 32, generator=g)

            yield encode, f"{self.model_key}|text"

        return session()


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


class FakeCaptionBackend(FakeBackend):
    """Stands in for the captioning model: answers the JSON the prompt asks for.

    By default a photo shows the real subject, whole and as the main subject, and every
    other field gets its first choice (free text gets "grey"). `answer(image, keys)` can
    return other answers; `available` says whether a model folder is "installed"."""

    kind = "caption"
    model_id = "example/fake-vlm@000000000000"

    def __init__(self, answer=None, available: bool = True, step_delay: float = 0.0) -> None:
        super().__init__(step_delay=step_delay)
        self.answer = answer
        self._available = available

    def available(self) -> bool:
        return self._available

    def describe(self, image, prompt: str, max_new_tokens: int = 256) -> str:
        import json
        import re

        if self.step_delay:
            time.sleep(self.step_delay)
        self.calls.append(("describe", image.size))
        fields = {}
        for key, rest in re.findall(r'^- "([^"]+)": (.*)$', prompt, re.M):
            choices = re.search(r"One of: (.*)\.$", rest)
            fields[key] = choices.group(1).split(" | ")[0] if choices else "grey"
        if self.answer is not None:
            fields.update(self.answer(image, list(fields)))
        return "Here you go:\n" + json.dumps(fields)
