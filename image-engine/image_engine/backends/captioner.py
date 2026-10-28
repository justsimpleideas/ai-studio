"""The captioning model: a vision-language model that answers questions about a photo.

It describes training photos (training/captions.py builds the questions and
judges the answers). It is a third kind of model next to the image and video
ones and, like them, only one is on the GPU at a time (jobs.JobQueue).

Loaded only from a folder fetch_model.py verified (no code, hashes checked), with
remote code refused, in bf16 on the GPU. Photos are shrunk before they reach the
model, which bounds the number of image tokens and so the memory and time per photo.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from ..model_check import ModelFolderError, check_model_folder
from ..params import GenerationRequest, VideoRequest
from .base import Backend, BackendError, StatusCallback, StepCallback
from .diffusion import _provenance, gpu_device_info

#: Longest side a photo is shrunk to before the model sees it (about 900 image tokens at most).
MAX_SIDE = 960
MAX_NEW_TOKENS = 256


def shrink(image: Any, max_side: int = MAX_SIDE) -> Any:
    """An RGB copy no larger than max_side on its longest side."""
    img = image.convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    return img


class CaptionBackend(Backend):
    kind = "caption"

    def __init__(self, get_settings: Callable[[], Settings]) -> None:
        self._get_settings = get_settings
        self._model: Any = None
        self._processor: Any = None
        self._folder: Path | None = None
        self.model_id = "caption"

    def device_info(self) -> dict[str, Any]:
        return gpu_device_info(None)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def folder(self) -> Path:
        return Path(self._get_settings().caption_model_dir)

    def available(self) -> bool:
        """Whether a verified model folder is there (checked without loading anything)."""
        try:
            check_model_folder(self.folder(), "--model caption")
            return True
        except ModelFolderError:
            return False

    def load(self, on_status: StatusCallback) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise BackendError(f"The captioning model needs PyTorch and transformers: {exc}") from exc
        folder = self.folder()
        try:
            manifest = check_model_folder(folder, "--model caption")
        except ModelFolderError as exc:
            raise BackendError(str(exc)) from exc
        if not torch.cuda.is_available():
            raise BackendError("The captioning model needs an NVIDIA GPU.")
        on_status("Loading the captioning model")
        try:
            processor = AutoProcessor.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                folder, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
                device_map="cuda", attn_implementation="sdpa")
        except torch.cuda.OutOfMemoryError as exc:
            gc.collect()
            torch.cuda.empty_cache()
            raise BackendError("Not enough GPU memory for the captioning model. Close other programs "
                               "that use the GPU and try again.") from exc
        model.eval()
        self._model, self._processor, self._folder = model, processor, folder
        repo, revision = manifest.get("repo", "caption"), str(manifest.get("revision", ""))[:12]
        self.model_id = f"{repo}@{revision}" if revision else str(repo)

    def unload(self) -> None:
        self._model = self._processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def describe(self, image: Any, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """The model's reply to one question about one photo (greedy, so the same photo and
        question always give the same answer)."""
        if self._model is None:
            raise BackendError("The captioning model is not loaded.")
        import torch

        messages = [{"role": "user", "content": [{"type": "image", "image": shrink(image)},
                                                 {"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_dict=True, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        try:
            with torch.inference_mode():
                out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise BackendError("Ran out of GPU memory while describing a photo.") from exc
        reply = out[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(reply, skip_special_tokens=True)[0]

    def provenance(self) -> dict[str, Any]:
        return _provenance(self._folder) if self._folder else {}

    # A captioning model does not make pictures.
    def encode(self, prompt: str, negative_prompt: str, guidance: float) -> Any:
        raise BackendError("The captioning model does not generate images.")

    def sample(self, cond: Any, request: GenerationRequest | VideoRequest, seed: int, on_step: StepCallback,
               start_frame: Any = None) -> Any:
        raise BackendError("The captioning model does not generate images.")

    def decode(self, latents: Any) -> Any:
        raise BackendError("The captioning model does not generate images.")
