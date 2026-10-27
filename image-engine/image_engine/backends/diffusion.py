"""The supported image model (6B, few-step) through Hugging Face diffusers.

The model folder is the one fetch_model.py downloads. The pipeline class is
imported by name below rather than looked up from the folder's model_index.json.

Heavy imports (torch, diffusers) happen lazily so the rest of the engine, and its
tests, run without them.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from ..memory import MemoryPlan, choose_plan
from ..model_check import ModelFolderError, check_model_folder
from ..params import GenerationRequest
from .base import Backend, BackendError, StatusCallback, StepCallback

MAX_SEQUENCE_LENGTH = 512


def _provenance(folder: Path) -> dict[str, Any]:
    """The model revision from the manifest fetch_model.py writes, and the library versions.

    A folder without a manifest (put together by hand) simply records no revision.
    """
    out: dict[str, Any] = {}
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        repo, revision = manifest.get("repo"), manifest.get("revision")
        if isinstance(repo, str) and isinstance(revision, str):
            out["model_revision"] = f"{repo}@{revision}"
    except (OSError, ValueError):
        pass
    libraries = {}
    for name in ("diffusers", "transformers", "torch"):
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if libraries:
        out["libraries"] = libraries
    return out


def gpu_device_info(plan_label: str | None) -> dict[str, Any]:
    """GPU name and memory, without loading anything. Shared by the image and video backends."""
    try:
        import torch
    except ImportError:
        # The engine still answers so the studio can say what is missing.
        return {"cuda": False, "name": "PyTorch is not installed in this Python environment",
                "total_bytes": 0, "used_bytes": 0, "torch": False}

    info: dict[str, Any] = {"cuda": torch.cuda.is_available(), "name": "CPU",
                            "total_bytes": 0, "used_bytes": 0}
    if info["cuda"]:
        free, total = torch.cuda.mem_get_info()
        info.update(name=torch.cuda.get_device_name(0), total_bytes=total, used_bytes=total - free)
    if plan_label is not None:
        info["memory_plan"] = plan_label
    return info


class DiffusionBackend(Backend):
    model_id = "turbo-6b"

    def __init__(self, get_settings: Callable[[], Settings]) -> None:
        self._get_settings = get_settings
        self._pipe: Any = None
        self._plan: MemoryPlan | None = None
        self._last_stats: dict[str, Any] = {}
        self._provenance: dict[str, Any] = {}

    # ------------------------------------------------------------ device

    def device_info(self) -> dict[str, Any]:
        return gpu_device_info(self._plan.label if self._plan is not None else None)

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    # ------------------------------------------------------------ lifecycle

    def load(self, on_status: StatusCallback) -> None:
        try:
            import torch
        except ImportError as exc:
            raise BackendError("PyTorch is not installed in the Python environment the studio uses. "
                               "See the README for setup.") from exc

        settings = self._get_settings()
        folder = Path(settings.model_dir)
        on_status("Checking model files")
        try:
            check_model_folder(folder)
        except ModelFolderError as exc:
            raise BackendError(str(exc)) from exc
        if not torch.cuda.is_available():
            raise BackendError("No CUDA GPU found. The image model needs an NVIDIA GPU with at least 6 GB of memory.")

        _, total = torch.cuda.mem_get_info()
        plan = choose_plan(settings.memory_mode, total)

        on_status("Reading weights from disk")
        from diffusers import ZImagePipeline

        try:
            pipe = ZImagePipeline.from_pretrained(
                folder,
                torch_dtype=torch.bfloat16,
                local_files_only=True,   # never reach the network
                use_safetensors=True,    # never unpickle
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            raise BackendError(f"Could not load the model: {exc}") from exc

        on_status(f"Placing model on GPU ({plan.label})")
        try:
            if plan.transformer_fp8:
                # transformer.dtype then reports the compute dtype (bf16), which is what the
                # pipeline casts latents to; the fp8 copies stay internal to each layer.
                pipe.transformer.enable_layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
            if plan.cpu_offload:
                pipe.enable_model_cpu_offload()
            elif plan.swap_text_encoder:
                # diffusers 0.42 reports the accelerator even with the text encoder left on the CPU;
                # the VAE still goes first so a version that took the first module agrees.
                pipe.vae.to("cuda")
                pipe.transformer.to("cuda")
            else:
                pipe.to("cuda")
        except torch.cuda.OutOfMemoryError as exc:
            del pipe
            self._free()
            raise BackendError(
                f"Not enough GPU memory for the '{plan.name}' memory mode. "
                "Choose a lower mode in Settings.") from exc

        pipe.set_progress_bar_config(disable=True)
        self._pipe, self._plan = pipe, plan
        self._provenance = _provenance(folder)

    def unload(self) -> None:
        self._pipe = None
        self._plan = None
        self._free()

    @staticmethod
    def _free() -> None:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ stages

    def encode(self, prompt: str, negative_prompt: str, guidance: float) -> Any:
        import torch

        pipe, plan = self._require()
        swap = plan.swap_text_encoder and not plan.cpu_offload
        try:
            if swap:
                pipe.text_encoder.to("cuda")
            with torch.inference_mode():
                embeds, negative = pipe.encode_prompt(
                    prompt=prompt,
                    device=torch.device("cuda"),
                    do_classifier_free_guidance=guidance > 0.0,  # the pipeline runs CFG for any scale above 0
                    negative_prompt=negative_prompt or None,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                )
        except torch.cuda.OutOfMemoryError as exc:
            raise BackendError("Out of GPU memory while reading the prompt. Try a lower memory mode.") from exc
        finally:
            if swap:
                pipe.text_encoder.to("cpu")
                self._free()
        return {"embeds": embeds, "negative": negative}

    def sample(self, cond: Any, request: GenerationRequest, seed: int, on_step: StepCallback,
               start_frame: Any = None) -> Any:
        import torch

        pipe, plan = self._require()
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        total = request.steps

        def callback(_pipe: Any, step_index: int, _timestep: Any, kwargs: dict) -> dict:
            on_step(step_index + 1, total)  # raising here (cancel) unwinds out of the pipeline
            return kwargs

        try:
            with torch.inference_mode():
                out = pipe(
                    prompt_embeds=cond["embeds"],
                    negative_prompt_embeds=cond["negative"] or None,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=request.steps,
                    guidance_scale=request.guidance,
                    generator=generator,
                    output_type="latent",
                    callback_on_step_end=callback,
                )
        except torch.cuda.OutOfMemoryError as exc:
            self._free()
            raise BackendError(
                f"Out of GPU memory at {request.width}×{request.height}. "
                "Try a smaller size or a lower memory mode.") from exc
        return out.images

    def decode(self, latents: Any) -> Any:
        import torch

        pipe, plan = self._require()
        vae = pipe.vae
        with torch.inference_mode():
            latents = latents.to(device=vae.device, dtype=vae.dtype)
            latents = latents / vae.config.scaling_factor + vae.config.shift_factor
            image = vae.decode(latents, return_dict=False)[0]
            pil = pipe.image_processor.postprocess(image, output_type="pil")[0]
        self._last_stats = {"peak_vram_bytes": torch.cuda.max_memory_allocated(), "memory_plan": plan.label}
        return pil

    def stats(self) -> dict[str, Any]:
        return dict(self._last_stats)

    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def _require(self) -> tuple[Any, MemoryPlan]:
        if self._pipe is None or self._plan is None:
            raise BackendError("Model is not loaded")
        return self._pipe, self._plan

