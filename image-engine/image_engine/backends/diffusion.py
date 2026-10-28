"""The 6B image model through Hugging Face diffusers.

The model folder is the one fetch_model.py downloads, or one scripts/import_model.py
put together around a single transformer file (fp8 or bf16): the rest of that folder
(text encoder, tokenizer, VAE, scheduler and the transformer's config.json) comes from
the downloaded model, and model_index.json names the file (`transformer_file`). Such a
file is checked tensor by tensor before it loads (single_file.inspect), and its fp8
weights stay in fp8 (fp8.ScaledFP8Linear). The pipeline class is imported by name
below rather than looked up from the folder's model_index.json.

Heavy imports (torch, diffusers) happen lazily so the rest of the engine, and its
tests, run without them.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .. import single_file
from ..config import Settings
from ..image_models import TURBO, ImageProfile, UnknownImageModel, profile_of, read_index, try_profile_of
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


def single_transformer(folder: Path) -> Path | None:
    """The transformer file an imported folder names in model_index.json, or None for a downloaded folder."""
    if not (folder / "model_index.json").is_file():
        return None
    try:
        named = read_index(folder).get("transformer_file")
    except UnknownImageModel:
        return None
    if not isinstance(named, str) or not named:
        return None
    if "/" in named or "\\" in named or not named.endswith(".safetensors"):
        raise BackendError("transformer_file in model_index.json must be a .safetensors file name")
    return folder / "transformer" / named


class DiffusionBackend(Backend):
    def __init__(self, get_settings: Callable[[], Settings]) -> None:
        self._get_settings = get_settings
        self._pipe: Any = None
        self._plan: MemoryPlan | None = None
        self._profile: ImageProfile | None = None
        self._last_stats: dict[str, Any] = {}
        self._provenance: dict[str, Any] = {}
        self._loras: dict[str, tuple[str, float]] = {}  # library id -> (file, strength) now on the transformer

    @property
    def profile(self) -> ImageProfile:
        """The loaded model's profile, else that of the folder in settings."""
        if self._profile is not None:
            return self._profile
        folder = try_profile_of(self._get_settings().model_dir)
        return folder if folder is not None and folder.family == "diffusion" else TURBO

    @property
    def model_id(self) -> str:  # type: ignore[override]
        return self.profile.model_id

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
        single = single_transformer(folder)
        inspection = config = None
        try:
            check_model_folder(folder, script="scripts/import_model.py" if single else None)
            profile = profile_of(folder) if (folder / "model_index.json").is_file() else TURBO
            if single is not None:
                on_status("Checking the transformer file")
                inspection = single_file.inspect(single)
                if inspection.family != "diffusion":
                    raise BackendError(f"{single.name} holds the {inspection.label}, not the 6B one")
                config = json.loads((folder / "transformer" / "config.json").read_text(encoding="utf-8"))
        except (ModelFolderError, UnknownImageModel, single_file.SingleFileError, OSError, ValueError) as exc:
            raise BackendError(str(exc)) from exc
        if not torch.cuda.is_available():
            raise BackendError("No CUDA GPU found. The image model needs an NVIDIA GPU with at least 6 GB of memory.")

        _, total = torch.cuda.mem_get_info()
        plan = choose_plan(settings.memory_mode, total)

        on_status("Reading weights from disk")
        from diffusers import ZImagePipeline

        try:
            parts = {}
            if single is not None:
                from ..transformer_file import build_six_b

                parts["transformer"] = build_six_b(single, inspection, config)
            pipe = ZImagePipeline.from_pretrained(
                folder,
                **parts,
                torch_dtype=torch.bfloat16,
                local_files_only=True,   # never reach the network
                use_safetensors=True,    # never unpickle
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            raise BackendError(f"Could not load the model: {exc}") from exc

        on_status(f"Placing model on GPU ({plan.label})")
        stored_fp8 = inspection is not None and bool(inspection.quantized)
        try:
            # A file stored in fp8 already keeps its weights in fp8, whatever the plan.
            if plan.transformer_fp8 and not stored_fp8:
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
        self._pipe, self._plan, self._profile = pipe, plan, profile
        self._provenance = _provenance(folder)
        if inspection is not None:
            self._provenance["weights"] = inspection.storage

    def unload(self) -> None:
        self._pipe = None
        self._plan = None
        self._profile = None
        self._loras = {}
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
        # With every part offloaded the VAE waits in RAM and its hook moves only the VAE to the GPU
        # when it runs, so the latents must be sent there themselves.
        device = torch.device("cuda") if plan.cpu_offload else vae.device
        with torch.inference_mode():
            latents = latents.to(device=device, dtype=vae.dtype)
            latents = latents / vae.config.scaling_factor + vae.config.shift_factor
            image = vae.decode(latents, return_dict=False)[0]
            pil = pipe.image_processor.postprocess(image, output_type="pil")[0]
        self._last_stats = {"peak_vram_bytes": torch.cuda.max_memory_allocated(), "memory_plan": plan.label}
        return pil

    def stats(self) -> dict[str, Any]:
        return dict(self._last_stats)

    # ------------------------------------------------------------ LoRA

    def set_loras(self, loras: list[tuple[str, Any, float]]) -> None:
        from safetensors import safe_open

        from .. import lora

        pipe, _ = self._require()
        transformer = pipe.transformer
        wanted = {lora_id: (str(path), strength) for lora_id, path, strength in loras}
        for lora_id, (path, _) in list(self._loras.items()):
            if wanted.get(lora_id, (None,))[0] != path:
                lora.remove(transformer, _lora_name(lora_id))
                del self._loras[lora_id]
        for lora_id, (path, strength) in wanted.items():
            name = _lora_name(lora_id)
            if lora_id not in self._loras:
                with safe_open(path, "pt") as fh:
                    tensors = {k: fh.get_tensor(k) for k in fh.keys()}
                    meta = fh.metadata()
                try:
                    lora.load(transformer, name, tensors, meta, strength=strength, dtype=torch_bf16())
                except lora.LoRAError as exc:
                    raise BackendError(f"This LoRA does not fit the image model: {exc}") from exc
            else:
                lora.set_strength(transformer, name, strength)
            self._loras[lora_id] = (path, strength)

    # ------------------------------------------------------------ training

    @contextmanager
    def training_session(self) -> Iterator["TrainingParts"]:
        """The loaded pipeline's parts, arranged for training; put back as they were afterwards.

        The text encoder waits on the CPU while the transformer trains (it is only needed to
        cache caption embeddings), generation LoRAs come off, and the transformer leaves in
        inference shape: no LoRA from training, no gradient checkpointing, eval mode."""
        import torch

        from .. import lora

        pipe, plan = self._require()
        if plan.cpu_offload:
            raise BackendError("Training needs the model on the GPU. Choose a memory mode other than Low.")
        self.set_loras([])
        te_device = next(pipe.text_encoder.parameters()).device
        pipe.text_encoder.to("cpu")
        self._free()
        parts = TrainingParts(self, pipe)
        try:
            yield parts
        finally:
            transformer = pipe.transformer
            for name in list(lora.adapters(transformer)):
                lora.remove(transformer, name)
            if getattr(transformer, "gradient_checkpointing", False):
                transformer.disable_gradient_checkpointing()
            transformer.eval()
            for p in transformer.parameters():
                p.requires_grad_(False)
            pipe.text_encoder.to(te_device)
            self._free()
            torch.cuda.reset_peak_memory_stats()

    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def model_key(self) -> str:
        """Identifies the weights in cache names: the verified revision, else the folder."""
        return self._provenance.get("model_revision") or str(Path(self._get_settings().model_dir).resolve())

    def _require(self) -> tuple[Any, MemoryPlan]:
        if self._pipe is None or self._plan is None:
            raise BackendError("Model is not loaded")
        return self._pipe, self._plan



def _lora_name(lora_id: str) -> str:
    return f"lib_{lora_id}"


def torch_bf16() -> Any:
    import torch

    return torch.bfloat16


class TrainingParts:
    """What the training service needs from the loaded image model, and nothing more."""

    def __init__(self, backend: DiffusionBackend, pipe: Any) -> None:
        import torch

        self._backend, self._pipe = backend, pipe
        self.transformer = pipe.transformer
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.shift = float(pipe.scheduler.config.get("shift", 3.0))
        self.model_key = backend.model_key()

    def vae_encoder(self) -> tuple[Callable, str]:
        from ..training.cache import vae_encoder

        self._pipe.vae.to("cuda")
        return vae_encoder(self._pipe.vae), f"{self.model_key}|vae"

    @contextmanager
    def text_encoder(self) -> Iterator[tuple[Callable[[str], Any], str]]:
        """Caption -> embedding (tokens x features), with the text encoder on the GPU meanwhile."""
        import torch

        pipe = self._pipe
        pipe.text_encoder.to("cuda")

        @torch.inference_mode()
        def encode(text: str) -> Any:
            embeds, _ = pipe.encode_prompt(prompt=text, device=torch.device("cuda"),
                                           do_classifier_free_guidance=False,
                                           max_sequence_length=MAX_SEQUENCE_LENGTH)
            return embeds[0]

        try:
            yield encode, f"{self.model_key}|text|{MAX_SEQUENCE_LENGTH}"
        finally:
            pipe.text_encoder.to("cpu")
            self._backend._free()
