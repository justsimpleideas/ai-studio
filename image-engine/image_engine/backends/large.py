"""The 12B single-stream image model through Hugging Face diffusers.

The model folder is the one scripts/import_model.py puts together:

    model_index.json        names the pipeline; says whether the model is the distilled one
    transformer/<file>      the transformer exactly as published: one .safetensors file,
                            fp8 or bf16, in the authors' layout or diffusers'
    text_encoder/, tokenizer/   a 4B vision-language model, of which only the language part runs
    vae/                    the VAE the model was trained with
    manifest.json           every file, verified once, re-checked before each load

The transformer's structure is read off its tensor shapes and checked tensor by tensor
(single_file.inspect) before a single weight is read. Its linear weights are kept in
fp8 (fp8.ScaledFP8Linear); a bf16 file is quantized per output row as it loads. The
pipeline class is imported by name below; nothing is looked up from the folder.

Where each part lives follows the memory plan (memory.py, LargeMemoryPlan). On a 12 GB
card the transformer streams from pinned RAM block by block while it samples, and the
text encoder visits the GPU only to read a prompt.

Heavy imports (torch, diffusers, transformers) happen lazily so the rest of the engine,
and its tests, run without them.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable

from .. import single_file
from ..config import Settings
from ..image_models import LARGE, ImageProfile, UnknownImageModel, profile_of, read_index, try_profile_of
from ..memory import LargeMemoryPlan, choose_large_plan
from ..model_check import ModelFolderError, check_model_folder
from ..params import GenerationRequest
from .base import Backend, BackendError, StatusCallback, StepCallback
from .diffusion import _provenance, gpu_device_info

MAX_SEQUENCE_LENGTH = 512
# The sampler: flow matching with the resolution-aware exponential time shift the model was
# trained with. Built here rather than read from the folder.
SCHEDULER = {"num_train_timesteps": 1000, "shift": 1.0, "use_dynamic_shifting": True, "base_shift": 0.5,
             "max_shift": 1.15, "base_image_seq_len": 256, "max_image_seq_len": 6400,
             "time_shift_type": "exponential"}


HOST_MEMORY_REFUSED = ("The system would not lock enough RAM to stream the model to the GPU. "
                       "Close other programs that use a lot of memory, or restart the studio, and try again.")


def host_memory_refused(exc: BaseException) -> bool:
    """A CUDA call refused page-locked RAM. PyTorch reports it as "CUDA error: out of memory",
    like a GPU shortage in wording but not in type: a GPU shortage is torch.OutOfMemoryError."""
    import torch

    return (isinstance(exc, RuntimeError) and not isinstance(exc, torch.cuda.OutOfMemoryError)
            and "CUDA error: out of memory" in str(exc))


def transformer_file(folder: Path) -> Path:
    """The folder's transformer file: the one model_index.json names, else the only one there."""
    named = read_index(folder).get("transformer_file")
    if isinstance(named, str) and named and "/" not in named and "\\" not in named:
        path = folder / "transformer" / named
        if not path.is_file():
            raise BackendError(f"{path} is missing")
        return path
    files = sorted((folder / "transformer").glob("*.safetensors"))
    if len(files) != 1:
        raise BackendError(f"{folder / 'transformer'} must hold exactly one .safetensors file, found {len(files)}")
    return files[0]


def load_transformer(path: Path, inspection: single_file.Inspection, pin: bool) -> tuple[Any, bool]:
    """The transformer on the CPU, its linear weights in fp8. With `pin`, every tensor goes to
    page-locked memory as it is read, so streaming needs no second copy; if the system refuses,
    the rest stays pageable. Returns (model, whether everything is pinned)."""
    from ..transformer_file import build_large

    return build_large(path, inspection, pin)


class LargeImageBackend(Backend):
    def __init__(self, get_settings: Callable[[], Settings]) -> None:
        self._get_settings = get_settings
        self._pipe: Any = None
        self._plan: LargeMemoryPlan | None = None
        self._profile: ImageProfile | None = None
        self._last_stats: dict[str, Any] = {}
        self._provenance: dict[str, Any] = {}

    @property
    def profile(self) -> ImageProfile:
        """The loaded model's profile, else that of the folder in settings."""
        if self._profile is not None:
            return self._profile
        folder = try_profile_of(self._get_settings().model_dir)
        return folder if folder is not None and folder.family == "large" else LARGE

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

        folder = Path(self._get_settings().model_dir)
        on_status("Checking model files")
        try:
            check_model_folder(folder, script="scripts/import_model.py")
            profile = profile_of(folder)
            if profile.family != "large":
                raise BackendError(f"{folder} does not hold the 12B image model")
            path = transformer_file(folder)
            on_status("Checking the transformer file")
            inspection = single_file.inspect(path)
            if inspection.family != "large":
                raise BackendError(f"{path.name} holds the {inspection.label}, not the 12B one")
            layers = read_index(folder).get("text_encoder_select_layers")
            if layers is not None and not (isinstance(layers, list) and len(layers) == inspection.config.num_text_layers
                                           and all(type(i) is int and i >= 0 for i in layers)):
                raise BackendError(f"text_encoder_select_layers in model_index.json must list "
                                   f"{inspection.config.num_text_layers} layer numbers")
        except (ModelFolderError, UnknownImageModel, single_file.SingleFileError) as exc:
            raise BackendError(str(exc)) from exc
        if not torch.cuda.is_available():
            raise BackendError("No CUDA GPU found. The 12B image model needs an NVIDIA GPU with at least 8 GB of memory.")

        _, total = torch.cuda.mem_get_info()
        plan = choose_large_plan(self._get_settings().memory_mode, total)

        from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Pipeline
        from transformers import AutoTokenizer, Qwen3VLModel

        safe = {"local_files_only": True,  # never reach the network
                "use_safetensors": True}   # never unpickle
        try:
            on_status("Reading the transformer")
            transformer, pinned = load_transformer(path, inspection, pin=plan.stream_transformer)
            on_status(f"Placing the transformer ({plan.label})")
            if plan.stream_transformer:
                # Blocks wait in RAM; each is copied in on a side stream while the one before computes.
                # low_cpu_mem_usage: tensors already pinned are used as they are, any others are pinned
                # only while they are copied. Without it diffusers pins every remaining tensor up front,
                # in power-of-two blocks, which is where a 32 GB machine runs out of lockable memory.
                transformer.enable_group_offload(
                    onload_device=torch.device("cuda"), offload_type="block_level", num_blocks_per_group=1,
                    use_stream=True, low_cpu_mem_usage=True)
            else:
                transformer.to("cuda")

            on_status("Reading the text encoder")
            text_encoder = Qwen3VLModel.from_pretrained(folder / "text_encoder", dtype=torch.bfloat16, **safe)
            text_encoder.visual = None  # prompts are text only; the vision tower would only take memory
            text_encoder.eval()
            if plan.text_encoder == "gpu":
                text_encoder.to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer", local_files_only=True)

            on_status("Reading the VAE")
            vae = AutoencoderKLQwenImage.from_pretrained(folder, subfolder="vae", torch_dtype=torch.bfloat16, **safe)
            vae.to("cuda")

            pipe = Krea2Pipeline(
                scheduler=FlowMatchEulerDiscreteScheduler(**SCHEDULER), vae=vae, text_encoder=text_encoder,
                tokenizer=tokenizer, transformer=transformer, text_encoder_select_layers=layers,
                is_distilled=profile.distilled)
        except torch.cuda.OutOfMemoryError as exc:
            self._free()
            raise BackendError(f"Not enough GPU memory for the '{plan.name}' memory mode. "
                               "Choose a lower mode in Settings.") from exc
        except BackendError:
            self._free()
            raise
        except Exception as exc:
            self._free()
            if host_memory_refused(exc):
                from ..transformer_file import clear_refused_lock

                clear_refused_lock()
                raise BackendError(HOST_MEMORY_REFUSED) from exc
            raise BackendError(f"Could not load the image model: {exc}") from exc
        if not pinned and plan.stream_transformer:
            on_status("Part of the transformer stays in pageable RAM; images will take a little longer")

        pipe.set_progress_bar_config(disable=True)
        self._pipe, self._plan, self._profile = pipe, plan, profile
        self._provenance = {**_provenance(folder), "weights": inspection.storage}

    def unload(self) -> None:
        self._pipe = None
        self._plan = None
        self._profile = None
        self._free()

    @staticmethod
    def _free() -> None:
        try:
            import torch
        except ImportError:
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ stages

    def encode(self, prompt: str, negative_prompt: str, guidance: float) -> Any:
        import torch

        pipe, plan = self._require()
        swap = plan.text_encoder == "swap"
        device = torch.device("cpu" if plan.text_encoder == "cpu" else "cuda")

        def run(text: str) -> tuple[Any, Any]:
            embeds, mask = pipe.encode_prompt(prompt=text, device=device, max_sequence_length=MAX_SEQUENCE_LENGTH)
            return embeds.to("cuda", torch.bfloat16), mask.to("cuda")

        try:
            if swap:
                pipe.text_encoder.to("cuda")
            with torch.inference_mode():
                embeds, mask = run(prompt)
                # The pipeline's own default for a missing negative prompt is the empty one.
                negative = run(negative_prompt) if self.uses_cfg(guidance) else None
        except torch.cuda.OutOfMemoryError as exc:
            raise BackendError("Out of GPU memory while reading the prompt. Try a lower memory mode.") from exc
        finally:
            if swap:
                pipe.text_encoder.to("cpu")
                self._free()
        return {"embeds": embeds, "mask": mask, "negative": negative}

    def sample(self, cond: Any, request: GenerationRequest, seed: int, on_step: StepCallback,
               start_frame: Any = None) -> Any:
        import torch

        pipe, _plan = self._require()
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        total = request.steps
        negative = cond["negative"]

        def callback(_pipe: Any, step_index: int, _timestep: Any, kwargs: dict) -> dict:
            on_step(step_index + 1, total)  # raising here (cancel) unwinds out of the pipeline
            return kwargs

        try:
            with torch.inference_mode():
                out = pipe(
                    prompt_embeds=cond["embeds"],
                    prompt_embeds_mask=cond["mask"],
                    negative_prompt_embeds=negative[0] if negative else None,
                    negative_prompt_embeds_mask=negative[1] if negative else None,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=request.steps,
                    # Guidance needs the negative encoding, which exists exactly when uses_cfg said so.
                    guidance_scale=request.guidance if negative else 0.0,
                    generator=generator,
                    output_type="latent",
                    callback_on_step_end=callback,
                )
        except torch.cuda.OutOfMemoryError as exc:
            self._free()
            raise BackendError(f"Out of GPU memory at {request.width}×{request.height}. "
                               "Try a smaller size or a lower memory mode.") from exc
        except RuntimeError as exc:
            if host_memory_refused(exc):
                from ..transformer_file import clear_refused_lock

                self._free()
                clear_refused_lock()
                raise BackendError(HOST_MEMORY_REFUSED) from exc
            raise
        # Latents come packed into 2x2 patches; unpacking needs the size they were made for.
        return {"latents": out.images, "height": request.height, "width": request.width}

    def decode(self, latents: Any) -> Any:
        import torch

        pipe, plan = self._require()
        vae = pipe.vae

        def run() -> Any:
            with torch.inference_mode():
                lat = pipe._unpack_latents(latents["latents"], latents["height"], latents["width"])
                lat = lat.to(device=vae.device, dtype=vae.dtype)
                shape = (1, vae.config.z_dim, 1, 1, 1)
                mean = torch.tensor(vae.config.latents_mean).view(shape).to(lat.device, lat.dtype)
                inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(shape).to(lat.device, lat.dtype)
                image = vae.decode(lat / inv_std + mean, return_dict=False)[0][:, :, 0]  # as the pipeline does
                return pipe.image_processor.postprocess(image, output_type="pil")[0]

        try:
            pil = run()
        except torch.cuda.OutOfMemoryError:
            # Large images: decode in overlapping tiles rather than fail.
            self._free()
            vae.enable_tiling()
            try:
                pil = run()
            except torch.cuda.OutOfMemoryError as exc:
                raise BackendError("Out of GPU memory while decoding the image. Try a smaller size.") from exc
            finally:
                vae.disable_tiling()
        self._last_stats = {"peak_vram_bytes": torch.cuda.max_memory_allocated(), "memory_plan": plan.label}
        return pil

    def stats(self) -> dict[str, Any]:
        return dict(self._last_stats)

    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def _require(self) -> tuple[Any, LargeMemoryPlan]:
        if self._pipe is None or self._plan is None:
            raise BackendError("Model is not loaded")
        return self._pipe, self._plan
