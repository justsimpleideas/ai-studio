"""The supported video model (5B, text or image to video) through Hugging Face diffusers.

One model folder serves both modes. Text to video runs the folder's own pipeline;
image to video runs the image-to-video pipeline over the same loaded components,
which this model was trained for (the start frame becomes the first latent frame,
so nothing extra is loaded).

The model folder is the one `fetch_model.py --model video` (or `--model video-fast`)
downloads, or one scripts/import_model.py put together around a single transformer
file (fp8 or bf16) with the downloaded model's other parts; model_index.json then names
the file (`transformer_file`), which is checked tensor by tensor before it loads
(single_file.inspect) and keeps its fp8 weights in fp8 (fp8.ScaledFP8Linear).
The pipeline classes are imported by name below rather than looked up from
model_index.json; the folder's pipeline name only picks the profile (video_models.py).
The distilled model runs the same pipeline with a fixed three-step schedule that
noises each step's clean estimate again (make_dmd_scheduler), and no guidance.

Where each part lives follows the memory plan (memory.py, VideoMemoryPlan). The
components are read one at a time, largest first, so the transformer is already
on the GPU before the text encoder is read and the two never sit in RAM together.

A 14B model (two experts, video_models.py) loads only from an imported folder, whose
model_index.json names both experts' files. The experts are always fp8 (a bf16 file is
quantized as it loads) and placed by a TwoExpertPlan: resident on a large card, streamed
block by block from RAM otherwise. The first expert runs down to the boundary; at the end
of the step before it, the second is read (Experts.ensure), and unless both fit in RAM
the first is dropped first, so each clip reads both experts from disk once.

Heavy imports (torch, diffusers, transformers) happen lazily so the rest of the
engine, and its tests, run without them.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Callable

from .. import single_file
from ..config import Settings
from ..memory import (GIB, TwoExpertPlan, VideoMemoryPlan, choose_two_expert_plan, choose_video_plan,
                      keep_both_experts)
from ..model_check import ModelFolderError, check_model_folder
from ..params import VideoRequest
from ..video_models import STANDARD, UnknownVideoModel, VideoProfile, profile_of, try_profile_of
from .base import Backend, BackendError, StatusCallback, StepCallback
from .diffusion import _provenance, gpu_device_info, single_transformer

MAX_SEQUENCE_LENGTH = 512
# The text encoder in bf16, the same for every model of the family (memory.py).
TEXT_ENCODER_BYTES = int(11.4 * GIB)


def _total_ram() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return 0  # unknown: keep one expert at a time


class Experts:
    """The two experts of a 14B model: built from their files when needed, placed as the plan
    says, and, unless both fit in RAM, the one not running dropped to make room for the other."""

    def __init__(self, files: dict[str, tuple[Path, Any, dict]], plan: TwoExpertPlan, keep_both: bool,
                 on_status: StatusCallback | None = None) -> None:
        self.files, self.plan, self.keep_both = files, plan, keep_both
        self.models: dict[str, Any] = {"high": None, "low": None}
        self.on_status = on_status
        self.builds = 0  # how many times an expert was read from its file (tests, and the status line)
        self.pipe: Any = None  # the pipeline holding the experts, whose references are dropped with them

    def ensure(self, which: str) -> Any:
        """The expert `which` ("high" or "low"), ready to run, and set on the pipeline."""
        other = "low" if which == "high" else "high"
        if self.models[which] is None:
            if not self.keep_both and self.models[other] is not None:
                self.drop(other)
            self.models[which] = self._build(which)
        elif not self.plan.stream_experts:
            if self.models[other] is not None:
                self.models[other].to("cpu")
                VideoBackend._free()
            self.models[which].to("cuda")
        if self.pipe is not None:
            setattr(self.pipe, self.SLOTS[which], self.models[which])
        return self.models[which]

    SLOTS = {"high": "transformer", "low": "transformer_2"}

    def _build(self, which: str) -> Any:
        from ..transformer_file import build_video

        path, inspection, config = self.files[which]
        if self.on_status:
            self.on_status(f"Reading the {'first' if which == 'high' else 'second'} expert ({self.plan.label})")
        model = build_video(path, inspection, config, quantize=True)
        self.builds += 1
        if self.plan.stream_experts:
            import torch

            # Blocks wait in RAM and visit the GPU one at a time.
            model.enable_group_offload(onload_device=torch.device("cuda"), offload_type="block_level",
                                       num_blocks_per_group=1)
        else:
            model.to("cuda")
        return model

    def drop(self, which: str) -> None:
        self.models[which] = None
        if self.pipe is not None:
            setattr(self.pipe, self.SLOTS[which], None)
        VideoBackend._free()

    def clear(self) -> None:
        self.models = {"high": None, "low": None}
        self.pipe = None


def make_dmd_scheduler(sigmas: tuple[float, ...]) -> Any:
    """A flow-matching scheduler for a DMD-distilled model: whatever step count the pipeline
    asks for, it runs the fixed noise levels `sigmas`, and each step takes the model's clean
    estimate (x0 = x - sigma * v) and noises it again to the next level with fresh noise.
    The noise comes from `scheduler.generator`, so a seed reproduces the same clip."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    class DmdScheduler(FlowMatchEulerDiscreteScheduler):
        generator: Any = None

        def set_timesteps(self, num_inference_steps: Any = None, device: Any = None, **_: Any) -> None:
            super().set_timesteps(sigmas=list(sigmas), device=device)

        def step(self, *args: Any, generator: Any = None, **kwargs: Any) -> Any:
            return super().step(*args, generator=generator if generator is not None else self.generator, **kwargs)

    # shift 1: the levels are used exactly as given; timesteps are sigma x 1000 (1000, 757, 522).
    return DmdScheduler(num_train_timesteps=1000, shift=1.0, stochastic_sampling=True)


class VideoBackend(Backend):
    kind = "video"
    cfg_above = 1.0  # the video pipelines run CFG only for guidance above 1

    def __init__(self, get_settings: Callable[[], Settings]) -> None:
        self._get_settings = get_settings
        self._pipe: Any = None
        self._i2v: Any = None
        self._experts: Experts | None = None
        self._prompts: Any = None  # the pipeline prompts are encoded with, when not the main one
        self._plan: VideoMemoryPlan | TwoExpertPlan | None = None
        self._profile: VideoProfile | None = None
        self._last_stats: dict[str, Any] = {}
        self._provenance: dict[str, Any] = {}

    @property
    def profile(self) -> VideoProfile | None:
        """The loaded model's profile, else that of the folder in settings (None if unknown)."""
        return self._profile or try_profile_of(self._get_settings().video_model_dir)

    @property
    def model_id(self) -> str:  # type: ignore[override]
        profile = self.profile
        if profile is not None and profile.experts == 2:
            return profile.name
        return "ti2v-5b-fast" if profile is not None and profile.fixed else "ti2v-5b"

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
        folder = Path(settings.video_model_dir)
        on_status("Checking model files")
        single = single_transformer(folder)
        inspection = config = None
        try:
            check_model_folder(folder, fetch_args="--model video", script="scripts/import_model.py" if single else None)
            profile = profile_of(folder)
            if profile.experts == 2:
                experts = self._check_experts(folder, profile, on_status)
            elif single is not None:
                on_status("Checking the transformer file")
                inspection = single_file.inspect(single)
                if inspection.family != "video":
                    raise BackendError(f"{single.name} holds the {inspection.label}, not a video model")
                config = json.loads((folder / "transformer" / "config.json").read_text(encoding="utf-8"))
        except (ModelFolderError, UnknownVideoModel, single_file.SingleFileError, OSError, ValueError) as exc:
            raise BackendError(str(exc)) from exc
        if not torch.cuda.is_available():
            raise BackendError("No CUDA GPU found. The video model needs an NVIDIA GPU with at least 8 GB of memory.")

        _, total = torch.cuda.mem_get_info()
        if profile.experts == 2:
            self._load_two_experts(folder, profile, experts, choose_two_expert_plan(settings.video_memory_mode, total),
                                   on_status)
            return
        plan = choose_video_plan(settings.video_memory_mode, total)

        from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
        from transformers import UMT5EncoderModel

        safe = {"local_files_only": True,  # never reach the network
                "use_safetensors": True}   # never unpickle
        try:
            on_status("Reading the transformer")
            if single is not None:
                from ..transformer_file import build_video

                transformer = build_video(single, inspection, config)
            else:
                transformer = WanTransformer3DModel.from_pretrained(
                    folder, subfolder="transformer", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **safe)
            on_status(f"Placing the transformer ({plan.label})")
            self._place_transformer(transformer, plan, stored_fp8=bool(inspection and inspection.quantized))

            on_status("Reading the text encoder")
            text_encoder = UMT5EncoderModel.from_pretrained(
                folder, subfolder="text_encoder", dtype=torch.bfloat16, **safe)
            if plan.text_encoder_on_gpu:
                text_encoder.to("cuda")

            on_status("Reading the VAE")
            # fp32, as the model's authors run it; tiling bounds the memory a decode needs.
            vae = AutoencoderKLWan.from_pretrained(
                folder, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True, **safe)
            vae.enable_tiling()
            if plan.vae_on_gpu:
                vae.to("cuda")

            # The tokenizer and scheduler are small; the pipeline reads them and its own config.
            # A distilled model brings its own schedule in place of the folder's scheduler.
            extra = {"scheduler": make_dmd_scheduler(profile.sigmas)} if profile.sigmas else {}
            pipe = WanPipeline.from_pretrained(
                folder, transformer=transformer, text_encoder=text_encoder, vae=vae, **extra, **safe)
        except torch.cuda.OutOfMemoryError as exc:
            self._free()
            raise BackendError(
                f"Not enough GPU memory for the '{plan.name}' video memory mode. "
                "Choose a lower mode in Settings.") from exc
        except Exception as exc:
            self._free()
            raise BackendError(f"Could not load the video model: {exc}") from exc

        pipe.set_progress_bar_config(disable=True)
        self._pipe, self._plan, self._profile = pipe, plan, profile
        self._provenance = _provenance(folder)
        if inspection is not None:
            self._provenance["weights"] = inspection.storage

    # ------------------------------------------------------------ two experts (14B)

    @staticmethod
    def _check_experts(folder: Path, profile: VideoProfile, on_status: StatusCallback) -> dict[str, tuple[Path, Any, dict]]:
        """Each expert's file, its inspection and its config.json, checked against the folder's model."""
        try:
            index = json.loads((folder / "model_index.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BackendError(f"{folder} has no readable model_index.json") from exc
        kind = "14b-text" if profile.start == "text" else "14b-image"
        out = {}
        for which, key, sub in (("high", "transformer_file", "transformer"), ("low", "transformer_2_file", "transformer_2")):
            named = index.get(key)
            if not isinstance(named, str) or "/" in named or "\\" in named or not named.endswith(".safetensors"):
                raise BackendError("A 14B video model loads from a folder made with Import a model file…, "
                                   f"whose model_index.json names both experts' files ({key} is missing)")
            path = folder / sub / named
            on_status(f"Checking the {'first' if which == 'high' else 'second'} expert's file")
            inspection = single_file.inspect(path)
            if inspection.video_kind != kind:
                raise BackendError(f"{named} holds the {inspection.label}, not the {single_file.VIDEO_KIND_LABELS[kind]}")
            config = json.loads((folder / sub / "config.json").read_text(encoding="utf-8"))
            single_file.check_video_config(config, inspection.config)
            out[which] = (path, inspection, config)
        return out

    def _load_two_experts(self, folder: Path, profile: VideoProfile, files: dict[str, tuple[Path, Any, dict]],
                          plan: TwoExpertPlan, on_status: StatusCallback) -> None:
        import torch
        from diffusers import (AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler,
                               WanImageToVideoPipeline, WanPipeline)
        from transformers import AutoTokenizer, UMT5EncoderModel

        safe = {"local_files_only": True, "use_safetensors": True}
        expert_bytes = max(i.parameters for _, i, _ in files.values())  # ~1 byte per weight in fp8
        keep_both = keep_both_experts(_total_ram(), expert_bytes, TEXT_ENCODER_BYTES, plan)
        experts = Experts(files, plan, keep_both, on_status)
        try:
            experts.ensure("high")

            on_status("Reading the text encoder")
            text_encoder = UMT5EncoderModel.from_pretrained(folder, subfolder="text_encoder", dtype=torch.bfloat16, **safe)
            if plan.text_encoder_on_gpu:
                text_encoder.to("cuda")

            on_status("Reading the VAE")
            vae = AutoencoderKLWan.from_pretrained(
                folder, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True, **safe)
            vae.enable_tiling()
            vae.to("cuda")

            tokenizer = AutoTokenizer.from_pretrained(folder, subfolder="tokenizer", local_files_only=True)
            if profile.shift is not None:
                # A few-step distillation: plain Euler steps on a schedule shifted as its publisher says.
                scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=profile.shift)
            else:
                scheduler = UniPCMultistepScheduler.from_pretrained(folder, subfolder="scheduler", local_files_only=True)
            parts = {"tokenizer": tokenizer, "text_encoder": text_encoder, "vae": vae, "scheduler": scheduler,
                     "transformer": experts.models["high"], "transformer_2": None, "boundary_ratio": profile.boundary}
            if profile.start == "image":
                pipe = WanImageToVideoPipeline(**parts, image_encoder=None, image_processor=None)
                # Prompts are read through the text pipeline, whose code for it is the same except that
                # the image pipeline calls an optional text-fixing package without checking it is installed.
                self._prompts = WanPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler)
            else:
                pipe = WanPipeline(**parts)
        except torch.cuda.OutOfMemoryError as exc:
            experts.clear()
            self._free()
            raise BackendError(f"Not enough GPU memory for the '{plan.name}' video memory mode. "
                               "Choose a lower mode in Settings.") from exc
        except Exception as exc:
            experts.clear()
            self._free()
            raise BackendError(f"Could not load the video model: {exc}") from exc

        pipe.set_progress_bar_config(disable=True)
        experts.pipe = pipe
        self._pipe, self._plan, self._profile, self._experts = pipe, plan, profile, experts
        self._provenance = {**_provenance(folder), "weights": files["high"][1].storage}

    @staticmethod
    def _place_transformer(transformer: Any, plan: VideoMemoryPlan, stored_fp8: bool = False) -> None:
        """`stored_fp8`: the linear weights are fp8 already (an imported fp8 file), so the plan's
        fp8 storage holds without casting, and the bf16 plans keep them in fp8 too."""
        import torch

        if plan.group_offload:
            # Blocks wait in RAM and visit the GPU one at a time: slow, but needs little VRAM.
            transformer.enable_group_offload(
                onload_device=torch.device("cuda"), offload_type="block_level", num_blocks_per_group=1)
            return
        if plan.transformer_fp8 and not stored_fp8:
            # transformer.dtype then reports bf16, which is what the pipeline casts its inputs to.
            transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        transformer.to("cuda")

    def unload(self) -> None:
        if self._experts is not None:
            self._experts.clear()
        self._experts = None
        self._prompts = None
        self._pipe = None
        self._i2v = None
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

        pipe, _plan = self._require()
        pipe = self._prompts or pipe
        try:
            with torch.inference_mode():
                # Runs wherever the text encoder is (the CPU on most plans); the result goes to the GPU.
                embeds, negative = pipe.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    do_classifier_free_guidance=self.uses_cfg(guidance),
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                    device=torch.device("cuda"),
                    dtype=torch.bfloat16,
                )
        except torch.cuda.OutOfMemoryError as exc:
            raise BackendError("Out of GPU memory while reading the prompt. Try a lower memory mode.") from exc
        return {"embeds": embeds, "negative": negative}

    def sample(self, cond: Any, request: VideoRequest, seed: int, on_step: StepCallback,
               start_frame: Any = None) -> Any:
        if self._experts is not None:
            return self._sample_two_experts(cond, request, seed, on_step, start_frame)
        import torch

        pipe, _plan = self._require()
        profile = self._profile or STANDARD
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        # The server already fits requests to a fixed profile; this keeps the backend right on its own.
        steps, guidance = (profile.steps, profile.guidance) if profile.fixed else (request.steps, request.guidance)
        if profile.sigmas:
            # The re-noising draws from the same seeded generator as the starting noise.
            pipe.scheduler.generator = generator
        total = steps

        def callback(_pipe: Any, step_index: int, _timestep: Any, kwargs: dict) -> dict:
            on_step(step_index + 1, total)  # raising here (cancel) unwinds out of the pipeline
            return kwargs

        args = {
            "prompt_embeds": cond["embeds"],
            "negative_prompt_embeds": cond["negative"],
            "height": request.height,
            "width": request.width,
            "num_frames": request.frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
            "output_type": "latent",
            "callback_on_step_end": callback,
        }
        try:
            with torch.inference_mode():
                if start_frame is None:
                    out = pipe(**args)
                else:
                    # The first frame is encoded wherever the VAE is; on plans that keep it in
                    # RAM that is the CPU, which for a single frame takes seconds.
                    out = self._image_to_video()(image=start_frame, **args)
        except torch.cuda.OutOfMemoryError as exc:
            self._free()
            raise BackendError(
                f"Out of GPU memory at {request.width}×{request.height}, {request.frames} frames. "
                "Try fewer frames, a smaller size, or a lower memory mode.") from exc
        return out.frames

    def _sample_two_experts(self, cond: Any, request: VideoRequest, seed: int, on_step: StepCallback,
                            start_frame: Any) -> Any:
        """The first expert down to the boundary, then the second: the second is read (or moved
        to the GPU) at the end of the last step before the boundary, the first dropped first
        unless both fit in RAM. The next clip starts again from the first."""
        import torch

        pipe, _plan = self._require()
        profile, experts = self._profile, self._experts
        assert profile is not None and experts is not None
        if profile.start == "image" and start_frame is None:
            raise BackendError("This video model starts from a picture: choose a start picture.")
        if profile.start == "text" and start_frame is not None:
            raise BackendError("This video model starts from text only: remove the start picture.")
        steps, guidance = (profile.steps, profile.guidance) if profile.fixed else (request.steps, request.guidance)
        # The second expert's guidance keeps the model's own ratio to the first's.
        guidance_2 = guidance * (profile.guidance_2 / profile.guidance) if profile.guidance_2 and profile.guidance else guidance

        experts.ensure("high")  # drops the second expert first unless both fit in RAM
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        boundary = profile.boundary * pipe.scheduler.config.num_train_timesteps

        def callback(_pipe: Any, step_index: int, _timestep: Any, kwargs: dict) -> dict:
            on_step(step_index + 1, steps)  # raising here (cancel) unwinds out of the pipeline
            timesteps = pipe.scheduler.timesteps
            if step_index + 1 < len(timesteps) and timesteps[step_index + 1] < boundary:
                experts.ensure("low")
            return kwargs

        args = {
            "prompt_embeds": cond["embeds"],
            "negative_prompt_embeds": cond["negative"],
            "height": request.height,
            "width": request.width,
            "num_frames": request.frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "guidance_scale_2": guidance_2,
            "generator": generator,
            "output_type": "latent",
            "callback_on_step_end": callback,
        }
        if start_frame is not None:
            args["image"] = start_frame
        try:
            with torch.inference_mode():
                out = pipe(**args)
        except torch.cuda.OutOfMemoryError as exc:
            self._free()
            raise BackendError(
                f"Out of GPU memory at {request.width}×{request.height}, {request.frames} frames. "
                "Try fewer frames, a smaller size, or a lower memory mode.") from exc
        return out.frames

    def _image_to_video(self) -> Any:
        """The image-to-video pipeline over the already loaded components (nothing new on the GPU)."""
        if self._i2v is None:
            from diffusers import WanImageToVideoPipeline

            p = self._pipe
            self._i2v = WanImageToVideoPipeline(
                tokenizer=p.tokenizer, text_encoder=p.text_encoder, vae=p.vae, scheduler=p.scheduler,
                transformer=p.transformer, expand_timesteps=p.config.expand_timesteps)
            self._i2v.set_progress_bar_config(disable=True)
        return self._i2v

    def decode(self, latents: Any) -> Any:
        import torch

        pipe, plan = self._require()
        vae = pipe.vae
        parked = False
        try:
            if plan.park_transformer:
                # Room for the VAE and its activations: a few seconds over PCIe, against minutes of sampling.
                pipe.transformer.to("cpu")
                parked = True
                self._free()
            if not plan.vae_on_gpu:
                vae.to("cuda")
            with torch.inference_mode():
                latents = latents.to(device=vae.device, dtype=vae.dtype)
                shape = (1, vae.config.z_dim, 1, 1, 1)
                mean = torch.tensor(vae.config.latents_mean).view(shape).to(latents.device, latents.dtype)
                std = torch.tensor(vae.config.latents_std).view(shape).to(latents.device, latents.dtype)
                video = vae.decode(latents * std + mean, return_dict=False)[0]
                frames = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
        except torch.cuda.OutOfMemoryError as exc:
            raise BackendError("Out of GPU memory while decoding the video. "
                               "Try fewer frames, a smaller size, or a lower memory mode.") from exc
        finally:
            if not plan.vae_on_gpu:
                vae.to("cpu")
            if parked:
                pipe.transformer.to("cuda")
            self._free()
        self._last_stats = {"peak_vram_bytes": torch.cuda.max_memory_allocated(), "memory_plan": plan.label}
        return frames

    def stats(self) -> dict[str, Any]:
        return dict(self._last_stats)

    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def _require(self) -> tuple[Any, VideoMemoryPlan]:
        if self._pipe is None or self._plan is None:
            raise BackendError("Video model is not loaded")
        return self._pipe, self._plan
