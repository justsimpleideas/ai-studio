"""The LoRA training loop (docs/training.md, "Train").

Each step takes one cached photo latent x0 and its cached caption embedding,
draws noise e and a noise level s, and builds the noisy latent the model would
see while generating:

    x_s = (1 - s) * x0 + s * e

The image transformer is called the way the pipeline calls it — timestep
1 - s, one latent per list entry with a frame axis — and is trained to output
x0 - e. (The pipeline negates that output and takes an Euler step along it.)
The loss is the mean squared error against that target, and only the LoRA's
A and B matrices receive gradients.

Noise levels follow the generation schedule: a logit-normal draw u, then the
model's timestep shift, s = shift * u / (1 + (shift - 1) * u). The shift is
read from the model's scheduler config (it differs between model variants).

Memory: only the transformer is loaded; one latent and one embedding are read
from disk per step and nothing else is kept in RAM. CPU threads are capped.

Runs are resumable to the exact same result: every random draw is derived from
(seed, step) rather than from a running generator, and the LoRA and optimiser
state are saved together. All files are safetensors or JSON — nothing pickled.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

TRAINED = "subject"
ADAPTER = "adapter"
STATE_TENSORS = "trainer.safetensors"
STATE_JSON = "trainer.json"
STATE_VERSION = 1


class TrainError(Exception):
    """Training cannot start or resume. The message is safe to show to the user."""


@dataclass(frozen=True)
class TrainConfig:
    rank: int = 16
    alpha: float | None = None
    learning_rate: float = 1e-4
    steps: int = 2000
    checkpoint_every: int = 250
    caption_dropout: float = 0.1
    shift: float = 3.0
    grad_clip: float = 1.0
    seed: int = 0
    cpu_threads: int = 4
    gradient_checkpointing: bool = True

    def validate(self) -> None:
        if not 1 <= self.rank <= 256:
            raise TrainError("rank must be between 1 and 256")
        if not 0 < self.learning_rate < 1:
            raise TrainError("learning_rate must be between 0 and 1")
        if self.steps < 1 or self.checkpoint_every < 1:
            raise TrainError("steps and checkpoint_every must be at least 1")
        if not 0 <= self.caption_dropout < 1:
            raise TrainError("caption_dropout must be in [0, 1)")
        if self.shift <= 0:
            raise TrainError("shift must be positive")

    def run_key(self) -> dict:
        """The settings a resumed run must share. steps and checkpoint_every may change (to extend a run)."""
        d = asdict(self)
        for k in ("steps", "checkpoint_every", "cpu_threads", "gradient_checkpointing"):
            d.pop(k)
        return d


@dataclass(frozen=True)
class Sample:
    latent: Path
    embedding: Path


@dataclass
class TrainResult:
    status: str  # "finished" | "stopped"
    step: int
    checkpoint: str
    loss_avg: float
    seconds: float
    peak_vram_bytes: int = 0
    log: list[dict] = field(default_factory=list)


def sample_sigma(u: float, shift: float) -> float:
    return shift * u / (1 + (shift - 1) * u)


class Trainer:
    def __init__(self, transformer: Any, samples: list[Sample], empty_embedding: Path, out_dir: Path,
                 config: TrainConfig, *, device: str = "cuda", dtype: Any = None,
                 adapter: tuple[dict, dict | None] | None = None) -> None:
        import torch

        config.validate()
        if not samples:
            raise TrainError("There are no training photos (all excluded, or the caches are empty)")
        self.torch = torch
        self.model = transformer
        self.samples = samples
        self.empty_embedding = empty_embedding
        self.out_dir = out_dir
        self.config = config
        self.device = device
        self.dtype = dtype or torch.bfloat16
        self.adapter = adapter

    # -- setup -------------------------------------------------------------------------------------

    def _prepare(self) -> list:
        from .. import lora

        torch = self.torch
        for p in self.model.parameters():
            p.requires_grad_(False)
        present = lora.adapters(self.model)
        for name in (TRAINED, ADAPTER):
            if name in present:
                lora.remove(self.model, name)
        if self.adapter is not None:
            tensors, meta = self.adapter
            lora.load(self.model, ADAPTER, tensors, meta, dtype=self.dtype)
        gen = torch.Generator(device="cpu").manual_seed(self.config.seed)
        params = lora.add_trainable(self.model, TRAINED, self.config.rank, alpha=self.config.alpha,
                                    dtype=torch.float32, generator=None)
        with torch.no_grad():  # seeded init that does not depend on the device's generator
            for p in params[0::2]:
                bound = 1 / math.sqrt(p.shape[1])
                p.copy_(torch.empty(p.shape).uniform_(-bound, bound, generator=gen))
        if self.config.gradient_checkpointing and hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()
        self.model.train()
        return params

    def _run_hash(self) -> str:
        data = {"config": self.config.run_key(), "samples": [[s.latent.name, s.embedding.name] for s in self.samples],
                "adapter": self._adapter_hash()}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _adapter_hash(self) -> str | None:
        if self.adapter is None:
            return None
        digest = hashlib.sha256()
        for key in sorted(self.adapter[0]):
            t = self.adapter[0][key]
            digest.update(key.encode())
            digest.update(str(tuple(t.shape)).encode())
            digest.update(t.float().sum().item().hex().encode())
        return digest.hexdigest()

    # -- one step ----------------------------------------------------------------------------------

    def _draws(self, step: int) -> tuple[int, float, bool, int]:
        """(sample index, sigma, caption dropped, noise seed) for a step; a pure function of (seed, step)."""
        n = len(self.samples)
        epoch, pos = divmod(step, n)
        order = list(range(n))
        random.Random(f"{self.config.seed}/order/{epoch}").shuffle(order)
        rng = random.Random(f"{self.config.seed}/step/{step}")
        u = 1 / (1 + math.exp(-rng.gauss(0.0, 1.0)))  # logit-normal
        dropped = rng.random() < self.config.caption_dropout
        return order[pos], sample_sigma(u, self.config.shift), dropped, rng.getrandbits(63)

    def _load(self, path: Path, key: str):
        from safetensors.torch import load_file

        return load_file(str(path))[key].to(device=self.device, dtype=self.dtype)

    def loss_at(self, step: int):
        """The loss for one step's draws (also used by tests to measure progress at fixed draws)."""
        torch = self.torch
        index, sigma, dropped, noise_seed = self._draws(step)
        sample = self.samples[index]
        x0 = self._load(sample.latent, "latent")
        cond = self._load(self.empty_embedding if dropped else sample.embedding, "embedding")
        noise = torch.randn(x0.shape, generator=torch.Generator(device="cpu").manual_seed(noise_seed))
        noise = noise.to(device=self.device, dtype=self.dtype)
        noisy = (1 - sigma) * x0 + sigma * noise
        t = torch.tensor([1 - sigma], device=self.device, dtype=self.dtype)
        out = self.model([noisy.unsqueeze(1)], t, [cond], return_dict=False)[0][0].squeeze(1)
        target = (x0 - noise).float()
        loss = torch.nn.functional.mse_loss(out.float(), target)
        return loss, {"sample": index, "sigma": round(sigma, 4), "dropped": dropped}

    # -- checkpoints -------------------------------------------------------------------------------

    def _save(self, step: int, params: list, optimizer: Any, log: list[dict]) -> str:
        from safetensors.torch import save_file

        from .. import lora

        ckpt_dir = self.out_dir / "checkpoints"
        state_dir = self.out_dir / "state"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        tensors, meta = lora.state_dict(self.model, TRAINED)
        name = f"step-{step:06d}.safetensors"
        _atomic(lambda p: save_file(tensors, str(p), metadata=meta), ckpt_dir / name)

        opt = optimizer.state_dict()
        state = {f"param.{i}": p.detach().cpu().contiguous() for i, p in enumerate(params)}
        for i, s in opt["state"].items():
            for k, v in s.items():
                state[f"opt.{i}.{k}"] = (v if hasattr(v, "detach") else self.torch.tensor(v)).detach().cpu().contiguous()
        _atomic(lambda p: save_file(state, str(p), metadata={"step": str(step)}), state_dir / STATE_TENSORS)
        info = {"version": STATE_VERSION, "step": step, "run": self._run_hash(), "config": asdict(self.config),
                "checkpoint": name, "log": log}
        _atomic(lambda p: p.write_text(json.dumps(info), encoding="utf-8"), state_dir / STATE_JSON)
        return name

    def _resume(self, params: list, optimizer: Any) -> tuple[int, list[dict]]:
        from safetensors import safe_open

        info_path = self.out_dir / "state" / STATE_JSON
        if not info_path.is_file():
            return 0, []
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise TrainError(f"Unreadable training state {info_path}: {exc}") from exc
        if info.get("version") != STATE_VERSION:
            raise TrainError("The saved training state is from another version; start a new run")
        if info.get("run") != self._run_hash():
            raise TrainError("This folder holds a run with different settings, photos or adapter; "
                             "use a new output folder or the same settings")
        with safe_open(str(self.out_dir / "state" / STATE_TENSORS), "pt") as fh:
            if fh.metadata().get("step") != str(info["step"]):
                raise TrainError("The saved training state is incomplete (interrupted while saving)")
            tensors = {k: fh.get_tensor(k) for k in fh.keys()}
        torch = self.torch
        with torch.no_grad():
            for i, p in enumerate(params):
                p.copy_(tensors[f"param.{i}"])
        opt = optimizer.state_dict()
        opt["state"] = {}
        for key, value in tensors.items():
            if key.startswith("opt."):
                _, i, k = key.split(".", 2)
                opt["state"].setdefault(int(i), {})[k] = value.to(params[int(i)].device) if value.ndim else value
        optimizer.load_state_dict(opt)
        return int(info["step"]), list(info["log"])

    # -- the loop ----------------------------------------------------------------------------------

    def run(self, on_progress: Callable[[int, int, float], None] | None = None,
            should_stop: Callable[[], bool] | None = None) -> TrainResult:
        torch = self.torch
        threads = torch.get_num_threads()
        torch.set_num_threads(max(1, self.config.cpu_threads))
        try:
            return self._run(on_progress, should_stop)
        finally:
            torch.set_num_threads(threads)

    def _run(self, on_progress, should_stop) -> TrainResult:
        torch = self.torch
        params = self._prepare()
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=0.0)
        step, log = self._resume(params, optimizer)
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        checkpoint = ""
        status = "finished"
        while step < self.config.steps:
            if should_stop is not None and should_stop():
                status = "stopped"
                break
            loss, info = self.loss_at(step)
            if not torch.isfinite(loss):
                raise TrainError(f"The loss became {loss.item()} at step {step + 1}; lower the learning rate")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            log.append({"step": step, "loss": round(loss.item(), 6), **info})
            if on_progress is not None:
                on_progress(step, self.config.steps, loss.item())
            if step % self.config.checkpoint_every == 0:
                checkpoint = self._save(step, params, optimizer, log)
        if not checkpoint or not checkpoint.endswith(f"{step:06d}.safetensors"):
            checkpoint = self._save(step, params, optimizer, log)
        recent = [e["loss"] for e in log[-50:]]
        peak = torch.cuda.max_memory_allocated() if self.device.startswith("cuda") else 0
        return TrainResult(status=status, step=step, checkpoint=checkpoint,
                           loss_avg=sum(recent) / len(recent) if recent else float("nan"),
                           seconds=time.monotonic() - start, peak_vram_bytes=peak, log=log)


def _atomic(write: Callable[[Path], None], target: Path) -> None:
    tmp = target.with_name(target.name + ".tmp")
    write(tmp)
    os.replace(tmp, target)
