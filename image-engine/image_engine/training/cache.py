"""The latent cache: every training photo, encoded by the VAE once, kept on disk.

Training never runs the VAE. Each photo is cropped to its bucket, encoded once,
and the latent is saved as a small safetensors file. The training loop then
loads only the transformer, which is what makes a 6B model trainable in 12 GB
(docs/training.md, "Train").

A cache entry is named by everything that determines its contents: the photo's
content hash, its bucket and crop, and the VAE. Change any of them and the
entry is simply a different file, so a stale latent can never be used. Building
is resumable: entries already on disk are skipped.

torch and safetensors are imported lazily, as in the backends, so the rest of
the engine (and its tests) runs without them.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps

from .buckets import assign, crop_plan, make_buckets
from .dataset import Dataset

INDEX_FILE = "latents.json"
INDEX_VERSION = 1

Encoder = Callable[[Image.Image], Any]
"""Turns one cropped RGB photo into its latent tensor (no batch dimension)."""


class CacheError(Exception):
    """The cache cannot be built or read. The message is safe to show to the user."""


class Stopped(Exception):
    """Raised by build() when should_stop() returns true. Finished entries are kept."""


@dataclass(frozen=True)
class Entry:
    item_id: str
    bucket: tuple[int, int]
    file: str  # relative to the cache folder


def prepare_image(path: Path, bucket: tuple[int, int]) -> Image.Image:
    """The photo upright, scaled to cover `bucket`, centre-cropped to exactly its size."""
    with Image.open(path) as img:
        upright = ImageOps.exif_transpose(img).convert("RGB")
    sw, sh, left, top = crop_plan(upright.width, upright.height, bucket)
    scaled = upright.resize((sw, sh), Image.Resampling.LANCZOS)
    return scaled.crop((left, top, left + bucket[0], top + bucket[1]))


def entry_key(sha256: str, bucket: tuple[int, int], encoder_key: str) -> str:
    return hashlib.sha256(f"{sha256}|{bucket[0]}x{bucket[1]}|{encoder_key}".encode()).hexdigest()


def build(dataset: Dataset, cache_dir: Path, encoder: Encoder, encoder_key: str, *,
          target_side: int, include_holdout: bool = False,
          on_progress: Callable[[int, int], None] | None = None,
          should_stop: Callable[[], bool] | None = None) -> list[Entry]:
    """Encode every usable photo that is not cached yet, and write the index.

    `encoder_key` identifies the VAE (e.g. its repo revision); it is part of every
    entry's name. Returns the entries in dataset order."""
    from safetensors.torch import save_file

    cache_dir.mkdir(parents=True, exist_ok=True)
    buckets = make_buckets(target_side)
    root = Path(dataset.root)
    todo = [i for i in dataset.items if i.usable and (include_holdout or not i.holdout)]
    entries: list[Entry] = []
    for done, item in enumerate(todo, start=1):
        if should_stop is not None and should_stop():
            _write_index(cache_dir, entries, encoder_key, target_side)
            raise Stopped(f"Stopped after {done - 1} of {len(todo)} photos")
        bucket = assign(item.width, item.height, buckets)
        key = entry_key(item.sha256, bucket, encoder_key)
        rel = f"{key[:2]}/{key}.safetensors"
        target = cache_dir / rel
        if not target.is_file():
            try:
                image = prepare_image(root / item.path, bucket)
            except OSError as exc:
                raise CacheError(f"Could not read {item.path}: {exc}") from exc
            latent = encoder(image)
            target.parent.mkdir(exist_ok=True)
            tmp = target.with_suffix(".tmp")
            save_file({"latent": latent.detach().to("cpu").contiguous()}, str(tmp),
                      metadata={"item": item.id, "bucket": f"{bucket[0]}x{bucket[1]}"})
            os.replace(tmp, target)  # a half-written file never looks like a finished entry
        entries.append(Entry(item.id, bucket, rel))
        if on_progress is not None:
            on_progress(done, len(todo))
    _write_index(cache_dir, entries, encoder_key, target_side)
    return entries


def _write_index(cache_dir: Path, entries: list[Entry], encoder_key: str, target_side: int) -> None:
    data = {"version": INDEX_VERSION, "encoder": encoder_key, "target_side": target_side,
            "entries": [{"item": e.item_id, "bucket": list(e.bucket), "file": e.file} for e in entries]}
    tmp = cache_dir / (INDEX_FILE + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, cache_dir / INDEX_FILE)


def read_index(cache_dir: Path, encoder_key: str) -> list[Entry]:
    try:
        data = json.loads((cache_dir / INDEX_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CacheError(f"No usable latent cache in {cache_dir}: {exc}") from exc
    if data.get("version") != INDEX_VERSION:
        raise CacheError(f"Unsupported latent cache version: {data.get('version')!r}")
    if data.get("encoder") != encoder_key:
        raise CacheError("The latent cache was built with a different VAE; rebuild it.")
    return [Entry(e["item"], tuple(e["bucket"]), e["file"]) for e in data["entries"]]


def load_latent(cache_dir: Path, entry: Entry) -> Any:
    from safetensors.torch import load_file

    return load_file(str(cache_dir / entry.file))["latent"]


def vae_encoder(vae: Any) -> Encoder:
    """An Encoder for a diffusers AutoencoderKL, normalised the way the transformer sees latents.

    This is the exact inverse of the backend's decode (backends/diffusion.py):
    decode divides by scaling_factor and adds shift_factor, so encode subtracts
    and multiplies. The distribution's mean is used, not a sample, so the cache
    is deterministic."""
    import torch

    scale = vae.config.scaling_factor
    shift = vae.config.shift_factor or 0.0

    @torch.no_grad()
    def encode(image: Image.Image) -> Any:
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        pixels = pixels.view(image.height, image.width, 3).permute(2, 0, 1)
        pixels = pixels.to(device=vae.device, dtype=vae.dtype).unsqueeze(0) / 127.5 - 1.0
        mean = vae.encode(pixels).latent_dist.mode()
        return ((mean - shift) * scale)[0]

    return encode
