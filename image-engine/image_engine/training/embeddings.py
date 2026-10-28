"""The caption-embedding cache: each caption run through the text encoder once, kept on disk.

Like the latent cache (cache.py), this is what lets training unload the text
encoder: the training loop reads one small file per step instead. An entry is
named by the caption text and the text encoder, so editing a caption in review
or switching encoders simply makes a different file.

The empty caption is always cached too: caption dropout trains on it, so that
guidance still has an unconditional prediction to work against.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Iterable

TextEncoder = Callable[[str], Any]
"""Turns one caption into its embedding (tokens x features, no batch dimension)."""


def entry_name(text: str, encoder_key: str) -> str:
    key = hashlib.sha256(f"{encoder_key}|{text}".encode("utf-8")).hexdigest()
    return f"{key[:2]}/{key}.safetensors"


def build(texts: Iterable[str], cache_dir: Path, encoder: TextEncoder, encoder_key: str, *,
          on_progress: Callable[[int, int], None] | None = None,
          should_stop: Callable[[], bool] | None = None) -> dict[str, str]:
    """Encode every caption not cached yet (plus the empty one). Returns caption -> file, relative."""
    from safetensors.torch import save_file

    from .cache import Stopped

    unique = list(dict.fromkeys(["", *texts]))
    files: dict[str, str] = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for done, text in enumerate(unique, start=1):
        if should_stop is not None and should_stop():
            raise Stopped(f"Stopped after {done - 1} of {len(unique)} captions")
        rel = entry_name(text, encoder_key)
        target = cache_dir / rel
        if not target.is_file():
            embedding = encoder(text)
            if embedding.ndim != 2 or embedding.shape[0] < 1:
                raise ValueError(f"The text encoder returned shape {tuple(embedding.shape)}, expected tokens x features")
            target.parent.mkdir(exist_ok=True)
            tmp = target.with_suffix(".tmp")
            save_file({"embedding": embedding.detach().to("cpu").contiguous()}, str(tmp))
            os.replace(tmp, target)
        files[text] = rel
        if on_progress is not None:
            on_progress(done, len(unique))
    return files
