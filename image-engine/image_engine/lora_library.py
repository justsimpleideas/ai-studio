"""The LoRA library: LoRA files the Image tab can apply, kept in the engine's data folder.

A LoRA gets here by being chosen from a training run ("use this checkpoint").
Each entry is a copy of the file, so deleting a run or moving the source never
breaks an image's recipe, plus its content hash, which is recorded with every
image made with it.

Only safetensors files whose every key looks like a LoRA weight are accepted;
whether they fit the model is checked when they are applied (lora.load).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any

INDEX = "index.json"
NAME_MAX = 60
MAX_FILE_BYTES = 2 * 1024**3
LORA_KEY = re.compile(r"\.(lora_[AB](\.default)?\.weight|lora_(down|up)\.weight|alpha)$")


class LoraLibraryError(Exception):
    """The message is safe to show to the user."""


def check_lora_file(path: Path) -> dict[str, Any]:
    """Size, hash and rank of a LoRA file, or LoraLibraryError if it is not one."""
    from safetensors import safe_open

    if not path.is_file() or path.suffix != ".safetensors":
        raise LoraLibraryError("A LoRA must be a .safetensors file.")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise LoraLibraryError("That file is too large to be a LoRA.")
    try:
        with safe_open(str(path), "pt") as fh:
            keys = list(fh.keys())
            rank = next((fh.get_slice(k).get_shape()[0] for k in keys if ".lora_A." in k or ".lora_down." in k), 0)
    except Exception as exc:  # safetensors raises its own error types
        raise LoraLibraryError(f"Not a readable safetensors file: {exc}") from exc
    if not keys or not all(LORA_KEY.search(k) for k in keys):
        raise LoraLibraryError("That file is not a LoRA (it holds weights other than LoRA pairs).")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return {"size": size, "sha256": digest.hexdigest(), "rank": int(rank)}


class LoraLibrary:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = self._read()

    def _read(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads((self.root / INDEX).read_text(encoding="utf-8"))
            return {item["id"]: item for item in data["items"] if (self.root / item["file"]).is_file()}
        except (OSError, ValueError, KeyError, TypeError):
            return {}

    def _write(self) -> None:
        tmp = self.root / (INDEX + ".tmp")
        tmp.write_text(json.dumps({"items": list(self._items.values())}, indent=1), encoding="utf-8")
        os.replace(tmp, self.root / INDEX)

    @staticmethod
    def clean_name(name: object) -> str:
        if not isinstance(name, str) or not name.strip():
            raise LoraLibraryError("A LoRA needs a name.")
        name = " ".join(name.split())
        if len(name) > NAME_MAX:
            raise LoraLibraryError(f"A LoRA name can be at most {NAME_MAX} characters.")
        return name

    def add(self, source: Path, name: object, origin: dict[str, Any]) -> dict[str, Any]:
        cleaned = self.clean_name(name)
        info = check_lora_file(source)
        with self._lock:
            same = next((i for i in self._items.values() if i["sha256"] == info["sha256"]), None)
            if same is not None:
                raise LoraLibraryError(f"This LoRA is already in the library as “{same['name']}”.")
            lora_id = secrets.token_hex(8)
            rel = f"{lora_id}.safetensors"
            tmp = self.root / (rel + ".tmp")
            shutil.copyfile(source, tmp)
            os.replace(tmp, self.root / rel)
            item = {"id": lora_id, "name": cleaned, "file": rel, "created": time.time(), "origin": origin, **info}
            self._items[lora_id] = item
            self._write()
        return dict(item)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted((dict(i) for i in self._items.values()), key=lambda i: -i["created"])

    def get(self, lora_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(lora_id)
            return dict(item) if item else None

    def path_of(self, lora_id: str) -> Path | None:
        item = self.get(lora_id)
        return self.root / item["file"] if item else None

    def rename(self, lora_id: str, name: object) -> dict[str, Any] | None:
        cleaned = self.clean_name(name)
        with self._lock:
            item = self._items.get(lora_id)
            if item is None:
                return None
            item["name"] = cleaned
            self._write()
            return dict(item)

    def delete(self, lora_id: str) -> bool:
        with self._lock:
            item = self._items.pop(lora_id, None)
            if item is None:
                return False
            self._write()
        (self.root / item["file"]).unlink(missing_ok=True)
        return True
