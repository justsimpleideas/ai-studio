"""Fast pre-load check of a model folder.

The full byte-level verification (hashes, safetensors headers) is done once by
scripts/fetch_model.py, which leaves a manifest.json behind. Before every load
the engine re-checks the cheap part: the manifest is there, every listed file
still has its recorded size, and nothing that could execute code has appeared
in the folder since.
"""

from __future__ import annotations

import json
from pathlib import Path

CODE_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".py", ".pyc", ".so", ".dll", ".exe"}


class ModelFolderError(Exception):
    pass


def check_model_folder(folder: Path, fetch_args: str = "") -> dict:
    """Return the manifest if the folder is safe to load, else raise ModelFolderError.

    `fetch_args` goes into the suggested fetch command, e.g. "--model video"."""
    script = f"scripts/fetch_model.py{' ' + fetch_args if fetch_args else ''}"
    if not folder.is_dir():
        raise ModelFolderError(
            f"Model folder not found: {folder}. Download it with "
            f"`python {script} --dest \"{folder}\"`, or set the folder in Settings.")
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise ModelFolderError(
            f"{folder} has no manifest.json, so it was not verified by {script}. "
            "Run the script on this folder (with --verify-only if the files are already there).")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelFolderError(f"Unreadable manifest.json in {folder}: {exc}") from exc

    listed = set()
    for entry in files:
        rel = Path(entry["path"])
        path = folder / rel
        listed.add(str(rel))
        if not path.is_file():
            raise ModelFolderError(f"Missing model file: {rel}")
        if path.stat().st_size != entry["size"]:
            raise ModelFolderError(f"Model file changed since it was verified: {rel}. Run the fetch script again.")

    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(folder)
        suffix = path.suffix.lower()
        if suffix in CODE_SUFFIXES:
            raise ModelFolderError(f"Refusing to load: executable or pickle file in model folder: {rel}")
        # Even a stray config file matters: a config can point a loader at remote code.
        if str(rel) not in listed and str(rel) != "manifest.json":
            raise ModelFolderError(f"Unverified file in model folder: {rel}")
    return manifest
