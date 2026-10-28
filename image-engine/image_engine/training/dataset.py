"""A training dataset: a folder of the user's photos, described by a manifest.

The photos are never modified, moved or copied, and nothing is written into the
user's folder. The manifest (dataset.json, kept in the app's data folder) points
at the originals and records what the import learned about each one:

- facts, read from the folder path (docs/training.md, "Import"): each folder
  level below the root fills one level of the project's subject, e.g.
  <root>/<brand>/<model>/photo.jpg (subject.py). A labels.json in any folder sets
  facts for everything below it, overriding the folder names;
- size, a content hash, and a perceptual hash for spotting near-duplicates;
- issues that keep the photo out of training (unreadable, too small, duplicate,
  facts missing).

Rescanning reuses everything it can from the previous manifest, so a dataset of
thousands of photos only re-reads the files that changed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps, UnidentifiedImageError

from .subject import Subject

MANIFEST_VERSION = 1
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
LABELS_FILE = "labels.json"
DEFAULT_MIN_SIDE = 512
DEFAULT_HOLDOUT_PERCENT = 5
# dHash bits that may differ for two photos to count as the same picture
# (re-saved, resized, lightly recompressed). 4 of 64 is conservative.
DUPLICATE_DISTANCE = 4

ProgressCallback = Callable[[int, int], None]


class DatasetError(Exception):
    """The dataset cannot be scanned. The message is safe to show to the user."""


@dataclass
class Item:
    #: Stable id: the start of the content hash.
    id: str
    #: Relative to the dataset root, with forward slashes.
    path: str
    size: int
    mtime_ns: int
    sha256: str
    #: Upright size (EXIF orientation applied); 0 when unreadable.
    width: int
    height: int
    #: 64-bit difference hash as 16 hex digits; "" when unreadable.
    dhash: str
    facts: dict[str, str]
    #: Held out of training and used to compare generated against real photos.
    holdout: bool
    issues: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.issues


@dataclass
class Dataset:
    root: str
    #: The subject's preset name, for reference only; the project holds the definition.
    schema: str
    min_side: int
    items: list[Item]
    #: Problems that are not about one photo, e.g. a broken labels.json.
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        issues: dict[str, int] = {}
        for item in self.items:
            for issue in item.issues:
                kind = issue.split(":", 1)[0]
                issues[kind] = issues.get(kind, 0) + 1
        usable = [i for i in self.items if i.usable]
        return {
            "total": len(self.items),
            "usable": len(usable),
            "train": sum(1 for i in usable if not i.holdout),
            "holdout": sum(1 for i in usable if i.holdout),
            "issues": issues,
            "warnings": len(self.warnings),
        }

    def to_json(self) -> dict:
        return {"version": MANIFEST_VERSION, "root": self.root, "schema": self.schema,
                "min_side": self.min_side, "warnings": self.warnings,
                "items": [asdict(item) for item in self.items]}

    @classmethod
    def from_json(cls, data: dict) -> "Dataset":
        if data.get("version") != MANIFEST_VERSION:
            raise DatasetError(f"Unsupported dataset manifest version: {data.get('version')!r}")
        return cls(root=data["root"], schema=data["schema"], min_side=data["min_side"],
                   warnings=list(data.get("warnings", [])),
                   items=[Item(**item) for item in data["items"]])

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        try:
            return cls.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DatasetError(f"Unreadable dataset manifest {path}: {exc}") from exc


def scan(root: Path, subject: Subject, *, min_side: int = DEFAULT_MIN_SIDE,
         holdout_percent: int = DEFAULT_HOLDOUT_PERCENT, previous: Dataset | None = None,
         on_progress: ProgressCallback | None = None) -> Dataset:
    """Index every photo under `root`. Reads files, never writes them."""
    root = root.resolve()
    if not root.is_dir():
        raise DatasetError(f"Dataset folder not found: {root}")
    if not 0 <= holdout_percent <= 50:
        raise DatasetError("holdout_percent must be between 0 and 50")

    warnings: list[str] = []
    paths = _find_images(root)
    reusable = {}
    if previous is not None and Path(previous.root) == root:
        reusable = {item.path: item for item in previous.items}

    items: list[Item] = []
    labels_cache: dict[Path, dict[str, str]] = {}
    for index, path in enumerate(paths):
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        old = reusable.get(rel)
        if old is not None and old.size == stat.st_size and old.mtime_ns == stat.st_mtime_ns:
            sha, width, height, dhash = old.sha256, old.width, old.height, old.dhash
        else:
            sha = _file_sha256(path)
            width, height, dhash = _inspect_image(path)
        facts = _facts_for(path, root, subject, labels_cache, warnings)
        items.append(Item(id=sha[:16], path=rel, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                          sha256=sha, width=width, height=height, dhash=dhash, facts=facts,
                          holdout=int(sha[:8], 16) % 100 < holdout_percent))
        if on_progress is not None:
            on_progress(index + 1, len(paths))

    for item in items:
        if not item.dhash:
            item.issues.append("unreadable")
            continue
        if min(item.width, item.height) < min_side:
            item.issues.append("too_small")
        missing = [f for f in subject.required_facts if not item.facts.get(f)]
        if missing:
            item.issues.append("missing_facts:" + ",".join(missing))
    _mark_duplicates(items)
    return Dataset(root=str(root), schema=subject.preset, min_side=min_side, items=items, warnings=warnings)


def _find_images(root: Path) -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):  # does not follow symlinks
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if not name.startswith(".") and Path(name).suffix.lower() in IMAGE_SUFFIXES:
                found.append(Path(dirpath) / name)
    return found


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_image(path: Path) -> tuple[int, int, str]:
    """Upright size and difference hash, or (0, 0, "") if the file is not a usable image."""
    try:
        with Image.open(path) as img:
            width, height = img.size  # from the header, before draft() shrinks the decode
            if img.getexif().get(0x0112, 1) in (5, 6, 7, 8):  # EXIF orientation: rotated 90 degrees
                width, height = height, width
            img.draft("L", (128, 128))  # JPEG: decode at reduced size, much faster for big photos
            return width, height, dhash(ImageOps.exif_transpose(img))
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
        return 0, 0, ""


def dhash(img: Image.Image) -> str:
    """64-bit difference hash: is each pixel brighter than its right neighbour, on a 9x8 thumbnail."""
    small = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = small.load()
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (px[x, y] > px[x + 1, y])
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def _mark_duplicates(items: list[Item]) -> None:
    """Flag every copy of a picture but one: the largest copy with no other issue.

    Near-duplicates are found without comparing every pair: split the 64 bits into
    DUPLICATE_DISTANCE + 1 bands; two hashes that differ in at most that many bits
    must agree exactly on at least one band (pigeonhole), so only photos sharing a
    band are compared."""
    bands = DUPLICATE_DISTANCE + 1
    width = 64 // bands
    edges = [b * width for b in range(bands)] + [64]
    buckets: dict[tuple[int, int], list[Item]] = {}
    by_sha: dict[str, Item] = {}
    for item in sorted(items, key=lambda i: (bool(i.issues), -i.width * i.height, i.path)):
        if not item.dhash:
            continue
        original = by_sha.get(item.sha256)
        if original is None:
            value = int(item.dhash, 16)
            keys = [(b, (value >> (64 - edges[b + 1])) & ((1 << (edges[b + 1] - edges[b])) - 1))
                    for b in range(bands)]
            candidates = {id(c): c for k in keys for c in buckets.get(k, [])}
            for cand in candidates.values():
                if hamming(cand.dhash, item.dhash) <= DUPLICATE_DISTANCE:
                    original = cand
                    break
        if original is not None:
            item.issues.append(f"duplicate_of:{original.id}")
            continue
        by_sha[item.sha256] = item
        for key in keys:
            buckets.setdefault(key, []).append(item)


def _facts_for(path: Path, root: Path, subject: Subject,
               labels_cache: dict[Path, dict[str, str]], warnings: list[str]) -> dict[str, str]:
    """Folder levels fill the fact fields in order; labels.json files override, nearest wins."""
    parts = path.relative_to(root).parts[:-1]
    facts = {name: _folder_label(part) for name, part in zip(subject.fact_fields, parts)}
    folder = root
    for part in (None, *parts):
        if part is not None:
            folder = folder / part
        if folder not in labels_cache:
            labels_cache[folder] = _read_labels(folder / LABELS_FILE, subject, root, warnings)
        facts.update(labels_cache[folder])
    return {k: v for k, v in facts.items() if v}


def _folder_label(name: str) -> str:
    """Underscores become spaces; anything else (hyphens in "F-150") is kept as written."""
    return " ".join(name.replace("_", " ").split())


def _read_labels(path: Path, subject: Subject, root: Path, warnings: list[str]) -> dict[str, str]:
    if not path.is_file():
        return {}
    where = path.relative_to(root).as_posix()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"{where}: ignored, not valid JSON ({exc})")
        return {}
    if not isinstance(data, dict):
        warnings.append(f"{where}: ignored, expected an object like {{\"{subject.fact_fields[0]}\": \"...\"}}")
        return {}
    labels = {}
    for key, value in data.items():
        if key not in subject.label_fields:
            warnings.append(f"{where}: unknown field {key!r} ignored "
                            f"(fields: {', '.join(subject.label_fields)})")
        elif not isinstance(value, str):
            warnings.append(f"{where}: {key!r} must be text, ignored")
        else:
            labels[key] = " ".join(value.split())
    return labels
