"""Merging other studio galleries (their "outputs" folders) into this one.

Nothing is lost and nothing is overwritten. Every item of every source either comes
across or is counted in the report with the reason (already here, or unreadable); no
file here is ever replaced (a name in use gets a number), no folder here changes unless
it is renamed on purpose, and a failure part way leaves this gallery as it was.

Everything is copied, never moved: the other galleries are opened read-only and stay as
they were. What comes across is each image and video with its files, its prompt and
settings, its provenance and stats, whether it is a favourite and which folder it is in;
and each folder with its name, whether it is hidden and its PIN (the stored hash, so the
same PIN still opens it). Items keep their creation time, so they fall into place among
this gallery's own.

A gallery copied from another (a backup, or one merged before) holds the same items. An
item is the same as one already here, or one already coming from another source, only if
its id, its file byte for byte and its whole generation record (prompt, settings, seed,
model, provenance, when it was made) all match; it is then skipped and counted. Anything
less and both are kept, the newcomer with a new id. Being a favourite and the folder it is
in are how it is organised, not what it is: they do not count, and where only the other
gallery has them, that is reported. A folder with the same id and exactly the same name as
one here is the same folder, whatever each holds: its items join it, and this gallery's
hidden setting and PIN stay. The same name alone is a clash to resolve.

Folder names are unique regardless of case, as in the gallery itself; clashes are shown
before anything is copied, each can be renamed, and suggestions are made for all of them.
As in the gallery, a hidden folder's name only counts while hidden folders are shown, so a
merge never tells that a hidden folder of some name exists.

A source gallery is not trusted: every row is checked (ids, paths that must stay inside
its folder and name a file there, JSON fields) and a row that fails is left out and
counted. Galleries from older versions lack later columns, which take their defaults as
in Gallery; columns from a newer version that this one does not know are left out, and
named in the report.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .gallery import (ADDED_COLUMNS, COLUMNS, FOLDER_ADDED_COLUMNS, FOLDER_NAME_MAX, ID_PATTERN, MEDIA, FolderError,
                      Gallery)

MAX_SOURCES = 16
# The first release's columns: a database without them is not a studio gallery.
BASE_COLUMNS = tuple(c for c in COLUMNS if c not in ADDED_COLUMNS)
FOLDER_BASE_COLUMNS = ("id", "name", "created")
FOLDER_COLUMNS = (*FOLDER_BASE_COLUMNS, *FOLDER_ADDED_COLUMNS)
FILE_COLUMNS = ("file", "thumb", "last_frame")
TOP_DIRS = {"images", "videos", "thumbs", "frames"}
SUFFIXES = {".png", ".webp", ".mp4"}
PART = re.compile(r"^[\w.-]{1,160}$")
# A stored PIN: this version's scrypt form, or (from a newer version) anything printable. One
# this version cannot check keeps its folder locked here rather than opening it.
PIN_STORED = re.compile(r"^[\x21-\x7e]{1,512}$")
SPACE_MARGIN = 1.05  # copies need a little more than their bytes on disk
TARGET = "target"  # the key prefix of this gallery's own folders
# What an item is, as opposed to where it is filed: all of these must match for two to be the same.
RECORD_COLUMNS = tuple(c for c in COLUMNS if c not in ("id", *FILE_COLUMNS, "favorite", "folder_id"))


def _same_record(a: dict[str, Any], b: dict[str, Any]) -> bool:
    for column in RECORD_COLUMNS:
        x, y = a.get(column), b.get(column)
        if column in ("stats", "provenance"):
            try:
                x, y = json.loads(x or "{}"), json.loads(y or "{}")
            except (TypeError, ValueError):
                return False
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            x, y = float(x), float(y)
        if x != y:
            return False
    return True


class MergeError(ValueError):
    """Something the merge cannot do. The message is safe to show."""


def _default_of(definition: str) -> Any:
    """The DEFAULT of a column definition such as "INTEGER NOT NULL DEFAULT 1"."""
    match = re.search(r"DEFAULT\s+('(?:[^']|'')*'|-?\d+(?:\.\d+)?)", definition)
    if match is None:
        return None
    raw = match.group(1)
    if raw.startswith("'"):
        return raw[1:-1].replace("''", "'")
    return float(raw) if "." in raw else int(raw)


def _safe_rel(root: Path, rel: object) -> str | None:
    """A file path from a source database, if it names a file inside that gallery's own
    folders: forward slashes, plain names, a known folder and kind of file."""
    if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel:
        return None
    parts = PurePosixPath(rel).parts
    if len(parts) < 2 or parts[0] not in TOP_DIRS or any(not PART.match(p) or p in {".", ".."} for p in parts):
        return None
    if PurePosixPath(rel).suffix.lower() not in SUFFIXES:
        return None
    path = root.joinpath(*parts)
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file() or path.is_symlink():
        return None
    return rel


def _json_object(value: object) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return "{}"
        return value if isinstance(parsed, dict) else "{}"
    return "{}"


# Hashes by path, size and modification time: looking at a backup compares every file, and
# the window looks again after each change of a folder name. Bounded, as a merge is.
_HASHES: dict[Path, tuple[int, int, str]] = {}
_HASHES_MAX = 50_000
_hashes_lock = threading.Lock()


def _sha256(path: Path) -> str:
    stat = path.stat()
    key = (stat.st_size, stat.st_mtime_ns)
    with _hashes_lock:
        known = _HASHES.get(path)
        if known is not None and known[:2] == key:
            return known[2]
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    with _hashes_lock:
        if len(_HASHES) >= _HASHES_MAX:
            _HASHES.clear()
        _HASHES[path] = (*key, digest.hexdigest())
    return _HASHES[path][2]


@dataclass
class SourceFolder:
    id: str
    name: str
    created: float
    hidden: bool
    pin: str


@dataclass
class Source:
    root: Path
    label: str
    items: list[dict[str, Any]]
    folders: list[SourceFolder]
    unreadable: int = 0  # rows left out because they failed a check or their file is missing
    unknown_columns: list[str] = field(default_factory=list)
    no_folder: int = 0  # items whose folder is not in the source's folder list: they come in no folder
    no_last_frame: int = 0  # videos whose last frame is missing: they come without one
    new_thumbs: int = 0  # items whose thumbnail is missing: one is made from the picture


def open_source(root: Path) -> Source:
    """Read another gallery without changing it."""
    root = root.resolve()
    db_path = root / "gallery.db"
    if not db_path.is_file():
        raise MergeError(f"“{root.name}” is not a studio gallery: it has no gallery.db.")
    try:
        db = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise MergeError(f"“{root.name}”: its gallery.db cannot be opened ({exc}).") from exc
    db.row_factory = sqlite3.Row
    try:
        tables = {r["name"] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "images" not in tables:
            raise MergeError(f"“{root.name}” is not a studio gallery: its database has no images.")
        columns = [r["name"] for r in db.execute("PRAGMA table_info(images)")]
        if any(c not in columns for c in BASE_COLUMNS):
            raise MergeError(f"“{root.name}” is not a studio gallery: its database is missing columns.")
        unknown = [c for c in columns if c not in COLUMNS]
        rows = [dict(r) for r in db.execute("SELECT * FROM images")]
        folder_rows: list[dict[str, Any]] = []
        if "folders" in tables:
            folder_columns = [r["name"] for r in db.execute("PRAGMA table_info(folders)")]
            unknown += [f"folders.{c}" for c in folder_columns if c not in FOLDER_COLUMNS]
            if all(c in folder_columns for c in FOLDER_BASE_COLUMNS):
                folder_rows = [dict(r) for r in db.execute("SELECT * FROM folders")]
    except sqlite3.Error as exc:
        raise MergeError(f"“{root.name}”: its gallery.db cannot be read ({exc}).") from exc
    finally:
        db.close()

    folders: list[SourceFolder] = []
    for row in folder_rows:
        try:
            name = Gallery._clean_name(row.get("name"))
        except FolderError:
            continue
        if not isinstance(row.get("id"), str) or not ID_PATTERN.match(row["id"]):
            continue
        pin = row.get("pin") if isinstance(row.get("pin"), str) else ""
        folders.append(SourceFolder(
            id=row["id"], name=name,
            created=float(row["created"]) if isinstance(row.get("created"), (int, float)) else time.time(),
            hidden=bool(row.get("hidden")) if isinstance(row.get("hidden"), int) else False,
            pin=pin if PIN_STORED.match(pin) else ("?" if pin else "")))
    folder_ids = {f.id for f in folders}

    # Galleries are all called "outputs": the folder holding one says which it is ("backup1").
    label = root.parent.name if root.name.casefold() == "outputs" and root.parent.name else root.name
    source = Source(root=root, label=label, items=[], folders=folders, unknown_columns=sorted(set(unknown)))
    for row in rows:
        item = _checked_item(root, row, folder_ids, source)
        if item is None:
            source.unreadable += 1
        else:
            source.items.append(item)
    return source


def _checked_item(root: Path, row: dict[str, Any], folder_ids: set[str], source: Source) -> dict[str, Any] | None:
    """A source row with every known column, in this version's types, or None if it fails a check."""
    item: dict[str, Any] = {}
    for column in COLUMNS:
        value = row.get(column, _default_of(ADDED_COLUMNS[column]) if column in ADDED_COLUMNS else None)
        item[column] = value
    if not isinstance(item["id"], str) or not ID_PATTERN.match(item["id"]):
        return None
    if item["media"] not in MEDIA:
        return None
    for column in ("job_id", "model", "prompt", "negative_prompt", "mime"):
        if not isinstance(item[column], str):
            return None
    for column in ("width", "height", "steps", "seed", "duration_ms", "frames", "fps"):
        if not isinstance(item[column], int) or isinstance(item[column], bool):
            return None
    for column in ("created", "guidance"):
        if not isinstance(item[column], (int, float)) or isinstance(item[column], bool):
            return None
    item["favorite"] = 1 if item["favorite"] else 0
    item["stats"] = _json_object(item["stats"])
    item["provenance"] = _json_object(item["provenance"])
    item["file"] = _safe_rel(root, item["file"])
    if item["file"] is None:
        return None
    if item["media"] == "video":
        last = _safe_rel(root, item["last_frame"])
        if last is None and item["last_frame"]:
            source.no_last_frame += 1
        item["last_frame"] = last or ""
    else:
        item["last_frame"] = ""
    item["thumb"] = _safe_rel(root, item["thumb"])
    if item["thumb"] is None:
        # Made again from the picture (or a video's last frame) when copied; without either, left out.
        if item["media"] == "video" and not item["last_frame"]:
            return None
        source.new_thumbs += 1
    start = item["start_image"]
    item["start_image"] = start if isinstance(start, str) and ID_PATTERN.match(start) else ""
    folder = item["folder_id"]
    if isinstance(folder, str) and folder in folder_ids:
        item["folder_id"] = folder
    else:
        if folder:
            source.no_folder += 1
        item["folder_id"] = ""
    return item


# ---------------------------------------------------------------- plan


@dataclass
class NewFolder:
    key: str  # "<source index>:<folder id>"
    source: int
    old_id: str
    id: str
    name: str
    created: float
    hidden: bool
    pin: str


@dataclass
class Plan:
    sources: list[Source]
    folder_map: list[dict[str, str]]  # per source: old folder id -> id in this gallery
    new_folders: list[NewFolder]
    target_renames: dict[str, str]  # this gallery's folder id -> new name
    items: list[tuple[int, dict[str, Any], str]]  # (source, row, id here)
    duplicates: int
    conflicts: list[dict[str, Any]]
    joined: list[dict[str, Any]]  # folders found to be the same as one here
    differing: int
    bytes: int
    free_bytes: int
    missing_loras: int
    inputs: list[tuple[int, str]]  # (source, input id) to copy

    @property
    def problems(self) -> list[str]:
        out = []
        if self.conflicts:
            out.append("Some folders have the same name. Rename them first.")
        if self.bytes * SPACE_MARGIN > self.free_bytes:
            out.append("There is not enough free space on the drive this gallery is on.")
        return out


def _suggest(name: str, label: str, taken: set[str]) -> str:
    """A name like "Trips (outputs2)", shortened to fit, and numbered if that is taken too."""
    for n in range(1, 1000):
        tail = f" ({label})" if n == 1 else f" ({label} {n})"
        candidate = name[: max(1, FOLDER_NAME_MAX - len(tail))].rstrip() + tail
        if candidate.casefold() not in taken:
            return candidate
    raise MergeError("Could not find a free folder name.")


def plan_merge(gallery: Gallery, sources: list[Source], renames: dict[str, str] | None = None,
               lora_ids: set[str] | None = None) -> Plan:
    """What a merge would do, without doing it."""
    renames = dict(renames or {})
    show_hidden = gallery.show_hidden
    here = {f["id"]: f for f in gallery.list_folders(everything=True)}

    # Folders: the same folder as one here (id and name), else a new one.
    folder_map: list[dict[str, str]] = []
    new_folders: list[NewFolder] = []
    joined: list[dict[str, Any]] = []
    used_ids = set(here)
    by_old_id: dict[str, NewFolder] = {}
    for index, source in enumerate(sources):
        mapping: dict[str, str] = {}
        for folder in source.folders:
            mine = here.get(folder.id)
            if mine is not None and mine["name"] == folder.name:
                mapping[folder.id] = folder.id
                if not (mine["hidden"] and not show_hidden):
                    joined.append({"name": mine["name"], "source": source.label,
                                   "settings_differ": bool(folder.hidden) != mine["hidden"]
                                   or bool(folder.pin) != mine["locked"]})
                continue
            earlier = by_old_id.get(folder.id)
            if earlier is not None and earlier.name == folder.name:
                mapping[folder.id] = earlier.id
                continue
            new_id = folder.id if folder.id not in used_ids else secrets.token_hex(8)
            while new_id in used_ids:
                new_id = secrets.token_hex(8)
            used_ids.add(new_id)
            key = f"{index}:{folder.id}"
            entry = NewFolder(key=key, source=index, old_id=folder.id, id=new_id, name=folder.name,
                              created=folder.created, hidden=folder.hidden, pin=folder.pin)
            if key in renames:
                entry.name = Gallery._clean_name(renames.pop(key))
            new_folders.append(entry)
            by_old_id.setdefault(folder.id, entry)
            mapping[folder.id] = new_id
        folder_map.append(mapping)

    target_renames: dict[str, str] = {}
    for key, name in renames.items():
        prefix, _, folder_id = key.partition(":")
        mine = here.get(folder_id)
        if prefix != TARGET or mine is None or (mine["hidden"] and not show_hidden):
            continue  # a stale key, or a folder this session does not see
        if mine["locked"] and not mine["unlocked"]:
            raise MergeError(f"“{mine['name']}” is locked here. Unlock it to rename it.")
        target_renames[folder_id] = Gallery._clean_name(name)

    # Name clashes among the folders this session sees, regardless of case.
    seen: dict[str, list[dict[str, Any]]] = {}
    for folder_id, mine in here.items():
        if mine["hidden"] and not show_hidden:
            continue
        name = target_renames.get(folder_id, mine["name"])
        seen.setdefault(name.casefold(), []).append(
            {"key": f"{TARGET}:{folder_id}", "where": "This gallery", "name": name,
             "locked": mine["locked"] and not mine["unlocked"], "hidden": mine["hidden"]})
    for entry in new_folders:
        if entry.hidden and not show_hidden:
            continue
        seen.setdefault(entry.name.casefold(), []).append(
            {"key": entry.key, "where": sources[entry.source].label, "name": entry.name,
             "locked": bool(entry.pin), "hidden": entry.hidden})
    taken = set(seen)
    conflicts = []
    for group in seen.values():
        if len(group) < 2:
            continue
        # This gallery's folder keeps its name; the first source's does when none is here.
        keeper = next((g for g in group if g["key"].startswith(f"{TARGET}:")), group[0])
        for g in group:
            if g is keeper:
                g["suggested"] = g["name"]
            else:
                g["suggested"] = _suggest(g["name"], g["where"], taken)
                taken.add(g["suggested"].casefold())
        conflicts.append({"name": keeper["name"], "entries": group})

    # Items. Each id here or already coming stands for one item. An item with such an id is the
    # same item only if its file is byte for byte the same and its record matches; otherwise
    # both are kept, the newcomer with a new id.
    with gallery._lock:
        here_rows = {r["id"]: dict(r) for r in gallery._db.execute("SELECT * FROM images")}
    known: dict[str, tuple[Path, dict[str, Any], bool]] = {  # id -> (file, record, here)
        item_id: (gallery.root / r["file"], r, True) for item_id, r in here_rows.items()}
    items: list[tuple[int, dict[str, Any], str]] = []
    duplicates = 0
    differing = 0  # skipped as the same item, but a favourite or in a folder only in the source
    total = 0
    missing_loras = 0
    inputs: list[tuple[int, str]] = []
    wanted_inputs: set[str] = set()
    def same_file(a: Path, b: Path) -> bool:
        try:
            if a.stat().st_size != b.stat().st_size:
                return False
            return _sha256(a) == _sha256(b)
        except OSError:
            return False

    for index, source in enumerate(sources):
        for row in source.items:
            path = source.root / row["file"]
            item_id = row["id"]
            if item_id in known and _same_record(known[item_id][1], row) and same_file(known[item_id][0], path):
                duplicates += 1
                mine, here = known[item_id][1], known[item_id][2]
                if here and ((row["favorite"] and not mine["favorite"]) or (row["folder_id"] and not mine["folder_id"])):
                    differing += 1
                continue
            new_id = item_id
            while new_id in known:
                new_id = secrets.token_hex(8)
            known[new_id] = (path, row, False)
            items.append((index, row, new_id))
            for column in FILE_COLUMNS:
                if row[column]:
                    total += (source.root / row[column]).stat().st_size
            start = row["start_image"]
            if start and start not in wanted_inputs:
                start_path = source.root / "inputs" / f"{start}.png"
                if start_path.is_file() and not (gallery.inputs_dir / f"{start}.png").exists():
                    wanted_inputs.add(start)
                    inputs.append((index, start))
                    total += start_path.stat().st_size
            if lora_ids is not None:
                used = json.loads(row["provenance"]).get("loras") or []
                if isinstance(used, list) and any(isinstance(u, dict) and u.get("id") not in lora_ids for u in used):
                    missing_loras += 1

    free = shutil.disk_usage(gallery.root).free
    return Plan(sources=sources, folder_map=folder_map, new_folders=new_folders, target_renames=target_renames,
                items=items, duplicates=duplicates, conflicts=conflicts, joined=joined, differing=differing, bytes=total,
                free_bytes=free, missing_loras=missing_loras, inputs=inputs)


def describe(plan: Plan, show_hidden: bool) -> dict[str, Any]:
    """The plan as the window shows it. Hidden folders are named only while hidden folders are shown."""
    sources = []
    for index, source in enumerate(plan.sources):
        coming = [row for i, row, _ in plan.items if i == index]
        folders = []
        for folder in source.folders:
            secret = folder.hidden and not show_hidden
            folders.append({"name": None if secret else folder.name, "hidden": folder.hidden,
                            "locked": bool(folder.pin)})
        sources.append({
            "label": source.label,
            "items": len(source.items),
            "images": sum(1 for r in source.items if r["media"] == "image"),
            "videos": sum(1 for r in source.items if r["media"] == "video"),
            "favorites": sum(1 for r in source.items if r["favorite"]),
            "new_items": len(coming),
            "folders": folders,
            "unreadable": source.unreadable,
            "no_folder": source.no_folder,
            "no_last_frame": source.no_last_frame,
            "new_thumbs": source.new_thumbs,
            "unknown_columns": source.unknown_columns,
        })
    return {
        "sources": sources,
        "new_items": len(plan.items),
        "new_folders": len(plan.new_folders),
        "duplicates": plan.duplicates,
        "duplicates_differing": plan.differing,
        "conflicts": plan.conflicts,
        "joined": plan.joined,
        "bytes": plan.bytes,
        "free_bytes": plan.free_bytes,
        "missing_loras": plan.missing_loras,
        "problems": plan.problems,
    }


# ---------------------------------------------------------------- merge


def _free_rel(gallery: Gallery, rel: str, claimed: set[str]) -> str:
    """`rel` if nothing here has that path, else the same name with a number added."""
    path = PurePosixPath(rel)
    candidate = rel
    n = 1
    while candidate in claimed or (gallery.root / candidate).exists():
        n += 1
        candidate = str(path.with_name(f"{path.stem}-{n}{path.suffix}"))
    claimed.add(candidate)
    return candidate


def _copy(src: Path, dst: Path, cancelled: threading.Event | None) -> None:
    """Copy one file, stopping between chunks if the merge is cancelled (a video can be gigabytes)."""
    with src.open("rb") as f, dst.open("wb") as out:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            if cancelled is not None and cancelled.is_set():
                raise MergeError("The merge was cancelled. Nothing was added.")
            out.write(chunk)


def _id_rel(rel: str, old_id: str, new_id: str) -> str:
    """Thumbnails and last frames are named by item id; a new id renames them too."""
    path = PurePosixPath(rel)
    return str(path.with_name(path.name.replace(old_id, new_id))) if old_id != new_id else rel


def run_merge(gallery: Gallery, plan: Plan, progress: Callable[[int, int], None] | None = None,
              cancelled: threading.Event | None = None) -> dict[str, Any]:
    """Copy the files, then add everything to the index in one transaction. On any failure
    nothing is added and every file copied so far is removed."""
    if plan.problems:
        raise MergeError(" ".join(plan.problems))
    copies: list[tuple[Path, Path]] = []  # (from, to)
    thumbs: list[tuple[Path, str]] = []  # (picture, thumbnail to make here)
    rows: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for index, row, new_id in plan.items:
        source = plan.sources[index]
        out = dict(row)
        out["id"] = new_id
        for column in FILE_COLUMNS:
            rel = row[column]
            if not rel:
                continue
            target_rel = _free_rel(gallery, _id_rel(rel, row["id"], new_id), claimed)
            copies.append((source.root / rel, gallery.root / target_rel))
            out[column] = target_rel
        if not row["thumb"]:
            thumb_rel = _free_rel(gallery, f"thumbs/{new_id}.webp", claimed)
            picture = row["file"] if row["media"] == "image" else row["last_frame"]
            thumbs.append((source.root / picture, thumb_rel))
            out["thumb"] = thumb_rel
        out["folder_id"] = plan.folder_map[index].get(row["folder_id"], "") if row["folder_id"] else ""
        rows.append(out)
    for index, input_id in plan.inputs:
        copies.append((plan.sources[index].root / "inputs" / f"{input_id}.png", gallery.inputs_dir / f"{input_id}.png"))

    made: list[Path] = []
    try:
        for done, (src, dst) in enumerate(copies):
            if cancelled is not None and cancelled.is_set():
                raise MergeError("The merge was cancelled. Nothing was added.")
            if dst.exists():
                if dst.parent == gallery.inputs_dir:
                    continue  # a start picture is named by its content: one already here is the same
                raise MergeError("A file appeared where the merge was about to copy one. Nothing was added; try again.")
            dst.parent.mkdir(parents=True, exist_ok=True)
            part = dst.with_name(f"{dst.name}.{secrets.token_hex(4)}.part")
            made.append(part)
            _copy(src, part, cancelled)
            try:
                os.link(part, dst)  # fails rather than replace a file that appeared meanwhile
                part.unlink()
            except FileExistsError:
                raise MergeError("A file appeared where the merge was about to copy one. Nothing was added; try again.")
            except OSError:  # a drive without hard links: checked just before instead
                if dst.exists():
                    raise MergeError("A file appeared where the merge was about to copy one. Nothing was added; try again.")
                os.replace(part, dst)
            made.remove(part)
            made.append(dst)
            if progress is not None:
                progress(done + 1, len(copies))
        for picture, thumb_rel in thumbs:
            from PIL import Image
            with Image.open(picture) as image:
                gallery._thumb(image.convert("RGB"), thumb_rel)
            made.append(gallery.root / thumb_rel)
        with gallery._lock:
            db = gallery._db
            # A folder named meanwhile, here, with a name a new one takes: stop rather than have two.
            visible = {plan.target_renames.get(r["id"], r["name"]).casefold() for r in db.execute(
                "SELECT id, name FROM folders" + ("" if gallery.show_hidden else " WHERE hidden = 0"))}
            for f in plan.new_folders:
                if not (f.hidden and not gallery.show_hidden) and f.name.casefold() in visible:
                    raise MergeError(f"A folder called “{f.name}” was made here meanwhile. Nothing was added; try again.")
            try:
                for folder_id, name in plan.target_renames.items():
                    db.execute("UPDATE folders SET name = ? WHERE id = ?", (name, folder_id))
                for f in plan.new_folders:
                    db.execute("INSERT INTO folders (id, name, created, hidden, pin) VALUES (?, ?, ?, ?, ?)",
                               (f.id, f.name, f.created, int(f.hidden), f.pin))
                for out in rows:
                    db.execute(f"INSERT INTO images ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                               [out[c] for c in COLUMNS])
                db.commit()
            except Exception:
                db.rollback()
                raise
    except BaseException:
        for path in made:
            path.unlink(missing_ok=True)
        raise
    return {
        "added": len(rows),
        "images": sum(1 for r in rows if r["media"] == "image"),
        "videos": sum(1 for r in rows if r["media"] == "video"),
        "favorites": sum(1 for r in rows if r["favorite"]),
        "folders": len(plan.new_folders),
        "joined": len(plan.joined),
        "duplicates": plan.duplicates,
        "duplicates_differing": plan.differing,
        "unreadable": sum(s.unreadable for s in plan.sources),
        "no_folder": sum(s.no_folder for s in plan.sources),
        "no_last_frame": sum(s.no_last_frame for s in plan.sources),
        "new_thumbs": sum(s.new_thumbs for s in plan.sources),
        "bytes": plan.bytes,
    }
