"""Stores generated images and videos on disk and indexes them in SQLite.

Each PNG in the gallery carries its full generation request in a text chunk,
with where the model and code came from (provenance), so the image says exactly
how to reproduce it. Copies saved out of the studio leave that chunk out unless
the user chooses otherwise; the studio strips it on export.

Videos carry nothing inside the file (docs/image-engine-decisions.md, D10): their
settings and provenance live in the index only. Next to each video the gallery
keeps a thumbnail and the last frame as a PNG, which "continue this clip" starts from.

Start pictures for image-to-video ("inputs") are stored here too, named by a hash
of their content, so a request that names one stays reproducible.

Folders are the user's own grouping: each image or video sits in at most one.
They only organise the index; files never move on disk, deleting a folder leaves
everything that was in it in place, and favourites are independent of folders.

A folder can be hidden, locked with a six-digit PIN, or both. What a hidden folder
holds is left out of every view until hidden folders are shown; what a locked one
holds, until it is unlocked with its PIN. Both choices are per session: shown hidden
folders and unlocked folders are remembered in memory only, and reset_privacy() (the
studio calls it when its window closes) or a restart hides and locks them again. This
keeps the studio's own views private; it does not encrypt anything, and the files
stay readable on disk to anyone who can open the folder.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from . import video_io

PNG_METADATA_KEY = "local-ai-studio"
# Bumped when the embedded JSON changes shape, so readers can tell versions apart.
METADATA_FORMAT = 1
THUMB_SIZE = 384
INPUT_MAX_SIDE = 4096
ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
MEDIA = ("image", "video")

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id          TEXT PRIMARY KEY,
    created     REAL NOT NULL,
    file        TEXT NOT NULL,
    thumb       TEXT NOT NULL,
    job_id      TEXT NOT NULL,
    model       TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    negative_prompt TEXT NOT NULL DEFAULT '',
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    steps       INTEGER NOT NULL,
    guidance    REAL NOT NULL,
    seed        INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    favorite    INTEGER NOT NULL DEFAULT 0,
    stats       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS images_created ON images(created DESC);
CREATE TABLE IF NOT EXISTS folders (
    id      TEXT PRIMARY KEY,
    name    TEXT NOT NULL,
    created REAL NOT NULL
);
"""

FOLDER_NAME_MAX = 60
PIN_PATTERN = re.compile(r"[0-9]{6}")
# scrypt at 32 MiB and about a tenth of a second per guess. Six digits are a million
# guesses, so this only slows down someone who has copied gallery.db; inside the studio
# the limit on wrong attempts below is what protects a PIN.
PIN_SCRYPT = {"n": 2**15, "r": 8, "p": 1}
PIN_FREE_TRIES = 5
PIN_FIRST_WAIT = 30.0
PIN_MAX_WAIT = 15 * 60.0

# Columns added to folders after the first release.
FOLDER_ADDED_COLUMNS = {
    "hidden": "INTEGER NOT NULL DEFAULT 0",
    "pin": "TEXT NOT NULL DEFAULT ''",  # '' = no PIN; else scrypt$n$r$p$salt$hash
    # The order the user dragged the folders into. 0 everywhere means no order of
    # their own, which is how every gallery starts: then the list is alphabetical.
    "position": "INTEGER NOT NULL DEFAULT 0",
}


class FolderError(ValueError):
    """A folder name or id the gallery cannot accept. The message is safe to show."""


class FolderLocked(FolderError):
    """The folder is locked and this session has not unlocked it, or a PIN was wrong."""


class TooManyTries(FolderError):
    """Too many wrong PINs in a row; `wait` seconds until the next try."""

    def __init__(self, wait: float) -> None:
        self.wait = wait
        super().__init__(f"Too many wrong PINs. Try again in {max(1, round(wait))} seconds.")


def _hash_pin(pin: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(pin.encode(), salt=salt, maxmem=64 * 2**20, dklen=32, **PIN_SCRYPT)
    return "$".join(["scrypt", str(PIN_SCRYPT["n"]), str(PIN_SCRYPT["r"]), str(PIN_SCRYPT["p"]),
                     salt.hex(), digest.hex()])


def _pin_matches(pin: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        got = hashlib.scrypt(pin.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                             maxmem=64 * 2**20, dklen=32)
    except ValueError:
        return False
    return hmac.compare_digest(got, bytes.fromhex(digest))


def _check_pin_format(pin: object) -> str:
    if not isinstance(pin, str) or not PIN_PATTERN.fullmatch(pin):  # fullmatch: "$" would let "123456\n" in
        raise FolderError("A PIN is six digits.")
    return pin

# Columns added after the first release. A gallery made by an older version gains them,
# with these defaults describing what its rows are: still images.
ADDED_COLUMNS = {
    "provenance": "TEXT NOT NULL DEFAULT '{}'",
    "media": "TEXT NOT NULL DEFAULT 'image'",
    "mime": "TEXT NOT NULL DEFAULT 'image/png'",
    "frames": "INTEGER NOT NULL DEFAULT 1",
    "fps": "INTEGER NOT NULL DEFAULT 0",
    "start_image": "TEXT NOT NULL DEFAULT ''",
    "last_frame": "TEXT NOT NULL DEFAULT ''",
    "folder_id": "TEXT NOT NULL DEFAULT ''",  # '' = in no folder
}

COLUMNS = ("id", "created", "file", "thumb", "job_id", "model", "prompt", "negative_prompt",
           "width", "height", "steps", "guidance", "seed", "duration_ms", "favorite", "stats",
           *ADDED_COLUMNS)
PRIVATE = ("file", "thumb", "last_frame")


@dataclass(frozen=True)
class SavedImage:
    record: dict[str, Any]


def valid_id(image_id: str) -> bool:
    return bool(ID_PATTERN.match(image_id))


class Gallery:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.images_dir = root / "images"
        self.videos_dir = root / "videos"
        self.thumbs_dir = root / "thumbs"
        self.frames_dir = root / "frames"
        self.inputs_dir = root / "inputs"
        for folder in (self.images_dir, self.videos_dir, self.thumbs_dir, self.frames_dir, self.inputs_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(root / "gallery.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        for table, added in (("images", ADDED_COLUMNS), ("folders", FOLDER_ADDED_COLUMNS)):
            existing = {row["name"] for row in self._db.execute(f"PRAGMA table_info({table})")}
            for name, definition in added.items():
                if name not in existing:
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
        self._db.commit()
        # This session's privacy choices; never written anywhere (module docstring).
        self._show_hidden = False
        self._unlocked: set[str] = set()
        self._wrong_pins: dict[str, tuple[int, float]] = {}  # folder id -> (wrong in a row, no tries before)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------ write

    def _names(self, seed: int) -> tuple[str, float, str, str]:
        image_id = secrets.token_hex(8)
        created = time.time()
        stamp = time.localtime(created)
        day = time.strftime("%Y-%m-%d", stamp)
        name = f"{time.strftime('%Y%m%d-%H%M%S', stamp)}-{seed}-{image_id[:6]}"
        return image_id, created, day, name

    def _thumb(self, image: Image.Image, rel: str) -> None:
        thumb = image.copy()
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
        thumb.save(self.root / rel, format="WEBP", quality=82)

    def _insert(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._db.execute(
                f"INSERT INTO images ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                [record[c] for c in COLUMNS],
            )
            self._db.commit()
        return self._public(record)

    def save(self, image: Image.Image, *, job_id: str, model: str, request: dict[str, Any],
             seed: int, duration_ms: int, stats: dict[str, Any] | None = None,
             provenance: dict[str, Any] | None = None) -> dict[str, Any]:
        """Store one image. `provenance` names the app, the exact model revision and the
        library versions behind it, when known; it goes into the PNG and the index."""
        image_id, created, day, name = self._names(seed)
        provenance = dict(provenance or {})

        # The request's own fields come first, so provenance can never overwrite them.
        meta = {"format": METADATA_FORMAT, **provenance, **request, "seed": seed, "model": model}
        info = PngInfo()
        info.add_text(PNG_METADATA_KEY, json.dumps(meta, ensure_ascii=False))

        (self.images_dir / day).mkdir(exist_ok=True)
        file_rel = f"images/{day}/{name}.png"
        thumb_rel = f"thumbs/{image_id}.webp"
        image.save(self.root / file_rel, format="PNG", pnginfo=info)
        self._thumb(image, thumb_rel)

        return self._insert({
            "id": image_id, "created": created, "file": file_rel, "thumb": thumb_rel,
            "job_id": job_id, "model": model,
            "prompt": request["prompt"], "negative_prompt": request.get("negative_prompt", ""),
            "width": image.width, "height": image.height,
            "steps": request["steps"], "guidance": request["guidance"], "seed": seed,
            "duration_ms": duration_ms, "favorite": 0, "stats": json.dumps(stats or {}),
            "provenance": json.dumps(provenance), "media": "image", "mime": "image/png",
            "frames": 1, "fps": 0, "start_image": "", "last_frame": "", "folder_id": "",
        })

    def save_video(self, frames: list[Image.Image], *, job_id: str, model: str, request: dict[str, Any],
                   seed: int, duration_ms: int, ffmpeg: str | None, stats: dict[str, Any] | None = None,
                   provenance: dict[str, Any] | None = None) -> dict[str, Any]:
        """Encode and store one video (see video_io for the format), with its thumbnail and last frame."""
        if not frames:
            raise ValueError("a video needs at least one frame")
        image_id, created, day, name = self._names(seed)
        (self.videos_dir / day).mkdir(exist_ok=True)
        encoded = video_io.encode(frames, request["fps"], self.videos_dir / day / name, ffmpeg)
        file_rel = encoded.path.relative_to(self.root).as_posix()
        thumb_rel = f"thumbs/{image_id}.webp"
        last_rel = f"frames/{image_id}-last.png"
        self._thumb(frames[len(frames) // 2], thumb_rel)
        frames[-1].convert("RGB").save(self.root / last_rel, format="PNG")

        width, height = frames[0].size
        return self._insert({
            "id": image_id, "created": created, "file": file_rel, "thumb": thumb_rel,
            "job_id": job_id, "model": model,
            "prompt": request["prompt"], "negative_prompt": request.get("negative_prompt", ""),
            "width": width, "height": height,
            "steps": request["steps"], "guidance": request["guidance"], "seed": seed,
            "duration_ms": duration_ms, "favorite": 0,
            "stats": json.dumps({**(stats or {}), "encoder": encoded.encoder}),
            "provenance": json.dumps(dict(provenance or {})), "media": "video", "mime": encoded.mime,
            "frames": len(frames), "fps": request["fps"], "start_image": request.get("start_image") or "",
            "last_frame": last_rel, "folder_id": "",
        })

    def set_favorite(self, image_id: str, favorite: bool) -> dict[str, Any] | None:
        with self._lock:
            self._db.execute("UPDATE images SET favorite = ? WHERE id = ?", (int(favorite), image_id))
            self._db.commit()
        return self.get(image_id)

    def set_folder(self, image_id: str, folder_id: str | None, *, everything: bool = False) -> dict[str, Any] | None:
        """Put an item in a folder, or (None) in none. Raises FolderError for a folder that does not
        exist or (unless `everything`, the engine's own use) is hidden from this session. A locked
        folder takes items without its PIN; they are then out of view until it is unlocked."""
        with self._lock:
            if folder_id is not None and not (self._folder_exists(folder_id) if everything
                                              else self._folder_row(folder_id) is not None):
                raise FolderError("No such folder.")
            self._db.execute("UPDATE images SET folder_id = ? WHERE id = ?", (folder_id or "", image_id))
            self._db.commit()
        return self.get(image_id)

    def set_folder_many(self, image_ids: list[str], folder_id: str | None) -> list[dict[str, Any]]:
        """set_folder for several items in one go (one check, one commit). Items that are gone, or
        out of view this session, are skipped; returns the records of the ones moved."""
        with self._lock:
            if folder_id is not None and self._folder_row(folder_id) is None:
                raise FolderError("No such folder.")
            blocked = self._blocked()
            moved = []
            for image_id in dict.fromkeys(image_ids):
                row = self._db.execute("SELECT folder_id FROM images WHERE id = ?", (image_id,)).fetchone()
                if row is None or (row["folder_id"] and row["folder_id"] in blocked):
                    continue
                self._db.execute("UPDATE images SET folder_id = ? WHERE id = ?", (folder_id or "", image_id))
                moved.append(image_id)
            self._db.commit()
        return [r for r in (self.get(i) for i in moved) if r is not None]

    # ------------------------------------------------------------ folders

    @staticmethod
    def _clean_name(name: object) -> str:
        if not isinstance(name, str):
            raise FolderError("A folder name must be text.")
        cleaned = " ".join(name.split())  # no line breaks or runs of spaces
        if not cleaned:
            raise FolderError("A folder needs a name.")
        if len(cleaned) > FOLDER_NAME_MAX:
            raise FolderError(f"A folder name can be at most {FOLDER_NAME_MAX} characters.")
        if any(ord(c) < 32 or ord(c) == 127 for c in cleaned):
            raise FolderError("A folder name cannot contain control characters.")
        return cleaned

    def _folder_exists(self, folder_id: str) -> bool:
        return self._db.execute("SELECT 1 FROM folders WHERE id = ?", (folder_id,)).fetchone() is not None

    def _name_taken(self, name: str, except_id: str = "") -> bool:
        """Among the folders this session sees: refusing a name because a hidden folder has it
        would tell that the hidden folder exists."""
        hidden = "" if self._show_hidden else " AND hidden = 0"
        row = self._db.execute(f"SELECT 1 FROM folders WHERE lower(name) = lower(?) AND id != ?{hidden}",
                               (name, except_id)).fetchone()
        return row is not None

    def list_folders(self, *, everything: bool = False) -> list[dict[str, Any]]:
        """The folders this session may see, by name, with how many items each holds (None for
        one still locked) and whether it is hidden, has a PIN, and is unlocked. `everything`
        includes hidden ones regardless, for the engine's own use (never shown as is)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT f.id, f.name, f.created, f.hidden, f.pin, COUNT(i.id) AS count FROM folders f "
                "LEFT JOIN images i ON i.folder_id = f.id GROUP BY f.id "
                "ORDER BY f.position, lower(f.name)").fetchall()
            out = []
            for r in rows:
                if r["hidden"] and not (self._show_hidden or everything):
                    continue
                locked = bool(r["pin"])
                unlocked = locked and r["id"] in self._unlocked
                out.append({"id": r["id"], "name": r["name"], "created": r["created"],
                            "count": r["count"] if unlocked or not locked else None,
                            "hidden": bool(r["hidden"]), "locked": locked, "unlocked": unlocked})
        return out

    # ------------------------------------------------------------ privacy

    def _blocked(self) -> set[str]:
        """Folders whose items this session may not see: hidden ones (unless shown) and locked
        ones not unlocked. Call with self._lock held."""
        rows = self._db.execute("SELECT id, hidden, pin FROM folders WHERE hidden = 1 OR pin != ''").fetchall()
        return {r["id"] for r in rows
                if (r["hidden"] and not self._show_hidden) or (r["pin"] and r["id"] not in self._unlocked)}

    def _folder_row(self, folder_id: str) -> sqlite3.Row | None:
        """A folder this session can see (hidden ones only while shown), else None. Lock held."""
        row = self._db.execute("SELECT id, hidden, pin FROM folders WHERE id = ?", (folder_id,)).fetchone()
        if row is None or (row["hidden"] and not self._show_hidden):
            return None
        return row

    def _require_open(self, row: sqlite3.Row) -> None:
        if row["pin"] and row["id"] not in self._unlocked:
            raise FolderLocked("This folder is locked. Unlock it with its PIN first.")

    @property
    def show_hidden(self) -> bool:
        return self._show_hidden

    def set_show_hidden(self, show: bool) -> None:
        with self._lock:
            self._show_hidden = bool(show)

    def reset_privacy(self) -> None:
        """Hide hidden folders and lock every unlocked one again, as at start."""
        with self._lock:
            self._show_hidden = False
            self._unlocked.clear()

    def blocked_folders(self) -> list[str]:
        with self._lock:
            return sorted(self._blocked())

    def set_hidden(self, folder_id: str, hidden: bool) -> dict[str, Any] | None:
        """Hide or show a folder. Needs no PIN: hiding shows nothing, and showing a locked
        folder still keeps it locked."""
        with self._lock:
            if self._folder_row(folder_id) is None:
                return None
            self._db.execute("UPDATE folders SET hidden = ? WHERE id = ?", (int(bool(hidden)), folder_id))
            self._db.commit()
        return self._listed(folder_id)

    def _check_tries(self, folder_id: str) -> None:
        _, not_before = self._wrong_pins.get(folder_id, (0, 0.0))
        wait = not_before - time.monotonic()
        if wait > 0:
            raise TooManyTries(wait)

    def _verify(self, folder_id: str, stored: str, pin: object) -> None:
        """Check a PIN against a folder's, counting wrong ones. Raises FolderLocked or TooManyTries."""
        pin = _check_pin_format(pin)
        with self._lock:
            self._check_tries(folder_id)
        if _pin_matches(pin, stored):  # slow on purpose; not under the lock
            with self._lock:
                self._wrong_pins.pop(folder_id, None)
            return
        with self._lock:
            wrong, _ = self._wrong_pins.get(folder_id, (0, 0.0))
            wrong += 1
            wait = 0.0 if wrong < PIN_FREE_TRIES else min(PIN_MAX_WAIT, PIN_FIRST_WAIT * 2 ** (wrong - PIN_FREE_TRIES))
            self._wrong_pins[folder_id] = (wrong, time.monotonic() + wait)
        raise FolderLocked("Wrong PIN.")

    def unlock(self, folder_id: str, pin: object) -> dict[str, Any] | None:
        """Unlock a folder for this session."""
        with self._lock:
            row = self._folder_row(folder_id)
        if row is None:
            return None
        if not row["pin"]:
            raise FolderError("This folder has no PIN.")
        self._verify(folder_id, row["pin"], pin)
        with self._lock:
            self._unlocked.add(folder_id)
        return self._listed(folder_id)

    def lock_again(self, folder_id: str) -> dict[str, Any] | None:
        with self._lock:
            if self._folder_row(folder_id) is None:
                return None
            self._unlocked.discard(folder_id)
        return self._listed(folder_id)

    def set_pin(self, folder_id: str, pin: object, current: object = None) -> dict[str, Any] | None:
        """Give a folder a PIN, change it, or (pin None) remove it. A folder that has one needs
        it as `current`, even while unlocked, so an unattended unlocked studio cannot change it.
        Setting a PIN leaves the folder unlocked for this session; it locks when the session ends."""
        with self._lock:
            row = self._folder_row(folder_id)
        if row is None:
            return None
        if row["pin"]:
            if current is None:
                raise FolderLocked("Enter the folder's current PIN.")
            self._verify(folder_id, row["pin"], current)
        stored = "" if pin is None else _hash_pin(_check_pin_format(pin))
        with self._lock:
            self._db.execute("UPDATE folders SET pin = ? WHERE id = ?", (stored, folder_id))
            self._db.commit()
            if stored:
                self._unlocked.add(folder_id)
            else:
                self._unlocked.discard(folder_id)
        return self._listed(folder_id)

    def _listed(self, folder_id: str) -> dict[str, Any] | None:
        return next((f for f in self.list_folders(everything=True) if f["id"] == folder_id), None)

    def _next_position(self) -> int:
        """Where a new folder goes: nowhere in particular (0) until the user has
        put the folders in an order, then after the last of them."""
        row = self._db.execute("SELECT MAX(position) AS top FROM folders").fetchone()
        top = row["top"] if row and row["top"] is not None else 0
        return top + 1 if top else 0

    def _ordered_folder_ids(self) -> list[str]:
        """Every folder, hidden ones included, in the order the list shows them."""
        rows = self._db.execute("SELECT id FROM folders ORDER BY position, lower(name)").fetchall()
        return [r["id"] for r in rows]

    def move_folder(self, folder_id: str, before: object) -> list[str]:
        """Put `folder_id` just before the folder `before`, or last when that is None.

        The whole order is rewritten, hidden folders included, so it stays total:
        a folder this session cannot see keeps its place among the ones it can.
        Returns the new order (every folder). Nothing is moved on disk and no item
        changes folder: this is only the order the list is shown in.
        """
        with self._lock:
            if self._folder_row(folder_id) is None:
                raise FolderError("That folder is no longer there.")
            target = None
            if before is not None:
                if not isinstance(before, str) or self._folder_row(before) is None:
                    raise FolderError("That folder is no longer there.")
                if before == folder_id:
                    raise FolderError("A folder cannot be moved before itself.")
                target = before
            order = [i for i in self._ordered_folder_ids() if i != folder_id]
            at = order.index(target) if target is not None else len(order)
            order.insert(at, folder_id)
            # Positions start at 1: 0 is reserved for "no order of their own".
            self._db.executemany("UPDATE folders SET position = ? WHERE id = ?",
                                 [(n, i) for n, i in enumerate(order, start=1)])
            self._db.commit()
        return order

    def create_folder(self, name: object) -> dict[str, Any]:
        cleaned = self._clean_name(name)
        folder = {"id": secrets.token_hex(8), "name": cleaned, "created": time.time(), "count": 0}
        with self._lock:
            if self._name_taken(cleaned):
                raise FolderError(f"There is already a folder called “{cleaned}”.")
            # Alphabetical while the gallery has no order of its own; once it has
            # one, a new folder joins the end rather than jumping into the middle.
            self._db.execute("INSERT INTO folders (id, name, created, position) VALUES (?, ?, ?, ?)",
                             (folder["id"], folder["name"], folder["created"], self._next_position()))
            self._db.commit()
        return folder

    def rename_folder(self, folder_id: str, name: object) -> dict[str, Any] | None:
        cleaned = self._clean_name(name)
        with self._lock:
            row = self._folder_row(folder_id)
            if row is None:
                return None
            self._require_open(row)
            if self._name_taken(cleaned, except_id=folder_id):
                raise FolderError(f"There is already a folder called “{cleaned}”.")
            self._db.execute("UPDATE folders SET name = ? WHERE id = ?", (cleaned, folder_id))
            self._db.commit()
        return self._listed(folder_id)

    def delete_folder(self, folder_id: str) -> int | None:
        """Remove a folder. What was in it stays in the gallery, in no folder. Returns how many items that was.
        A locked folder must be unlocked first: deleting it would put what it holds in plain view."""
        with self._lock:
            row = self._folder_row(folder_id)
            if row is None:
                return None
            self._require_open(row)
            moved = self._db.execute("UPDATE images SET folder_id = '' WHERE folder_id = ?", (folder_id,)).rowcount
            self._db.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
            self._db.commit()
            self._unlocked.discard(folder_id)
            self._wrong_pins.pop(folder_id, None)
        return moved

    def delete(self, image_id: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT file, thumb, last_frame, folder_id FROM images WHERE id = ?",
                                   (image_id,)).fetchone()
            if row is None or (row["folder_id"] and row["folder_id"] in self._blocked()):
                return False
            self._db.execute("DELETE FROM images WHERE id = ?", (image_id,))
            self._db.commit()
        for rel in (row["file"], row["thumb"], row["last_frame"]):
            if rel:
                (self.root / rel).unlink(missing_ok=True)
        return True

    # ------------------------------------------------------------ start pictures

    def save_input(self, image: Image.Image) -> str:
        """Keep a picture a video can start from. Returns its id, taken from the content,
        so choosing the same picture twice stores it once. Nothing of the original file
        but its pixels is kept."""
        rgb = image.convert("RGB")
        if max(rgb.size) > INPUT_MAX_SIDE:
            rgb.thumbnail((INPUT_MAX_SIDE, INPUT_MAX_SIDE))
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
        data = buffer.getvalue()
        input_id = hashlib.sha256(data).hexdigest()[:16]
        path = self.inputs_dir / f"{input_id}.png"
        if not path.exists():
            part = path.with_name(path.name + ".part")
            part.write_bytes(data)
            os.replace(part, path)
        return input_id

    def input_path(self, input_id: str) -> Path | None:
        if not valid_id(input_id):
            return None
        path = self.inputs_dir / f"{input_id}.png"
        return path if path.is_file() else None

    def load_input(self, input_id: str) -> Image.Image | None:
        path = self.input_path(input_id)
        if path is None:
            return None
        with Image.open(path) as img:
            return img.convert("RGB")

    def input_from(self, image_id: str, frame: str) -> str | None:
        """Make a start picture from a gallery image ("image") or a video's last frame ("last")."""
        path = self.path_of(image_id, "image" if frame == "image" else "last")
        if path is None or not path.is_file():
            return None
        with Image.open(path) as img:
            return self.save_input(img)

    # ------------------------------------------------------------ read

    def get(self, image_id: str) -> dict[str, Any] | None:
        """An item by id; None too for one in a hidden or locked folder this session cannot see."""
        with self._lock:
            row = self._db.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            if row is not None and row["folder_id"] and row["folder_id"] in self._blocked():
                row = None
        return self._public(dict(row)) if row else None

    def list(self, *, offset: int = 0, limit: int = 60, query: str = "",
             favorites_only: bool = False, media: str = "", folder: str = "") -> dict[str, Any]:
        """`folder` narrows to one folder's items. Favourites are not narrowed by folder
        unless a folder is given too: the favourites view shows every favourite. Items in
        hidden or locked folders this session cannot see are left out of every view; asking
        for such a folder raises FolderError (hidden: as if it did not exist) or FolderLocked."""
        if folder:
            with self._lock:
                row = self._folder_row(folder)
                if row is None:
                    raise FolderError("No such folder.")
                self._require_open(row)
        where, args = [], []
        if query:
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where.append("prompt LIKE ? ESCAPE '\\'")
            args.append(f"%{escaped}%")
        if favorites_only:
            where.append("favorite = 1")
        if media in MEDIA:
            where.append("media = ?")
            args.append(media)
        if folder:
            where.append("folder_id = ?")
            args.append(folder)
        with self._lock:
            blocked = sorted(self._blocked())
            if blocked:
                where.append(f"folder_id NOT IN ({', '.join('?' * len(blocked))})")
                args.extend(blocked)
            clause = f"WHERE {' AND '.join(where)}" if where else ""
            total = self._db.execute(f"SELECT COUNT(*) FROM images {clause}", args).fetchone()[0]
            rows = self._db.execute(
                f"SELECT * FROM images {clause} ORDER BY created DESC LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        return {"total": total, "items": [self._public(dict(r)) for r in rows]}

    def path_of(self, image_id: str, kind: str) -> Path | None:
        """Absolute path of a file behind an index entry. Only ids from the index resolve, and
        only the kinds that entry has: "image" for images, "video" and "last" for videos,
        "thumb" for both."""
        column, media = {"image": ("file", "image"), "video": ("file", "video"),
                         "last": ("last_frame", "video"), "thumb": ("thumb", None)}[kind]
        with self._lock:
            row = self._db.execute(f"SELECT {column}, media, folder_id FROM images WHERE id = ?",
                                   (image_id,)).fetchone()
            if row is not None and row["folder_id"] and row["folder_id"] in self._blocked():
                row = None
        if row is None or not row[0] or (media is not None and row["media"] != media):
            return None
        path = (self.root / row[0]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            return None
        return path

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in record.items() if k not in PRIVATE}
        out["favorite"] = bool(out["favorite"])
        for key in ("stats", "provenance"):
            value = out.get(key) or "{}"
            out[key] = json.loads(value) if isinstance(value, str) else value
        out["start_image"] = out.get("start_image") or None
        out["folder_id"] = out.get("folder_id") or None
        image_id = record["id"]
        out["thumb_url"] = f"/images/{image_id}.webp"
        out["file_name"] = Path(record["file"]).name
        if out.get("media") == "video":
            out["video_url"] = f"/videos/{image_id}"
            out["last_frame_url"] = f"/frames/{image_id}-last.png"
            out["image_url"] = out["thumb_url"]  # a still for anything that shows one picture
        else:
            out["image_url"] = f"/images/{image_id}.png"
        return out
