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
"""

from __future__ import annotations

import hashlib
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


class FolderError(ValueError):
    """A folder name or id the gallery cannot accept. The message is safe to show."""

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
        existing = {row["name"] for row in self._db.execute("PRAGMA table_info(images)")}
        for name, definition in ADDED_COLUMNS.items():
            if name not in existing:
                self._db.execute(f"ALTER TABLE images ADD COLUMN {name} {definition}")
        self._db.commit()

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

    def set_folder(self, image_id: str, folder_id: str | None) -> dict[str, Any] | None:
        """Put an item in a folder, or (None) in none. Raises FolderError for a folder that does not exist."""
        with self._lock:
            if folder_id is not None and not self._folder_exists(folder_id):
                raise FolderError("No such folder.")
            self._db.execute("UPDATE images SET folder_id = ? WHERE id = ?", (folder_id or "", image_id))
            self._db.commit()
        return self.get(image_id)

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
        row = self._db.execute("SELECT 1 FROM folders WHERE lower(name) = lower(?) AND id != ?",
                               (name, except_id)).fetchone()
        return row is not None

    def list_folders(self) -> list[dict[str, Any]]:
        """Every folder, by name, with how many items it holds."""
        with self._lock:
            rows = self._db.execute(
                "SELECT f.id, f.name, f.created, COUNT(i.id) AS count FROM folders f "
                "LEFT JOIN images i ON i.folder_id = f.id GROUP BY f.id ORDER BY lower(f.name)").fetchall()
        return [dict(r) for r in rows]

    def create_folder(self, name: object) -> dict[str, Any]:
        cleaned = self._clean_name(name)
        folder = {"id": secrets.token_hex(8), "name": cleaned, "created": time.time(), "count": 0}
        with self._lock:
            if self._name_taken(cleaned):
                raise FolderError(f"There is already a folder called “{cleaned}”.")
            self._db.execute("INSERT INTO folders (id, name, created) VALUES (?, ?, ?)",
                             (folder["id"], folder["name"], folder["created"]))
            self._db.commit()
        return folder

    def rename_folder(self, folder_id: str, name: object) -> dict[str, Any] | None:
        cleaned = self._clean_name(name)
        with self._lock:
            if not self._folder_exists(folder_id):
                return None
            if self._name_taken(cleaned, except_id=folder_id):
                raise FolderError(f"There is already a folder called “{cleaned}”.")
            self._db.execute("UPDATE folders SET name = ? WHERE id = ?", (cleaned, folder_id))
            self._db.commit()
        return next((f for f in self.list_folders() if f["id"] == folder_id), None)

    def delete_folder(self, folder_id: str) -> int | None:
        """Remove a folder. What was in it stays in the gallery, in no folder. Returns how many items that was."""
        with self._lock:
            if not self._folder_exists(folder_id):
                return None
            moved = self._db.execute("UPDATE images SET folder_id = '' WHERE folder_id = ?", (folder_id,)).rowcount
            self._db.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
            self._db.commit()
        return moved

    def delete(self, image_id: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT file, thumb, last_frame FROM images WHERE id = ?", (image_id,)).fetchone()
            if row is None:
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
        with self._lock:
            row = self._db.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return self._public(dict(row)) if row else None

    def list(self, *, offset: int = 0, limit: int = 60, query: str = "",
             favorites_only: bool = False, media: str = "", folder: str = "") -> dict[str, Any]:
        """`folder` narrows to one folder's items. Favourites are not narrowed by folder
        unless a folder is given too: the favourites view shows every favourite."""
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
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
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
            row = self._db.execute(f"SELECT {column}, media FROM images WHERE id = ?", (image_id,)).fetchone()
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
