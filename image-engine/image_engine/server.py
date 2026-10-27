"""Local HTTP API of the image engine (standard library only).

Its only intended client is the studio's main process. Nothing else on the
machine, and certainly no web page, may drive it:

* bind to loopback only;
* require a per-launch token on every request (the studio passes it through the
  environment, so it never appears in a process list);
* reject any Host header that is not this machine (DNS rebinding);
* refuse anything that looks like a browser: an Origin header or a cross-site
  Sec-Fetch-Site (CSRF);
* require application/json on writes, which a cross-site form cannot send (the one
  exception, a start picture, must be an image type, which no form can send either);
* send no CORS headers at all, plus Cross-Origin-Resource-Policy.
"""

from __future__ import annotations

import hmac
import io
import json
import queue
import re
import socket
import sys
import tempfile
import threading
from dataclasses import asdict, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

from . import __version__, video_io
from .backends import Backend
from .config import SettingsStore
from .events import EventHub
from .gallery import MEDIA, FolderError, Gallery, valid_id
from .jobs import JobQueue
from .params import (DEFAULTS, MAX_COUNT, MAX_SIDE, MAX_STEPS, MIN_SIDE, SIDE_MULTIPLE, VIDEO_DEFAULTS, VIDEO_FPS,
                     VIDEO_FRAME_STEP, VIDEO_MAX_COUNT, VIDEO_MAX_FRAMES, VIDEO_MAX_PIXELS, VIDEO_MAX_SIDE,
                     VIDEO_MIN_FRAMES, VIDEO_MIN_SIDE, VIDEO_SIDE_MULTIPLE, GenerationRequest, RequestError,
                     VideoRequest, parse_job)
from .video_models import VideoProfile, try_profile_of

MAX_BODY = 64 * 1024
# A start picture for image-to-video: a photo straight off a phone fits comfortably.
MAX_INPUT_BYTES = 40 * 1024 * 1024
MAX_INPUT_PIXELS = 64_000_000
INPUT_TYPES = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
VIDEO_TYPES = ("video/mp4", "image/webp")
FILE_CHUNK = 256 * 1024
MEDIA_PREFIXES = ("/images/", "/videos/", "/frames/", "/inputs/")
# Files the gallery serves: path pattern -> (Gallery.path_of kind, content type; None = from the index).
MEDIA_ROUTES = (
    (re.compile(r"^/images/([0-9a-f]{16})\.png$"), "image", "image/png"),
    (re.compile(r"^/images/([0-9a-f]{16})\.webp$"), "thumb", "image/webp"),
    (re.compile(r"^/videos/([0-9a-f]{16})$"), "video", None),
    (re.compile(r"^/frames/([0-9a-f]{16})-last\.png$"), "last", "image/png"),
)
RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
# An oversized body up to this size is read and dropped before the 413, so the answer
# arrives intact; anything larger just gets the connection closed.
DRAIN_LIMIT = 1024 * 1024
DRAIN_SECONDS = 2.0
SSE_KEEPALIVE_SECONDS = 15
TOKEN_HEADER = "X-Studio-Token"
# Returned by a route that has already sent its own answer (a file rather than JSON).
SENT = object()


class HttpError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class App:
    """Everything a request handler needs, wired together once at startup."""

    def __init__(self, *, settings: SettingsStore, backend: Backend, gallery: Gallery,
                 events: EventHub, jobs: JobQueue, port: int, token: str | None) -> None:
        self.settings, self.backend, self.gallery = settings, backend, gallery
        self.events, self.jobs = events, jobs
        self.token = token
        self.stopping = threading.Event()
        self.set_port(port)

    def set_port(self, port: int) -> None:
        self.port = port
        self.allowed_hosts = {f"{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]")}


    def status(self) -> dict[str, Any]:
        models = {kind: {"id": backend.model_id, **self.jobs.models[kind]}
                  for kind, backend in self.jobs.backends.items()}
        # Device numbers come from whichever backend is loaded, so the memory plan shown is the live one.
        loaded = next((b for b in self.jobs.backends.values() if b.loaded), self.backend)
        status = {
            "version": __version__,
            "model": models.get("image", {"id": self.backend.model_id, "state": "unloaded", "message": ""}),
            "models": models,
            "device": loaded.device_info(),
            "queue": self.jobs.queue_summary(),
            "limits": {"min_side": MIN_SIDE, "max_side": MAX_SIDE, "side_multiple": SIDE_MULTIPLE,
                       "max_steps": MAX_STEPS, "max_count": MAX_COUNT},
            "defaults": DEFAULTS,
        }
        if "video" in self.jobs.backends:
            ffmpeg = video_io.find_ffmpeg(self.settings.get().ffmpeg_path)
            profile = self.video_profile()
            status["video"] = {
                "limits": {"min_side": VIDEO_MIN_SIDE, "max_side": VIDEO_MAX_SIDE,
                           "side_multiple": VIDEO_SIDE_MULTIPLE, "max_pixels": VIDEO_MAX_PIXELS,
                           "min_frames": VIDEO_MIN_FRAMES, "max_frames": VIDEO_MAX_FRAMES,
                           "frame_step": VIDEO_FRAME_STEP, "fps": VIDEO_FPS,
                           "max_steps": MAX_STEPS, "max_count": VIDEO_MAX_COUNT},
                "defaults": {**VIDEO_DEFAULTS, **({"steps": profile.steps, "guidance": profile.guidance} if profile else {})},
                # Which model the video folder holds (None until a supported one is chosen).
                "profile": profile.to_dict() if profile else None,
                # Without ffmpeg, videos are saved as animated WebP instead of MP4.
                "encoder": "mp4" if ffmpeg else "webp",
                "ffmpeg": ffmpeg,
            }
        return status

    def video_profile(self) -> VideoProfile | None:
        return try_profile_of(self.settings.get().video_model_dir)

    def fit_to_model(self, request: GenerationRequest | VideoRequest) -> GenerationRequest | VideoRequest:
        """A video request as the configured model will run it: a distilled model has fixed
        steps and no guidance (so no negative prompt), and the gallery should record that."""
        profile = self.video_profile() if isinstance(request, VideoRequest) else None
        if profile is None or not profile.fixed:
            return request
        return replace(request, steps=profile.steps, guidance=profile.guidance,
                       negative_prompt="" if profile.guidance <= 1 else request.negative_prompt)

    def load_model_async(self, kind: str = "image") -> None:
        def run() -> None:
            try:
                self.jobs.load_model(kind)
            except Exception:
                pass  # state and message are already published by the queue
        threading.Thread(target=run, name="model-loader", daemon=True).start()


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    routes: list[tuple[str, re.Pattern, Callable]] = []

    def route(method: str, pattern: str):
        def register(fn):
            routes.append((method, re.compile(f"^{pattern}$"), fn))
            return fn
        return register

    # ---------------------------------------------------------------- API

    @route("GET", r"/api/status")
    def get_status(h, m):
        return app.status()

    @route("GET", r"/api/settings")
    def get_settings(h, m):
        return asdict(app.settings.get())

    @route("PUT", r"/api/settings")
    def put_settings(h, m):
        before = app.settings.get()
        try:
            after = app.settings.update(h.json_body())
        except (TypeError, ValueError) as exc:
            raise HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        # A model whose folder or memory plan changed is unloaded; its next job reloads it.
        for kind, keys in (("image", ("model_dir", "memory_mode")), ("video", ("video_model_dir", "video_memory_mode"))):
            backend = app.jobs.backends.get(kind)
            changed = any(getattr(after, k) != getattr(before, k) for k in keys)
            if backend is not None and backend.loaded and changed:
                app.jobs.unload_model(kind)
        return asdict(after)

    def model_kind(h) -> str | None:
        body = h.json_body()
        kind = body.get("kind") if isinstance(body, dict) else None
        if kind is not None and kind not in app.jobs.backends:
            raise HttpError(HTTPStatus.BAD_REQUEST, "unknown model kind")
        return kind

    @route("POST", r"/api/model/load")
    def post_load(h, m):
        app.load_model_async(model_kind(h) or "image")
        return {"state": "loading"}

    @route("POST", r"/api/model/unload")
    def post_unload(h, m):
        # Without a kind every model goes, which is what freeing the GPU for the chat model needs.
        app.jobs.unload_model(model_kind(h))
        return {"state": app.jobs.model_state}

    @route("POST", r"/api/jobs")
    def post_job(h, m):
        try:
            request = parse_job(h.json_body())
        except RequestError as exc:
            raise HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        if request.kind not in app.jobs.backends:
            raise HttpError(HTTPStatus.BAD_REQUEST, f"this engine cannot make {request.kind}s")
        if isinstance(request, VideoRequest) and request.start_image and app.gallery.input_path(request.start_image) is None:
            raise HttpError(HTTPStatus.BAD_REQUEST, "the start picture is not in the engine's inputs")
        return app.jobs.submit(app.fit_to_model(request)).to_dict()

    @route("POST", r"/api/inputs")
    def post_input(h, m):
        """A start picture for image-to-video, sent as the image bytes themselves."""
        picture = h.image_body()
        return {"id": app.gallery.save_input(picture), "width": picture.width, "height": picture.height}

    @route("POST", r"/api/inputs/from-gallery")
    def post_input_from_gallery(h, m):
        body = h.json_body()
        image_id = body.get("id") if isinstance(body, dict) else None
        frame = body.get("frame", "image") if isinstance(body, dict) else None
        if not isinstance(image_id, str) or not valid_id(image_id) or frame not in ("image", "last"):
            raise HttpError(HTTPStatus.BAD_REQUEST, "expected {\"id\": <gallery id>, \"frame\": \"image\"|\"last\"}")
        input_id = app.gallery.input_from(image_id, frame)
        if input_id is None:
            raise HttpError(HTTPStatus.NOT_FOUND, "no such image" if frame == "image" else "no such video")
        picture = app.gallery.load_input(input_id)
        return {"id": input_id, "width": picture.width, "height": picture.height}

    @route("GET", r"/api/jobs")
    def get_jobs(h, m):
        return {"items": app.jobs.list()}

    @route("GET", r"/api/jobs/(?P<id>[0-9a-f]{16})")
    def get_job(h, m):
        job = app.jobs.get(m["id"])
        if job is None:
            raise HttpError(HTTPStatus.NOT_FOUND, "no such job")
        return job.to_dict()

    @route("POST", r"/api/jobs/(?P<id>[0-9a-f]{16})/cancel")
    def cancel_job(h, m):
        h.json_body()
        job = app.jobs.cancel(m["id"])
        if job is None:
            raise HttpError(HTTPStatus.NOT_FOUND, "no such job")
        return job.to_dict()

    @route("GET", r"/api/gallery")
    def list_gallery(h, m):
        q = h.query()
        try:
            offset = max(0, int(q.get("offset", "0")))
            limit = min(200, max(1, int(q.get("limit", "60"))))
        except ValueError as exc:
            raise HttpError(HTTPStatus.BAD_REQUEST, "offset and limit must be integers") from exc
        media = q.get("media", "")
        if media and media not in MEDIA:
            raise HttpError(HTTPStatus.BAD_REQUEST, "media must be image or video")
        folder = q.get("folder", "")
        if folder and not valid_id(folder):
            raise HttpError(HTTPStatus.BAD_REQUEST, "folder must be a folder id")
        return app.gallery.list(offset=offset, limit=limit, query=q.get("q", "")[:200],
                                favorites_only=q.get("favorites") == "1", media=media, folder=folder)

    @route("GET", r"/api/folders")
    def list_folders(h, m):
        return {"items": app.gallery.list_folders()}

    @route("POST", r"/api/folders")
    def create_folder(h, m):
        body = h.json_body()
        try:
            return app.gallery.create_folder(body.get("name") if isinstance(body, dict) else None)
        except FolderError as exc:
            raise HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from exc

    @route("PATCH", r"/api/folders/(?P<id>[0-9a-f]{16})")
    def rename_folder(h, m):
        body = h.json_body()
        try:
            folder = app.gallery.rename_folder(m["id"], body.get("name") if isinstance(body, dict) else None)
        except FolderError as exc:
            raise HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        if folder is None:
            raise HttpError(HTTPStatus.NOT_FOUND, "no such folder")
        return folder

    @route("DELETE", r"/api/folders/(?P<id>[0-9a-f]{16})")
    def delete_folder(h, m):
        moved = app.gallery.delete_folder(m["id"])
        if moved is None:
            raise HttpError(HTTPStatus.NOT_FOUND, "no such folder")
        return {"deleted": m["id"], "moved": moved}

    @route("GET", r"/api/gallery/(?P<id>[0-9a-f]{16})")
    def get_image(h, m):
        record = app.gallery.get(m["id"])
        if record is None:
            raise HttpError(HTTPStatus.NOT_FOUND, "no such image")
        return record

    @route("PATCH", r"/api/gallery/(?P<id>[0-9a-f]{16})")
    def patch_image(h, m):
        body = h.json_body()
        usage = "expected {\"favorite\": true|false} and/or {\"folder\": <folder id>|null}"
        if not isinstance(body, dict) or not body or set(body) - {"favorite", "folder"}:
            raise HttpError(HTTPStatus.BAD_REQUEST, usage)
        if "favorite" in body and not isinstance(body["favorite"], bool):
            raise HttpError(HTTPStatus.BAD_REQUEST, usage)
        folder = body.get("folder")
        if "folder" in body and folder is not None and not (isinstance(folder, str) and valid_id(folder)):
            raise HttpError(HTTPStatus.BAD_REQUEST, usage)
        if app.gallery.get(m["id"]) is None:
            raise HttpError(HTTPStatus.NOT_FOUND, "no such image")
        record = None
        if "favorite" in body:
            record = app.gallery.set_favorite(m["id"], body["favorite"])
        if "folder" in body:
            try:
                record = app.gallery.set_folder(m["id"], folder)
            except FolderError as exc:
                raise HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        return record

    @route("DELETE", r"/api/gallery/(?P<id>[0-9a-f]{16})")
    def delete_image(h, m):
        if not app.gallery.delete(m["id"]):
            raise HttpError(HTTPStatus.NOT_FOUND, "no such image")
        return {"deleted": m["id"]}

    @route("POST", r"/api/gallery/(?P<id>[0-9a-f]{16})/gif")
    def export_gif(h, m):
        """A video as a looping GIF, sent back as the file itself. Made on request, kept nowhere."""
        body = h.json_body()
        usage = (f"expected {{\"width\": {video_io.GIF_MIN_WIDTH}..{VIDEO_MAX_SIDE}, "
                 f"\"fps\": 1..{video_io.GIF_MAX_FPS}}}")
        if not isinstance(body, dict) or set(body) != {"width", "fps"}:
            raise HttpError(HTTPStatus.BAD_REQUEST, usage)
        width, fps = body["width"], body["fps"]
        for value, low, high in ((width, video_io.GIF_MIN_WIDTH, VIDEO_MAX_SIDE), (fps, 1, video_io.GIF_MAX_FPS)):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise HttpError(HTTPStatus.BAD_REQUEST, usage)
        record = app.gallery.get(m["id"])
        source = app.gallery.path_of(m["id"], "video")
        if record is None or source is None or not source.is_file():
            raise HttpError(HTTPStatus.NOT_FOUND, "no such video")
        ffmpeg = video_io.find_ffmpeg(app.settings.get().ffmpeg_path)
        with tempfile.TemporaryDirectory(prefix="studio-gif-") as tmp:
            dest = Path(tmp) / "export.gif"
            try:
                video_io.to_gif(source, record["mime"], record["fps"], (record["width"], record["height"]), dest,
                                width=width, gif_fps=fps, ffmpeg=ffmpeg)
            except video_io.GifError as exc:
                raise HttpError(HTTPStatus.CONFLICT, str(exc)) from exc
            h._send_file(dest, "image/gif", "no-store")
        return SENT

    # ---------------------------------------------------------------- handler

    class Handler(BaseHTTPRequestHandler):
        server_version = f"LocalImageStudio/{__version__}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # keep the console readable
            if not self.path.startswith(("/api/events", "/assets/", *MEDIA_PREFIXES)):
                super().log_message(fmt, *args)

        def do_GET(self) -> None: self._dispatch("GET")
        def do_POST(self) -> None: self._dispatch("POST")
        def do_PUT(self) -> None: self._dispatch("PUT")
        def do_PATCH(self) -> None: self._dispatch("PATCH")
        def do_DELETE(self) -> None: self._dispatch("DELETE")

        # ------------------------------------------------------------ guards

        def _guard(self, method: str) -> None:
            if self.headers.get("Host", "") not in app.allowed_hosts:
                raise HttpError(HTTPStatus.FORBIDDEN, "unexpected Host header")
            if app.token is not None:
                supplied = self.headers.get(TOKEN_HEADER, "")
                if not hmac.compare_digest(supplied.encode(), app.token.encode()):
                    raise HttpError(HTTPStatus.UNAUTHORIZED, "missing or wrong token")
            # The studio's main process sends neither header. A browser always does.
            if self.headers.get("Sec-Fetch-Site", "none") != "none":
                raise HttpError(HTTPStatus.FORBIDDEN, "browser request refused")
            if self.headers.get("Origin") is not None:
                raise HttpError(HTTPStatus.FORBIDDEN, "cross-origin request refused")
            if method != "GET":
                ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
                length = self.headers.get("Content-Length")
                if method == "POST" and urlsplit(self.path).path == "/api/inputs":
                    if ctype not in INPUT_TYPES:
                        raise HttpError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "expected a PNG, JPEG or WebP image")
                    return
                if length not in (None, "0") and ctype != "application/json":
                    raise HttpError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "expected application/json")
                if method == "POST" and ctype != "application/json":
                    raise HttpError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "expected application/json")

        # ------------------------------------------------------------ dispatch

        def _dispatch(self, method: str) -> None:
            self._body_cache: Any = None
            self._body_read = False
            try:
                self._guard(method)
                path = urlsplit(self.path).path
                if method == "GET" and path == "/api/events":
                    return self._events()
                if method == "GET" and path.startswith(MEDIA_PREFIXES):
                    return self._media(path)
                for verb, pattern, fn in routes:
                    match = pattern.match(path)
                    if match and verb == method:
                        result = fn(self, match.groupdict())
                        if result is not SENT:
                            self._send_json(HTTPStatus.OK, result)
                        return
                if any(p.match(path) for _, p, _ in routes):
                    raise HttpError(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed")
                raise HttpError(HTTPStatus.NOT_FOUND, "not found")
            except HttpError as exc:
                self._send_json(exc.status, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except Exception as exc:
                self.log_error("unhandled error: %r", exc)
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal error"})

        # ------------------------------------------------------------ io helpers

        def json_body(self) -> Any:
            if self._body_read:
                return self._body_cache
            self._body_read = True
            # If the body is not consumed, the bytes left on a keep-alive connection would be
            # parsed as the next request. Close the connection instead of draining unbounded input.
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError as exc:
                self.close_connection = True
                raise HttpError(HTTPStatus.BAD_REQUEST, "bad Content-Length") from exc
            if length < 0 or length > MAX_BODY:
                self.close_connection = True
                if length <= DRAIN_LIMIT:
                    self._discard(length)
                raise HttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body too large")
            raw = self.rfile.read(length) if length else b""
            if not raw:
                self._body_cache = {}
                return self._body_cache
            try:
                self._body_cache = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HttpError(HTTPStatus.BAD_REQUEST, "body is not valid JSON") from exc
            return self._body_cache

        def image_body(self) -> Image.Image:
            """The request body as a picture (PNG, JPEG or WebP), upright, in RGB."""
            self._body_read = True
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError as exc:
                self.close_connection = True
                raise HttpError(HTTPStatus.BAD_REQUEST, "bad Content-Length") from exc
            if length <= 0 or length > MAX_INPUT_BYTES:
                self.close_connection = True
                if 0 < length <= DRAIN_LIMIT:
                    self._discard(length)
                raise HttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE if length > 0 else HTTPStatus.BAD_REQUEST,
                                f"a start picture must be between 1 byte and {MAX_INPUT_BYTES // 2**20} MB")
            raw = self.rfile.read(length)
            ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            try:
                with Image.open(io.BytesIO(raw), formats=[INPUT_TYPES[ctype]]) as img:
                    if img.width * img.height > MAX_INPUT_PIXELS:
                        raise HttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "that picture has too many pixels")
                    img.load()
                    # Phone photos are stored sideways with a rotation tag; apply it, then drop all metadata.
                    return ImageOps.exif_transpose(img).convert("RGB")
            except HttpError:
                raise
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
                raise HttpError(HTTPStatus.BAD_REQUEST, f"not a readable {INPUT_TYPES[ctype]} image") from exc

        def _discard(self, length: int) -> None:
            """Read and drop a body that will not be used. Closing a socket with unread data
            makes Windows reset the connection, which can destroy the 413 before the client
            reads it. Bounded in size (DRAIN_LIMIT) and in time, so it cannot stall a thread."""
            self.connection.settimeout(DRAIN_SECONDS)
            try:
                while length > 0:
                    chunk = self.rfile.read(min(length, 64 * 1024))
                    if not chunk:
                        break
                    length -= len(chunk)
            except OSError:
                pass

        def query(self) -> dict[str, str]:
            return {k: v[-1] for k, v in parse_qs(urlsplit(self.path).query).items()}

        def _common_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Referrer-Policy", "no-referrer")

        def _send_json(self, status: HTTPStatus, payload: Any) -> None:
            if not self._body_read and self.command in ("POST", "PUT", "PATCH", "DELETE"):
                try:
                    self.json_body()  # drain so keep-alive stays in sync
                except HttpError:
                    pass
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if self.close_connection:
                self.send_header("Connection", "close")
            self._common_headers()
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str, cache: str) -> None:
            """Send a file, or the one byte range asked for: a video player seeks with ranges."""
            size = path.stat().st_size
            start, end, status = 0, size - 1, HTTPStatus.OK
            asked = self.headers.get("Range")
            if asked:
                match = RANGE.match(asked.strip())
                if match and (match[1] or match[2]):
                    if match[1]:
                        start = int(match[1])
                        end = min(int(match[2]), size - 1) if match[2] else size - 1
                    else:  # "bytes=-N": the last N bytes
                        start = max(0, size - int(match[2]))
                    status = HTTPStatus.PARTIAL_CONTENT
                if status != HTTPStatus.PARTIAL_CONTENT or start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self._common_headers()
                    self.end_headers()
                    return
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", cache)
            self._common_headers()
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = fh.read(min(FILE_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _media(self, path: str) -> None:
            immutable = "private, max-age=31536000, immutable"
            match = re.match(r"^/inputs/([0-9a-f]{16})\.png$", path)
            if match:
                file = app.gallery.input_path(match[1])
                if file is None:
                    raise HttpError(HTTPStatus.NOT_FOUND, "not found")
                return self._send_file(file, "image/png", immutable)
            for pattern, kind, ctype in MEDIA_ROUTES:
                match = pattern.match(path)
                if not match:
                    continue
                file = app.gallery.path_of(match[1], kind)
                if file is None or not file.is_file():
                    break
                if ctype is None:
                    record = app.gallery.get(match[1])
                    ctype = record["mime"] if record and record["mime"] in VIDEO_TYPES else None
                    if ctype is None:
                        break
                return self._send_file(file, ctype, immutable)
            raise HttpError(HTTPStatus.NOT_FOUND, "not found")

        def _events(self) -> None:
            q = app.events.subscribe()
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self._common_headers()
                self.end_headers()
                self._write_event({"type": "hello", "status": app.status(), "jobs": app.jobs.list()})
                while not app.stopping.is_set():
                    try:
                        event = q.get(timeout=SSE_KEEPALIVE_SECONDS)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    self._write_event(event)
            finally:
                app.events.unsubscribe(q)
                self.close_connection = True

        def _write_event(self, event: dict) -> None:
            data = json.dumps(event, ensure_ascii=False)
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

    return Handler


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False  # a second instance on the same port must fail, not share it
    # The gallery asks for every thumbnail at once. With the default backlog of 5,
    # Windows refuses the connections beyond it, and those pictures show as broken.
    request_queue_size = 128

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client hanging up on an idle keep-alive connection is normal; on Windows it
        # surfaces as a reset. Log only what is actually unexpected.
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)

    def server_bind(self) -> None:
        # On Windows another process could otherwise bind the same port and intercept requests.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(app: App, host: str) -> StudioServer:
    httpd = StudioServer((host, app.port), make_handler(app))
    app.set_port(httpd.server_address[1])  # resolves port 0 (used by tests) to the real one
    return httpd
