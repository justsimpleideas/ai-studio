"""Start the image engine.

Normally started by the studio's main process:

    python -m image_engine --data-dir <dir> --port 5198 --watch-stdin
    (with STUDIO_IMAGE_TOKEN set in the environment)

Without a token the engine refuses to start, unless --no-token is given for
local experiments.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable

# Generation must never touch the network. Set before any Hugging Face library is imported.
for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
             "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
    os.environ[_var] = "1"

from . import __version__  # noqa: E402

APP_VERSION_ENV = "STUDIO_APP_VERSION"


def app_label() -> str:
    """What made the image, recorded in it: the studio's version when the studio
    started the engine, the engine's own otherwise."""
    version = os.environ.get(APP_VERSION_ENV, "")
    if re.fullmatch(r"[0-9A-Za-z.+-]{1,32}", version):
        return f"Local AI Studio {version}"
    return f"image engine {__version__}"
from . import video_io  # noqa: E402
from .config import DEFAULT_PORT, SettingsStore, default_data_dir  # noqa: E402
from .events import EventHub  # noqa: E402
from .gallery import Gallery  # noqa: E402
from .jobs import JobQueue  # noqa: E402
from .lora_library import LoraLibrary  # noqa: E402
from .training.service import TrainingService  # noqa: E402
from .training.service import register as register_training  # noqa: E402
from .server import App, serve  # noqa: E402

LOOPBACK = ("127.0.0.1", "::1", "localhost")
TOKEN_ENV = "STUDIO_IMAGE_TOKEN"
MIN_TOKEN_LENGTH = 32
PIPE_POLL_SECONDS = 0.5


def watch_stdin(on_eof) -> None:
    """Exit when the parent goes away.

    The studio keeps our stdin open for as long as it runs. If it crashes or is
    killed, the pipe closes, and an engine left behind would keep the port and the
    GPU memory for nothing.
    """
    def run() -> None:
        try:
            if sys.platform == "win32":
                _wait_for_pipe_close(sys.stdin.fileno())
            else:
                while sys.stdin.buffer.read(4096):
                    pass
        except (OSError, ValueError):
            pass
        on_eof()
    threading.Thread(target=run, name="parent-watch", daemon=True).start()


def _wait_for_pipe_close(fd: int) -> None:
    """Return once the pipe on `fd` is closed, without ever waiting inside a read.

    On Windows, a thread blocked reading a synchronous pipe also blocks some queries
    on that handle from other threads (GetFileInformationByHandle, for one). Loading
    numpy's C core, which `import torch` does, waited on one forever and froze the
    whole engine on its first status request. Polling never leaves a read pending.
    """
    import ctypes
    import msvcrt
    from ctypes import wintypes

    peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
    peek.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
                     ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    peek.restype = wintypes.BOOL
    handle = msvcrt.get_osfhandle(fd)
    available = wintypes.DWORD()
    # Fails once the studio's end is closed (or if stdin is not a pipe at all).
    while peek(handle, None, 0, None, ctypes.byref(available), None):
        if available.value:
            os.read(fd, available.value)  # data is waiting, so this returns at once
        else:
            time.sleep(PIPE_POLL_SECONDS)


def main(argv: list[str] | None = None, backend_factory: Callable | None = None,
         video_backend_factory: Callable | None = None, caption_backend_factory: Callable | None = None) -> int:
    """Run the engine. `backend_factory(settings_getter)`, `video_backend_factory(settings_getter)` and
    `caption_backend_factory(settings_getter)` replace the real models in tests; a test that passes
    only the first runs without video and without the captioning model."""
    ap = argparse.ArgumentParser(prog="python -m image_engine", description="Local AI Studio image engine")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1", help="loopback address to bind (default 127.0.0.1)")
    ap.add_argument("--data-dir", type=Path, default=None, help="settings and the gallery")
    ap.add_argument("--watch-stdin", action="store_true", help="exit when stdin closes (set by the studio)")
    ap.add_argument("--no-token", action="store_true", help="run without a token (local experiments only)")
    args = ap.parse_args(argv)

    if args.host not in LOOPBACK:
        ap.error("the engine only listens on loopback")
    token = os.environ.pop(TOKEN_ENV, None)  # removed so child processes never inherit it
    if token is None and not args.no_token:
        ap.error(f"set {TOKEN_ENV} (the studio does this), or pass --no-token")
    if token is not None and len(token) < MIN_TOKEN_LENGTH:
        ap.error(f"{TOKEN_ENV} must be at least {MIN_TOKEN_LENGTH} characters")

    data_dir = (args.data_dir or default_data_dir()).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    settings = SettingsStore(data_dir)

    if backend_factory is None:
        from .backends.image import ImageBackend
        from .backends.video import VideoBackend
        from .backends.captioner import CaptionBackend
        backend_factory, video_backend_factory = ImageBackend, VideoBackend
        caption_backend_factory = CaptionBackend

    events = EventHub()
    gallery = Gallery(data_dir / "outputs")
    backend = backend_factory(settings.get)
    backends = {"image": backend}
    if video_backend_factory is not None:
        backends["video"] = video_backend_factory(settings.get)
    if caption_backend_factory is not None:
        backends["caption"] = caption_backend_factory(settings.get)
    loras = LoraLibrary(data_dir / "loras")
    jobs = JobQueue(backends, gallery, events, app=app_label(),
                    ffmpeg=lambda: video_io.find_ffmpeg(settings.get().ffmpeg_path), loras=loras)
    training = TrainingService(data_dir / "training", loras, gallery)
    register_training(jobs, training)
    app = App(settings=settings, backend=backend, gallery=gallery, events=events, jobs=jobs,
              port=args.port, token=token, training=training, loras=loras)

    try:
        httpd = serve(app, args.host)
    except OSError as exc:
        print(f"Cannot listen on {args.host}:{args.port}: {exc}", file=sys.stderr, flush=True)
        return 1

    jobs.start()
    if args.watch_stdin:
        watch_stdin(lambda: threading.Thread(target=httpd.shutdown, daemon=True).start())
    if settings.get().load_on_start:
        app.load_model_async()
    print(f"Image engine {__version__} listening on http://{args.host}:{app.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stopping.set()
        httpd.server_close()
        jobs.stop()
        gallery.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
