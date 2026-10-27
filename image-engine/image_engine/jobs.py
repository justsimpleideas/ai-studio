"""Job queue and the single GPU worker.

The GPU runs one job at a time. Jobs queue behind it, can be cancelled while
queued or between sampling steps, and report every stage change through the
event hub so the UI can show live progress.

The engine can hold two models, one for images and one for video, but only one
is ever loaded: they do not fit on a consumer GPU together, so loading one first
unloads the other (the same rule the studio applies between chat and images).
"""

from __future__ import annotations

import secrets
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image, ImageOps

from .backends import Backend, BackendError, Cancelled
from .events import EventHub
from .gallery import Gallery
from .params import GenerationRequest, VideoRequest

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
FINISHED = (DONE, FAILED, CANCELLED)
HISTORY_LIMIT = 200
COND_CACHE_SIZE = 4
KINDS = ("image", "video")


@dataclass
class Job:
    id: str
    request: GenerationRequest | VideoRequest
    created: float = field(default_factory=time.time)
    status: str = QUEUED
    stage: str = "queued"
    index: int = 0          # image or video within the batch currently being made
    step: int = 0
    steps: int = 0
    images: list[str] = field(default_factory=list)
    error: str | None = None
    started: float | None = None
    finished: float | None = None
    cancel_requested: bool = False

    @property
    def kind(self) -> str:
        return self.request.kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "request": self.request.to_dict(), "created": self.created,
            "status": self.status, "stage": self.stage, "index": self.index,
            "count": self.request.count, "step": self.step, "steps": self.steps,
            "images": list(self.images), "error": self.error,
            "started": self.started, "finished": self.finished,
        }


class JobQueue:
    def __init__(self, backends: Backend | dict[str, Backend], gallery: Gallery, events: EventHub,
                 app: str | None = None, ffmpeg: Callable[[], str | None] = lambda: None) -> None:
        if isinstance(backends, Backend):
            backends = {backends.kind: backends}
        self.backends = dict(backends)
        self.gallery = gallery
        self.events = events
        self.app = app  # recorded with every image, e.g. "Local AI Studio 0.1.0"
        self.ffmpeg = ffmpeg  # where ffmpeg is right now; asked again for every video
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._pending: list[str] = []
        self._cv = threading.Condition()
        self._stopping = False
        self._cond_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._backend_lock = threading.Lock()  # load/unload vs. generation
        # unloaded | loading | ready | error, per model
        self.models = {kind: {"state": "unloaded", "message": ""} for kind in self.backends}
        self._worker = threading.Thread(target=self._run, name="gpu-worker", daemon=True)

    @property
    def backend(self) -> Backend | None:
        """The image backend (what the engine had before video existed)."""
        return self.backends.get("image")

    @property
    def model_state(self) -> str:
        return self.models.get("image", {"state": "unloaded"})["state"]

    @property
    def model_message(self) -> str:
        return self.models.get("image", {"message": ""})["message"]

    # ------------------------------------------------------------ control

    def start(self) -> None:
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._stopping = True
            for job in self._jobs.values():
                if job.status in (QUEUED, RUNNING):
                    job.cancel_requested = True
            self._cv.notify_all()
        self._worker.join(timeout)

    def submit(self, request: GenerationRequest | VideoRequest) -> Job:
        job = Job(id=secrets.token_hex(8), request=request)
        with self._cv:
            self._jobs[job.id] = job
            self._pending.append(job.id)
            self._trim_history()
            self._cv.notify_all()
        self._publish(job)
        return job

    def cancel(self, job_id: str) -> Job | None:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None or job.status in FINISHED:
                return job
            job.cancel_requested = True
            if job.status == QUEUED:
                self._pending.remove(job.id)
                self._finish(job, CANCELLED)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._cv:
            return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._cv:
            return [j.to_dict() for j in reversed(self._jobs.values())]

    def queue_summary(self) -> dict[str, int]:
        with self._cv:
            running = sum(1 for j in self._jobs.values() if j.status == RUNNING)
            return {"queued": len(self._pending), "running": running}

    # ------------------------------------------------------------ model

    def load_model(self, kind: str = "image") -> None:
        """Load a model now (otherwise it loads on its first job). Blocking."""
        with self._backend_lock:
            self._ensure_loaded(kind)

    def unload_model(self, kind: str | None = None) -> None:
        """Unload one model, or with no kind every model, which frees the GPU."""
        with self._backend_lock:
            for k in ([kind] if kind else list(self.backends)):
                self._unload(k)

    def _unload(self, kind: str) -> None:
        backend = self.backends.get(kind)
        if backend is None:
            return
        backend.unload()
        # Cached prompt encodings are GPU tensors of this model; they go with it.
        for key in [k for k in self._cond_cache if k[0] == kind]:
            del self._cond_cache[key]
        self._set_model_state(kind, "unloaded", "")

    def _ensure_loaded(self, kind: str) -> None:
        backend = self.backends.get(kind)
        if backend is None:
            raise BackendError(f"This engine has no {kind} model.")
        if backend.loaded:
            return
        for other, b in self.backends.items():
            if other != kind and b.loaded:
                self._unload(other)  # one model on the GPU at a time
        self._set_model_state(kind, "loading", "Loading model")
        try:
            backend.load(lambda msg: self._set_model_state(kind, "loading", msg))
        except Exception as exc:
            self._set_model_state(kind, "error", str(exc) or exc.__class__.__name__)
            raise
        self._set_model_state(kind, "ready", "")

    def _set_model_state(self, kind: str, state: str, message: str) -> None:
        self.models[kind] = {"state": state, "message": message}
        self.events.publish("model", {"kind": kind, "state": state, "message": message})

    # ------------------------------------------------------------ worker

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._stopping:
                    self._cv.wait()
                if self._stopping:
                    return
                job = self._jobs[self._pending.pop(0)]
                job.status, job.started = RUNNING, time.time()
            self._publish(job)
            try:
                with self._backend_lock:
                    self._execute(job)
                self._finish(job, DONE)
            except Cancelled:
                self._finish(job, CANCELLED)
            except BackendError as exc:
                job.error = str(exc)
                self._finish(job, FAILED)
            except Exception as exc:  # keep the worker alive whatever the backend does
                job.error = f"{exc.__class__.__name__}: {exc}"
                traceback.print_exc()
                self._finish(job, FAILED)

    def _execute(self, job: Job) -> None:
        req = job.request
        self._stage(job, "loading model")
        self._ensure_loaded(job.kind)
        backend = self.backends[job.kind]
        self._check_cancel(job)

        start_frame = self._start_frame(req) if isinstance(req, VideoRequest) else None

        key = (job.kind, req.prompt, req.negative_prompt, backend.uses_cfg(req.guidance))
        cond = self._cond_cache.get(key)
        if cond is None:
            self._stage(job, "encoding prompt")
            cond = backend.encode(req.prompt, req.negative_prompt, req.guidance)
            self._cond_cache[key] = cond
            while len(self._cond_cache) > COND_CACHE_SIZE:
                self._cond_cache.popitem(last=False)
        else:
            self._cond_cache.move_to_end(key)

        for index in range(req.count):
            self._check_cancel(job)
            job.index, job.step, job.steps = index, 0, req.steps
            self._stage(job, "sampling")
            seed = req.seed_for(index)
            started = time.perf_counter()

            def on_step(done: int, total: int) -> None:
                job.step, job.steps = done, total
                self._publish(job)
                self._check_cancel(job)

            latents = backend.sample(cond, req, seed, on_step, start_frame=start_frame)
            self._stage(job, "decoding")
            output = backend.decode(latents)
            duration_ms = int((time.perf_counter() - started) * 1000)
            provenance = {**({"app": self.app} if self.app else {}), **backend.provenance()}
            common = {"job_id": job.id, "model": backend.model_id, "request": req.to_dict(), "seed": seed,
                      "duration_ms": duration_ms, "stats": backend.stats(), "provenance": provenance}
            if isinstance(req, VideoRequest):
                self._stage(job, "encoding video")
                record = self.gallery.save_video(output, ffmpeg=self.ffmpeg(), **common)
            else:
                record = self.gallery.save(output, **common)
            job.images.append(record["id"])
            self.events.publish("image", {"job_id": job.id, "image": record})

    def _start_frame(self, req: VideoRequest) -> Image.Image | None:
        if not req.start_image:
            return None
        picture = self.gallery.load_input(req.start_image)
        if picture is None:
            raise BackendError("The start picture is no longer in the engine's inputs. Choose it again.")
        # Cropped to the video's shape rather than squashed into it.
        return ImageOps.fit(picture, (req.width, req.height), Image.Resampling.LANCZOS)

    # ------------------------------------------------------------ helpers

    def _check_cancel(self, job: Job) -> None:
        if job.cancel_requested:
            raise Cancelled()

    def _stage(self, job: Job, stage: str) -> None:
        job.stage = stage
        self._publish(job)

    def _finish(self, job: Job, status: str) -> None:
        job.status, job.stage, job.finished = status, status, time.time()
        self._publish(job)

    def _publish(self, job: Job) -> None:
        self.events.publish("job", {"job": job.to_dict()})

    def _trim_history(self) -> None:
        while len(self._jobs) > HISTORY_LIMIT:
            oldest_id, oldest = next(iter(self._jobs.items()))
            if oldest.status not in FINISHED:
                break
            del self._jobs[oldest_id]
