"""Training projects: the engine side of the Train tab (docs/training.md).

A project is one photo folder being turned into one subject LoRA. Everything it
produces lives under <data>/training/<project id>/ — the photos themselves are
only ever read:

    project.json   name, photo folder, subject definition (subject.py)
    dataset.json   what the import found (dataset.py)
    captions.json  the caption records and review decisions (captions.py)
    latents/       the VAE latent cache (cache.py)
    embeddings/    the caption-embedding cache (embeddings.py)
    runs/<id>/     one training run: run.json, checkpoints/, state/, compare.json

The GPU work (scanning shares the worker too, so it never races a run) is done
by tasks the job queue runs one at a time: scan, caption, train and compare.

Until the captioning model has looked at a photo its caption is "facts only",
built from the folder facts. The caption task asks the model about every photo
(captions.py: whether it shows the subject, then the subject's own fields) and
can be stopped and run again: it carries on with the photos not yet described.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from ..backends import BackendError
from ..params import DEFAULTS, GenerationRequest
from . import cache, embeddings
from .captions import (SHOWN, SIZE, UNKNOWN, CaptionRecord, CaptionStore, all_specs, audit_estimate, caption_text,
                       choose_audit, flag_outliers, judge, prompts, review, review_queue)
from .dataset import Dataset, DatasetError, scan
from .subject import GENERAL, PRESETS, Subject, SubjectError, from_project, parse_subject
from .trainer import TrainConfig, Trainer, TrainError, Sample

PROJECT_ID = re.compile(r"^[0-9a-f]{16}$")
RUN_ID = re.compile(r"^[0-9a-f]{12}$")
NAME_MAX = 60
CAPTIONER_FACTS = "facts-only@1"
# Bump when the questions change: photos described with older ones are described again.
PROMPTS_VERSION = "prompts@2"
SAVE_EVERY = 10
MAX_COMPARE_PROMPTS = 4
MAX_COMPARE_CHECKPOINTS = 4
TRAIN_LIMITS = {"rank": (1, 128), "steps": (1, 20000), "checkpoint_every": (1, 20000), "seed": (0, 2**31 - 1)}
TARGET_SIDES = (512, 768, 1024)


class TrainingError(Exception):
    """The message is safe to show to the user."""


def subject_presets() -> list[dict[str, Any]]:
    """The starting points offered when a project's subject is defined."""
    return [s.to_json() for s in PRESETS.values()]


def _parse_subject(data: object) -> Subject:
    try:
        return parse_subject(data)
    except SubjectError as exc:
        raise TrainingError(str(exc)) from exc


class TrainingService:
    def __init__(self, root: Path, loras: Any, gallery: Any) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.loras = loras
        self.gallery = gallery
        self._lock = threading.RLock()

    # ------------------------------------------------------------ projects

    def _dir(self, project_id: str) -> Path:
        if not PROJECT_ID.match(project_id or ""):
            raise TrainingError("No such training project.")
        folder = self.root / project_id
        if not (folder / "project.json").is_file():
            raise TrainingError("No such training project.")
        return folder

    def _meta(self, project_id: str) -> dict[str, Any]:
        return json.loads((self._dir(project_id) / "project.json").read_text(encoding="utf-8"))

    def _subject(self, project_id: str) -> Subject:
        try:
            return from_project(self._meta(project_id))
        except SubjectError as exc:
            raise TrainingError(f"This project's subject definition is broken: {exc}") from exc

    def create(self, name: object, photo_root: object, subject: object = None) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > NAME_MAX:
            raise TrainingError(f"A project needs a name of at most {NAME_MAX} characters.")
        if not isinstance(photo_root, str) or not Path(photo_root).is_absolute() or not Path(photo_root).is_dir():
            raise TrainingError("Choose an existing photo folder.")
        definition = GENERAL if subject is None else _parse_subject(subject)
        project_id = secrets.token_hex(8)
        folder = self.root / project_id
        folder.mkdir()
        meta = {"id": project_id, "name": " ".join(name.split()), "photo_root": str(Path(photo_root).resolve()),
                "subject": definition.to_json(), "created": time.time(), "captioner": CAPTIONER_FACTS}
        _write_json(folder / "project.json", meta)
        return self.summary(project_id)

    def set_subject(self, project_id: str, subject: object) -> dict[str, Any]:
        """Replace the subject definition. Facts come from the folder levels, so scan again after."""
        definition = _parse_subject(subject)
        with self._lock:
            meta = self._meta(project_id)
            meta.pop("schema", None)
            meta["subject"] = definition.to_json()
            _write_json(self._dir(project_id) / "project.json", meta)
        return self.summary(project_id)

    def list(self) -> list[dict[str, Any]]:
        out = []
        for folder in sorted(self.root.iterdir()) if self.root.is_dir() else []:
            if (folder / "project.json").is_file() and PROJECT_ID.match(folder.name):
                out.append(self.summary(folder.name))
        return sorted(out, key=lambda p: -p["created"])

    def delete(self, project_id: str) -> None:
        folder = self._dir(project_id)
        with self._lock:
            shutil.rmtree(folder)  # only the engine's own files: the photos live elsewhere

    def summary(self, project_id: str) -> dict[str, Any]:
        folder = self._dir(project_id)
        meta = self._meta(project_id)
        dataset = self._dataset(project_id)
        store = self._captions(project_id)
        caps = None
        if store is not None:
            records = store.records
            caps = {"total": len(records), "flagged": sum(1 for r in records if r.flags),
                    "to_review": len(review_queue(records)),
                    "excluded": sum(1 for r in records if r.excluded),
                    "off_subject": sum(1 for r in records if r.relevant is False),
                    "described": sum(1 for r in records if r.by),
                    "edited": sum(1 for r in records if r.status == "edited"),
                    "audit": audit_estimate(records)}
        meta.pop("schema", None)  # projects made before subjects existed
        return {**meta, "subject": self._subject(project_id).to_json(),
                "dataset": dataset.summary() if dataset else None,
                "warnings": dataset.warnings if dataset else [], "captions": caps,
                "runs": [self._run_summary(folder, d.name) for d in sorted((folder / "runs").iterdir())
                         if RUN_ID.match(d.name)] if (folder / "runs").is_dir() else []}

    def _dataset(self, project_id: str) -> Dataset | None:
        path = self._dir(project_id) / "dataset.json"
        return Dataset.load(path) if path.is_file() else None

    def _captions(self, project_id: str) -> CaptionStore | None:
        path = self._dir(project_id) / "captions.json"
        return CaptionStore.load(path) if path.is_file() else None

    # ------------------------------------------------------------ captions

    def captions(self, project_id: str, offset: int = 0, limit: int = 60, view: str = "all") -> dict[str, Any]:
        dataset, store = self._dataset(project_id), self._captions(project_id)
        if dataset is None or store is None:
            return {"items": [], "total": 0}
        items = {i.id: i for i in dataset.items}
        if view == "review":
            records = review_queue(store.records)
        elif view == "excluded":
            records = [r for r in store.records if r.excluded]
        elif view == "all":
            records = store.records
        else:
            raise TrainingError("view must be all, review or excluded")
        page = records[offset:offset + limit]
        subject = self._subject(project_id)
        return {"total": len(records), "items": [self._caption_view(items[r.item_id], r, subject) for r in page
                                                 if r.item_id in items]}

    @staticmethod
    def _caption_view(item: Any, record: CaptionRecord, subject: Subject) -> dict[str, Any]:
        return {"item": item.id, "path": item.path, "facts": item.facts, "holdout": item.holdout,
                "width": item.width, "height": item.height, "caption": caption_text(subject, item.facts, record),
                "description": record.current(subject.fields), "flags": record.flags, "status": record.status,
                "audit": record.audit, "excluded": record.excluded,
                "relevance": {k: record.effective().get(k, UNKNOWN) for k in (SHOWN, SIZE)} if record.by else None}

    def review(self, project_id: str, item_id: str, action: str, edits: dict | None) -> dict[str, Any]:
        with self._lock:
            store = self._captions(project_id)
            dataset = self._dataset(project_id)
            record = store.by_item().get(item_id) if store else None
            if record is None or dataset is None:
                raise TrainingError("No such photo in this project.")
            subject = self._subject(project_id)
            try:
                review(record, action, edits, subject.fields)
            except ValueError as exc:
                raise TrainingError(str(exc)) from exc
            store.save(self._dir(project_id) / "captions.json")
            item = next(i for i in dataset.items if i.id == item_id)
            return self._caption_view(item, record, subject)

    def photo_path(self, project_id: str, item_id: str) -> Path | None:
        """A photo's file, for the review grid. Only photos the scan indexed are reachable."""
        dataset = self._dataset(project_id)
        item = next((i for i in dataset.items if i.id == item_id), None) if dataset else None
        return Path(dataset.root) / item.path if item else None

    # ------------------------------------------------------------ runs

    def _run_dir(self, project_id: str, run_id: str) -> Path:
        if not RUN_ID.match(run_id or ""):
            raise TrainingError("No such training run.")
        folder = self._dir(project_id) / "runs" / run_id
        if not (folder / "run.json").is_file():
            raise TrainingError("No such training run.")
        return folder

    @staticmethod
    def _run_summary(project_dir: Path, run_id: str) -> dict[str, Any]:
        folder = project_dir / "runs" / run_id
        try:
            info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"id": run_id, "status": "broken"}
        ckpts = sorted(int(p.stem.split("-")[1]) for p in (folder / "checkpoints").glob("step-*.safetensors")) \
            if (folder / "checkpoints").is_dir() else []
        compare = folder / "compare.json"
        return {**info, "checkpoints": ckpts,
                "compare": json.loads(compare.read_text(encoding="utf-8")) if compare.is_file() else None}

    def run_log(self, project_id: str, run_id: str) -> list[dict]:
        state = self._run_dir(project_id, run_id) / "state" / "trainer.json"
        return json.loads(state.read_text(encoding="utf-8"))["log"] if state.is_file() else []

    def use_checkpoint(self, project_id: str, run_id: str, step: object, name: object) -> dict[str, Any]:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TrainingError("step must be a checkpoint step")
        path = self._run_dir(project_id, run_id) / "checkpoints" / f"step-{step:06d}.safetensors"
        if not path.is_file():
            raise TrainingError("No such checkpoint.")
        meta = self._meta(project_id)
        run = self._run_summary(self._dir(project_id), run_id)
        origin = {"project": project_id, "project_name": meta["name"], "run": run_id, "step": step}
        if run.get("photo_licences"):
            origin["photo_licences"] = run["photo_licences"]
        return self.loras.add(path, name, origin)

    # ------------------------------------------------------------ parameters

    @staticmethod
    def train_params(data: object) -> dict[str, Any]:
        """The training settings a request may set, checked; anything unset takes TrainConfig's default."""
        if not isinstance(data, dict):
            raise TrainingError("expected a JSON object of training settings")
        allowed = {"rank", "learning_rate", "steps", "checkpoint_every", "caption_dropout", "seed",
                   "target_side", "run"}
        unknown = set(data) - allowed
        if unknown:
            raise TrainingError(f"unknown training setting: {sorted(unknown)[0]}")
        out: dict[str, Any] = {}
        for key, (lo, hi) in TRAIN_LIMITS.items():
            if key in data:
                v = data[key]
                if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                    raise TrainingError(f"{key} must be an integer between {lo} and {hi}")
                out[key] = v
        for key, lo, hi in (("learning_rate", 1e-6, 1e-2), ("caption_dropout", 0.0, 0.5)):
            if key in data:
                v = data[key]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                    raise TrainingError(f"{key} must be between {lo:g} and {hi:g}")
                out[key] = float(v)
        if "target_side" in data:
            if data["target_side"] not in TARGET_SIDES:
                raise TrainingError(f"target_side must be one of {', '.join(map(str, TARGET_SIDES))}")
            out["target_side"] = data["target_side"]
        if "run" in data:
            if not isinstance(data["run"], str) or not RUN_ID.match(data["run"]):
                raise TrainingError("run must be a run id")
            out["run"] = data["run"]
        return out

    def compare_params(self, project_id: str, data: object) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise TrainingError("expected a JSON object")
        run_id = data.get("run")
        self._run_dir(project_id, run_id if isinstance(run_id, str) else "")
        steps = data.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_COMPARE_CHECKPOINTS or \
                any(isinstance(s, bool) or not isinstance(s, int) for s in steps):
            raise TrainingError(f"steps must list 1 to {MAX_COMPARE_CHECKPOINTS} checkpoint steps")
        prompts = data.get("prompts", [])
        if not isinstance(prompts, list) or len(prompts) > MAX_COMPARE_PROMPTS or \
                any(not isinstance(p, str) or not p.strip() or len(p) > 1000 for p in prompts):
            raise TrainingError(f"prompts must list up to {MAX_COMPARE_PROMPTS} prompts")
        size = data.get("size", 768)
        if size not in (512, 768, 1024):
            raise TrainingError("size must be 512, 768 or 1024")
        seed = data.get("seed", 1)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise TrainingError("seed must be a non-negative integer")
        return {"run": run_id, "steps": sorted(set(steps)), "prompts": [p.strip() for p in prompts],
                "size": size, "seed": seed}

    # ------------------------------------------------------------ tasks (run by the GPU worker)

    def task_scan(self, job: Any, ctx: Any) -> dict[str, Any]:
        project_id = job.request.project
        meta = self._meta(project_id)
        folder = self._dir(project_id)
        subject = self._subject(project_id)
        previous = self._dataset(project_id)
        try:
            dataset = scan(Path(meta["photo_root"]), subject, previous=previous,
                           on_progress=lambda d, t: (ctx.check_cancel(), ctx.progress(d, t, "scanning photos")))
        except DatasetError as exc:
            raise BackendError(str(exc)) from exc
        with self._lock:
            dataset.save(folder / "dataset.json")
            old = self._captions(project_id)
            kept = old.by_item() if old else {}
            records = []
            for item in dataset.items:
                if not item.usable:
                    continue
                records.append(kept.get(item.id)
                               or CaptionRecord(item.id, {s.name: UNKNOWN for s in subject.fields}, []))
            store = CaptionStore(schema=subject.preset, captioner=meta["captioner"], records=records)
            store.save(folder / "captions.json")
        return {"dataset": dataset.summary(), "warnings": len(dataset.warnings)}

    def task_caption(self, job: Any, ctx: Any) -> dict[str, Any]:
        """Ask the captioning model about every photo not yet described with these questions.
        Stopping keeps what is done; running again carries on."""
        project_id = job.request.project
        folder = self._dir(project_id)
        dataset, store = self._dataset(project_id), self._captions(project_id)
        if dataset is None or store is None:
            raise BackendError("Scan the photo folder first.")
        subject = self._subject(project_id)
        backend = ctx.backend("caption")
        fingerprint = json.dumps({"noun": subject.noun, "fields": subject.to_json()["fields"]}, sort_keys=True)
        captioner = f"{backend.model_id}|{PROMPTS_VERSION}|{_digest(fingerprint)}"
        specs, (first, second) = all_specs(subject), prompts(subject)
        items = {i.id: i for i in dataset.items if i.usable}
        todo = [r for r in store.records if r.item_id in items and r.by != captioner]
        root = Path(dataset.root)
        done = 0
        for record in todo:
            if ctx.stop_requested():
                break
            item = items[record.item_id]
            try:
                with Image.open(root / item.path) as img:
                    photo = ImageOps.exif_transpose(img).convert("RGB")
                replies = (backend.describe(photo, first), backend.describe(photo, second))
                fresh = judge(item.id, replies, specs, item.facts)
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
                fresh = CaptionRecord(item.id, {s.name: UNKNOWN for s in specs}, ["unparseable"])
            # The user's decisions and corrections stay; the model's answers are replaced.
            record.description, record.flags, record.relevant, record.by = (fresh.description, fresh.flags,
                                                                            fresh.relevant, captioner)
            done += 1
            ctx.progress(done, len(todo), "describing photos")
            if done % SAVE_EVERY == 0:
                with self._lock:
                    store.save(folder / "captions.json")
        with self._lock:
            flag_outliers(store.records, dataset, subject)
            if not any(r.audit for r in store.records):
                choose_audit(store.records)
            store.captioner = captioner
            store.save(folder / "captions.json")
            meta = self._meta(project_id)
            meta["captioner"] = captioner
            _write_json(folder / "project.json", meta)
        finished = done == len(todo)
        return {"status": "finished" if finished else "stopped", "described": done, "left": len(todo) - done,
                "off_subject": sum(1 for r in store.records if r.relevant is False),
                "flagged": sum(1 for r in store.records if r.flags)}

    def task_train(self, job: Any, ctx: Any) -> dict[str, Any]:
        project_id, params = job.request.project, dict(job.request.params)
        folder = self._dir(project_id)
        dataset, store = self._dataset(project_id), self._captions(project_id)
        if dataset is None or store is None:
            raise BackendError("Scan the photo folder first.")
        records = store.by_item()
        subject = self._subject(project_id)
        chosen = [i for i in dataset.items if i.usable and not i.holdout
                  and i.id in records and not records[i.id].excluded]
        if not chosen:
            raise BackendError("There are no photos to train on: every photo is excluded, held out or unusable.")
        run_id = params.pop("run", None) or secrets.token_hex(6)
        run_dir = folder / "runs" / run_id
        target_side = params.pop("target_side", 768)
        profile = getattr(ctx.unloaded_backend("image"), "profile", None)
        if profile is not None and not profile.training:  # say so before loading a model for nothing
            raise BackendError("LoRA training needs the 6B image model. Choose its folder in Settings.")
        backend = ctx.backend("image")
        if not hasattr(backend, "training_session"):
            raise BackendError("This image model cannot be trained.")
        ctx.forget_prompt_cache()
        with backend.training_session() as parts:
            subset = Dataset(root=dataset.root, schema=dataset.schema, min_side=dataset.min_side, items=chosen)
            texts = {i.id: caption_text(subject, i.facts, records[i.id]) for i in chosen}
            try:  # stopping during caching keeps what is cached; nothing else to save yet
                encoder, vae_key = parts.vae_encoder()
                entries = cache.build(subset, folder / "latents", encoder, vae_key, target_side=target_side,
                                      on_progress=lambda d, t: ctx.progress(d, t, "caching photos"),
                                      should_stop=ctx.stop_requested)
                with parts.text_encoder() as (text_encoder, text_key):
                    files = embeddings.build(texts.values(), folder / "embeddings", text_encoder, text_key,
                                             on_progress=lambda d, t: ctx.progress(d, t, "caching captions"),
                                             should_stop=ctx.stop_requested)
            except cache.Stopped:
                return {"status": "stopped", "run": run_id, "step": 0}
            except cache.CacheError as exc:
                raise BackendError(str(exc)) from exc
            samples = [Sample(folder / "latents" / e.file, folder / "embeddings" / files[texts[e.item_id]])
                       for e in entries]
            try:
                config = TrainConfig(shift=parts.shift, **params)
                trainer = Trainer(parts.transformer, samples, folder / "embeddings" / files[""], run_dir, config,
                                  device=parts.device, dtype=parts.dtype)
            except TrainError as exc:
                raise BackendError(str(exc)) from exc
            info = {"id": run_id, "created": time.time(), "status": "running", "photos": len(samples),
                    "target_side": target_side, "config": asdict(config), "step": 0, "steps": config.steps,
                    "photo_licences": photo_licences(Path(dataset.root), [i.path for i in chosen])}
            run_dir.mkdir(parents=True, exist_ok=True)
            previous = run_dir / "run.json"
            if previous.is_file():
                info["created"] = json.loads(previous.read_text(encoding="utf-8")).get("created", info["created"])
            _write_json(previous, info)
            try:
                result = trainer.run(
                    on_progress=lambda s, t, loss: ctx.progress(s, t, "training", loss=round(loss, 5)),
                    should_stop=ctx.stop_requested)
            except TrainError as exc:
                _write_json(previous, {**info, "status": "failed", "error": str(exc)})
                raise BackendError(str(exc)) from exc
        _write_json(previous, {**info, "status": result.status, "step": result.step,
                               "loss_avg": result.loss_avg, "seconds": round(result.seconds, 1),
                               "peak_vram_bytes": result.peak_vram_bytes})
        return {"status": result.status, "run": run_id, "step": result.step, "checkpoint": result.checkpoint,
                "loss_avg": result.loss_avg, "peak_vram_bytes": result.peak_vram_bytes}

    def task_compare(self, job: Any, ctx: Any) -> dict[str, Any]:
        project_id, params = job.request.project, job.request.params
        run_dir = self._run_dir(project_id, params["run"])
        meta = self._meta(project_id)
        prompts = params["prompts"] or self._default_prompts(project_id)
        if not prompts:
            raise BackendError("Give at least one prompt to compare with.")
        columns: list[tuple[str, Path | None]] = [("base", None)]
        for step in params["steps"]:
            path = run_dir / "checkpoints" / f"step-{step:06d}.safetensors"
            if not path.is_file():
                raise BackendError(f"There is no checkpoint at step {step}.")
            columns.append((str(step), path))
        backend = ctx.backend("image")
        folder_id = self._gallery_folder(meta["name"])
        size, seed = params["size"], params["seed"]
        total = len(columns) * len(prompts)
        grid: list[list[str]] = [[] for _ in prompts]
        done = 0
        try:
            for label, path in columns:
                backend.set_loras([] if path is None else [(f"ckpt{label}", path, 1.0)])
                for row, prompt in enumerate(prompts):
                    ctx.check_cancel()
                    request = GenerationRequest(prompt=prompt, negative_prompt="", width=size, height=size,
                                                steps=DEFAULTS["steps"], guidance=DEFAULTS["guidance"],
                                                seed=seed, count=1)
                    started = time.perf_counter()
                    cond = backend.encode(prompt, "", request.guidance)
                    latents = backend.sample(cond, request, seed, lambda d, t: ctx.check_cancel())
                    image = backend.decode(latents)
                    provenance = {**({"app": ctx.app} if ctx.app else {}), **backend.provenance(),
                                  "training_checkpoint": {"project": project_id, "run": params["run"],
                                                          "step": None if path is None else int(label)}}
                    record = ctx.gallery.save(image, job_id=job.id, model=backend.model_id,
                                              request=request.to_dict(), seed=seed,
                                              duration_ms=int((time.perf_counter() - started) * 1000),
                                              stats=backend.stats(), provenance=provenance)
                    if folder_id:
                        # The user may have hidden or locked the folder; the comparison still files there.
                        record = ctx.gallery.set_folder(record["id"], folder_id, everything=True) or record
                    ctx.publish_image(record)
                    grid[row].append(record["id"])
                    done += 1
                    ctx.progress(done, total, "comparing")
        finally:
            backend.set_loras([])
        result = {"columns": [c for c, _ in columns], "prompts": prompts, "images": grid, "size": size,
                  "seed": seed, "created": time.time()}
        _write_json(run_dir / "compare.json", result)
        return result

    def _default_prompts(self, project_id: str) -> list[str]:
        """Captions of held-out photos (never trained on), else of the first photos."""
        dataset, store = self._dataset(project_id), self._captions(project_id)
        if dataset is None or store is None:
            return []
        records = store.by_item()
        items = [i for i in dataset.items if i.id in records and not records[i.id].excluded]
        items.sort(key=lambda i: (not i.holdout, i.path))
        subject = self._subject(project_id)
        prompts = list(dict.fromkeys(caption_text(subject, i.facts, records[i.id]) for i in items))
        return prompts[:3]

    def _gallery_folder(self, project_name: str) -> str | None:
        name = f"Training: {project_name}"[:60]
        existing = next((f for f in self.gallery.list_folders(everything=True)
                         if f["name"].lower() == name.lower()), None)
        if existing:
            return existing["id"]
        try:
            return self.gallery.create_folder(name)["id"]
        except Exception:
            return None


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def photo_licences(root: Path, paths: list[str]) -> dict[str, int]:
    """How many of these photos came under each licence, when the folder was collected by
    the studio (its ATTRIBUTION.json says); empty for the user's own photos."""
    try:
        credits = json.loads((root / "ATTRIBUTION.json").read_text(encoding="utf-8"))
        by_file = {c["file"]: c["license"] for c in credits if isinstance(c, dict)}
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    counts: dict[str, int] = {}
    for path in paths:
        lic = by_file.get(path)
        if isinstance(lic, str):
            counts[lic] = counts.get(lic, 0) + 1
    return counts


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def register(jobs: Any, service: TrainingService) -> None:
    jobs.register_task("training.scan", service.task_scan)
    jobs.register_task("training.caption", service.task_caption)
    jobs.register_task("training.train", service.task_train)
    jobs.register_task("training.compare", service.task_compare)

