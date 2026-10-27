"""Download a model snapshot from Hugging Face and verify every byte of it.

Standard library only, so it can run before any dependency is installed.

What it guarantees once it exits with status 0:
  * every file comes from one pinned commit, not from a moving branch;
  * only data formats are present (safetensors, JSON, plain-text tokenizer
    files) - no pickle checkpoints and no Python files, so loading the model
    cannot execute code shipped by the repository;
  * each file matches the hash Hugging Face publishes for it (SHA-256 for
    LFS files, git blob SHA-1 for small files);
  * each safetensors header is well-formed: tensors are laid out back to back,
    stay inside the file, and use known dtypes.

Usage:
  python scripts/fetch_model.py --dest /path/to/models/image-model
  python scripts/fetch_model.py --dest /path/to/models/image-model --verify-only
  python scripts/fetch_model.py --model video --dest /path/to/models/video-model
  python scripts/fetch_model.py --model video-fast --dest /path/to/models/video-model-fast \
      --reuse-from /path/to/models/video-model

With --json, progress goes to stdout as one JSON object per line (the studio's
download button reads it) and the human-readable messages go to stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPO = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
# The video model (5B, text or image to video, Apache 2.0): 34.2 GB, of which the
# transformer is 20 GB because it is published in fp32.
VIDEO_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
VIDEO_REVISION = "b8fff7315c768468a5333511427288870b2e9635"
# A 3-step distillation of the same model (Apache 2.0): 24.2 GB, but only the transformer
# (10 GB, bf16) and two small configs differ; the rest can be reused with --reuse-from.
FAST_VIDEO_REPO = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
FAST_VIDEO_REVISION = "3e187042a324f6f5fb68fd22110a78725253de8f"
MODELS = {"image": (DEFAULT_REPO, DEFAULT_REVISION), "video": (VIDEO_REPO, VIDEO_REVISION),
          "video-fast": (FAST_VIDEO_REPO, FAST_VIDEO_REVISION)}

# Only the pipeline itself; the repos' assets/ and examples/ folders (sample images, a PDF) are not needed.
WANTED_DIRS = ("scheduler/", "text_encoder/", "tokenizer/", "transformer/", "vae/")
WANTED_ROOT_FILES = ("model_index.json",)
DATA_EXTENSIONS = (".json", ".txt", ".safetensors")
# A SentencePiece vocabulary is a protobuf data file. It is accepted only inside tokenizer/.
TOKENIZER_EXTENSIONS = (".model",)
# Formats that can run code on load. Their presence anywhere we download from is fatal.
CODE_EXTENSIONS = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".py", ".pyc", ".so", ".dll", ".exe")
SAFETENSORS_DTYPES = {
    "BOOL", "U8", "I8", "I16", "U16", "I32", "U32", "I64", "U64",
    "F8_E4M3", "F8_E5M2", "F16", "BF16", "F32", "F64",
}
DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}
MAX_HEADER_BYTES = 100 * 1024 * 1024
CHUNK = 4 * 1024 * 1024
USER_AGENT = "local-ai-studio-fetch/1.0"
# Keep retrying through outages (a dropped connection, a sleeping laptop) as long as this
# much time has not passed without a single byte of progress.
GIVE_UP_AFTER_SECONDS = 12 * 60 * 60
# How often --json reports the bytes of the file being downloaded.
JSON_PROGRESS_SECONDS = 1.0

# Set by --json: events on stdout, messages on stderr.
_json_events = False


def say(message: str) -> None:
    print(message, file=sys.stderr if _json_events else sys.stdout, flush=True)


def emit(event: str, **fields: object) -> None:
    if _json_events:
        print(json.dumps({"event": event, **fields}), flush=True)


class VerificationError(Exception):
    pass


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None  # LFS files
    git_sha1: str       # every file

    @property
    def is_lfs(self) -> bool:
        return self.sha256 is not None


# ---------------------------------------------------------------- listing

def http_get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def list_remote(repo: str, revision: str) -> list[RemoteFile]:
    entries = http_get_json(f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true")
    if not isinstance(entries, list):
        raise VerificationError(f"unexpected tree listing: {entries!r:.200}")
    files = []
    for e in entries:
        if e.get("type") != "file":
            continue
        lfs = e.get("lfs")
        files.append(RemoteFile(
            path=e["path"],
            size=int(lfs["size"] if lfs else e["size"]),
            sha256=lfs["oid"] if lfs else None,
            git_sha1=e["oid"],
        ))
    return files


def select_files(files: list[RemoteFile]) -> list[RemoteFile]:
    """Pick the files the pipeline needs, refusing anything that could execute code."""
    chosen = []
    for f in files:
        wanted = f.path in WANTED_ROOT_FILES or f.path.startswith(WANTED_DIRS)
        if not wanted:
            continue
        lower = f.path.lower()
        if lower.endswith(CODE_EXTENSIONS):
            raise VerificationError(f"refusing executable/pickle file in model folder: {f.path}")
        tokenizer_data = f.path.startswith("tokenizer/") and lower.endswith(TOKENIZER_EXTENSIONS)
        if not (lower.endswith(DATA_EXTENSIONS) or tokenizer_data):
            raise VerificationError(f"unexpected file type in model folder: {f.path}")
        if lower.endswith(".safetensors") and not f.is_lfs:
            raise VerificationError(f"weights file is not stored in LFS, no SHA-256 published: {f.path}")
        chosen.append(f)
    if not any(f.path == "model_index.json" for f in chosen):
        raise VerificationError("model_index.json missing - not a diffusers pipeline repo")
    return chosen


# ---------------------------------------------------------------- hashing

def file_digests(path: Path) -> tuple[str, str]:
    """Return (sha256, git blob sha1) of a local file in one pass."""
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git = hashlib.sha1(b"blob %d\0" % size)
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            sha256.update(chunk)
            git.update(chunk)
    return sha256.hexdigest(), git.hexdigest()


def verify_hash(local: Path, remote: RemoteFile) -> None:
    actual_size = local.stat().st_size
    if actual_size != remote.size:
        raise VerificationError(f"{remote.path}: size {actual_size} != expected {remote.size}")
    sha256, git_sha1 = file_digests(local)
    if remote.is_lfs:
        if sha256 != remote.sha256:
            raise VerificationError(f"{remote.path}: SHA-256 mismatch ({sha256} != {remote.sha256})")
    elif git_sha1 != remote.git_sha1:
        raise VerificationError(f"{remote.path}: git SHA-1 mismatch ({git_sha1} != {remote.git_sha1})")


# ---------------------------------------------------------------- safetensors

def check_safetensors(path: Path) -> dict[str, int]:
    """Validate the header layout. Returns a {dtype: tensor count} summary."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise VerificationError(f"{path.name}: truncated header length")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len <= 0 or header_len > MAX_HEADER_BYTES or 8 + header_len > size:
            raise VerificationError(f"{path.name}: implausible header length {header_len}")
        try:
            header = json.loads(fh.read(header_len))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VerificationError(f"{path.name}: header is not valid JSON ({exc})") from exc
    if not isinstance(header, dict):
        raise VerificationError(f"{path.name}: header is not an object")

    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values()):
        raise VerificationError(f"{path.name}: __metadata__ must map strings to strings")

    data_len = size - 8 - header_len
    spans = []
    summary: dict[str, int] = {}
    for name, info in header.items():
        dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        if dtype not in SAFETENSORS_DTYPES:
            raise VerificationError(f"{path.name}: tensor {name} has unknown dtype {dtype!r}")
        if not (isinstance(shape, list) and all(isinstance(d, int) and d >= 0 for d in shape)):
            raise VerificationError(f"{path.name}: tensor {name} has invalid shape {shape!r}")
        if not (isinstance(offsets, list) and len(offsets) == 2):
            raise VerificationError(f"{path.name}: tensor {name} has invalid offsets {offsets!r}")
        start, end = offsets
        count = 1
        for d in shape:
            count *= d
        if end - start != count * DTYPE_SIZES[dtype]:
            raise VerificationError(f"{path.name}: tensor {name} byte length does not match its shape")
        spans.append((start, end, name))
        summary[dtype] = summary.get(dtype, 0) + 1

    # Tensors must tile the data section exactly: no gaps, no overlaps, nothing hidden after.
    cursor = 0
    for start, end, name in sorted(spans):
        if start != cursor:
            raise VerificationError(f"{path.name}: gap or overlap before tensor {name}")
        cursor = end
    if cursor != data_len:
        raise VerificationError(f"{path.name}: {data_len - cursor} unaccounted bytes after the last tensor")
    return summary


# ---------------------------------------------------------------- download

def download(repo: str, revision: str, remote: RemoteFile, dest: Path) -> None:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{remote.path}"
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    last_progress = time.monotonic()
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have > remote.size:
            part.unlink()
            have = 0
        if have == remote.size:
            break
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:
                    have = 0  # server ignored the range; start over
                mode = "ab" if have else "wb"
                done, last_report, started = have, time.monotonic(), time.monotonic()
                emit("progress", path=remote.path, done=done, size=remote.size)
                last_event = time.monotonic()
                with part.open(mode) as out:
                    while chunk := resp.read(CHUNK):
                        out.write(chunk)
                        done += len(chunk)
                        last_progress = time.monotonic()
                        now = time.monotonic()
                        if now - last_event >= JSON_PROGRESS_SECONDS:
                            emit("progress", path=remote.path, done=done, size=remote.size)
                            last_event = now
                        if now - last_report > 15:
                            rate = (done - have) / max(now - started, 1e-6) / 1e6
                            say(f"    {remote.path}: {done / 1e9:.2f}/{remote.size / 1e9:.2f} GB ({rate:.1f} MB/s)")
                            last_report = now
            attempts = 0
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            attempts += 1
            if time.monotonic() - last_progress > GIVE_UP_AFTER_SECONDS:
                raise
            wait = min(60, 2 ** min(attempts, 6))
            say(f"    {remote.path}: {exc} - retrying in {wait}s")
            emit("retry", path=remote.path, message=str(exc), wait=wait)
            time.sleep(wait)
    os.replace(part, dest)


# ---------------------------------------------------------------- main

def reuse(source: Path, local: Path, remote: RemoteFile) -> bool:
    """Put the same-named file from another model folder at `local`, as a hard link (no extra
    disk space) or else a copy, if its size matches. Its hash is then checked like any other
    file already present, so a different file is simply downloaded again."""
    candidate = source / remote.path
    if not candidate.is_file() or candidate.stat().st_size != remote.size:
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(candidate, local)
        how = "linked"
    except OSError:
        shutil.copyfile(candidate, local)
        how = "copied"
    say(f"  {how:<7} {remote.path} from {source}")
    emit("file", path=remote.path, size=remote.size, action=how)
    return True


def run(repo: str, revision: str, dest: Path, verify_only: bool, reuse_from: Path | None = None) -> int:
    say(f"Listing {repo} @ {revision}")
    files = select_files(list_remote(repo, revision))
    total = sum(f.size for f in files)
    say(f"{len(files)} files, {total / 1e9:.2f} GB")
    emit("plan", repo=repo, revision=revision, files=len(files), bytes=total)

    manifest = []
    for remote in files:
        local = dest / remote.path
        if not local.exists() and reuse_from is not None and not verify_only:
            reuse(reuse_from, local, remote)
        if local.exists():
            emit("verify", path=remote.path, size=remote.size)
            try:
                verify_hash(local, remote)
                say(f"  ok      {remote.path}")
            except VerificationError as exc:
                if verify_only:
                    raise
                say(f"  stale   {remote.path} ({exc}) - downloading again")
                emit("file", path=remote.path, size=remote.size, action="stale")
                local.unlink()
        if not local.exists():
            if verify_only:
                raise VerificationError(f"{remote.path}: missing")
            say(f"  fetch   {remote.path} ({remote.size / 1e9:.2f} GB)")
            emit("file", path=remote.path, size=remote.size, action="fetch")
            download(repo, revision, remote, local)
            emit("verify", path=remote.path, size=remote.size)
            verify_hash(local, remote)
            say(f"  ok      {remote.path}")
        emit("file", path=remote.path, size=remote.size, action="ok")
        entry = {"path": remote.path, "size": remote.size,
                 "sha256" if remote.is_lfs else "git_sha1": remote.sha256 or remote.git_sha1}
        if remote.path.endswith(".safetensors"):
            entry["dtypes"] = check_safetensors(local)
            say(f"  header  {remote.path} {entry['dtypes']}")
        manifest.append(entry)

    # Nothing may sit in the model folder that we did not verify.
    expected = {str(Path(f.path)) for f in files} | {"manifest.json"}
    extras = [str(p.relative_to(dest)) for p in dest.rglob("*")
              if p.is_file() and str(p.relative_to(dest)) not in expected]
    if extras:
        raise VerificationError(f"unverified files in model folder: {extras}")

    (dest / "manifest.json").write_text(json.dumps(
        {"repo": repo, "revision": revision, "files": manifest}, indent=2), encoding="utf-8")
    say(f"Verified {len(files)} files. Manifest written to {dest / 'manifest.json'}")
    emit("done", files=len(files), bytes=total)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="image",
                    help="which supported model to fetch (sets --repo and --revision)")
    ap.add_argument("--repo", default=None, help="override the model's repository")
    ap.add_argument("--revision", default=None, help="commit SHA to pin (not a branch name)")
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--verify-only", action="store_true", help="check an existing download, fetch nothing")
    ap.add_argument("--reuse-from", type=Path, default=None,
                    help="another model folder whose identical files are linked instead of downloaded "
                         "(each is still checked against this model's published hashes)")
    ap.add_argument("--json", action="store_true",
                    help="report progress as JSON lines on stdout (messages go to stderr)")
    args = ap.parse_args(argv)
    global _json_events
    _json_events = args.json
    repo, revision = MODELS[args.model]
    repo, revision = args.repo or repo, args.revision or revision
    if len(revision) != 40:
        ap.error("--revision must be a full 40-character commit SHA")
    if args.reuse_from is not None and args.reuse_from.resolve() == args.dest.resolve():
        ap.error("--reuse-from must be a different folder from --dest")
    try:
        return run(repo, revision, args.dest, args.verify_only, args.reuse_from)
    except VerificationError as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        emit("error", message=f"Verification failed: {exc}")
        return 1
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        emit("error", message=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
