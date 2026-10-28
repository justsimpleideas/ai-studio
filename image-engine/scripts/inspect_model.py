"""Inspect a downloaded model file, without loading or running any of it.

For the studio's "Inspect a model file" panel, and usable on its own. It reads the
file's header and the few small tensors that hold text rather than weights, then
hashes the whole file. Nothing is unpickled, imported or executed, and nothing
leaves the machine.

What it reports:
  * the SHA-256 of the file, to compare with the hash on its download page;
  * the format, from the file's first bytes (a pickle-based checkpoint can run code
    when loaded; safetensors cannot);
  * for safetensors: whether the header is well-formed (known dtypes, tensors tiling
    the data exactly, no name given twice, nothing hidden after the last tensor),
    what the metadata holds and anything odd in it (code-like text, embedded files,
    long encoded blobs, control characters), text stored in byte tensors, and whether
    the tensors match a model structure the studio knows.

It does not scan for malware (run your antivirus on the file for that), and it says
nothing about how the model behaves.

Usage:
  python scripts/inspect_model.py --file /path/to/model.safetensors
  python scripts/inspect_model.py --file /path/to/model.safetensors --json

With --json, progress and the report go to stdout as one JSON object per line.
Standard library only, plus the engine's own pure-Python header reader.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

from image_engine import single_file  # noqa: E402

CHUNK = 8 * 1024 * 1024
PROGRESS_SECONDS = 0.5
PREVIEW_CHARS = 240
LONG_VALUE = 64 * 1024
SMALL_BYTES = 4096        # byte tensors up to this size are read and looked at
BYTE_DTYPES = {"U8", "I8", "BOOL"}

# First bytes of formats that have no business inside model metadata or a byte tensor.
SIGNATURES = {
    b"MZ": "a Windows program (MZ header)",
    b"\x7fELF": "a Linux program (ELF header)",
    b"\xcf\xfa\xed\xfe": "a macOS program (Mach-O header)",
    b"PK\x03\x04": "a zip archive",
    b"%PDF": "a PDF document",
    b"\x80\x02": "a pickle", b"\x80\x03": "a pickle", b"\x80\x04": "a pickle", b"\x80\x05": "a pickle",
}
CODE_LIKE = re.compile(
    r"(\bimport\s+\w|\bexec\s*\(|\beval\s*\(|__import__|\bsubprocess\b|\bos\.system\b|\bpowershell\b|"
    r"\bcmd(\.exe)?\s+/c\b|<script\b|\bInvoke-Expression\b|\bcurl\s+-|\bwget\s+http)", re.IGNORECASE)
URL = re.compile(r"\b(?:https?|ftp)://[^\s\"'<>]+", re.IGNORECASE)
BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{400,}={0,2}")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_json = False


def emit(event: str, **fields: object) -> None:
    if _json:
        print(json.dumps({"event": event, **fields}), flush=True)


def say(message: str) -> None:
    if not _json:
        print(message, flush=True)


class Report:
    def __init__(self, path: Path) -> None:
        self.data: dict = {"file": path.name, "folder": str(path.parent), "size": path.stat().st_size,
                           "format": "unknown", "sha256": None, "findings": [], "metadata": [], "summary": None}

    def add(self, level: str, title: str, detail: str = "") -> None:
        """level: ok, info, warn or danger."""
        self.data["findings"].append({"level": level, "title": title, "detail": detail})

    @property
    def verdict(self) -> str:
        levels = {f["level"] for f in self.data["findings"]}
        return "unsafe" if "danger" in levels else "warnings" if "warn" in levels else "clean"


def preview(text: str) -> str:
    text = CONTROL.sub("�", text)
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"


def signature_of(data: bytes) -> str | None:
    for magic, what in SIGNATURES.items():
        if data.startswith(magic):
            return what
    return None


def look_at_text(report: Report, where: str, text: str) -> None:
    """Flag what does not belong in descriptive text."""
    if len(text) > LONG_VALUE:
        report.add("warn", f"{where} is unusually long", f"{len(text):,} characters.")
    if CONTROL.search(text):
        report.add("warn", f"{where} contains control characters", "Text meant for people has no reason to.")
    match = CODE_LIKE.search(text)
    if match:
        report.add("warn", f"{where} contains something that looks like code",
                   f"…{preview(text[max(0, match.start() - 40): match.end() + 40])}…")
    signature = signature_of(text.encode("utf-8", "replace"))
    if signature:
        report.add("danger", f"{where} starts like {signature}", "Model metadata is descriptive text, not a file.")
    for run in BASE64_RUN.findall(text)[:3]:
        try:
            decoded = base64.b64decode(run + "=" * (-len(run) % 4), validate=False)
        except (binascii.Error, ValueError):
            continue
        embedded = signature_of(decoded)
        if embedded:
            report.add("danger", f"{where} hides {embedded}, base64-encoded", f"{len(run):,} encoded characters.")
        else:
            report.add("info", f"{where} holds a long encoded blob",
                       f"{len(run):,} base64 characters ({len(decoded):,} bytes). Often a thumbnail or training data.")
    urls = sorted(set(URL.findall(text)))
    if urls:
        report.add("info", f"{where} mentions {len(urls)} web address{'es' if len(urls) > 1 else ''}",
                   ", ".join(preview(u) for u in urls[:5]) + (" …" if len(urls) > 5 else ""))


def inspect_safetensors(path: Path, report: Report) -> None:
    try:
        header = single_file.read_header(path)
    except single_file.SingleFileError as exc:
        report.add("danger", "The header is not well-formed", str(exc))
        return
    report.add("ok", "The header is well-formed",
               "Known dtypes, no name given twice, tensors tile the data exactly, nothing after the last one.")

    for key, value in sorted(header.metadata.items()):
        report.data["metadata"].append({"key": preview(key), "chars": len(value), "preview": preview(value)})
        look_at_text(report, f"Metadata '{preview(key)}'", value)
        if CONTROL.search(key) or len(key) > 256:
            report.add("warn", "A metadata name is odd", preview(key))
    if not header.metadata:
        report.add("info", "No metadata", "The file carries tensors only.")

    tensors = header.tensors
    odd_names = [n for n in tensors if CONTROL.search(n) or len(n) > 512]
    if odd_names:
        report.add("warn", f"{len(odd_names)} tensor names contain control characters or are very long",
                   preview(odd_names[0]))

    tags, texts, full_precision = 0, 0, 0
    with path.open("rb") as fh:
        for name, t in tensors.items():
            if t.dtype not in BYTE_DTYPES:
                continue
            length = t.end - t.start
            fh.seek(header.data_start + t.start)
            head = fh.read(min(length, SMALL_BYTES))
            embedded = signature_of(head)
            if embedded:
                report.add("danger", f"Byte tensor '{preview(name)}' starts like {embedded}",
                           f"{length:,} bytes. Weights are numbers, not files.")
                continue
            if length > SMALL_BYTES:
                weight_like = name.endswith((".weight", ".weight_scale", ".scale", ".bias", ".qweight", ".scales"))
                if not weight_like:
                    report.add("warn", f"Large byte tensor '{preview(name)}' is not named like a weight",
                               f"{length:,} bytes. Packed low-bit weights are stored like this; other data could be too.")
                continue
            try:
                text = head.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if name.endswith(single_file.QUANT_TAG):
                try:
                    tag = single_file.decode_tag(head, name)
                    tags += 1
                    full_precision += tag.get("full_precision_matrix_mult") is True
                    continue
                except single_file.SingleFileError as exc:
                    report.add("warn", "A quantization tag is not of a known kind", str(exc))
                    continue
            texts += 1
            look_at_text(report, f"Text in tensor '{preview(name)}'", text)
            report.add("info", f"Tensor '{preview(name)}' holds text", preview(text))
    if tags:
        detail = "Short JSON objects naming the fp8 format, with no settings the engine does not know."
        if full_precision:
            detail += (f" {full_precision} of them ask for the layer's weight to be widened before multiplying"
                       " (full_precision_matrix_mult), which the engine does for every layer.")
        report.add("ok", f"{tags} quantization tags, all of a known kind", detail)

    dtypes: dict[str, int] = {}
    parameters = 0
    for t in tensors.values():
        dtypes[t.dtype] = dtypes.get(t.dtype, 0) + 1
        if t.dtype not in BYTE_DTYPES:
            parameters += t.numel
    structure = "unknown"
    try:
        inspection = single_file.inspect(path, header)
        structure = f"the {inspection.label} ({inspection.storage}, {inspection.config.num_layers} blocks)"
        section = "Video model" if inspection.family == "video" else "Image model"
        where = f"It can be imported under Settings → {section}."
        if inspection.family == "video":
            try:
                if single_file.video_kind(inspection.config) != "5b":
                    where = f"It can be imported under Settings → {section}, together with the model's other expert."
            except single_file.SingleFileError as exc:
                where = f"It cannot be imported yet: {exc}."
        report.add("ok", f"Matches the {inspection.label}'s structure exactly",
                   f"{inspection.parameters / 1e9:.2f}B parameters, every tensor present with the expected shape. {where}")
    except single_file.SingleFileError as exc:
        if "not a model the engine supports" in str(exc):
            report.add("info", "Not a structure the studio checks tensor by tensor",
                       "The header was still checked; the tensors were not compared with a known model.")
        else:
            report.add("warn", "Looks like a model the studio supports, but does not match its structure", str(exc))
    report.data["summary"] = {"tensors": len(tensors), "parameters": parameters, "dtypes": dtypes,
                              "metadata_keys": len(header.metadata), "structure": structure}


def sniff_format(path: Path) -> str:
    with path.open("rb") as fh:
        head = fh.read(16)
    if head.startswith(b"GGUF"):
        return "gguf"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head[:1] == b"\x80" and head[1:2] in (b"\x02", b"\x03", b"\x04", b"\x05"):
        return "pickle"
    if len(head) >= 9 and head[8:9] == b"{":
        return "safetensors"
    return "unknown"


def sha256_of(path: Path) -> str:
    size = path.stat().st_size
    digest, done, last = hashlib.sha256(), 0, 0.0
    emit("progress", stage="hashing", done=0, size=size)
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last >= PROGRESS_SECONDS:
                emit("progress", stage="hashing", done=done, size=size)
                last = now
    emit("progress", stage="hashing", done=size, size=size)
    return digest.hexdigest()


def inspect_file(path: Path) -> dict:
    report = Report(path)
    emit("progress", stage="header", done=0, size=report.data["size"])
    fmt = sniff_format(path)
    report.data["format"] = fmt
    if fmt == "safetensors":
        inspect_safetensors(path, report)
    elif fmt in ("zip", "pickle"):
        report.add("danger", "A pickle-based checkpoint",
                   "Loading this format can run code hidden in it. The studio never loads it; prefer a .safetensors "
                   "copy of the model.")
    elif fmt == "gguf":
        report.add("info", "A GGUF file", "Hashed only; its contents are not checked here.")
    else:
        report.add("warn", "Not a known model format", "Neither safetensors, GGUF nor a pickle checkpoint.")
    if fmt != "safetensors" and path.suffix.lower() == ".safetensors":
        report.add("danger", "Named .safetensors but is not one", f"Its first bytes are those of a {fmt} file.")
    report.data["sha256"] = sha256_of(path)
    report.data["verdict"] = report.verdict
    return report.data


def print_report(data: dict) -> None:
    say(f"{data['file']}  ({data['size'] / 1e9:.2f} GB, {data['format']})")
    say(f"SHA-256  {data['sha256']}")
    for f in data["findings"]:
        say(f"  [{f['level']:<6}] {f['title']}" + (f"\n           {f['detail']}" if f["detail"] else ""))
    if data["metadata"]:
        say("Metadata:")
        for m in data["metadata"]:
            say(f"  {m['key']} ({m['chars']} chars): {m['preview']}")
    say(f"Verdict: {data['verdict']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, type=Path)
    ap.add_argument("--json", action="store_true", help="report as JSON lines on stdout")
    args = ap.parse_args(argv)
    global _json
    _json = args.json
    if not args.file.is_file():
        emit("error", message=f"Not a file: {args.file}")
        print(f"Not a file: {args.file}", file=sys.stderr)
        return 1
    try:
        data = inspect_file(args.file)
    except OSError as exc:
        emit("error", message=f"Cannot read the file: {exc}")
        print(f"Cannot read the file: {exc}", file=sys.stderr)
        return 1
    emit("report", report=data)
    print_report(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
