"""Verify a folder of downloaded wheels against the official package indexes.

Standard library only. For every wheel it looks up the SHA-256 that the official
index publishes for that exact file name, hashes the local file, and compares:

  * PyTorch builds (a local version such as "+cu128") against download.pytorch.org;
  * everything else against pypi.org.

A wheel whose hash does not match, or that the official index does not list at
all, is deleted (or only reported with --report-only). Anything in the folder that
is not a wheel is reported too, since installation will only ever use wheels.

Each verified wheel is then opened and inspected:

  * RECORD integrity: every file inside matches the hash in the wheel's own RECORD;
  * .pth files are listed. Python runs import lines in a .pth file at every startup,
    which is the one way a wheel can execute code without being imported, and a known
    trick of malicious packages. Each one found is shown for review;
  * native binaries and executables are counted, for the report.

Usage:
  python scripts/verify_wheels.py <wheel folder> [--report-only]

Run it before installing with:
  pip install --no-index --find-links <wheel folder> ...
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

USER_AGENT = "local-ai-studio-verify-wheels/1.0"
PYTORCH_INDEX = "https://download.pytorch.org/whl/{variant}/{name}/"
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"


def get(url: str, attempts: int = 4) -> bytes:
    """Fetch from an official index, retrying slow or dropped connections."""
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except (OSError, TimeoutError):
            if attempt == attempts:
                raise
            time.sleep(10 * attempt)
    raise AssertionError("unreachable")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def parse_wheel_name(filename: str) -> tuple[str, str]:
    """(normalized project name, version) from a wheel file name (PEP 427)."""
    parts = filename[: -len(".whl")].split("-")
    if len(parts) < 5:
        raise ValueError(f"not a valid wheel name: {filename}")
    name = re.sub(r"[-_.]+", "-", parts[0]).lower()
    return name, parts[1]


def official_sha256(filename: str) -> str | None:
    name, version = parse_wheel_name(filename)
    if "+" in version:
        # Local versions (e.g. 2.11.0+cu128) only exist on the PyTorch index.
        variant = version.split("+", 1)[1]
        page = get(PYTORCH_INDEX.format(variant=variant, name=name)).decode("utf-8", "replace")
        quoted = urllib.parse.quote(filename)
        for candidate in (filename, quoted):
            match = re.search(re.escape(candidate) + r"#sha256=([0-9a-f]{64})", page)
            if match:
                return match.group(1)
        return None
    data = json.loads(get(PYPI_JSON.format(name=name, version=version)))
    for entry in data.get("urls", []):
        if entry.get("filename") == filename:
            return entry.get("digests", {}).get("sha256")
    return None


def inspect_wheel(path: Path) -> list[str]:
    """Problems found inside the archive (empty if none). Prints .pth contents for review."""
    problems: list[str] = []
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        # The wheel's own RECORD sits in a top-level "<name>-<version>.dist-info" folder.
        # Packages that vendor others (setuptools does) also carry nested RECORD files,
        # which describe only the vendored copy.
        record_name = next((n for n in names if n.endswith(".dist-info/RECORD") and n.count("/") == 1), None)
        if record_name is None:
            return ["no RECORD file"]
        recorded: dict[str, str] = {}
        for line in zf.read(record_name).decode("utf-8").splitlines():
            parts = line.rsplit(",", 2)
            if len(parts) == 3 and parts[1].startswith("sha256="):
                # Some Windows builds (torchvision's DLLs added after the build) write the
                # path with backslashes; zip entries always use "/". Same file, same check.
                recorded[parts[0].replace("\\", "/")] = parts[1][len("sha256="):]
        for name in names:
            if name.endswith("/") or name == record_name or name.endswith((".dist-info/RECORD.jws", ".dist-info/RECORD.p7s")):
                continue
            expected = recorded.get(name)
            if expected is None:
                problems.append(f"file not listed in RECORD: {name}")
                continue
            digest = base64.urlsafe_b64encode(hashlib.sha256(zf.read(name)).digest()).rstrip(b"=").decode()
            if digest != expected:
                problems.append(f"content does not match RECORD: {name}")
        pth = [n for n in names if n.endswith(".pth")]
        for n in pth:
            body = zf.read(n).decode("utf-8", "replace").strip()
            print(f"               .pth file {n}:")
            for line in body.splitlines()[:10]:
                print(f"                 | {line}")
        native = sum(1 for n in names if n.lower().endswith((".pyd", ".dll", ".so", ".dylib")))
        exes = [n for n in names if n.lower().endswith((".exe", ".bat", ".cmd", ".ps1"))]
        print(f"               {len(names)} files, {native} native libraries, {len(exes)} executables, {len(pth)} .pth")
        for n in exes[:10]:
            print(f"                 executable: {n}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--report-only", action="store_true", help="never delete, only report")
    ap.add_argument("--only", nargs="+", metavar="FILE", help="check only these file names in the folder")
    args = ap.parse_args(argv)

    files = sorted(p for p in args.folder.iterdir() if p.is_file() and (not args.only or p.name in args.only))
    bad = 0
    for path in files:
        if path.suffix != ".whl":
            print(f"  not a wheel  {path.name}")
            bad += 1
            continue
        try:
            expected = official_sha256(path.name)
        except Exception as exc:  # network trouble: never treat as verified
            print(f"  UNCHECKED    {path.name} (could not reach the official index: {exc})")
            bad += 1
            continue
        actual = sha256_of(path)
        if expected and expected == actual:
            print(f"  ok           {path.name}")
            problems = inspect_wheel(path)
            for problem in problems[:10]:
                print(f"  PROBLEM      {problem}")
            if problems:
                bad += 1
            continue
        reason = "not listed by the official index" if not expected else "SHA-256 differs from the official index"
        bad += 1
        if args.report_only:
            print(f"  MISMATCH     {path.name} ({reason})")
        else:
            path.unlink()
            print(f"  DELETED      {path.name} ({reason})")
    print(f"{len(files) - bad}/{len(files)} files verified")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
