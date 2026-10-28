# Python package security audit (image engine)

Audit of the Python packages the image engine runs on, carried out 2026-10-25
before anything was installed. The process is described in
[image-engine-security.md](image-engine-security.md); this page records what it
found and what was decided.

## Scope

45 wheels: PyTorch 2.11.0 (CUDA 12.8 build) and everything `diffusers`,
`transformers`, `accelerate`, `safetensors` and `huggingface-hub` pull in for
Python 3.14 on Windows.

## Method

1. **Binary wheels only, from official indexes.** PyTorch from download.pytorch.org
   with `--no-deps`; everything else from pypi.org. No package ran any code during
   download (`--only-binary=:all:`).
2. **Official hash.** Every wheel's SHA-256 compared with the hash the official
   index publishes for that exact file name (`image-engine/scripts/verify_wheels.py`).
   A mismatch, or a file the index does not list, is deleted. A file whose hash
   could not be fetched is reported as *unchecked*, never as passed, and re-checked.
3. **Archive inspection** (same script):
   - every file inside matches the wheel's own top-level `RECORD`;
   - `.pth` files are listed with their contents, since Python runs their import
     lines at every startup;
   - executables and native libraries are counted and named.
4. **Antivirus.** Windows Defender on-demand scan of the whole wheel folder
   (real-time protection was also on during download).
5. **Name review.** Every package name checked by hand for look-alikes of popular
   packages (typosquatting).
6. **Provenance and age** for anything that looked unfamiliar, and the official
   upload date of every package.
7. **Resolution check.** `pip install --dry-run --no-index` against the verified
   folder only, to confirm the set installs without reaching any index.

## Results

| Check | Result |
|---|---|
| Official hash | 45 / 45 match |
| RECORD integrity | 45 / 45 consistent |
| `.pth` files | 1: `distutils-precedence.pth` in setuptools, its standard distutils shim |
| Executables | `torch/bin/protoc.exe` (protobuf compiler shipped with PyTorch); 8 launcher stubs in setuptools (`cli*.exe`, `gui*.exe`). None run unless called. |
| Defender | No threats |
| Names | All well known except `httpx2` and `httpcore2`, examined below |
| Offline resolution | Resolves completely from the verified folder |

One defect in the audit tool was found and fixed along the way: setuptools vendors
other packages, each with its own `RECORD`, and the script had picked a nested one.
It reported false problems (never false passes); it now uses only the top-level
`RECORD`, and all 45 wheels were checked again with the fix.

## Decision P1: `httpx2` and `httpcore2` pinned to 2.13.1

**Context.** `huggingface-hub` 2.3.0 depends on `httpx2>=2.0,<3`, which brought in
`httpx2` and `httpcore2` 2.16.0. The names resemble the well-known `httpx` and
`httpcore`, so they were examined as possible typosquats.

**Evidence that they are genuine.**

- The dependency is declared by the official, hash-verified `huggingface-hub`.
- The project URL is `github.com/pydantic/httpx2`, owned by the Pydantic
  organization, not a fork, created the same day as the PyPI project (2026-05-11),
  with 22 releases since.
- The GitHub release v2.16.0 was published 30 seconds before the PyPI upload,
  the pattern of an automated release pipeline.
- The files match the official hashes, their RECORDs are consistent, and they
  contain no `.pth` files, executables or native code.

**What could not be established.** PyPI holds no provenance attestation for these
files, so nothing cryptographically ties the upload to that repository's build.
And 2.16.0 was four days old when downloaded. A hijacked maintainer account
publishing a malicious new release is the supply-chain attack that a waiting
period protects against; such releases are usually caught within days.

**Decision.** Use 2.13.1 (released 2026-09-23, a month earlier) for both. The pair
is consistent (`httpx2` 2.13.1 requires `httpcore2==2.13.1`), it satisfies
`huggingface-hub`'s range, and it passed the same checks as everything else.

**Cost.** None functional for this project: the engine runs Hugging Face libraries
fully offline, so these HTTP clients are imported but make no requests.

## Finding: other releases younger than 14 days

The npm side of the project only accepts package versions that have been public for
at least 14 days. Measured against the same rule on 2026-10-25, these Python
packages are newer:

| Package | Version | Uploaded | Age |
|---|---|---|---|
| diffusers | 0.42.0 | 2026-10-23 | 1 day |
| huggingface-hub | 2.3.0 | 2026-10-23 | 1 day |
| certifi | 2026.10.23 | 2026-10-23 | 1 day |
| importlib-metadata | 10.1.0 | 2026-10-24 | 0 days |
| zipp | 4.1.2 | 2026-10-21 | 3 days |
| filelock | 4.2.4 | 2026-10-19 | 5 days |
| transformers | 5.20.2 | 2026-10-19 | 5 days |
| pillow | 13.0.0 | 2026-10-16 | 8 days |
| truststore | 0.10.5 | 2026-10-12 | 12 days |

All of them passed every check above. Whether to hold them to the 14-day rule as
well is written up in [fresh-release-risk.md](fresh-release-risk.md). `diffusers` matters most: the engine's image-model code was
checked against 0.42.0's pipeline source, and earlier versions differ in how they
count sampling steps, so moving it back means re-checking that code.

## Decision P2: publish provenance first, the 14-day rule for the rest

**Decision (2026-10-25).** Each of the nine is kept only if its publish provenance
(PEP 740 attestation on PyPI) shows it was built by the project's own repository;
the others are held back to a version older than 14 days.

| Package | Version | Provenance | Outcome |
|---|---|---|---|
| diffusers | 0.42.0 | huggingface/diffusers, `pypi_publish.yaml` | kept |
| huggingface-hub | 2.3.0 | huggingface/huggingface_hub, `release.yml` | kept |
| transformers | 5.20.2 | huggingface/transformers, `release.yml` | kept |
| certifi | 2026.10.23 | certifi/python-certifi, `release.yml` | kept |
| filelock | 4.2.4 | tox-dev/filelock, `release.yaml` | kept |
| pillow | 13.0.0 | python-pillow/Pillow, `wheels.yml` | kept |
| truststore | 0.10.5 | sethmlarson/truststore, `deploy.yml` | kept |
| importlib-metadata | 10.1.0 | none (the project has never published any) | **9.0.1** (2026-08-28) |
| zipp | 4.1.2 | none (the project has never published any) | **4.1.1** (2026-10-03) |

**How provenance was checked.** For each file: PyPI's sha256 equals the local file;
the provenance object names a Trusted Publisher whose repository is the project's
official one; and the attested subject is exactly this file (name and sha256). PyPI
verifies the Sigstore signature when an attestation is uploaded; the signatures were
not re-verified independently here.

**The two replacements** were checked like every other wheel: official hash, every
file against RECORD, no `.pth`, native or script files, a clean antivirus scan, and a
scan of the code for network, process, dynamic-code and file-writing calls (none).
Both are pinned in `image-engine/requirements.txt`.

## Addition 2026-10-27: torchvision 0.26.0+cu128

**Why.** The captioning model of the Train tab (Qwen3-VL) is fed through its
transformers image processor, and in transformers 5.20 both the fast and the
slow image processor import torchvision. Without it no photo can be described.
It is PyTorch's own companion package, built and published together with torch
on the same index; 0.26.0 is the release paired with torch 2.11.0.

**Checks** (the wheel for Python 3.14, the version this engine was audited on):

| Check | Result |
|---|---|
| Official hash | matches download.pytorch.org: `6319e1ba49c6f62ac9902f73d0eab207b8a4dc6b4d3392fe9edd9903fff1be0a`, 9,678,588 bytes |
| RECORD integrity | 214 / 214 files match (see the note below) |
| `.pth` files | none |
| Executables | none; 11 native libraries (the package's own `_C.pyd` and `image.pyd`, and the DLLs they load: libjpeg, libpng, libwebp, nvjpeg, the CUDA runtime, and a `python314.dll`) |
| Defender | No threats |
| Dependencies | numpy, pillow and torch, all already pinned here; installed with `--no-deps`, `pip check` clean, nothing else changed |

The same release for Python 3.12 and 3.13 is published under the hashes
`8c0d1c4fbb2c9a4d5d41d0aaa87da20e525bcb2a154ce405725b0be59456804b` (cp312) and
`0232cb219927a52d6c98ff202f32d1cdf4802c2195a85fc1f1a0c1b0b4983a4d` (cp313).

**Note on RECORD.** The audit script first reported ten files "not listed in
RECORD". They were listed, with backslashes in their paths (`torchvision\image.pyd`):
the DLLs are added to the Windows wheel after it is built, and that step writes
Windows paths. Each of the ten matches its recorded hash. The script now reads
either separator; like the earlier fix, the defect could only cause false alarms.

## Ages of all audited packages

Official upload dates (PyPI). The PyTorch index does not publish upload dates.

| Package | Version | Uploaded |
|---|---|---|
| accelerate | 1.15.0 | 2026-09-09 |
| annotated-doc | 0.0.5 | 2026-07-28 |
| anyio | 4.15.1 | 2026-09-05 |
| certifi | 2026.10.23 | 2026-10-23 |
| charset-normalizer | 3.5.2 | 2026-09-30 |
| click | 8.5.0 | 2026-08-26 |
| colorama | 0.4.6 | 2022-10-25 |
| diffusers | 0.42.0 | 2026-10-23 |
| filelock | 4.2.4 | 2026-10-19 |
| fsspec | 2026.9.0 | 2026-09-18 |
| h11 | 0.16.0 | 2025-04-24 |
| hf-xet | 1.7.0 | 2026-10-06 |
| httpcore2 | 2.13.1 | 2026-09-23 |
| httpx2 | 2.13.1 | 2026-09-23 |
| huggingface-hub | 2.3.0 | 2026-10-23 |
| idna | 3.20 | 2026-09-17 |
| importlib-metadata | 9.0.1 (P2; 10.1.0 was 2026-10-24) | 2026-08-28 |
| jinja2 | 3.1.6 | 2025-03-05 |
| markdown-it-py | 4.2.0 | 2026-05-07 |
| markupsafe | 3.0.4 | 2026-10-02 |
| mdurl | 0.1.2 | 2022-08-14 |
| mpmath | 1.3.0 | 2023-03-07 |
| networkx | 3.7 | 2026-09-21 |
| numpy | 2.5.4 | 2026-10-10 |
| packaging | 26.3 | 2026-08-04 |
| pillow | 13.0.0 | 2026-10-16 |
| psutil | 7.2.2 | 2026-01-28 |
| pygments | 2.21.0 | 2026-08-17 |
| pyyaml | 6.0.3 | 2025-09-25 |
| regex | 2026.9.29 | 2026-09-29 |
| requests | 2.34.2 | 2026-05-14 |
| rich | 15.0.0 | 2026-04-12 |
| safetensors | 0.8.0 | 2026-06-09 |
| setuptools | 81.0.0 | 2026-02-06 |
| shellingham | 1.5.4 | 2023-10-24 |
| sympy | 1.14.0 | 2025-04-27 |
| tokenizers | 0.23.3 | 2026-10-09 |
| torch | 2.11.0+cu128 | not published by the index |
| torchvision | 0.26.0+cu128 | not published by the index |
| tqdm | 4.70.1 | 2026-09-11 |
| transformers | 5.20.2 | 2026-10-19 |
| truststore | 0.10.5 | 2026-10-12 |
| typer | 0.27.3 | 2026-10-06 |
| typing-extensions | 4.16.0 | 2026-07-02 |
| urllib3 | 2.8.0 | 2026-09-15 |
| zipp | 4.1.1 (P2; 4.1.2 was 2026-10-21) | 2026-10-03 |
