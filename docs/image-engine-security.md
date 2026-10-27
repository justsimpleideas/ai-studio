# Security and supply chain

Local Image Studio downloads two kinds of third-party material: model weights and
software packages. Both are checked before anything is executed. This page records
what is checked and how, so the checks can be repeated.

## Model weights

Fetched with `image-engine/scripts/fetch_model.py`, which uses only the Python standard library.

| Check | How |
|---|---|
| Pinned source | Files come from one 40-character commit of the model repository (`DEFAULT_REPO` and `DEFAULT_REVISION` in `fetch_model.py`), never from a branch that can move. |
| Data formats only | The download is refused if the pipeline folders contain pickle checkpoints (`.bin`, `.pt`, `.ckpt`, …) or code (`.py`, `.so`, `.dll`, …). Weights must be `.safetensors`, which cannot execute code when loaded. The one other format accepted is a SentencePiece vocabulary (`.model`, a protobuf data file), and only inside `tokenizer/`. |
| No remote code | Every component in `model_index.json` names a class built into diffusers or transformers, and no config carries an `auto_map` entry. |
| Integrity | Every file is hashed and compared with the hash Hugging Face publishes for it: SHA-256 for large (LFS) files, git blob SHA-1 for small ones. |
| Well-formed weights | Each safetensors header is parsed and checked: known dtypes, byte lengths matching tensor shapes, tensors tiling the data section exactly with no gaps, overlaps or trailing bytes. |
| Hub scan | Hugging Face's own malware and pickle scan reported no flagged files for the image model's pinned commit. For both video models it was checked at their pinned commits before downloading: every downloaded file safe or not flagged, the shared SentencePiece vocabulary still queued at one scanner, which is why its protobuf structure is walked field by field after download. |
| Nothing extra | After download, any file in the model folder that is not in the verified manifest fails the run. |

At every model load the engine repeats the cheap part of this (`image-engine/image_engine/model_check.py`):
the manifest exists, each file still has its verified size, and no unlisted file has
appeared. It then loads with `local_files_only=True` and `use_safetensors=True`.

## Python packages

- **Binary wheels only** (`--only-binary=:all:`), so no package runs a `setup.py`
  during installation.
- **Separate indexes.** PyTorch is fetched only from the official PyTorch index with
  `--no-deps`; everything else only from PyPI. Mixing indexes in one resolution is
  how dependency-confusion attacks work.
- **Download, check, then install offline.** Wheels go to a local folder first, each
  file's SHA-256 is compared with the index's published hash, package names are
  reviewed for look-alikes, and installation runs with `--no-index` from that folder.
- **Isolated environment.** A dedicated virtual environment, outside the repository.

## npm packages

The studio's JavaScript dependencies are the chat app's existing, locked set. The
image UI added none: it is React components built by the same toolchain. (When the
image side was a standalone web app, its own UI packages were resolved with
`npm install --before=<two weeks earlier>` and install scripts disabled; that build
is no longer part of this repository.)

## Runtime

- The engine binds to `127.0.0.1` and refuses to start on any other address.
- Its only client is the studio's main process. Every request must carry a token the
  studio generates at each launch and passes through the environment (not the
  command line, which other programs can read). Comparison is constant-time.
- Requests are refused if the `Host` header does not name this machine (blocks DNS
  rebinding), or if they carry an `Origin` or a non-`none` `Sec-Fetch-Site`, which
  only browsers send (blocks cross-site request forgery from any open web page).
- Write requests must be `application/json`, which an HTML form cannot send.
- No CORS headers are sent, and `Cross-Origin-Resource-Policy: same-origin` stops
  other sites from embedding generated images.
- Request bodies are capped at 64 KB. If a body is not read, the connection is
  closed rather than reused, so leftover bytes cannot be parsed as a second request.
- The engine serves no web UI at all, only its API and gallery images. The studio's
  window loads nothing from the network, and its Content-Security-Policy blocks any
  direct connection to the engine.
- The engine exits when its stdin closes, so it cannot outlive a crashed or killed
  studio and keep the port or GPU memory.
- Hugging Face libraries run with `HF_HUB_OFFLINE=1` and telemetry disabled.

## Video

- **Start pictures** arrive at `POST /api/inputs` as raw image bytes, the one write
  that is not JSON. It still needs the token and the same browser checks, and its
  content type must be `image/png`, `image/jpeg` or `image/webp`, which no HTML form
  can send. The main process checks the bytes' real type by their signature before
  sending; the engine decodes them with Pillow restricted to that one format, refuses
  more than 40 MB or 64 megapixels, and stores a re-encoded PNG named by its hash, so
  nothing but pixels is kept and nothing from the request becomes a path.
- **ffmpeg** runs as a child process with an argument list (no shell), reading raw
  frames on stdin and writing one file into the gallery. A path chosen in Settings
  must be a file named `ffmpeg` or `ffmpeg.exe`. It gets no metadata from the engine
  and is told to drop any (`-map_metadata -1`).
- **Byte ranges.** Videos are served with single `Range` requests so the player can
  seek; anything malformed or out of bounds gets a 416, never a read outside the file.
  The studio's protocol handler passes only a well-formed `Range` header on.
- **The window** may load media only from the private `studio-image:` protocol
  (`media-src studio-image:` in its Content-Security-Policy), as it already did for
  images.
