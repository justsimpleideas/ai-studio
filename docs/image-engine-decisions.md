# Decision log: image engine

Written when the image side was a standalone app (Local Image Studio). In Local AI
Studio the engine is a sidecar of the desktop app; where these entries mention a web
UI or ports, the studio-level decisions S2, S3 and S6 in [decisions.md](decisions.md)
now apply.

Each entry records what was decided, why, and what it rules out. Newest at the bottom.

## D1 — Scope: simple generator first, node graph later

**Decision.** Version 1 is a focused text-to-image app: a prompt box, generation
settings, live progress, and a gallery of past results. A node-graph
editor is a planned later phase, not part of v1.

**Why.** A node editor multiplies the work (graph model, execution scheduler,
per-node caching, editor UI) before a single image can be produced. Shipping the
simple generator first proves the hard part — running a modern diffusion model
locally on consumer hardware — and gives a usable tool right away.

**Constraint this puts on v1.** The generation backend must not be written as a
monolith tied to the v1 form. Generation is split into explicit stages
(encode prompt → denoise → decode → save) behind a small engine interface, and
every job is described by a plain JSON request. The v1 form is then just one
fixed graph, and the node editor can later assemble the same stages freely
without rewriting the engine.

## D2 — Model: a 6B few-step model with an official diffusers pipeline

**Decision.** The first supported model is a 6B-parameter text-to-image model
released under Apache 2.0. The exact repository and commit are pinned in one place,
`DEFAULT_REPO` and `DEFAULT_REVISION` in `image-engine/scripts/fetch_model.py`.

**Why.** 6B parameters, 8-step distilled sampling, strong prompt following and
in-image text rendering. It is one of the few current-generation models that
fits a 12 GB consumer GPU, and its license allows redistribution and commercial use.

## D3 — Engine: Python + PyTorch + diffusers

**Decision.** The engine is a local Python process using PyTorch (CUDA) and
Hugging Face diffusers. The UI talks to it over HTTP on localhost.

**Why.** diffusers ships an official pipeline class for the model, so it runs as its
authors intended, and features such as LoRA and img2img can be added without
writing kernels. The alternative considered was stable-diffusion.cpp (single
native binary, GGUF weights): easier to distribute, but every feature is bound
to what that engine implements, which conflicts with the node-graph goal in D1.

## D4 — Memory strategy for 12 GB GPUs

**Decision.** The transformer is kept on the GPU in bf16 when it fits; otherwise
weights are stored in fp8 and upcast per layer at compute time. The text encoder
is offloaded to system RAM after the prompt is encoded.

**Why.** In bf16 the transformer alone is ~12.3 GB and the text encoder ~8 GB,
which does not fit a 12 GB card at once. The encoder only runs once per prompt,
so moving it off the GPU costs little; fp8 storage halves the transformer's
footprint with a small quality cost and no extra dependencies.

## D5 — Fully local, models stored outside the repo

**Decision.** After the one-time model download, generation needs no network.
No telemetry, no remote calls. Model weights and the Python environment live in
a user-configurable data directory outside the repository.

**Why.** Local-only is the point of the project. Keeping weights (tens of GB) out
of the source tree also keeps them away from git and from any folder that a
cloud-sync client might upload.

## D6 — Ports

The UI dev server uses port 5194 and the engine API uses 5196, both pinned with
`strictPort` so the app fails loudly instead of drifting to another port.

## D7 — Video: a 5B text-and-image-to-video model, in the same engine

**Decision.** The engine gains a second model: a 5-billion-parameter video model
released under Apache 2.0, which makes clips from a prompt alone or from a picture,
up to 121 frames (5 seconds at 24 fps) at 1280×704. It is pinned in `fetch_model.py`
(`VIDEO_REPO`, `VIDEO_REVISION`) and loaded by `backends/video.py` through the
pipeline classes diffusers ships for it. Both modes run on one set of loaded
components: picture to video builds the image-to-video pipeline over the text-to-video
pipeline's own modules.

**Why.** Among open video models it is the one that fits a 12 GB consumer card with
a clean license; larger models (14B and up) need streaming from RAM there and become
impractically slow, and the strongest model that also makes sound needs 16–32 GB.
One model for both modes matters because longer videos are made by starting a clip
from the previous one's last frame, which needs picture to video.

**Same engine, not a second process.** The video model needs the same things the
image model does: PyTorch, diffusers, the gallery, the queue, the token-guarded API,
the GPU. A second engine would duplicate all of it and still have to coordinate the
GPU with the first. Inside one engine the rule is simple: the image and video models
are never loaded together (`jobs.py`); loading one unloads the other, and the studio's
GPU arbiter treats the engine as one side against chat.

## D8 — Video memory: the text encoder runs on the CPU below 30 GB

**Decision.** Four plans like the image model's (`memory.py`, `VideoMemoryPlan`), but
the 11.4 GB text encoder stays in system RAM and encodes on the CPU on every plan
except the largest. On a 12 GB card the automatic plan is Balanced: transformer stored
in fp8 on the GPU, the VAE moved in only to decode, and the transformer parked in RAM
while it does. Low streams transformer blocks from RAM (group offloading).

**Why.** Swapping the image model's 8 GB encoder in per prompt works because it fits
beside little else; this one does not fit on a 12 GB card at all next to anything.
Measured on the development machine, the CPU does bf16 matrix work at about 1.6 TFLOPS,
so a 512-token prompt (twice, with the negative prompt) takes seconds, and encodings are
cached across a batch. Components are read one at a time, largest first, so the
transformer is on the GPU before the text encoder is read; reading them all and then
placing them would need about 24 GB of RAM at once.

## D9 — Start pictures are stored by the engine, by content

**Decision.** A picture a video starts from is sent to the engine once (`POST
/api/inputs`, the image bytes themselves, PNG, JPEG or WebP) or made from a gallery
image or a video's last frame (`POST /api/inputs/from-gallery`). The engine re-encodes
it to a plain PNG named by the hash of its content and returns that id; a video
request names the id. The picture is cropped to the video's shape, never stretched.

**Why.** A request that names a picture by content stays reproducible: "Use these
settings" on a video brings back the exact picture it started from, even after the
gallery image it came from is deleted. Re-encoding drops everything but pixels (EXIF,
including location, never reaches the gallery), and phone photos are turned upright
first. The window never hands the engine a path; it sends bytes whose real type the
main process checks by their first bytes and the engine checks again by decoding.

## D10 — Video files: MP4 through the user's ffmpeg, else animated WebP; no metadata

**Decision.** Videos are encoded to H.264 MP4 by running ffmpeg as a child process
(raw frames on stdin, no shell, metadata stripped). ffmpeg is not bundled; the engine
uses the path chosen in Settings, else one on PATH, else where winget installs it.
Without ffmpeg, or if it fails, the frames are saved as an animated WebP with Pillow.
A video file carries nothing about how it was made: the prompt, settings and
provenance live in the gallery index only. Next to each video the gallery keeps its
thumbnail and its last frame as a PNG.

**Why.** Bundling ffmpeg means shipping, updating and verifying a large native binary
for one step; many users already have it, and the fallback means nobody loses a
finished generation. Animated WebP plays everywhere a picture does but is larger, so
MP4 is preferred. Putting settings inside MP4 would mean writing container metadata and
stripping it again on every export, for little use: a video is not dropped back into
the studio the way a PNG is. The images' default (exports leave the prompt out) is
kept by never writing it in the first place.

## D11 — Verifying video without the model

**Decision.** Before the 34 GB download, the video backend is tested on a tiny model
built from random weights (`tests/test_video_backend_tiny.py`): the same architecture
classes and folder layout, a few megabytes in size, run through the real diffusers
pipelines on the GPU. It covers loading, every memory plan, fp8 storage, picture to
video and the latent normalisation in decode. The rest of the engine and the studio are
tested with a fake video backend, end to end.

**What it does not cover.** Picture quality, speed and real memory use on a 12 GB card.
Those need the real model and are recorded separately once measured.

## D12 — Gallery folders: one folder per item, in the index only; favourites apart

**Decision.** The user can create, rename and delete folders in the gallery and
put each image or video in at most one of them (by dragging its tile onto a folder,
or choosing the folder in its details). Folders live in the gallery index (a
`folders` table and a `folder_id` per item); files never move on disk. Deleting a
folder removes the folder only: what was in it stays in the gallery, in no folder.
Favourites are separate from folders: the Favourites view shows every favourite,
wherever it is filed. Names are trimmed, up to 60 characters, unique ignoring case.

**Why.** One folder per item keeps the model simple to see and to trust: an item is
somewhere, or nowhere, and "All" always shows everything. Keeping folders out of the
file system means a rename is instant, nothing can be lost by a failed move, and the
image files (which carry their settings) are never rewritten. A folder deletion that
deleted its contents would be a way to lose work by tidying up, so it does not.

## D13 — GIF export: made on request from the stored video, never kept

**Decision.** "Save as GIF…" in a video's details asks for a width (320, 480, 640 or
the video's own; never enlarged) and a frame rate (10, 12, 15 or 24 fps; never above
the video's), then a save path. The main process asks the engine
(`POST /api/gallery/<id>/gif`), which converts the stored video in a temporary folder,
sends the GIF back and deletes it. With ffmpeg the conversion uses its palette filters
(`palettegen` with `stats_mode=diff`, `paletteuse` with Sierra dithering and
rectangle diffs). Without ffmpeg a WebP video is converted with Pillow (a median-cut
palette per frame, Floyd-Steinberg dithering); an MP4 cannot be read without ffmpeg,
and the error says so. The GIF loops and, like the video, carries no prompt.

**Why.** A GIF is a copy for sharing, not a second format to manage, so it is not
stored in the gallery and does not need to be generated with the video. Converting
the stored file (rather than keeping the decoded frames) costs a few seconds and no
disk space. GIF allows 256 colours per frame and compresses poorly, so the defaults
are small (480 wide, 12 fps); the user can raise both.
The size is computed the same way in the window and the engine (rounding half up),
so the size shown is the size made.

## D14 — Two video models: the original and its 3-step distillation

**Decision.** The video engine runs either the original 5B model or a published DMD
distillation of its transformer (`fetch_model.py --model video-fast`). The engine tells
them apart by the folder's `model_index.json` pipeline name (`video_models.py`); any
other name is refused. The distilled folder names a pipeline class diffusers does not
have. That name is only a label: the engine builds `WanPipeline` from its own imports as
for the original, with the same components and memory plans, and swaps in a scheduler
for the distilled schedule: three fixed noise levels (1.0, 0.757, 0.522), and after
each step the model's clean estimate (x − σ·v) is noised again to the next level. The
noise comes from the job's seeded generator, so a seed reproduces a clip. There is no
guidance, so one transformer pass per step and no negative prompt. The server fits a
video request to the configured model before queueing it, so the gallery records the
steps and guidance actually used (3 and 1). The window shows those as set by the model,
and warns that the distilled model was trained on 5-second clips.

The download reuses what the two models share. The text encoder, tokenizer, VAE and
scheduler config are byte-identical to the original's (same published hashes), so
`--reuse-from` hard-links them from the first folder, or copies them if a link is not
possible, and checks each against the distilled model's own published hashes like any
other file. Only the 10 GB transformer and two small configs are downloaded.

**Why.** The original takes 100 transformer passes for a clip (50 steps, guidance on);
the distillation takes 3. Keeping both, rather than replacing one with the other, lets
the user trade speed for the original's quality per clip, and costs 10 GB of disk. A
pipeline class picked from a folder's own config would let a folder decide what code
runs, so the class is fixed in the engine, as it is for the original.

**What is not verified.** The schedule is tested on the CPU, and the whole backend on
the tiny random-weight model of D11 labelled as the distilled one. Picture quality, and
how much faster a clip really is on a 12 GB card, need the real model and are recorded
separately once measured.
