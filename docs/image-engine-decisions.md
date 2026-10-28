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

**Hidden and locked folders.** A folder can be hidden, locked with a six-digit PIN, or
both (`hidden` and `pin` columns of `folders`). The engine keeps, in memory only, whether
this session shows hidden folders and which folders it has unlocked, and leaves the items
of every other hidden or locked folder out of everything it answers: listings, search,
favourites, an item's details, its files, and making a start picture from it. Asked for
directly, a hidden folder answers as if it did not exist (404) and a locked one 423, so the
window cannot reach their contents by a URL either. The studio resets both whenever its
window closes or goes to the background, and an engine restart starts reset. The PIN is
stored as salted scrypt; it is needed to change or remove it even while unlocked, and to
rename or delete the folder (deleting would put its items in plain view). Five wrong PINs
in a row make the next try wait 30 seconds, doubling up to 15 minutes. A locked folder
takes new items without its PIN. A name is refused only for a clash with a folder this
session sees, so creating a folder cannot reveal a hidden one.

**What it is not.** It keeps the studio's own views private from someone using the
studio. It does not encrypt anything: the files and `gallery.db` stay readable on disk,
and six digits fall quickly to a guesser who has copied `gallery.db`. Encrypting the
gallery would be a separate, much larger feature.

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

## D15 — A second image model, imported from a single transformer file

**Decision.** The image engine also runs a 12B single-stream model (`image_models.py`,
family `large`). The configured folder's `model_index.json` picks the family, among a
fixed list of pipeline names, and the engine's image slot (`backends/image.py`) loads
that family's backend, unloading the other first. Each backend builds its pipeline from
its own imports, never from a name in the folder.

The model's transformer is published as one safetensors file, not a diffusers folder, so
the folder is put together locally by `scripts/import_model.py`: the transformer file,
the studio's captioning model as its text encoder (the same 4B vision-language model at
the same pinned commit, of which only the language part runs), and its VAE, downloaded by
`fetch_model.py --model large-vae` from the pinned commit of another model's repository
that ships the same VAE. Files are hard-linked where possible, and the manifest records
every file's hash: the published ones for the two downloads, the computed SHA-256 for
the transformer, checked against the hash on its download page when one is given. The
load-time check (`model_check.py`) is the same as for every other model folder.

Nothing in the transformer file is trusted before it is read: its header alone
(`single_file.py`, pure Python) names the layout (the authors' tensor names, behind a
loader prefix or not, or diffusers'), the model's dimensions (read off the tensor
shapes), and the quantization. Every tensor the model has must be there with the right
shape, nothing else may be, fp8 is allowed only for linear weights and only with a scale
of one value per tensor or per output row, and the per-layer quantization tags must be
short JSON objects naming a known format with no unknown settings. The renaming rules
are the same as diffusers' own single-file converter for this model, which a test checks.

The linear weights stay in fp8 (`fp8.ScaledFP8Linear`) and are widened to bf16 right
before each matmul, so the arithmetic is a bf16 model's; a bf16 file is quantized per
output row as it loads, so both files need the same memory. On a 12 GB card the
transformer (about 13 GB) is streamed: its blocks wait in pinned RAM and visit the GPU
one at a time through diffusers' group offloading, each copied on a side stream while the
one before computes. The text encoder moves to the GPU only to read a new prompt
(`memory.py`, `LARGE_PLANS`).

Pinned RAM is handed out as slices of 256 MiB blocks (`transformer_file.PinnedArena`), up
to half of the machine's RAM; what does not fit stays pageable and is pinned only while it
is copied. PyTorch rounds each page-locked allocation up to a power of two, so pinning
tensor by tensor locked about 1.6 times the weights, and on a 32 GB Windows machine the
system refused partway. That refusal is raised as "CUDA error: out of memory", the same
words as a full GPU, and CUDA keeps it as its last error, so the next GPU operation,
wherever it is, fails with it too: the load stopped with that message even though the
loader had caught the refusal. A refusal is now followed by one small GPU operation that
takes the stale report (`clear_refused_lock`), and one that still escapes says that RAM,
not the GPU, ran out.

The model takes no LoRAs and cannot be trained yet. The server refuses a LoRA request for
it before queueing it, training says so before loading anything, and the window hides the
LoRA picker and fits steps and guidance to the model's own defaults unless the user
changed them.

**Why.** The model's quality per step is what makes it worth 13 GB, and its fp8 file is
how it fits a 12 GB card at all: a bf16 transformer would need 25 GB of RAM to stream,
and with the text encoder that exceeds a 32 GB machine. Weight-only fp8 keeps the
arithmetic in bf16, so it needs no fp8 matmul kernels and runs on any CUDA GPU that
diffusers supports. Streaming costs little because, at 1024², one block's matmuls take
longer than copying its 0.43 GB over PCIe. Reusing the captioning model as the text
encoder avoids a second 9 GB download of the same weights.

**What is not verified.** The loading code is tested on the CPU against diffusers'
converter and a tiny random-weight model (the fp8 layers reproduce the float model's
output), and the whole backend, every memory plan, streaming included, on that tiny model
on the GPU. Picture quality, speed, real VRAM and RAM use, and whether pinning 13 GB of
RAM works on every Windows machine (the loader falls back to pageable memory if not)
need the real model and are recorded separately once measured.

## D16 — Single-file models of either family, imported from the window

**Decision.** *Settings → Image model → Import a model file…* takes a single transformer
file of either image model. `import_model.py --probe` reads its header and says which
model it is (`single_file.detect_family`); the window then asks whether it is a distilled
or a base model (that only sets the default steps and guidance), optionally for the hash
on its download page, and for a folder name. The main process (`src/main/image/importer.ts`)
finds the parts that model needs among folders already verified at the pinned commits
(the configured folders first, then the models folder, then every verified folder it has
seen before and the folders next to the configured ones, so a download kept elsewhere is
still found after the image model folder has moved to an imported model), downloads the missing ones with
the same verified downloads as the model cards once the user has accepted their licenses
(`image-parts`, `large-vae`, `caption`; parts are never offered on their own and never
selected), runs the import, and points the engine at the new folder. A folder that holds
only a model file is refused at load with a pointer to the import.

For the 6B model the import links in everything but the transformer's weights from the
6B download (or from `fetch_model.py --model image-parts`, the same files without the
transformer, 8.2 GB), and its `model_index.json` gains `transformer_file` and
`is_distilled`. Its file is read in the authors' layout (q/k/v fused into one weight per
block, split here into three equal row blocks, with their fp8 scale cut alongside) or in
diffusers'; the model's dimensions are read off the shapes and checked against the
`config.json` it is assembled with. The 6B backend builds that transformer itself
(`transformer_file.build_six_b`) and hands it to the pipeline, so the rest of the folder
loads as for the download. fp8 weights stay in fp8 (layerwise casting is skipped for them),
and LoRAs wrap the fp8 layers as they wrap linear ones, so LoRAs work on such a model and
training works on a distilled one.

**Why.** Fine-tunes of both models are mostly shared this way, often only in fp8. Making
the user assemble a folder by hand, or download a second 32 GB copy of a model they have,
would put both out of reach; linking the parts costs no disk, and every file in the result
is still one the engine verified.

**Fixed along the way.** In the 6B model's Low VRAM mode the latents were moved to where the
VAE was waiting (RAM) rather than to the GPU it runs on, so decoding failed. Found by the
tiny-model test this change added.

**What is not verified.** As in D15: the loaders, the backend through its memory plans and
the import are tested on tiny random-weight models and synthetic files; real files, picture
quality and speed are the user's to check.

## D17 — The video model, imported from a single transformer file

**Decision.** *Settings → Video model → Import a model file…* takes a single transformer
file of the 5B video model, the same way D16 takes the image models'. The two cards share
one importer: an import shows in the card it was started from, and a file of the other
kind is sent to the other card by name rather than imported. `single_file` reads the
authors' layout with the same renaming rules, in the same order, as diffusers' own
converter for this model family (the last three swap `norm2` and `norm3`; a test checks
the two agree name by name and value by value), or diffusers' layout as it is. The
model's dimensions come from the shapes, except how its width splits into heads, which the
shapes do not show (q and k are normalized across all heads) and the folder's
`config.json` supplies; every shape-decided key, the head split, and the absence of an
image encoder or extra projections are checked against it. The expected tensors derived
for the published dimensions are exactly the 825 of the downloaded transformer, and their
size in fp32 is exactly the byte count its index states.

The parts are the original video download's files without the transformer's weights
(`fetch_model.py --model video-parts`, 14.2 GB; the full download is accepted too, the
distilled download is not, since it comes from another repository). Its `model_index.json`
gains `transformer_file`; a file the user says is distilled gets the distilled pipeline
label, so it runs the fixed 3-step schedule. The video backend builds the transformer
itself (`transformer_file.build_video`) and hands it to the pipeline. fp8 weights stay in
fp8 in every plan, including the streamed one, and layerwise casting is skipped for them.

`fetch_model.py`'s listed downloads now accept a SentencePiece vocabulary inside
`tokenizer/`, as its pipeline downloads already did; the video model's tokenizer has one.

**Why.** Video fine-tunes and fp8 copies are shared as single files like image ones, and
an fp8 transformer halves what the GPU holds, which matters most on the 8-12 GB cards this
studio is built for. The larger (14B) video models, shared as two files (one per noise
range), need a second transformer and a new memory plan; they are the next step, not
part of this one.

**What is not verified.** As in D15 and D16: tiny random-weight models and synthetic files
only. Real files, video quality and speed are the user's to check.

## D18 — The 14B video models: two experts, one at a time

**Decision.** The 14B text-to-video and image-to-video models load from imported single
files only (two per model, one per expert), with the parts of their pinned download
(`video-14b-text-parts`, `video-14b-image-parts`: everything but the experts). Their text
encoder and tokenizer are the 5B model's byte for byte (checked against the published
hashes), so the parts download links them from a 5B folder and fetches only the 0.5 GB VAE
and configurations. The expected tensors for the published 14B dimensions are exactly the
1,095 of the downloaded expert's index, and their fp32 size its stated byte count; the
header of a widely shared fp8 file (read over an HTTP range request, no weights) matched
them name for name and shape for shape.

That file uses an older fp8 layout than the image models' files: a `scale_weight` per
weight, a `scale_input` (used only by loaders that multiply in fp8, so it is checked and
left out), and an empty `scaled_fp8` marker. It also stores the timestep embedding's
linear layers in fp8; diffusers casts that layer's input to the layer's own dtype, so an
fp8 layer there would turn the timestep into fp8. Outside the blocks an fp8 weight is
therefore widened as it loads (`transformer_file.fill(widen=...)`); in the blocks it stays
fp8. The same applies to 5B files.

Sampling uses diffusers' own two-expert pipelines (`boundary_ratio`, `transformer_2`,
`guidance_scale_2`), built in place from the loaded parts. The second expert is only read
when needed: at the end of the step before the boundary, a step callback calls
`Experts.ensure("low")`, which drops the first expert first unless both fit in RAM
(`memory.keep_both_experts`), and sets it as the pipeline's `transformer_2`. On a 12 GB
card each expert is streamed block by block (`TwoExpertPlan`, *Balanced*); on 22 GB and up
the running expert is resident and the other waits in RAM. Steps, guidance (4/3 for text,
3.5/3.5 for image) and boundaries (0.875/0.9) are the models' own. A few-step distillation
records the steps and shift the user enters (from its download page) in model_index.json
and runs plain Euler steps on that shifted schedule without guidance. Requests are checked
against the configured model: 16 fps, sides in 16s up to 1280x720, 480p by default, and
the start the model takes (never a picture for text to video, always one for image to
video); the window's video form follows the model.

Found along the way: diffusers' image-to-video pipeline calls an optional text-fixing
package without checking it is installed, so its prompts are encoded through the
text-to-video pipeline, whose code for that is otherwise the same.

**Why.** The 14B models are the largest open video models of the family and the ones most
fine-tunes and distillations are shared for; a 12 GB card can run them only streamed and
one expert at a time, which is this studio's case. Keeping both experts when RAM allows
avoids reading 28 GB per clip on machines that can afford it.

**What is not verified.** Tiny random-weight models on the GPU (every plan, one or both
experts kept, few-step, image to video, the older fp8 layout) and synthetic files. Real
files, video quality, speed, and whether a given few-step file's shift suits plain Euler
steps are the user's to check.

## D19 — Merging other galleries into this one

**What.** Settings → Merge galleries takes one or more other galleries (an `outputs`
folder from a backup, an older install or another computer) and copies what they hold
into this studio's gallery: every image and video with its file, prompt, settings,
provenance and stats, whether it is a favourite and which folder it is in, and every
folder with its name, hidden setting and PIN (the stored hash, so the same PIN opens it).
Items keep their creation time. `image_engine/gallery_merge.py`.

**Why into this gallery.** The studio reads one gallery, so a merge anywhere else could not
be seen without a second feature to switch galleries. Merging in shows the result at once,
the engine that owns the index does the writing, and the other folders are only read.

**Nothing lost, nothing overwritten.**
- Every item either comes across or is counted with its reason: already here, or
  unreadable (a missing file or a damaged record).
- An item is "already here" only when its id, its file byte for byte and its whole
  generation record (prompt, settings, seed, model, provenance, when it was made) match
  one here or one already coming: a backup, or a gallery merged before. Anything less and
  both are kept, the newcomer with a new id. Being a favourite and the folder it is in are
  how it is filed, not what it is: they do not count, and differences are reported.
- A folder with the same id and exactly the same name as one here is that folder, whatever
  each holds: its items join it, and this gallery's hidden setting and PIN stay. The same
  name alone (in any case) is a clash to resolve.
- Folder names stay unique regardless of case. Clashes are listed before anything is
  copied; each side can be renamed, or names suggested ("Trips (backup1)"). This
  gallery's folders are renamed only on purpose, and a locked one only once unlocked.
- No file here is replaced: a path in use gets a number, and each copy is linked into
  place, which fails rather than replace a file that appeared meanwhile.
- Everything is copied before the index changes, in one transaction. On any failure
  every file copied so far is removed, and the gallery is as it was.
- There must be room for the copies on the gallery's drive before anything starts.
- A merge can be stopped from the window; what was copied so far is removed and nothing
  is added. A file's hash is remembered by its size and modification time, so looking
  again after a folder is renamed does not read every file again.

**Older and newer galleries.** A gallery from an older version lacks later columns; they
take the same defaults the gallery itself gives them. Columns from a newer version are
left out and named in the report. A PIN in a form this version cannot check is kept, so
its folder stays locked here rather than opening.

**Privacy.** As in the gallery, a hidden folder's name only counts, and is only shown,
while hidden folders are shown. A merge never tells that a hidden folder of some name
exists.

**Verified.** Engine tests on synthetic galleries: everything comes and the sources are
untouched byte for byte; a backup adds nothing; an id clash keeps both; no file here is
replaced; a failure part way and a full drive leave the gallery as it was; older and
newer schemas; unsafe paths; names differing in case. The end-to-end test merges two
galleries (a clash, a copied item, a hidden locked folder) from the window.
