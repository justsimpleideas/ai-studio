# Training — subject LoRAs on the local machine (design draft)

Status: built end to end — dataset import, the captioning model and its
relevance check, latent and caption-embedding caches, the in-house LoRA, the
training loop, the engine API, the Train tab, checkpoint comparison, the LoRA
library, and LoRAs in the Image tab. The captioning model has not described a
real photo yet (its path is tested with a tiny random model on the CPU); its
image processor needs torchvision, which the engine installs since 2026-10-27
(docs/package-security-audit.md). Until a photo is described its caption is
"facts only" (from folder names). Tested on a tiny random-weight transformer on the CPU (engine
tests and the end-to-end test); no real-model run yet. Nothing is tied to one
topic: each project defines its own subject (see "Subject").

## Goal

Let a user teach the image model a narrow subject (a kind of watch, a breed of
dog, a line of cars — anything they have photos of) from their own photos,
entirely on this machine, on a 12 GB GPU. The result is a LoRA file the Image
tab can load next to the base model.

The hard part is not the training loop. It is getting **thousands of correctly
captioned images** without the user typing thousands of captions. So the design
spends most of its effort on captioning and review.

## Scope

v1:
- Image LoRA for the current image model.
- Import → caption (local vision model) → review → train → compare → use.

Not in v1:
- Video LoRA. Same pipeline later, much higher cost; revisit once v1 works.
- Scraping. Photos come from the user's own folders, or are collected from
  two services with per-file licences (see "Collecting photos").
- Training text encoders or the VAE.

## The distillation trap

The installed image model is a few-step distilled ("turbo") variant. Training
a LoRA directly on it erodes what the distillation taught: images go soft or
need more steps again. Two known ways around it:

1. **Train on the non-distilled base checkpoint**, then apply the LoRA to the
   turbo model. Transfer is usually good but not perfect; v1 must measure it
   (eval grid below, base vs turbo).
2. **Train on turbo with a de-distillation adapter** merged in during training
   and removed at inference.

Both need an extra download. Decision deferred until candidates, official
hashes and sizes are listed and approved.

## Subject

What a project is about is defined by the user and stored in its
`project.json` (`image_engine/training/subject.py`; the window mirrors the
checks in `src/shared/subject.ts`). The engine has no built-in topic: the
presets "General" and "Cars" are only starting points, and every part of
either can be changed.

| Part | Example (wristwatches) | Used for |
|---|---|---|
| Noun | `wristwatch` | Every caption carries it, so the LoRA attaches to a concept the model already knows; the captioning prompts ask about it. |
| Trigger word (optional) | `mywatch` | Put first in every caption, so a prompt can call the LoRA up on purpose. |
| Folder levels (1–4) | Brand (required) / Model (required) / Reference (not in caption) | Each folder level below the photo folder fills one fact. Photos missing a required level are not trained on; the in-caption levels make up the subject's name. |
| Description fields (0–10) | View: `front, back, side` → `{value} view`; Dial colour: free text → `{value} dial` | What the captioning model is asked about each photo, and how an answer is written into the caption. |

A field with choices is answered from its list and can be cross-checked (two
prompts, outliers); a free-text field cannot, so the editor says so. A field
marked "same for the whole subject" (a car's body style, a watch's strap) must
have choices; `labels.json` may set it, and answers that contradict it or most
photos of the same subject are flagged.

Keys come from labels (`Dial colour` → `dial_colour`, any language's letters
allowed) and are what `labels.json` uses. Changing the subject rescans the
folder. Caption answers to a field that is later removed stay stored (nothing
is lost if it comes back) but are left out of captions; an answer that is no
longer one of a field's choices reads as unknown. Projects made before
subjects existed load as the "Cars" preset.

## Pipeline

### 1. Import

- The user picks a root folder. **Folder names carry the facts**: one folder
  level per level of the subject, e.g. `<brand>/<model>/<reference>/*.jpg`.
  Underscores become spaces; hyphens are kept (`F-150`). A `labels.json` in any
  folder (e.g. `{"brand": "Example Watch Co", "reference": "D-100"}`) sets facts
  for everything below it; the nearest one wins. A broken or unknown field is a
  warning, not a failure.
- Checks: decodable, shorter side ≥ 512 px (EXIF orientation applied),
  the subject's required levels present.
- Duplicates: exact (content hash) and near (64-bit dHash, ≤ 4 bits apart —
  re-saved or resized copies). All copies but one are flagged, not deleted;
  the one kept is the largest copy with no other issue. Near-duplicates are
  found by band lookup (pigeonhole), not by comparing every pair.
- Holdout: a deterministic 5% by content hash, so a rescan never moves a photo
  between train and holdout.
- Images are never modified, moved, or written next to. The dataset is a
  manifest (`dataset.json`, in the app's data folder) that points at the
  originals. Rescanning re-reads only files whose size or mtime changed.
- Latent cache: each photo is cropped to its bucket and VAE-encoded once
  (distribution mean, normalised as the inverse of the backend's decode).
  Entries are named by content hash + bucket + VAE, so a stale latent is never
  reused; building is resumable and stoppable.

### 2. Caption

Captions are split into two halves with different sources:

| Half | Fields | Source |
|---|---|---|
| Facts | the subject's folder levels (brand, model, ...) | folder labels (never the vision model) |
| Description | the subject's description fields (view, colour, setting, ...) | local vision model |

The vision model answers in a JSON object with one key per description field,
with `"unknown"` allowed for every field, and is never asked to name the
subject. The final caption is assembled from both halves, in this order:
trigger word, the in-caption levels, the noun, then each known answer through
its phrase:

```
mywatch, Seiko SKX007, wristwatch, front view, black dial, on a wrist, daylight
```

**Uncertainty is measured, not asked for.** Vision models state wrong answers
confidently, so an image is flagged when any of these trip:

- **Disagreement**: two passes with differently worded prompts give different
  values for a field.
- **Low token probability** on a field value.
- **Fact/description conflict**: e.g. labels.json says SUV, description says
  sedan (a field marked "same for the whole subject"). This also catches images
  filed in the wrong folder.
- **`"unknown"`** on a field marked important.
- **Outlier**: at least 5 photos of the same subject (same folder facts) and
  ≥ 70% agree on a "same for the whole subject" field; the rest are flagged.
  Catches misfiled photos even when no labels.json gives the value.
- **Single pass / unparseable**: one or both replies were not the JSON asked
  for, so nothing cross-checks the answer.

Off-list answers ("diagonal-ish") become `"unknown"`; common synonyms
(estate → wagon, profile → side) are mapped first when what they map to is
one of the field's choices.

The vision model runs in the Python engine (the chat engine has no image
input). It takes the GPU like any other model; captioning and training run
one after the other, never together.

**Relevance.** Before the subject's own fields, every photo is asked two
questions with fixed answers (`captions.RELEVANCE`; the keys `subject_shown`
and `subject_size` are reserved):

| Question | Answers | What happens |
|---|---|---|
| What the photo shows of the subject | the real thing, whole · the real thing, part or close-up · the inside of it · a toy, model or replica · a picture of it · not in the photo | toy, picture or absent: **off subject**; part or inside: flagged for a look |
| How the subject sits in the photo | the main subject · one of several things · small or in the background | small: **off subject** |

An off-subject photo is left out of training and shown first in "To review"
until the user includes it (or edits its caption); the answers never reach the
caption. A first collection of five car models from Commons had exactly these
problems: diecast models among the photos, and interiors, engine bays and
street scenes where the car was small.

**The model.** Qwen3-VL 4B Instruct (Apache 2.0), bf16, from a folder
`fetch_model.py --model caption` verified (12 listed files, nothing else; a
repository shipping code is refused), loaded with remote code refused. Photos
are shrunk to 960 px on the longest side first; decoding is greedy. The caption
task (`training.caption`) describes the photos not yet described with the
current model, questions and subject, saves every 10, stops gracefully and
carries on when run again; afterwards it flags outliers and picks the audit
sample.

### 3. Review

- Queue sorted by flag severity, then by flag count.
- Grid view; each card shows the image, the assembled caption, and *why* it
  was flagged.
- Keys: accept / edit / exclude. Bulk edit: select many, set one field.
- **Audit sample**: a random sample of *unflagged* images (default 50) is
  shown for review too. Its error rate is reported as "estimated error rate
  of auto-accepted captions". Without this there is no way to know whether
  the flags are catching anything.

Guidance shown in the UI: wrong facts (wrong model) hurt training badly; small
description slips (grey vs silver) mostly do not.

### 4. Train

Memory strategy for 12 GB:

- **Cache first, then unload**: VAE latents and text embeddings for every
  image are computed once and written to disk. Training loads only the
  transformer.
- Transformer weights frozen in fp8; LoRA weights trainable in bf16.
- Gradient checkpointing on.
- Aspect-ratio buckets near a target pixel count (start at ~768², then 1024²).
- Plain AdamW on the LoRA parameters (small enough that no 8-bit optimizer is
  needed).

Objective (`training/trainer.py`), matching how the pipeline samples:

- Noisy latent `x_s = (1 - s) * x0 + s * e`; the transformer gets timestep
  `1 - s` and is trained to output `x0 - e` (the pipeline negates that output
  and takes an Euler step along it). Loss: MSE.
- Noise level: logit-normal `u`, then the model's timestep shift,
  `s = shift * u / (1 + (shift - 1) * u)`. The shift comes from the model's
  scheduler config (3.0 for the turbo model, 6.0 for the base checkpoint).
- Caption dropout (default 10%) trains on the cached empty-caption embedding.
- Each epoch visits every photo once, in a shuffled order.

Defaults (to be tuned by measurement): rank 16, lr 1e-4, 2,000 steps,
batch 1, gradient clipping 1.0, checkpoint every 250 steps.

Runs as an engine job: progress events, cancel, and **resume from the last
checkpoint** (long runs on a laptop will be interrupted). Resuming gives
exactly the result of an uninterrupted run: every random draw is a function
of (seed, step), and the LoRA and optimiser state are saved together. A run
can be extended (more steps) but not changed — different settings, photos or
adapter in the same folder are refused.

Outputs, all safetensors or JSON (nothing pickled):
`checkpoints/step-NNNNNN.safetensors` (the trained LoRA only, diffusers
layout, loadable on its own) and `state/` (LoRA + AdamW state, step, loss log).

RAM and CPU: one latent and one embedding are read per step and nothing else
is held; CPU threads are capped during the run (default 4) and restored after.

### 5. Compare

- A fixed prompt set × fixed seeds, rendered at each checkpoint, shown as a
  grid: base model / with LoRA, per checkpoint.
- Held-out images (default 5%) are never trained on; their captions become
  prompts, so the user can compare generated vs real.
- If trained on the base checkpoint, the grid also shows the LoRA applied to
  the turbo model — this is where transfer quality is judged.

### 6. Use

- Image tab: pick one LoRA, strength slider.
- Provenance records the LoRA file hash and strength with every image.

## LoRA (in-house, no LoRA library)

`image_engine/lora.py` implements LoRA directly instead of depending on a LoRA
library (decision 2026-10-27). Training needs two LoRAs on the same layers at
once — the frozen training adapter and the one being trained — then removes the
adapter before generating, all on top of fp8-stored weights. That is simpler to
get right, and to test, as ~250 lines we control than as an integration of
diffusers' loader with an extra dependency.

- A targeted `nn.Linear` is replaced in its parent by a `LoRALinear` that calls
  the original layer first, so hooks on it (fp8 upcasting) keep working.
  `y = W x + Σ strength · (alpha / rank) · B (A x)` over the attached LoRAs.
- A and B are plain Parameters, not modules, so layerwise casting never stores
  them in fp8. A new LoRA starts at B = 0, an exact no-op.
- Default targets: `to_q`, `to_k`, `to_v`, `to_out.0`, `w1`, `w2`, `w3` of every
  main transformer block (7 per block).
- Removing the last LoRA from a layer puts the original `nn.Linear` back.
- Files are written in the diffusers layout (`transformer.<module>.lora_A.weight`,
  rank and alpha under `lora_adapter_metadata`). Loading also accepts the
  ComfyUI (`diffusion_model.`), PEFT (`base_model.model.`, `.default.`) and
  kohya (`lora_unet_…`, `lora_down`/`lora_up`, per-layer `.alpha`) layouts.
- Loading is all-or-nothing: a key the model has no place for, half a pair, or
  a shape that does not fit fails with a message and attaches nothing.
- Tested on a real `ZImageTransformer2DModel` with random weights: target names,
  exact no-op at init, clean removal, fp8 storage with gradients reaching B.
  Not yet tested: loading the actual training adapter file (not downloaded).

## Fit with the existing engine

- `backends/diffusion.py` gains LoRA loading (load / unload / strength).
- New `training/` package in the engine: `dataset.py`, `caption.py`,
  `cache.py`, `trainer.py`, each testable without a real model.
- Jobs go through the existing `JobQueue`; the main process GPU arbiter
  treats captioning and training as GPU owners like the image and video models.
- Tests follow the existing pattern: fakes for logic, a tiny random-weight
  model for one real end-to-end training step on the GPU (no real model
  load, no download).

## Engine API and the Train tab

Training runs in the image engine as *tasks* on the same GPU worker as image and
video jobs (`jobs.TaskRequest`), so it never competes with generation for the GPU:

| Route | What |
|---|---|
| `GET/POST /api/training/projects` | list; create from a name, a photo folder and a subject |
| `GET /api/training/subjects` | the subject presets to start from |
| `GET/DELETE /api/training/projects/<id>` | summary (subject, dataset, captions, runs); delete (photos untouched) |
| `PUT …/<id>/subject` | replace the subject definition (then scan again) |
| `POST …/<id>/scan` | task: index the folder, create fact captions |
| `POST …/<id>/caption` | task: describe the photos with the captioning model (409 when it is not installed) |
| `GET …/<id>/captions?view=all\|review\|excluded` | caption records, paged |
| `POST …/<id>/captions/<photo>` | accept / edit / exclude |
| `POST …/<id>/train` | task: cache latents and embeddings, then train (or resume a run) |
| `POST …/<id>/compare` | task: base model vs checkpoints on the same prompts and seed |
| `GET …/<id>/runs/<run>/log` | per-step loss |
| `POST …/<id>/runs/<run>/use` | copy a checkpoint into the LoRA library |
| `GET/PATCH/DELETE /api/loras[/<id>]` | the LoRA library |
| `GET /training/<id>/<photo>.webp` | a small thumbnail of an indexed photo |

- One task per project at a time (409 otherwise). Cancelling a training task
  stops it gracefully: it saves a checkpoint and the run can be resumed.
- During training the image backend lends its parts (`training_session`): the
  text encoder waits on the CPU, generation LoRAs come off, and afterwards the
  transformer is back in inference shape. The Low memory mode (CPU offload)
  cannot train.
- Image requests take `loras: [{id, strength}]` (up to 3, strength 0–2). Each
  image records the LoRAs' names, hashes and strengths in its provenance.
- Comparison images go into a gallery folder named "Training: <project>".
- The studio's main process checks every id and setting again before passing
  it on; the photo folder only ever comes from a native folder dialog.

## Collecting photos

Gathering thousands of photos by hand is the slowest part, so the Train tab can
collect openly licensed ones (`image-engine/scripts/collect_photos.py`):

- What to collect: the subject is defined first, in the same editor as a
  project's; its folder levels are the columns to fill in (one row per thing to
  collect, up to 10), and the project is made with it.
- Sources: Wikimedia Commons first (a category such as `Category:<brand>
  <model>` is the most precise; otherwise a file search, by the row's search
  words or else its names joined), then Openverse when Commons has too few.
- A Commons category for a subject holds almost no files itself: in a check of
  five car models the model categories held 0–6 files each and 19–40
  subcategories (by generation, year, colour...). Collecting therefore walks
  subcategories breadth first, three levels down and at most 150 categories,
  leaving out ones whose names say they hold logos, drawings, toys, maps,
  "in art" and the like.
- "Check what is available" asks both sources how many files a row would find
  and, from a sample of 50 (Commons) or 20 (Openverse), how many have a licence
  and size that would be kept; it downloads no photos. Openverse shows an
  anonymous client at most 240 results of any search (the collector pages
  through 200). No scraping: both are public APIs that
  state each file's licence.
- Licences kept: CC0, public domain, CC BY; CC BY-SA too when the user turns
  it on (off by default). Most Commons photos are CC BY-SA: in samples of car
  searches, 47 to 49 of the first 50 results were. ShareAlike binds only when
  adapted material is shared, and whether a LoRA or its images are adapted
  material is not settled, so a run records how many of its photos came under
  each licence (from the folder's ATTRIBUTION.json), the LoRA library keeps it,
  and the run card shows a CC BY-SA count. Non-commercial, no-derivatives and
  unknown licences are always skipped, and so are files Commons
  marks with personality rights (identifiable people). A free licence covers the
  photo, not its subject: logos and product designs can still be protected, and
  photos can show people or licence plates. A search can also find photos of
  something else that mention the same words. The UI says so.
- Credits: every photo is recorded with title, author, licence and source page
  in `ATTRIBUTION.json` and `CREDITS.md` next to the photos (what CC BY asks).
- Layout: one folder per level (`<folder>/<brand>/<model>/`), down to the first
  level left empty, plus a `labels.json` with all the names, so the folder opens
  as a training project as is; the studio
  makes the project and scans it when collecting finishes.
- Limits: photos per row (default 200, up to 1,000), a total size (1–20 GB,
  default 2 GB, checked against the drive's free space first), 8 MB per file,
  and photos fetched scaled to about 1600 px wide. Requests are sequential with
  a pause between them.
- Resumable: collecting into the same folder again keeps what is there.
- Who can see what (also shown in the Train tab, "Who can see what"):
  - Wikimedia and Openverse: this computer's internet address, the names,
    search words or category, which files are fetched and when, and the
    User-Agent `LocalAIStudio-collect/1.0 (<contact>; dataset tool for personal
    training)`. Wikimedia's User-Agent policy asks tools for a name and a way to
    contact the operator, and may block tools without one; the contact is what
    the user enters in the Train tab, or else the project's page
    (`https://github.com/justsimpleideas/ai-studio`), the same for everyone who
    runs the app. The field is optional and empty by default, so no one has to
    hand over personal details; the Train tab asks everyone who collects to
    enter their own (an email address, a website, or a Wikimedia user page such
    as `User:YourName`), since the project's page reaches the app's author, not
    the person running it. No account, cookies or referrer.
  - The hosts Openverse links to (often Flickr's image servers): the address,
    the User-Agent and which photo, not the search.
  - The internet provider, or a VPN provider: which hosts are contacted (DNS and
    the TLS server name are not hidden), when, and how much is downloaded. Not
    the searches or the photos: everything is fetched over HTTPS, and a photo
    offered only over plain HTTP is skipped ("not https").
  - Nothing about projects, files or other photos is sent.
- The engine's server still never reaches the network. The main process runs
  the script on the user's request, as it runs the model download script, and
  the window shows which hosts are contacted before anything is sent.
- Collected photos and credits stay on the user's machine; they are data, not
  part of this repository.
- Collecting is the user's responsibility: it runs from their computer, and
  they are the ones to follow Wikimedia's policies and each photo's licence,
  comply with the laws where they live, and collect sparingly. The Train tab
  says so above the collect form.
- Tested against a local stand-in for both services
  (`image-engine/tests/fake_photo_sources.py`), never the real ones.

## Video model (not built)

The same flow can train the video model; checked against the installed diffusers
source, not run. Most of it carries over (projects, scan, captions, review, the
in-house LoRA, the loop, the Train tab). What differs:

- Conventions: `WanTransformer3DModel` is trained to output `noise - x0` (the
  scheduler uses flow prediction, `x0 = sample - sigma * v`), the opposite sign
  of the image model; timesteps are `sigma * 1000`, given per token for the
  5B text/image-to-video model (`expand_timesteps`).
- VAE: 16x smaller in space and 4x in time, 48 channels, normalised with
  `latents_mean` / `latents_std` instead of a scaling factor and shift.
- LoRA targets: `to_q`, `to_k`, `to_v`, `to_out.0` of both attentions, and the
  two feed-forward projections, in every block.
- Distillation: the fast video model is a 3-step distillation, so the same trap
  applies. It is the only video model kept (the original takes tens of minutes
  per clip), so the route is to train it directly with short runs and a low
  learning rate, and to pick the last checkpoint before 3-step quality drops
  with the checkpoint comparison. Whether that holds has to be measured.
- Memory on 12 GB: a photo trained as a one-frame video is about 580 tokens and
  should fit; 33 frames at 480x832 is about 3,500 tokens (tight); five seconds
  at 720p is about 27,000 tokens, which is out of reach for training. The text
  encoder must be fully unloaded once captions are cached.
- Data: appearance can come from photos alone (the existing photo projects);
  motion needs short clips with captions that describe the movement.

Proposed order: photos as one-frame videos (code tested on a tiny random model),
then a short real run on the fast model with checkpoints compared, then clips
for motion.

## Milestones

1. **M1 Import + cache**: dataset manifest, checks, latent/embedding cache.
2. **M2 Caption + review**: vision model backend, flags, review UI, audit sample.
3. **M3 Trainer**: LoRA training loop, checkpoints, resume, tiny-model test.
4. **M4 Compare + use**: eval grid, LoRA loading in the Image tab, provenance.
5. **M5 First real run**: a small dataset end to end; record time, peak VRAM,
   and how the distilled model holds up.

## Open decisions


- Base checkpoint vs de-distillation adapter (needs a download; approval first).
- Which vision model for captioning (needs a download; approval first).

## Numbers to measure (none measured yet)

- Captioning: seconds per image, peak VRAM.
- Training: seconds per step at 768² and 1024², peak VRAM, total time for
  2,000 steps.
- Audit-sample error rate on a real dataset.
