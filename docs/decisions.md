# Decision log: Local AI Studio

Studio-level decisions. The image engine's own decisions (model choice, memory
plans, download verification) are in [image-engine-decisions.md](image-engine-decisions.md).
Newest at the bottom.

## S1. One studio for local text and image models

**Decision.** Local AI Studio merges two earlier projects into one desktop app:
a local LLM agent (chat with GGUF models, file tools behind a permission broker)
and a local image studio (text-to-image with a gallery). The two
original projects stay as they were; this one started from copies of both.

**Why.** The two halves are more useful together than apart. The chat model can
write and refine image prompts, and later call image generation as a tool. Each
alone mostly duplicates an existing tool; together they become one place to run
local models of both kinds.

## S2. Electron shell, Python image engine as a sidecar

**Decision.** The app is the LLM agent's Electron structure. The image engine stays
a Python process (PyTorch + diffusers), started and stopped by the main process.

**Why.** Each half keeps what it is good at. Rewriting the image side in Node would
lose diffusers. Rewriting the chat side in Python would lose node-llama-cpp's
grammar-constrained tool calls and the permission broker. A sidecar process also
matches how the chat engine already runs: in its own process, so a native crash
costs the model, not the app.

## S3. The renderer never talks to the image engine directly

**Decision.** All image requests go renderer → main (named IPC operations) →
engine (HTTP on 127.0.0.1). Generated images reach the window through a private
`studio-image://` protocol served by the main process.

**Why.** It keeps the trust boundary of the original app: the window is a view and
holds no authority. The engine additionally requires a token generated fresh at
every launch and passed through the environment, so other programs on the same
machine cannot drive it even though it listens on a local port.

## S4. One model on the GPU at a time

**Decision.** The chat model and the image model are never loaded on the GPU
together. Loading one unloads the other first. A switch is refused while the
other side is busy (a reply streaming, an image generating) instead of
interrupting it.

**Why.** On a 12 GB card the two do not fit side by side, and the user does not
intend to run them at the same time. A switch that waits for an explicit idle
state is predictable; a switch that silently cancels work is not.

## S5. "The chat model looks at generated images" is on hold

**Decision.** Not built for now. It is an idea, not a requirement.

**Why.** It needs a vision-capable runtime. node-llama-cpp has no image input, so
it would mean adding llama.cpp's `llama-server` with the model's separate
multimodal projector file (`mmproj-*.gguf`). That is a larger engine change plus a
binary download, recorded in [vision-support.md](vision-support.md) and left for
a later decision.

## S6. Ports

Renderer dev server: 5197. Image engine API: 5198 (loopback only). Both strict,
so a clash fails loudly. The chat engine uses no port; it talks to the main
process over Electron IPC.

## S7. A folder is a workspace, with its own agent state

**Decision.** The chat side works in an opened folder, the way a coding agent
works in a directory. Opening a folder makes it the working folder (shared
automatically, relative paths anchored to it, named in the system prompt) and
gives it its own chats, project rules, extra folders, grants and undo journal.
Global rules apply in every workspace, project rules after them. Modes and
confirm-everything stay global. All of it lives under the application's data
folder (`workspaces/<folder name>-<hash of path>/`); nothing is written into the
opened folder. A workspace is required to chat.

**Why.** Per-directory agents with their own history are the gap this project
started from: the local chat apps it was compared with have neither. Keeping grants and undo per workspace means
something allowed for one project is never in effect in another.

**How chats continue.** Each chat stores what the window showed and the
model's own chat history (node-llama-cpp `getChatHistory`). The engine holds one
conversation at a time; switching to another chat, or the first prompt after a
restart or model load, restores its history (`setChatHistory`) and the model
re-reads it, so that first reply is slower. A changed system prompt keeps the
conversation and only replaces the instructions at its head. Session builds in
the engine are serialized, so a measurement and a prompt for a chat being
switched to cannot race and start it empty. Verified with a real model
(a 9B GGUF at Q4_K_M): a fact read from a file in one chat is unknown in another,
and recalled from saved history after switching back and after a restart.

## S8. Assume the model will break its rules

**Decision.** Anything that must not happen is enforced by the broker, not asked
of the model. Four mechanisms, all independent of modes and approvals:

1. **Protected paths** per workspace, *no access* or *read-only*, matched relative
   to the shared folder (a bare name at any depth, a path with a slash from the
   top, a folder covering everything inside). Final denials: no rule or click
   lifts them, and no-access paths are left out of directory listings. Defaults
   cover secrets (no access) and `.git` / `node_modules` / virtualenvs
   (read-only), so nothing has to be set up per project.
2. **Read before change.** `write_file` over an existing file and `edit_file`
   are refused unless the file was read in this conversation and its size and
   modification time are unchanged since. Switching chats forgets what was read.
3. **Per-reply limits.** After 3 deletes or 20 writes in one reply, every further
   one is put to the user even in automatic mode.
4. **A persistent journal.** Backups before every change, grouped by the prompt
   that caused them, saved per workspace (`journal.json`). Reverting a group
   checks each file still holds what the agent left (hash), reports conflicts
   instead of overwriting, and records itself as a group so it can be undone.
5. **No reading through hard links.** Path resolution follows junctions and
   symbolic links, but a hard link is a second name for the same contents, and
   the other name may be outside every shared folder. A file with more than one
   name is refused for `read_file`, finally. Writes and deletes need no guard:
   a write is a temp file renamed over the name (the other name keeps the old
   contents) and a delete moves the name to the recycle bin.

**Why.** Local models make mistakes often, and text a model reads (a file, later
a web page) can talk it into things. A rule in the system prompt is a request;
these are properties of the only code that can touch the disk. Verified with a
real model: it edited a file only after reading it, a read of `.env` was refused
and the value never reached it, and the edit was reverted from the history.

## S9. Internet: offline by default, every request through one gate

**Decision.** Each workspace has an internet level: Offline (default), Search
only, or Search and download. There is no "everything" level. Offline means the
web tools are not offered to the model at all. Leaving Offline needs a
confirmation that spells out what changes.

Web tool calls never run in the model process; they reach `WebGate` in the main
process, the counterpart of the broker for files. Every request:

- is checked before anyone is asked: allowed at this level; query short and on
  one line; address public http(s) without credentials and with a short query
  string; host not this machine or the local network, by name, by DNS and on
  every redirect;
- is refused if it repeats six words in a row, or a long identifier, from text
  the agent read or wrote in this conversation (`LeakDetector`);
- is shown to the user, exactly as it will be sent, unless they set internet
  requests to run automatically (and a download is still asked about whenever
  the write itself would be);
- for a page or a file, is a GET with no body; for a search, carries the query
  and a result count in the chosen service's shape and nothing else; in no
  case is a header chosen by the model;
- is written to the workspace's `network-log.jsonl`, whatever its outcome.

Downloads land only as new files, never runnable types, at most 50 MB, through
the broker's path checks and the journal. Page text is returned with a note that
it comes from the web, not the user. Independently of the level, the window's
session cancels every request to a non-local address, and the chat engine is
started with `build: 'never'` so it can never fetch llama.cpp to compile.

**Why.** A request to the internet is the only action that can carry project
code off the machine. The automatic checks stop verbatim copies; a reworded
fragment can pass them (verified: a real model, refused for using a function
name, retried with the name split into ordinary words), which is why the
default is to show every request first. Search goes to one of a few search
services, chosen in Settings (`src/shared/search.ts`): two hosted APIs with the
user's own key, kept per service and encrypted with the OS store, never sent
to the window; or a SearXNG instance the user runs, whose address is the one
local address the gate will send to, because the user typed it to name their
own server and only the query goes there. The service's name and host are shown
next to every query, in the approval and in the log. Each service's request
shape and reply are described by pure functions, so the gate stays one code
path and adding a service does not touch the checks.

**Not covered.** DNS is checked before connecting, not pinned for the
connection, so a host that changes its answer in between (DNS rebinding) is not
fully excluded.

## S10. Closing the window quits, unless the user asks for the background

**Decision.** By default, closing the window quits the studio: both engines stop
and every model leaves the GPU. In the studio's Settings the user can choose to
keep running in the background instead. Then closing only hides the window; the
studio stays in the tray (open it from the icon, quit from its menu), work in
progress carries on, and for each model (chat, image, video) the user decides
whether it stays on the GPU (the default) or is unloaded. A model set to unload
goes once it is idle, never in the middle of a reply or a video
(`src/main/background.ts`), and the check repeats while the window is hidden,
because queued work can load a model again. The studio runs once per profile: a
second launch brings the existing window back.

**Why.** Quitting is the only choice that is certain to give the GPU back, which
is what most people expect from the close button, so it is the default. The
background is for long work (a video takes minutes) and for keeping a model
warm, where reloading costs a minute or more. Unloading only when idle means
closing the window can never lose a generation. An agent that needs an approval
while the window is hidden says so from the tray rather than waiting unseen.

**Not added.** Starting with Windows. The studio never registers itself to run
at sign-in; the background lasts until the user quits or signs out.

## S11. Code in the repository, everything per-user installed by the studio

**Decision.** The repository and the installer carry the studio's code and the
image engine's source, and nothing else. What differs from one machine to the
next is installed by each user, from inside the studio, into their own data
folder:

- **The engine's Python environment.** *Image → Settings → Install the engine*
  finds a Python 3.12 to 3.14 already on the machine, checks for an NVIDIA GPU,
  makes a virtual environment in the data folder, and runs pip: PyTorch's CUDA
  build alone from the PyTorch index (`--no-deps`), then the engine's
  requirements and PyTorch's dependencies from PyPI, every package pinned by
  `image-engine/constraints.txt` (the 45 audited versions) and binary wheels
  only. It ends by checking that PyTorch sees the GPU. Running it again repairs
  the same environment (`src/main/image/setup.ts`).
- **Model weights.** Each supported model has a download button. Before
  anything is fetched the studio shows the source repository and commit, the
  license (with a link to its text), the size and the destination folder, and
  waits for the user to accept the license. The download is
  `image-engine/scripts/fetch_model.py --json`, unchanged in what it guarantees:
  one pinned commit, data formats only, every file checked against its
  published hash. Stopping keeps finished and partial files, so a second click
  resumes; the two video models share 14 GB of files, which are hard-linked
  rather than downloaded twice. When it finishes the engine is pointed at the
  folder (`src/main/image/downloads.ts`).
- **Chat models** stay a file the user picks; any GGUF works.

**Why.** Hardware decides what is worth installing: a machine without an NVIDIA
GPU needs none of the 5 GB environment, and someone with little disk may want
one video model and not the other. Installing on the user's machine also means
the project redistributes neither PyTorch's CUDA libraries nor model weights:
pip and the model's own repository deliver them, under their own licenses,
which the user sees before accepting. Pinning everything keeps an install made
in a year identical to the one that was tested.

**Guarded by.** `.gitignore` covers every weight format, environment, database
and output folder, and `tests/repo-hygiene.test.ts` fails if any such file, any
binary other than the icon, any file over 1 MB, or a path into a real user's
home folder is ever part of the repository.

**Not added.** Bundling an embedded Python (the installer would grow by 5 GB
for users who only chat), a hash-locked lock file per Python version (the
pinned versions come from the official indexes, and pip checks each wheel
against the index's hash), and models other than the three the engine supports.

## S14. A model file is checked by the studio, and a model's behaviour is not trusted

**Decision.** Two things, because they answer two different questions.

The file: every GGUF goes through the studio's own strict reader
(`src/shared/gguf.ts`) before the inference library opens it, and the same
reader writes the Model check report. It trusts nothing in the header: counts
have ceilings, every length is checked against the file, tensors must be of
types it can size, fit, align and not overlap, nested arrays and unknown value
types are refused, and a tensor name the loader would truncate is refused. The
report adds what the metadata says and reads the chat template for text the
user would not see. The chat template is rendered by a JavaScript template
engine with no access to files, the network or the system, and the report shows
the rendered sample. Pickle-free formats aside, this closes the only two ways a
model *file* can act: a header built to trip the loader, and a template built to
whisper to the model.

The behaviour: not verifiable, so not relied on. Weights are numbers; a file can
be sound and hold a model trained to misbehave, and no reading of the file tells
them apart. The studio's answer is S8 and S9 — the model can only ask, and every
ask is judged elsewhere — plus **strict mode** per workspace: no tools given, none
described, none in the output grammar, and a call that arrives anyway refused in
the main process before any broker. The Model check tab says this in so many
words, with the folder's current settings and the one residual the structure
cannot catch (a reworded leak through an automatic search).

**Why.** "Is this model safe?" is asked of every downloaded file, and the honest
answer has two halves. The half that can be checked is checked here, without
Python and without loading anything. The half that cannot be checked is made not
to matter, and the user is told which is which rather than shown a green tick
that means less than it looks.

**Not added.** A behavioural sample test (fixed prompts, every tool refused,
watching for unasked-for calls): useful as a signal, but it needs the model
loaded and would read as a verdict. Left for a later decision.

## S13. Generation settings come from the model's header, not its name

**Decision.** At load the engine reads what the file says about the model
(architecture, expert layout, weight format, whether its chat template has a
thinking segment, the trained context size) and derives the defaults from that:
the context to open (trained size, capped at 32k, with the GPU layer count asked
to leave room for exactly that), the publisher's recommended sampling for the
family, and whether to offer thinking controls. The user can override each; the
overrides are plain values, never model names. Code: `src/shared/generation.ts`
(types, the sampling table, parsing), `src/engine/model-traits.ts` (header →
traits, load plan), `src/engine/thinking.ts` (thinking mode → library options).

**Why.** Before this, every model ran with the library's defaults: greedy
sampling, which makes a thinking model repeat itself; a context planned for the
trained size, which on a 262k-token model pushed most of the weights off the
GPU; and the chain of thought hidden, so a long think looked like a hang. Fixing
those per model would have meant a table of names that is out of date on the
next release. Reading the header makes the same code right for a model that
does not exist yet.

**Not added here.** Placing a mixture-of-experts model's experts in system
memory while the rest stays on the GPU: node-llama-cpp (3.22.2 checked) offers
only a layer count. Added as a second chat engine instead (S15).

## S15. llama-server as a second chat engine, beside the built-in one

**Decision.** The model menu has an Engine choice: Built-in (node-llama-cpp) or
llama-server (the default, after the measurement in S16). With llama-server the studio starts a llama.cpp server
the user installed and talks to it the way the image engine is talked to:
HTTP on 127.0.0.1, a port chosen as free, a key made for that start. A separate
engine host (`src/engine/server/host.ts`) speaks the same messages as the
built-in one, so the main process, the broker and the window are unchanged.

**Why.** A mixture-of-experts model bigger than the GPU, split by layer, runs
most of its work on the CPU: measured at 4.5 tokens/s for a 35B model with 3B
active on a consumer GPU. llama.cpp's `--cpu-moe` / `--n-cpu-moe` keep only
the experts in system memory, which is what such a model is built for. The
stock library cannot ask for it; the server can. (Since S16 the built-in engine
can too, on CUDA.)

**How the boundary holds.**
- The binary is the user's, chosen through a dialog; the studio never downloads
  or updates it. Its path is set by the main process, never named by the window.
- Loopback only (`--host 127.0.0.1`), its web page off (`--no-webui`), the key in
  `LLAMA_API_KEY` rather than on the command line, and every `LLAMA_*` / `HF_*`
  variable removed from its environment, since llama-server reads its options
  from those too and some of them download a model.
- Tool calls come back as data and go to the same validator and broker as the
  built-in engine's. The server's tool grammar is a convenience, not the guard.
- The engine host reports the server's process id to the main process, which
  ends it on quit and whenever the host itself ends, so a server is not left
  holding the GPU.
- The saved chat stays in the library's history format whichever engine wrote
  it; the server host converts to and from the chat-completions shape.

**Checked against a real server (llama.cpp b11669), no model loaded.** Every
flag the studio passes is accepted (a launch with a missing model file stops
only at "failed to load model"); `--offline`, `--no-cors-credentials` and
`--no-slots` were added after reading its help, `--n-gpu-layers all` replaced a
large number, and the per-request thinking cap is `reasoning_budget_tokens`, as
the server's source reads it. Its load log does not name the GPU at the default
verbosity, so the device comes from `--list-devices`, run before the server.

**Measured (2026-10).** The same 35B-A3B model on a consumer
GPU, thinking off, 8-bit context memory: 4.5 tokens/s with the built-in
engine (13 of its layers on the GPU, the rest split off to the CPU), 34 tokens/s
through llama-server with every layer's experts on the CPU and 8 CPU threads
(up to 38 on short replies). About 7.5 times faster.

## S16. The built-in engine keeps experts on the CPU, through a rebuilt addon

**Decision.** The studio rebuilds node-llama-cpp's CUDA addon
(`llama-addon.node`, about 1 MB) with one added load option,
`tensorBufferOverrides`, which fills llama.cpp's
`llama_model_params.tensor_buft_overrides`: the field llama-server's
`--cpu-moe` / `--n-cpu-moe` set. The built-in engine uses it for a
mixture-of-experts model on CUDA, with the same *Experts on CPU* and *CPU
threads* settings as the server. llama-server stays as an alternative engine
(S15), no longer the only way to get that speed.

**Why not more.** Writing an inference engine of its own would mean
re-implementing what makes llama.cpp fast (quantized CUDA kernels, flash
attention, the expert routing) to end up slower. The speed gap in S15 was never
the library's engine, which is llama.cpp too; it was one option the library does
not pass through.

**How it is built** (`scripts/native/`).
- Only the addon is compiled. It links against the prebuilt llama.cpp DLLs
  already in `node_modules` (import libraries are generated from their export
  tables), and those DLLs, the CUDA backend among them, are left byte for byte
  as shipped. So the build needs MSVC and git, nothing from NVIDIA, and
  downloads nothing: the llama.cpp headers come from the source bundle
  node-llama-cpp ships, at the commit its binaries were built from.
- The edits are exact text replacements at known anchors (`addon-patch.mjs`);
  an anchor found zero times or twice fails the build. The script also refuses
  any node-llama-cpp release but the one the patch was written for, and checks
  the new addon imports the same DLLs as the stock one.
- The patched addon reports a feature (`getStudioFeatures()`), and the engine
  passes the option only when it is there. The stock addon would silently drop
  the option and put every expert on the GPU; instead, without the feature (or
  on Vulkan), the model menu says why and the model is split by layer.
- With the experts placed, every layer goes to the GPU and the library's own
  memory checks are skipped, since they size whole layers and cannot plan for a
  split by tensor. A model whose remaining weights do not fit fails in
  llama.cpp's allocation with an error.

**Checked with a made-up model** (`tiny-moe-gguf.mjs`: two llama-architecture
layers of four experts, fixed-seed weights, about 1 MB; `addon-smoke.mjs`), on
CUDA. The rebuilt addon without the option gives the same buffer sizes and the
same 24 greedy tokens as the stock addon; with it, the six expert tensors go to
system memory (GPU buffer 0.91 MiB to 0.16 MiB) and the tokens are unchanged.
**Measured with a real model (2026-10).** The same 35B-A3B model as in S15, its
experts on the CPU through the rebuilt addon: 5.3 tokens/s with the built-in
engine, against llama-server's 34. Why the rest of the gap remains (memory
mapping of the CPU-side experts, threads, the library's evaluation loop) was not
looked into. The placement alone does not close it, so llama-server became the default
engine, with every layer's experts on the CPU, 8 CPU threads, a 32k context,
thinking off, recommended sampling and 8-bit context memory as the defaults
around it. The built-in placement stays, as the choice that needs nothing
installed.

## S17. Conversations are encrypted on disk

**Decision.** Each chat file (`chats/<id>.json`) and each workspace's chat list
(`chats.json`, whose titles are a chat's first words) is written sealed:
AES-256-GCM, a fresh 12-byte nonce per write, and the file's place in the store
(`<workspace>/chats/<id>.json`) as additional data, so a sealed chat copied over
another one does not open. One random 32-byte data key, made on first use, is
kept in `chat-key.json`, encrypted with the system store through Electron's
safeStorage, as the search keys already are (`secrets.ts`). On Windows that is
not DPAPI on each call: Chromium encrypts under a key of its own, which DPAPI
protects in the profile's `Local State` file.

**How it fails.**
- A key file that is there but does not open (another account, a reinstalled
  system) is never replaced: replacing it would make every existing chat
  unreadable for good. Chats are then neither read nor saved, and saving says why.
- A chat file that does not open (changed, or moved from elsewhere) counts as
  absent, as an unreadable file always has.
- With no system store at all (some Linux desktops), files are written plain, as
  before; encrypting under a key kept next to them would only look safe.
- Plain files from before are sealed at startup and whenever one is read.
- `Local State` is written lazily. A first run killed within seconds was seen
  to leave none, and a data key sealed under a system key that never reached
  the disk would be lost at the next start with every chat. So the data key is
  made only once `Local State` holds the system key; until then files are
  written as before, and sealed at the next start or a minute later. Checked
  with the packaged app: first run killed at once (left plain, nothing lost),
  normal quit, next start sealed everything, still readable after a kill.

**What stays plain.** Workspace settings, the change journal and file backups
(copies of the user's own files, which are plain where they came from), the
internet log, and the engine log, which holds no conversation text.

**Cost.** A copy of the data folder opens its chats only for the same Windows
account on the same machine; the README says so.

## S18. The data folders are moved by pointing, not by copying

**Decision.** Two folders can be chosen from Settings → Where things are kept:
the chats-and-settings folder (`userData`) and the gallery (`image/outputs`).
Choosing one only changes where new data is written. Nothing is moved, copied
or deleted, so the folder left behind keeps what it held and choosing it again
brings it back into view.

**Why pointing.** Moving a live store is the risky half of the feature and the
half nobody asked for: a copy interrupted part-way leaves two half-folders and
a chat key separated from the chats it opens. Pointing has one failure mode,
and it is visible and reversible — the new folder looks empty, because it is.
A gallery that was somewhere else can still be brought in deliberately, with
the merge that already exists (`gallery_merge.py`), and chats can be moved by
hand with the studio closed, then pointed at.

**Where the setting lives.** Not in `config.json`, which sits inside the folder
being named: `data-location.json`, next to the default folders in the
application data folder. Absent or unparsable means the defaults, so a studio
whose pointer file is edited by hand still starts. A folder equal to the
default is stored as no choice at all, which keeps a copied profile working
wherever it lands.

**When each takes effect.** `ConfigStore`, `WorkspaceStore`, `SecretStore` and
the sealer are opened once, at startup, and the running studio holds the old
folder’s chat key; so the chats folder changes on the next start, and the
window offers to restart (`app.relaunch()`). The gallery is held by the engine
(its SQLite handle and the images it serves over `image://`), so the engine is
stopped and started again, which is how the Python interpreter setting already
works. The folder is passed as `--outputs-dir`, read at every engine start, so
there is no second copy of the setting inside the engine.

**Refused.** The two folders may not be the same or nested in one another, and
neither may sit in the program folder, which an update or an uninstall empties.
Each folder is created and written to before the choice is kept, so a path that
cannot be used fails while the studio is still on the old one.

## S19. A queued job is edited in place; the running one is restarted

**Decision.** The queue window changes a waiting job's request where it stands: same
job, same place in line. The job being made is never changed mid-run. Editing it
cancels it at its next step and queues a changed copy first in line, which keeps the
original's `created` time so every list orders it where it runs. Both go through
the same checks as a new request (`PUT /api/jobs/<id>`), and a job that finished or
is already stopping is refused with 409 rather than silently replaced.

**Why.** A Cancel button beside a queue reads as "cancel what?", so the card says
*Cancel this one* and the rest of the queue gets its own window. Swapping a request
under a running sampler would make an image from half of each prompt; a restart
is honest about what happens, and whatever the job already finished is kept.

## S20. The gallery reads "the first N items", never a page in isolation

**Decision.** The gallery keeps how many items it wants on screen. Load more raises
that number and reads only what is missing; any change (a new image, a move, a
deletion) reads the list again from the top as far down as it was. Only the newest
read counts, and switching views empties the grid at once instead of leaving the
old view's items under the new heading.

**Why.** Reloading the first page after every change threw away pages brought in
with Load more, and a reload that started while Load more was waiting made the
extra page never appear. With a slow engine (busy loading a model, say) both looked
like a frozen gallery, as did a folder showing All's items until its answer came.

## S12. Judge-feedback training is on hold

**Decision.** Not built for now. If it is built, the judge is the local vision
model by default, with an external vision API as an opt-in option per run.

**Why.** The idea (generate, let a vision model check each image against its
prompt, train on the verdicts) is promising for small consistent errors, but it
is a large addition on top of a LoRA trainer that has not had a real-model run
yet, and no image model is installed at the time of the decision. The analysis,
methods and limits are recorded in
[judge-feedback-training.md](judge-feedback-training.md).
