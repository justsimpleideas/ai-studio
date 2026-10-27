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
- is a GET with no body and no headers chosen by the model;
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
default is to show every request first. Search goes to one search API
(`SEARCH_HOST` in `src/shared/network.ts`, shown to the user next to the key and
in every approval) with the user's own key, encrypted with the OS store and never
sent to the window.

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
