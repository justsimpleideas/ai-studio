# Vision model support (image to text): investigation

Status: **on hold by decision** (see S5 in [decisions.md](decisions.md)). No code has
been written for this yet.

## Hold decision (Local AI Studio)

With the image engine merged in, the natural use of vision is letting the chat model
look at images the studio generated, describe them, and suggest prompt changes. That is
kept as an idea, not a requirement, and nothing below is being built for now. The
options and open questions are left as they were so the decision can be picked up
later without redoing the investigation.

### Is `llama-server` still local?

Yes. "Server" describes how it is talked to, not where it runs. `llama-server` is a
program from llama.cpp that runs on this machine, loads the model from this machine's
disk, and answers HTTP requests. Bound to `127.0.0.1`, those requests never leave the
computer: loopback traffic does not touch the network card. The studio's image engine
already works the same way.

What would change is narrower:

- **One-time download.** The `llama-server` binary comes from llama.cpp's GitHub
  releases. It would be pinned to a release and verified by SHA-256, like the image
  model's files.
- **A local port.** Any program on the same computer can try to connect to a loopback
  port. The answer is the same as for the image engine: a random token generated at
  every launch, required on every request.
- **Wording.** The README's "no network call" claim would become "nothing leaves the
  computer; the engines talk over loopback".

### It is a new dependency, though

llama.cpp itself is already a dependency: node-llama-cpp ships prebuilt llama.cpp
binaries (CUDA, Vulkan and CPU builds) as npm packages, so npm pins them, checks their
integrity hashes and installs them. `llama-server` would be the same engine in a
different form, outside npm: a separate executable whose version pin, SHA-256 check,
updates and inclusion in the installer the project would have to manage itself. End
users would not install anything extra if it ships inside the installer; developers
would run a fetch script, as they already do for the image model. The image engine's
Python environment is the same kind of dependency, and a larger one.

Rechecked 2026-10-24: node-llama-cpp is now at 3.22.2, and its documentation still
mentions no image input. (Release notes could not be read at the time; worth
rechecking before deciding.)

### Option D: vision in the Python image engine

The image engine already has PyTorch and transformers, which can run vision-language
models directly, with no new runtime binary. Trade-offs:

- Different model files: safetensors from Hugging Face, not the GGUF + mmproj pair.
- More VRAM and slower than a quantized llama.cpp model of the same size, unless a
  quantization library is added (another dependency).
- It fits the one-model-on-the-GPU rule as a third model the arbiter switches to.
- Tool calls and token usage stay on the chat engine; the vision model would only
  describe or critique images and hand text back, closer to option B.

## The request

Support models with vision, so a user can attach an image and ask about it (image to
text). The 9B chat model used in testing is one example: it can take images when it is paired with its
multimodal projector file (`mmproj-*.gguf`).

## Findings

### node-llama-cpp cannot do this

- The version in use, **3.21.1**, is also the latest published release (checked against
  the npm registry, 2026-09-28), so upgrading does not help.
- Its public API has no image input: no `mmproj` option and no image message type in
  the chat history types.
- The prebuilt native binaries it ships (`@node-llama-cpp/win-x64*`) do not include
  llama.cpp's multimodal library. They export no `mtmd_*` or `clip_*` functions. The only
  `mmproj` strings in them are llama.cpp's command-line flag names (`--mmproj`,
  `--mmproj-offload`, …), which come from its shared argument parser. They are not
  working code.

It cannot be switched on by configuration. Supporting vision through node-llama-cpp
would mean writing and compiling a native binding for `mtmd` ourselves, which rules out
the prebuilt, no-compiler install this project relies on.

### llama.cpp itself does support it

llama.cpp's own **`llama-server`** loads a model together with an `--mmproj` projector
and accepts images through its HTTP chat-completions endpoint (base64 `image_url`
parts). It is published as prebuilt Windows binaries (CPU, Vulkan, CUDA) on the
official ggml-org/llama.cpp GitHub releases.

So any route to vision means adding llama.cpp's own binary alongside the current
engine or in place of it.

## Options

| | A. Switch the engine to llama-server | B. Hybrid: image → text pass | C. Both backends |
|---|---|---|---|
| What it means | One `llama-server` child process does text, images and tool calls | Chat and tools stay on node-llama-cpp; llama-server only turns an attached image into a description or OCR text, which goes into the chat | node-llama-cpp for text-only models, llama-server when a model has an mmproj |
| Model sees the image | Yes; follow-up questions about it work | No, only the text description | Yes, on the llama-server path |
| VRAM | One model load | Two loads of the same 9B model do not fit in 12 GB, so it needs a separate small vision model or a slow CPU pass | One model load at a time |
| Tool calls | llama-server's own grammar for tool calls (`--jinja`), replacing node-llama-cpp's function grammar | Unchanged | Two implementations |
| Token usage analyzer | Ported to `/apply-template` + `/tokenize` | Unchanged | Two implementations |
| Size of change | Large rewrite of `src/engine/` | Moderate | Largest; two engines to maintain and test |

Things that change with A or C:

- **The security story.** `llama-server` listens on a TCP port. It has to be bound to
  `127.0.0.1` with a random per-launch API key, and started and stopped by the app. The
  README's "no network call in the inference path" becomes "loopback only", and the
  README would need to say so.
- **Distribution.** The binary is not an npm dependency. It needs a fetch script that
  pins a release and its SHA-256 (the binary is never committed), and electron-builder
  has to ship it as an extra resource.
- **Backend choice.** Vulkan works on any modern GPU, including cards newer than the
  CUDA prebuilts cover (see Hardware notes in the README). CUDA is usually faster on
  NVIDIA cards that the prebuilt covers, but it is a larger download.

The recommendation was **A**: it is the only option where the model actually looks at
the image, and it loads the model only once.

## Open decisions

1. Which option: A, B or C.
2. Whether a llama.cpp release binary may be downloaded for development, and which
   build (Vulkan or CUDA). Nothing has been downloaded.
3. A complete main model and its mmproj file are needed to verify anything for real.
   Neither is available on the development machine at the moment.
