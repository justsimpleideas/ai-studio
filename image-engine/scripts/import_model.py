"""Put a model published as a single transformer file together into a folder the engine loads.

Two image models and the video model are shared this way, as one .safetensors file
holding only the transformer (fp8 or bf16), in their authors' layout or in diffusers':

  * the 6B image model: the rest of its folder (text encoder, tokenizer, VAE, scheduler
    and the transformer's config.json) comes from the downloaded 6B model
    (`fetch_model.py --model image`, or just those parts with `--model image-parts`);
  * the 12B image model: its text encoder is the studio's captioning model
    (`fetch_model.py --model caption`) and its VAE comes from `--model large-vae`;
  * the 5B video model: like the 6B one, the rest comes from the downloaded video model
    (`fetch_model.py --model video`, or just those parts with `--model video-parts`);
  * a 14B video model (text to video, or image to video): two files, one per expert
    (`--transformer` the first, high-noise one; `--transformer-2` the second, low-noise
    one), the rest from `--model video-14b-text-parts` or `--model video-14b-image-parts`.
    Published 14B files often use an older fp8 layout (`scale_weight`, `scale_input`, a
    `scaled_fp8` marker); single_file reads it.

The studio runs this from Settings → Image model (or Video model) → Import a model
file…; it works the same from a terminal. Files are hard links where the disk allows
(no extra space), copies otherwise:

    <dest>/model_index.json, transformer/<the file>, text_encoder/, tokenizer/, vae/, ..., manifest.json

What it guarantees once it exits with status 0:
  * the transformer file is well-formed safetensors (no pickle, tensors tile the data
    exactly, no duplicate names), every tensor the model needs is there with the right
    shape and nothing else is, and its fp8 scales and quantization tags are of a kind
    the engine understands (image_engine/single_file.py);
  * its SHA-256 is recorded, and matches --sha256 when given (copy it from the file's
    page on the hub to be sure the download is the published file);
  * the other parts come from folders fetch_model.py verified, at the pinned commits,
    and their files still have the recorded sizes;
  * the folder holds those files and nothing else.

Standard library only, plus the engine's own pure-Python header reader.

Usage:
  python scripts/import_model.py --transformer <file> --probe
  python scripts/import_model.py --transformer <file> --variant distilled \
      --parts-from /path/to/models/image-model --dest /path/to/models/my-6b-model [--sha256 <hex>]
  python scripts/import_model.py --transformer <file> --variant base \
      --text-encoder-from /path/to/models/caption-model --vae-from /path/to/models/large-vae \
      --dest /path/to/models/my-12b-model [--sha256 <hex>]
  python scripts/import_model.py --transformer <file> --variant base \
      --parts-from /path/to/models/video-parts --dest /path/to/models/my-video-model [--sha256 <hex>]
  python scripts/import_model.py --dest /path/to/models/my-12b-model --verify-only

  python scripts/import_model.py --transformer <high noise> --transformer-2 <low noise> --variant distilled \
      --steps 4 --shift 5 --parts-from /path/to/models/video-14b-text-parts --dest /path/to/models/my-14b-model

For the 5B video model, --variant distilled means the 3-step distillation the studio runs
on a fixed schedule (the same as `fetch_model.py --model video-fast`); base means the
original 50-step sampling with guidance. For a 14B model, distilled means a few-step
distillation run with plain Euler steps on a shifted schedule, with the --steps and
--shift its download page gives; base means the original 40 steps with guidance.

With --json, progress goes to stdout as one JSON object per line (the studio reads it)
and the human-readable messages go to stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(0, str(ENGINE_ROOT / "scripts"))

import fetch_model  # noqa: E402
from image_engine import single_file, video_models  # noqa: E402

LARGE_PIPELINE = "Krea2Pipeline"
# What the 12B model takes from each source folder, and where it goes.
TEXT_ENCODER_FILES = {
    "config.json": "text_encoder/config.json",
    "model.safetensors.index.json": "text_encoder/model.safetensors.index.json",
    "model-00001-of-00002.safetensors": "text_encoder/model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors": "text_encoder/model-00002-of-00002.safetensors",
    "tokenizer.json": "tokenizer/tokenizer.json",
    "tokenizer_config.json": "tokenizer/tokenizer_config.json",
    "vocab.json": "tokenizer/vocab.json",
    "merges.txt": "tokenizer/merges.txt",
}
VAE_FILES = {
    "vae/config.json": "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors": "vae/diffusion_pytorch_model.safetensors",
}
# What the 6B model takes from its downloaded folder: everything but the transformer's weights
# (the same list `fetch_model.py --model image-parts` downloads, minus model_index.json, which
# is rewritten here).
SIX_B_FILES = {name: name for name in fetch_model.MODEL_FILES["image-parts"] if name != "model_index.json"}
# The same for the video model (`fetch_model.py --model video-parts`).
VIDEO_FILES = {name: name for name in fetch_model.MODEL_FILES["video-parts"] if name != "model_index.json"}
# model_index.json `_class_name` of the video model's folders: the original, and the label
# video_models.py reads as the 3-step distilled schedule.
VIDEO_PIPELINE = "WanPipeline"
VIDEO_PIPELINE_DISTILLED = "WanDMDPipeline"
# Each video model the studio has parts for (single_file.VIDEO_KINDS): where its parts come
# from (the fetch_model.py download of them), and the pipeline its folder names.
VIDEO_SOURCES = {
    "5b": (fetch_model.VIDEO_REPO, fetch_model.VIDEO_REVISION, "video-parts", VIDEO_PIPELINE),
    "14b-text": (fetch_model.VIDEO_14B_TEXT_REPO, fetch_model.VIDEO_14B_TEXT_REVISION, "video-14b-text-parts",
                 "WanPipeline"),
    "14b-image": (fetch_model.VIDEO_14B_IMAGE_REPO, fetch_model.VIDEO_14B_IMAGE_REVISION, "video-14b-image-parts",
                  "WanImageToVideoPipeline"),
}
# The text encoder layer the 12B pipeline reads deepest (its taps are 2, 5, ..., 35).
DEEPEST_TAP = 35
CHUNK = 8 * 1024 * 1024
PROGRESS_SECONDS = 0.5

_json = False


class ImportFailed(Exception):
    pass


def say(message: str) -> None:
    print(message, file=sys.stderr if _json else sys.stdout, flush=True)


def emit(event: str, **fields: object) -> None:
    if _json:
        print(json.dumps({"event": event, **fields}), flush=True)


def sha256_of(path: Path, label: str, report: bool = False) -> str:
    size = path.stat().st_size
    digest, done, reported, last = hashlib.sha256(), 0, 0, 0.0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
            done += len(chunk)
            if report and time.monotonic() - last >= PROGRESS_SECONDS:
                emit("progress", stage="hashing", done=done, size=size)
                last = time.monotonic()
            if done - reported >= 2 * 1024**3:
                say(f"    hashing {label}: {done / 1e9:.1f}/{size / 1e9:.1f} GB")
                reported = done
    if report:
        emit("progress", stage="hashing", done=size, size=size)
    return digest.hexdigest()


def verified_source(folder: Path, repo: str, revision: str, wanted: dict[str, str], what: str) -> dict[str, dict]:
    """The manifest entries of `wanted` files in a folder fetch_model.py verified at repo@revision."""
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ImportFailed(f"{folder} has no readable manifest.json; fetch the {what} with fetch_model.py first") from exc
    if (manifest.get("repo"), manifest.get("revision")) != (repo, revision):
        raise ImportFailed(f"{folder} holds {manifest.get('repo')}@{manifest.get('revision')}, "
                           f"not the {what} this model needs ({repo}@{revision})")
    entries = {e["path"]: e for e in manifest.get("files", []) if isinstance(e, dict) and "path" in e}
    out = {}
    for name in wanted:
        entry = entries.get(name)
        if entry is None:
            raise ImportFailed(f"{folder}: {name} is not in its manifest")
        path = folder / name
        if not path.is_file() or path.stat().st_size != entry.get("size"):
            raise ImportFailed(f"{folder}: {name} is missing or changed since it was verified; "
                               "run fetch_model.py --verify-only on that folder")
        out[name] = entry
    return out


def place(source: Path, target: Path) -> str:
    """Hard-link (else copy) `source` to `target`. An identical file already there is kept."""
    if target.exists():
        if target.stat().st_size == source.stat().st_size and (
                os.path.samefile(source, target) or sha256_of(source, source.name) == sha256_of(target, target.name)):
            return "kept"
        raise ImportFailed(f"{target} already exists and is a different file")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "linked"
    except OSError:
        shutil.copyfile(source, target)
        return "copied"


def write_json(path: Path, data: dict) -> dict:
    text = json.dumps(data, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return {"size": len(text.encode("utf-8")), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def inspect_transformer(path: Path) -> single_file.Inspection:
    header = single_file.read_header(path)
    inspection = single_file.inspect(path, header)
    s = inspection.summary()
    say(f"  model     the {s['label']}: {s['parameters'] / 1e9:.2f}B parameters, {s['layers']} blocks, width {s['hidden_size']}")
    say(f"  storage   {s['storage']} ({s['fp8_weights']} fp8 linear weights; tensors by dtype: {s['dtypes']})")
    if inspection.tags:
        say(f"  tags      {len(inspection.tags)} quantization tags, all of a known kind")
    if header.metadata:
        sizes = ", ".join(f"{k} ({len(v)} chars)" for k, v in sorted(header.metadata.items()))
        say(f"  metadata  {sizes}")
    if inspection.family == "video":
        single_file.video_kind(inspection.config)  # a size the studio has parts for, before any is downloaded
    return inspection


def sources_for(args: argparse.Namespace, inspection: single_file.Inspection,
                second: single_file.Inspection | None = None) -> tuple[list[tuple[Path, dict[str, str], dict]], dict, dict]:
    """(folder, files wanted -> target, manifest entries) per source; the model_index.json to write; the sources to record.
    `second`: a 14B video model's second expert."""
    if inspection.family == "large":
        if args.text_encoder_from is None or args.vae_from is None:
            raise ImportFailed("The 12B model needs --text-encoder-from (the captioning model) and --vae-from")
        text = verified_source(args.text_encoder_from, fetch_model.CAPTION_REPO, fetch_model.CAPTION_REVISION,
                               TEXT_ENCODER_FILES, "text encoder (--model caption)")
        vae = verified_source(args.vae_from, fetch_model.LARGE_VAE_REPO, fetch_model.LARGE_VAE_REVISION,
                              VAE_FILES, "VAE (--model large-vae)")
        config = json.loads((args.text_encoder_from / "config.json").read_text(encoding="utf-8")).get("text_config", {})
        if config.get("hidden_size") != inspection.config.text_hidden_dim or config.get("num_hidden_layers", 0) <= DEEPEST_TAP:
            raise ImportFailed("The text encoder's shape does not match what the transformer expects")
        index = {"_class_name": LARGE_PIPELINE}
        record = {"text_encoder": f"{fetch_model.CAPTION_REPO}@{fetch_model.CAPTION_REVISION}",
                  "vae": f"{fetch_model.LARGE_VAE_REPO}@{fetch_model.LARGE_VAE_REVISION}"}
        return [(args.text_encoder_from, TEXT_ENCODER_FILES, text), (args.vae_from, VAE_FILES, vae)], index, record

    if inspection.family == "video":
        kind = single_file.video_kind(inspection.config)
        repo, revision, parts_id, pipeline = VIDEO_SOURCES[kind]
        two = kind != "5b"
        if two != (second is not None):
            raise ImportFailed("A 14B video model is two files, one per expert: give the second with --transformer-2"
                               if two else "The 5B video model is one file; --transformer-2 is for the 14B models")
        if second is not None and second.video_kind != kind:
            raise ImportFailed(f"The two files are not experts of the same model: {inspection.label}, {second.label}")
        if args.parts_from is None:
            raise ImportFailed(f"The video model needs --parts-from (its downloaded parts, --model {parts_id})")
        files = {name: name for name in fetch_model.MODEL_FILES[parts_id] if name != "model_index.json"}
        parts = verified_source(args.parts_from, repo, revision, {**files, "model_index.json": "model_index.json"},
                                f"video model's parts (--model {parts_id})")
        try:
            index = json.loads((args.parts_from / "model_index.json").read_text(encoding="utf-8"))
            for sub, which in (("transformer", inspection), ("transformer_2", second)):
                if which is not None:
                    config = json.loads((args.parts_from / sub / "config.json").read_text(encoding="utf-8"))
                    single_file.check_video_config(config, which.config)
        except (OSError, ValueError) as exc:
            raise ImportFailed(f"Cannot read the video model's configuration in {args.parts_from}") from exc
        if index.get("_class_name") != pipeline or (index.get("boundary_ratio") is not None) != two:
            raise ImportFailed(f"{args.parts_from} is not the video model's folder")
        if args.variant == "distilled" and not two:
            index = {**index, "_class_name": VIDEO_PIPELINE_DISTILLED}
        elif args.variant == "distilled":
            if args.steps is None or args.shift is None:
                raise ImportFailed("A distilled 14B model needs the --steps and --shift its download page gives")
            setting = {"steps": args.steps, "shift": args.shift}
            try:
                video_models.check_few_step(setting)
            except ValueError as exc:
                raise ImportFailed(str(exc)) from exc
            index = {**index, video_models.FEW_STEP_KEY: setting}
        del parts["model_index.json"]
        return [(args.parts_from, files, parts)], index, {"parts": f"{repo}@{revision}"}

    if args.parts_from is None:
        raise ImportFailed("The 6B model needs --parts-from (the downloaded 6B model, or its parts)")
    wanted = {**SIX_B_FILES, "model_index.json": "model_index.json"}
    parts = verified_source(args.parts_from, fetch_model.DEFAULT_REPO, fetch_model.DEFAULT_REVISION, wanted,
                            "6B model's parts (--model image-parts)")
    try:
        config = json.loads((args.parts_from / "transformer" / "config.json").read_text(encoding="utf-8"))
        index = json.loads((args.parts_from / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ImportFailed(f"Cannot read the 6B model's configuration in {args.parts_from}") from exc
    single_file.check_six_b_config(config, inspection.config)
    if index.get("_class_name") != "ZImagePipeline":
        raise ImportFailed(f"{args.parts_from} is not the 6B model's folder")
    del parts["model_index.json"]
    record = {"parts": f"{fetch_model.DEFAULT_REPO}@{fetch_model.DEFAULT_REVISION}"}
    return [(args.parts_from, SIX_B_FILES, parts)], index, record


def run_probe(path: Path) -> int:
    """What the file is, from its header alone."""
    if path.suffix.lower() != ".safetensors":
        raise ImportFailed("Only .safetensors files can be imported")
    say(f"Checking {path.name}")
    inspection = inspect_transformer(path)
    emit("probe", file=path.name, size=path.stat().st_size, **inspection.summary())
    return 0


def run_import(args: argparse.Namespace) -> int:
    source: Path = args.transformer
    dest: Path = args.dest
    # (folder in the model, file, the published hash to check it against): one transformer, or a 14B model's two experts.
    experts = [("transformer", source, args.sha256)]
    if args.transformer_2 is not None:
        experts.append(("transformer_2", args.transformer_2, args.sha256_2))
    for _, path, _ in experts:
        if path.suffix.lower() != ".safetensors":
            raise ImportFailed("Only .safetensors files can be imported")
    if len(experts) == 2 and os.path.samefile(source, args.transformer_2):
        raise ImportFailed("The two experts must be two different files")
    inspections = []
    for _, path, _ in experts:
        say(f"Checking {path.name}")
        emit("progress", stage="checking", done=0, size=path.stat().st_size)
        inspections.append(inspect_transformer(path))
    inspection = inspections[0]
    if len(experts) == 2 and inspection.family != "video":
        raise ImportFailed(f"--transformer-2 is for the 14B video models' second expert, not the {inspection.label}")

    say("Checking the other parts")
    sources, index, record = sources_for(args, inspection, inspections[1] if len(experts) == 2 else None)

    digests = []
    for _, path, published in experts:
        say(f"Hashing {path.name} ({path.stat().st_size / 1e9:.1f} GB)")
        digest = sha256_of(path, path.name, report=True)
        say(f"  sha256    {digest}")
        if published and published.lower() != digest:
            raise ImportFailed(f"SHA-256 mismatch: {path.name} is {digest}, the published hash is {published.lower()}")
        digests.append(digest)
    digest = digests[0]

    files: list[dict] = []
    say(f"Assembling {dest}")
    emit("progress", stage="assembling", done=0, size=1)
    for (folder, path, _), file_digest in zip(experts, digests):
        how = place(path, dest / folder / path.name)
        say(f"  {how:<7} {folder}/{path.name}")
        files.append({"path": f"{folder}/{path.name}", "size": path.stat().st_size, "sha256": file_digest})
    for folder, wanted, entries in sources:
        for name, target in wanted.items():
            how = place(folder / name, dest / target)
            say(f"  {how:<7} {target}")
            entry = {k: v for k, v in entries[name].items() if k in ("size", "sha256", "git_sha1")}
            files.append({"path": target, **entry})

    index = {**index, "is_distilled": args.variant == "distilled", "transformer_file": source.name}
    if len(experts) == 2:
        index["transformer_2_file"] = args.transformer_2.name
    files.append({"path": "model_index.json", **write_json(dest / "model_index.json", index)})

    expected = {str(Path(f["path"])) for f in files} | {"manifest.json"}
    extras = [str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file() and str(p.relative_to(dest)) not in expected]
    if extras:
        raise ImportFailed(f"Files the import did not put there are in {dest}: {extras}")

    manifest = {
        "repo": f"local/{source.name}",
        "revision": digest,
        "family": inspection.family,
        "variant": args.variant,
        "sources": {**{folder: {"file": path.name, "sha256": file_digest, "published_sha256_checked": bool(published)}
                       for (folder, path, published), file_digest in zip(experts, digests)},
                    **record},
        "transformer": inspection.summary(),
        "files": files,
    }
    if len(experts) == 2:
        manifest["transformer_2"] = inspections[1].summary()
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    kind = "video" if inspection.family == "video" else "image"
    say(f"Done: {len(files)} files. Choose {dest} as the {kind} model folder in Settings.")
    emit("done", dest=str(dest), family=inspection.family, variant=args.variant, sha256=digest)
    return 0


def run_verify(dest: Path) -> int:
    """Hash every file of an assembled folder again and re-check the transformer's structure."""
    try:
        manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
        files = manifest["files"]
    except (OSError, ValueError, KeyError) as exc:
        raise ImportFailed(f"{dest} has no readable manifest.json") from exc
    for entry in files:
        path = dest / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ImportFailed(f"{entry['path']}: missing or changed size")
        if "sha256" in entry:
            ok = sha256_of(path, entry["path"]) == entry["sha256"]
        else:
            ok = fetch_model.file_digests(path)[1] == entry["git_sha1"]
        if not ok:
            raise ImportFailed(f"{entry['path']}: hash mismatch")
        say(f"  ok      {entry['path']}")
    for name in [e["path"] for e in files
                 if e["path"].startswith(("transformer/", "transformer_2/")) and e["path"].endswith(".safetensors")]:
        say(f"Checking {name}")
        inspect_transformer(dest / name)
    say(f"Verified {len(files)} files.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, help="the model folder to create (or check)")
    ap.add_argument("--transformer", type=Path, help="the transformer .safetensors file")
    ap.add_argument("--probe", action="store_true", help="only say what the file is, from its header")
    ap.add_argument("--variant", choices=("base", "distilled"),
                    help="distilled: few steps, no guidance; base: more steps, with guidance")
    ap.add_argument("--parts-from", type=Path, help="6B model: its downloaded folder (fetch_model.py --model image or "
                    "image-parts); video model: likewise (--model video or video-parts)")
    ap.add_argument("--text-encoder-from", type=Path, help="12B model: the captioning model folder (fetch_model.py --model caption)")
    ap.add_argument("--vae-from", type=Path, help="12B model: the VAE folder (fetch_model.py --model large-vae)")
    ap.add_argument("--sha256", help="the transformer file's published SHA-256, to check the download against")
    ap.add_argument("--transformer-2", type=Path,
                    help="14B video model: the second expert's file (low noise); --transformer is the first (high noise)")
    ap.add_argument("--sha256-2", help="the second expert's published SHA-256")
    ap.add_argument("--steps", type=int, help="distilled 14B video model: the steps its download page gives")
    ap.add_argument("--shift", type=float, help="distilled 14B video model: the schedule shift its download page gives")
    ap.add_argument("--verify-only", action="store_true", help="hash an assembled folder again, change nothing")
    ap.add_argument("--json", action="store_true", help="report progress as JSON lines on stdout (messages go to stderr)")
    args = ap.parse_args(argv)
    global _json
    _json = args.json
    try:
        if args.probe:
            if args.transformer is None:
                ap.error("--probe needs --transformer")
            return run_probe(args.transformer)
        if args.dest is None:
            ap.error("--dest is required")
        if args.verify_only:
            return run_verify(args.dest)
        missing = [f"--{n.replace('_', '-')}" for n in ("transformer", "variant") if getattr(args, n) is None]
        if missing:
            ap.error(f"importing needs {', '.join(missing)}")
        for flag, value in (("--sha256", args.sha256), ("--sha256-2", args.sha256_2)):
            if value is not None and (len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value)):
                ap.error(f"{flag} must be 64 hexadecimal characters")
        return run_import(args)
    except (ImportFailed, single_file.SingleFileError, OSError) as exc:
        print(f"IMPORT FAILED: {exc}", file=sys.stderr)
        emit("error", message=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
