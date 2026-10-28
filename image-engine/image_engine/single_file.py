"""Reading a transformer published as one .safetensors file, without running anything.

Some image and video models are shared as a single file holding only the transformer,
rather than as a diffusers folder. Such a file names its tensors in the model authors'
original layout, often behind a loader prefix, and may store its linear weights in
fp8 with a scale per tensor (or per output row) to halve their size.

This module reads the file's header only. It is pure Python (no torch), so the
import script and the engine's plain tests can use it, and it never executes or
unpickles anything. It answers:

  * is the file well-formed (read_header): known dtypes, tensors tiling the data
    exactly, no duplicate names, string-only metadata;
  * which supported model it is (detect_family): the 12B single-stream image model
    (`large`), the 6B few-step image model (`diffusion`) or the 5B video model
    (`video`);
  * what each tensor is called in diffusers' layout (to_diffusers_name for the 12B
    model, six_b_diffusers_name for the 6B one, whose fused q/k/v weight is split,
    video_diffusers_name for the video model);
  * which linear weights are fp8, with what scale;
  * with what dimensions, and whether every tensor the engine needs is there with the
    right shape and nothing else is (inspect).
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_HEADER_BYTES = 100 * 1024 * 1024
DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}
FLOAT_DTYPES = {"F16", "BF16", "F32"}
FP8_DTYPE = "F8_E4M3"

# A loader prefix some tools put in front of every tensor name.
PREFIXES = ("model.diffusion_model.",)
# Per-layer quantization tags: a few bytes of JSON stored as a U8 tensor next to the weight.
QUANT_TAG = ".comfy_quant"
# An older layout of fp8 files: a scale per weight under another name, a scale for the layer's
# input (used only by loaders that multiply in fp8; the engine widens every weight first, so
# it is checked and left out), and an empty fp8 tensor marking the file as scaled.
SCALE_SUFFIXES = (".weight_scale", ".scale_weight")
INPUT_SCALE = ".scale_input"
SCALED_MARKER = "scaled_fp8"
MAX_TAG_BYTES = 1024
# Tag keys that change nothing for a weight that is dequantized before every matmul,
# which is what the engine does. Any other key is refused rather than guessed at.
#   format                       the storage format (only fp8 e4m3 is accepted)
#   full_precision_matrix_mult   asks a loader that could multiply in fp8 to widen this layer's
#                                weight first instead: the engine always does (fp8.ScaledFP8Linear)
TAG_KEYS = {"format", "full_precision_matrix_mult"}
TAG_FLAGS = {"full_precision_matrix_mult"}  # keys whose value must be true or false
TAG_FORMATS = {"float8_e4m3fn"}

# Bytes a file starts with when it is not safetensors at all.
FOREIGN_MAGIC = {b"PK\x03\x04": "a zip archive (a pickled checkpoint)", b"\x80": "a pickle"}


class SingleFileError(Exception):
    """The file is malformed, or not a model the engine can run. Safe to show to the user."""


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    start: int  # byte offsets within the data section
    end: int

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class Header:
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str]
    data_start: int  # absolute file offset of the data section
    file_size: int


# ---------------------------------------------------------------- header


def read_header(path: Path) -> Header:
    """Parse and validate the header. Reads at most the header, never the weights."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            head = fh.read(8)
            for magic, what in FOREIGN_MAGIC.items():
                if head.startswith(magic):
                    raise SingleFileError(f"{path.name} is {what}, not safetensors. Only safetensors files are accepted.")
            if len(head) != 8:
                raise SingleFileError(f"{path.name} is too short to be a safetensors file")
            (header_len,) = struct.unpack("<Q", head)
            if header_len <= 1 or header_len > MAX_HEADER_BYTES or 8 + header_len > size:
                raise SingleFileError(f"{path.name}: implausible header length {header_len}")
            raw = fh.read(header_len)
    except OSError as exc:
        raise SingleFileError(f"Cannot read {path}: {exc}") from exc

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                # Two readers could otherwise disagree on which entry counts.
                raise SingleFileError(f"{path.name}: the header names '{key}' twice")
            out[key] = value
        return out

    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, ValueError) as exc:
        raise SingleFileError(f"{path.name}: the header is not valid JSON ({exc})") from exc
    if not isinstance(header, dict):
        raise SingleFileError(f"{path.name}: the header is not a JSON object")

    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values()):
        raise SingleFileError(f"{path.name}: __metadata__ must map names to strings")

    data_start = 8 + header_len
    data_len = size - data_start
    tensors: dict[str, TensorInfo] = {}
    for name, info in header.items():
        if not isinstance(info, dict):
            raise SingleFileError(f"{path.name}: tensor {name} has no description")
        dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        if dtype not in DTYPE_SIZES:
            raise SingleFileError(f"{path.name}: tensor {name} has unknown dtype {dtype!r}")
        if not (isinstance(shape, list) and all(type(d) is int and d >= 0 for d in shape)):
            raise SingleFileError(f"{path.name}: tensor {name} has an invalid shape {shape!r}")
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(type(o) is int for o in offsets)):
            raise SingleFileError(f"{path.name}: tensor {name} has invalid offsets {offsets!r}")
        t = TensorInfo(dtype, tuple(shape), offsets[0], offsets[1])
        if t.end - t.start != t.numel * DTYPE_SIZES[dtype]:
            raise SingleFileError(f"{path.name}: tensor {name} is not as long as its shape says")
        tensors[name] = t

    # Tensors must tile the data section exactly: no gaps, no overlaps, nothing hidden after.
    cursor = 0
    for name, t in sorted(tensors.items(), key=lambda kv: (kv[1].start, kv[1].end)):
        if t.start != cursor:
            raise SingleFileError(f"{path.name}: gap or overlap before tensor {name}")
        cursor = t.end
    if cursor != data_len:
        raise SingleFileError(f"{path.name}: {data_len - cursor} bytes after the last tensor belong to no tensor")
    return Header(tensors=tensors, metadata=metadata, data_start=data_start, file_size=size)


def read_bytes(path: Path, header: Header, name: str) -> bytes:
    """The raw bytes of one (small) tensor."""
    t = header.tensors[name]
    with path.open("rb") as fh:
        fh.seek(header.data_start + t.start)
        return fh.read(t.end - t.start)


# ---------------------------------------------------------------- names

# The original layout of the 12B model, renamed to diffusers' (the same rules as diffusers'
# own single-file converter for this model; tests/test_large_backend_tiny.py checks that
# the two agree). Prefix rules apply to the start of a name, first match only;
# the inner rules then apply anywhere in it. Names already in diffusers' layout pass
# through unchanged.
LARGE_PREFIX_RULES = (
    ("first.", "img_in."),
    ("tmlp.0.", "time_embed.linear_1."),
    ("tmlp.2.", "time_embed.linear_2."),
    ("tproj.1.", "time_mod_proj."),
    ("txtmlp.0.scale", "txt_in.norm.weight"),
    ("txtmlp.1.", "txt_in.linear_1."),
    ("txtmlp.3.", "txt_in.linear_2."),
    ("txtfusion.", "text_fusion."),
    ("blocks.", "transformer_blocks."),
    ("last.linear.", "final_layer.linear."),
    ("last.norm.scale", "final_layer.norm.weight"),
    ("last.modulation.lin", "final_layer.scale_shift_table"),
)
LARGE_INNER_RULES = (
    (".attn.wq.", ".attn.to_q."),
    (".attn.wk.", ".attn.to_k."),
    (".attn.wv.", ".attn.to_v."),
    (".attn.wo.", ".attn.to_out.0."),
    (".attn.gate.", ".attn.to_gate."),
    (".attn.qknorm.qnorm.scale", ".attn.norm_q.weight"),
    (".attn.qknorm.knorm.scale", ".attn.norm_k.weight"),
    (".mlp.", ".ff."),
    (".prenorm.scale", ".norm1.weight"),
    (".postnorm.scale", ".norm2.weight"),
    (".mod.lin", ".scale_shift_table"),
)


def strip_prefix(names: list[str]) -> str:
    """The loader prefix every name carries, or "" (a prefix on only some names is refused)."""
    for prefix in PREFIXES:
        with_prefix = [n.startswith(prefix) for n in names]
        if all(with_prefix):
            return prefix
        if any(with_prefix):
            raise SingleFileError(f"Only some tensor names start with '{prefix}'; the file mixes two layouts")
    return ""


def to_diffusers_name(name: str, prefix: str = "") -> str:
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    for old, new in LARGE_PREFIX_RULES:
        if name.startswith(old):
            name = new + name[len(old):]
            break
    for old, new in LARGE_INNER_RULES:
        name = name.replace(old, new)
    return name


# ---------------------------------------------------------------- the 12B model


@dataclass(frozen=True)
class LargeConfig:
    """Krea2Transformer2DModel's constructor arguments, read off the tensor shapes."""
    in_channels: int
    num_layers: int
    attention_head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    timestep_embed_dim: int
    text_hidden_dim: int
    num_text_layers: int
    text_num_attention_heads: int
    text_num_key_value_heads: int
    text_intermediate_size: int
    num_layerwise_text_blocks: int
    num_refiner_text_blocks: int

    @property
    def hidden_size(self) -> int:
        return self.attention_head_dim * self.num_attention_heads

    def to_kwargs(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# Not readable from shapes; fixed by the architecture. The rotary split must add up to the head size.
LARGE_ROPE_AXES = (32, 48, 48)


def _linear(out: dict[str, tuple[int, ...]], name: str, n_out: int, n_in: int, bias: bool) -> None:
    out[f"{name}.weight"] = (n_out, n_in)
    if bias:
        out[f"{name}.bias"] = (n_out,)


def _attention(out: dict, name: str, dim: int, heads: int, kv_heads: int) -> None:
    head_dim = dim // heads
    _linear(out, f"{name}.to_q", head_dim * heads, dim, False)
    _linear(out, f"{name}.to_k", head_dim * kv_heads, dim, False)
    _linear(out, f"{name}.to_v", head_dim * kv_heads, dim, False)
    _linear(out, f"{name}.to_gate", dim, dim, False)
    _linear(out, f"{name}.to_out.0", dim, dim, False)
    out[f"{name}.norm_q.weight"] = (head_dim,)
    out[f"{name}.norm_k.weight"] = (head_dim,)


def _block(out: dict, name: str, dim: int, heads: int, kv_heads: int, inter: int) -> None:
    out[f"{name}.norm1.weight"] = (dim,)
    out[f"{name}.norm2.weight"] = (dim,)
    _attention(out, f"{name}.attn", dim, heads, kv_heads)
    _linear(out, f"{name}.ff.gate", inter, dim, False)
    _linear(out, f"{name}.ff.up", inter, dim, False)
    _linear(out, f"{name}.ff.down", dim, inter, False)


def large_expected_shapes(c: LargeConfig) -> dict[str, tuple[int, ...]]:
    """Every parameter of the model in diffusers' names, with its shape."""
    h = c.hidden_size
    out: dict[str, tuple[int, ...]] = {}
    _linear(out, "img_in", h, c.in_channels, True)
    _linear(out, "time_embed.linear_1", h, c.timestep_embed_dim, True)
    _linear(out, "time_embed.linear_2", h, h, True)
    _linear(out, "time_mod_proj", 6 * h, h, True)
    for kind, count in (("layerwise_blocks", c.num_layerwise_text_blocks), ("refiner_blocks", c.num_refiner_text_blocks)):
        for i in range(count):
            _block(out, f"text_fusion.{kind}.{i}", c.text_hidden_dim, c.text_num_attention_heads,
                   c.text_num_key_value_heads, c.text_intermediate_size)
    out["text_fusion.projector.weight"] = (1, c.num_text_layers)
    out["txt_in.norm.weight"] = (c.text_hidden_dim,)
    _linear(out, "txt_in.linear_1", h, c.text_hidden_dim, True)
    _linear(out, "txt_in.linear_2", h, h, True)
    for i in range(c.num_layers):
        out[f"transformer_blocks.{i}.scale_shift_table"] = (6, h)
        _block(out, f"transformer_blocks.{i}", h, c.num_attention_heads, c.num_key_value_heads, c.intermediate_size)
    out["final_layer.scale_shift_table"] = (2, h)
    out["final_layer.norm.weight"] = (h,)
    _linear(out, "final_layer.linear", c.in_channels, h, True)
    return out


def _count(names: set[str], pattern: str) -> int:
    rx = re.compile(pattern)
    found = {int(m.group(1)) for n in names if (m := rx.match(n))}
    if found and found != set(range(len(found))):
        raise SingleFileError(f"Block numbers are not consecutive ({pattern})")
    return len(found)


def _shape(shapes: dict[str, tuple[int, ...]], name: str) -> tuple[int, ...]:
    if name not in shapes:
        raise SingleFileError(f"The file has no tensor '{name}' (diffusers name); it is not a supported model")
    return shapes[name]


def infer_large_config(shapes: dict[str, tuple[int, ...]]) -> LargeConfig:
    names = set(shapes)
    hidden, in_channels = _shape(shapes, "img_in.weight")
    head_dim = _shape(shapes, "transformer_blocks.0.attn.norm_q.weight")[0]
    text_dim = _shape(shapes, "text_fusion.layerwise_blocks.0.attn.to_q.weight")[1]
    text_head_dim = _shape(shapes, "text_fusion.layerwise_blocks.0.attn.norm_q.weight")[0]
    for what, total, part in (("hidden size", hidden, head_dim), ("text width", text_dim, text_head_dim)):
        if part <= 0 or total % part:
            raise SingleFileError(f"The {what} {total} does not split into heads of {part}")
    if sum(LARGE_ROPE_AXES) != head_dim:
        raise SingleFileError(f"Head size {head_dim} is not the {sum(LARGE_ROPE_AXES)} this model family uses")
    return LargeConfig(
        in_channels=in_channels,
        num_layers=_count(names, r"^transformer_blocks\.(\d+)\."),
        attention_head_dim=head_dim,
        num_attention_heads=hidden // head_dim,
        num_key_value_heads=_shape(shapes, "transformer_blocks.0.attn.to_k.weight")[0] // head_dim,
        intermediate_size=_shape(shapes, "transformer_blocks.0.ff.gate.weight")[0],
        timestep_embed_dim=_shape(shapes, "time_embed.linear_1.weight")[1],
        text_hidden_dim=text_dim,
        num_text_layers=_shape(shapes, "text_fusion.projector.weight")[1],
        text_num_attention_heads=text_dim // text_head_dim,
        text_num_key_value_heads=_shape(shapes, "text_fusion.layerwise_blocks.0.attn.to_k.weight")[0] // text_head_dim,
        text_intermediate_size=_shape(shapes, "text_fusion.layerwise_blocks.0.ff.gate.weight")[0],
        num_layerwise_text_blocks=_count(names, r"^text_fusion\.layerwise_blocks\.(\d+)\."),
        num_refiner_text_blocks=_count(names, r"^text_fusion\.refiner_blocks\.(\d+)\."),
    )


# ---------------------------------------------------------------- the 6B model

# The 6B few-step model's original layout, renamed to diffusers' (the same rules as diffusers'
# own single-file converter for this model; tests/test_large_backend_tiny.py checks that the
# two agree). Every rule applies anywhere in a name. The fused q/k/v weight is split into
# three equal row blocks. Names already in diffusers' layout (they start with
# "all_x_embedder.") are left as they are.
SIX_B_RULES = (
    ("final_layer.", "all_final_layer.2-1."),
    ("x_embedder.", "all_x_embedder.2-1."),
    (".attention.out.bias", ".attention.to_out.0.bias"),
    (".attention.k_norm.weight", ".attention.norm_k.weight"),
    (".attention.q_norm.weight", ".attention.norm_q.weight"),
    (".attention.out.weight", ".attention.to_out.0.weight"),
)
SIX_B_FUSED = ".attention.qkv.weight"
SIX_B_SPLIT = (".attention.to_q.weight", ".attention.to_k.weight", ".attention.to_v.weight")
# Present in some files of this model, unused by it.
SIX_B_IGNORED = ("norm_final.weight",)
ADALN_DIM = 256  # the timestep embedding width the modulation layers read


def six_b_diffusers_name(name: str) -> str:
    for old, new in SIX_B_RULES:
        name = name.replace(old, new)
    return name


@dataclass(frozen=True)
class SixBConfig:
    """ZImageTransformer2DModel's shape arguments, read off the tensor shapes. The rest of its
    configuration (rotary axes, theta, ...) comes from the model's own config.json."""
    in_channels: int
    dim: int
    n_layers: int
    n_refiner_layers: int
    n_heads: int
    n_kv_heads: int
    cap_feat_dim: int
    patch_size: int = 2
    f_patch_size: int = 1

    @property
    def num_layers(self) -> int:
        return self.n_layers

    @property
    def hidden_size(self) -> int:
        return self.dim

    def to_kwargs(self) -> dict[str, Any]:
        return {"in_channels": self.in_channels, "dim": self.dim, "n_layers": self.n_layers,
                "n_refiner_layers": self.n_refiner_layers, "n_heads": self.n_heads, "n_kv_heads": self.n_kv_heads,
                "cap_feat_dim": self.cap_feat_dim, "all_patch_size": [self.patch_size],
                "all_f_patch_size": [self.f_patch_size]}


def _six_b_block(out: dict, name: str, c: SixBConfig, modulation: bool) -> None:
    head = c.dim // c.n_heads
    ff = int(c.dim / 3 * 8)
    _linear(out, f"{name}.attention.to_q", head * c.n_heads, c.dim, False)
    _linear(out, f"{name}.attention.to_k", head * c.n_kv_heads, c.dim, False)
    _linear(out, f"{name}.attention.to_v", head * c.n_kv_heads, c.dim, False)
    _linear(out, f"{name}.attention.to_out.0", c.dim, head * c.n_heads, False)
    out[f"{name}.attention.norm_q.weight"] = (head,)
    out[f"{name}.attention.norm_k.weight"] = (head,)
    _linear(out, f"{name}.feed_forward.w1", ff, c.dim, False)
    _linear(out, f"{name}.feed_forward.w2", c.dim, ff, False)
    _linear(out, f"{name}.feed_forward.w3", ff, c.dim, False)
    for norm in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"):
        out[f"{name}.{norm}.weight"] = (c.dim,)
    if modulation:
        _linear(out, f"{name}.adaLN_modulation.0", 4 * c.dim, min(c.dim, ADALN_DIM), True)


def six_b_expected_shapes(c: SixBConfig) -> dict[str, tuple[int, ...]]:
    """Every parameter of the 6B model in diffusers' names, with its shape."""
    out: dict[str, tuple[int, ...]] = {}
    key = f"{c.patch_size}-{c.f_patch_size}"
    patch = c.f_patch_size * c.patch_size * c.patch_size * c.in_channels
    _linear(out, f"all_x_embedder.{key}", c.dim, patch, True)
    _linear(out, f"all_final_layer.{key}.linear", patch, c.dim, True)
    _linear(out, f"all_final_layer.{key}.adaLN_modulation.1", c.dim, min(c.dim, ADALN_DIM), True)
    for i in range(c.n_refiner_layers):
        _six_b_block(out, f"noise_refiner.{i}", c, True)
        _six_b_block(out, f"context_refiner.{i}", c, False)
    for i in range(c.n_layers):
        _six_b_block(out, f"layers.{i}", c, True)
    _linear(out, "t_embedder.mlp.0", 1024, ADALN_DIM, True)
    _linear(out, "t_embedder.mlp.2", min(c.dim, ADALN_DIM), 1024, True)
    out["cap_embedder.0.weight"] = (c.cap_feat_dim,)
    _linear(out, "cap_embedder.1", c.dim, c.cap_feat_dim, True)
    out["x_pad_token"] = (1, c.dim)
    out["cap_pad_token"] = (1, c.dim)
    return out


def infer_six_b_config(shapes: dict[str, tuple[int, ...]]) -> SixBConfig:
    names = set(shapes)
    dim, cap = _shape(shapes, "cap_embedder.1.weight")
    head = _shape(shapes, "layers.0.attention.norm_q.weight")[0]
    if head <= 0 or dim % head:
        raise SingleFileError(f"The width {dim} does not split into heads of {head}")
    embedders = [n for n in names if re.match(r"^all_x_embedder\.\d+-\d+\.weight$", n)]
    if len(embedders) != 1:
        raise SingleFileError("The file must hold exactly one patch embedder")
    patch, f_patch = (int(x) for x in embedders[0].split(".")[1].split("-"))
    width = shapes[embedders[0]][1]
    if width % (patch * patch * f_patch):
        raise SingleFileError(f"The patch embedder's width {width} does not fit {patch}x{patch} patches")
    refiners = _count(names, r"^noise_refiner\.(\d+)\.")
    if _count(names, r"^context_refiner\.(\d+)\.") != refiners:
        raise SingleFileError("The file has different numbers of noise and context refiner blocks")
    return SixBConfig(in_channels=width // (patch * patch * f_patch), dim=dim, n_layers=_count(names, r"^layers\.(\d+)\."),
                      n_refiner_layers=refiners, n_heads=dim // head,
                      n_kv_heads=_shape(shapes, "layers.0.attention.to_k.weight")[0] // head, cap_feat_dim=cap,
                      patch_size=patch, f_patch_size=f_patch)


# The 6B model's configuration keys that the tensor shapes decide.
SIX_B_SHAPE_KEYS = ("in_channels", "dim", "n_layers", "n_refiner_layers", "n_heads", "n_kv_heads", "cap_feat_dim")


def check_six_b_config(config: dict[str, Any], shapes: SixBConfig) -> dict[str, Any]:
    """The model's config.json (diffusers' constructor arguments), checked against the file's shapes."""
    expected = shapes.to_kwargs()
    for key in SIX_B_SHAPE_KEYS:
        if config.get(key) != expected[key]:
            raise SingleFileError(f"The transformer file has {key} = {expected[key]}, its config.json says {config.get(key)}")
    for key in ("all_patch_size", "all_f_patch_size"):
        if list(config.get(key) or []) != expected[key]:
            raise SingleFileError(f"The transformer file's patch size does not match its config.json ({key})")
    return {k: v for k, v in config.items() if not k.startswith("_")}


# ---------------------------------------------------------------- the video model

# The 5B text/image-to-video model's original layout, renamed to diffusers' (the same rules,
# in the same order, as diffusers' own single-file converter for this model family;
# tests/test_video_single_file_tiny.py checks that the two agree). Every rule applies
# anywhere in a name, one after another: the last three swap norm2 and norm3. Names already
# in diffusers' layout (they start with "condition_embedder.") are left as they are.
VIDEO_RULES = (
    ("time_embedding.0", "condition_embedder.time_embedder.linear_1"),
    ("time_embedding.2", "condition_embedder.time_embedder.linear_2"),
    ("text_embedding.0", "condition_embedder.text_embedder.linear_1"),
    ("text_embedding.2", "condition_embedder.text_embedder.linear_2"),
    ("time_projection.1", "condition_embedder.time_proj"),
    ("cross_attn", "attn2"),
    ("self_attn", "attn1"),
    (".o.", ".to_out.0."),
    (".q.", ".to_q."),
    (".k.", ".to_k."),
    (".v.", ".to_v."),
    ("head.modulation", "scale_shift_table"),
    ("head.head", "proj_out"),
    ("modulation", "scale_shift_table"),
    ("ffn.0", "ffn.net.0.proj"),
    ("ffn.2", "ffn.net.2"),
    ("norm2", "norm__placeholder"),
    ("norm3", "norm2"),
    ("norm__placeholder", "norm3"),
)


def video_diffusers_name(name: str) -> str:
    for old, new in VIDEO_RULES:
        name = name.replace(old, new)
    return name


@dataclass(frozen=True)
class VideoConfig:
    """WanTransformer3DModel's shape arguments, read off the tensor shapes. How the width
    splits into heads is not in the shapes (q/k are normalized across all heads); it comes
    from the model's own config.json, like the rest of its configuration."""
    in_channels: int
    out_channels: int
    dim: int
    ffn_dim: int
    text_dim: int
    freq_dim: int
    num_layers: int
    patch_size: tuple[int, int, int]

    @property
    def hidden_size(self) -> int:
        return self.dim


def video_expected_shapes(c: VideoConfig) -> dict[str, tuple[int, ...]]:
    """Every parameter of the video model in diffusers' names, with its shape."""
    d = c.dim
    out: dict[str, tuple[int, ...]] = {"patch_embedding.weight": (d, c.in_channels, *c.patch_size),
                                       "patch_embedding.bias": (d,)}
    _linear(out, "condition_embedder.time_embedder.linear_1", d, c.freq_dim, True)
    _linear(out, "condition_embedder.time_embedder.linear_2", d, d, True)
    _linear(out, "condition_embedder.time_proj", 6 * d, d, True)
    _linear(out, "condition_embedder.text_embedder.linear_1", d, c.text_dim, True)
    _linear(out, "condition_embedder.text_embedder.linear_2", d, d, True)
    for i in range(c.num_layers):
        block = f"blocks.{i}"
        out[f"{block}.scale_shift_table"] = (1, 6, d)
        for attn in ("attn1", "attn2"):
            for proj in ("to_q", "to_k", "to_v", "to_out.0"):
                _linear(out, f"{block}.{attn}.{proj}", d, d, True)
            out[f"{block}.{attn}.norm_q.weight"] = (d,)
            out[f"{block}.{attn}.norm_k.weight"] = (d,)
        out[f"{block}.norm2.weight"] = (d,)  # the cross-attention's norm; norm1 and norm3 have no weights
        out[f"{block}.norm2.bias"] = (d,)
        _linear(out, f"{block}.ffn.net.0.proj", c.ffn_dim, d, True)
        _linear(out, f"{block}.ffn.net.2", d, c.ffn_dim, True)
    patch = c.patch_size[0] * c.patch_size[1] * c.patch_size[2]
    _linear(out, "proj_out", c.out_channels * patch, d, True)
    out["scale_shift_table"] = (1, 2, d)
    return out


def infer_video_config(shapes: dict[str, tuple[int, ...]]) -> VideoConfig:
    embed = _shape(shapes, "patch_embedding.weight")
    if len(embed) != 5:
        raise SingleFileError(f"patch_embedding.weight has shape {list(embed)}; a video patch embedding has five dimensions")
    dim, in_channels, *patch = embed
    volume = patch[0] * patch[1] * patch[2]
    projected = _shape(shapes, "proj_out.weight")[0]
    if volume <= 0 or projected % volume:
        raise SingleFileError(f"The output projection's {projected} rows do not fit {patch[0]}x{patch[1]}x{patch[2]} patches")
    return VideoConfig(in_channels=in_channels, out_channels=projected // volume, dim=dim,
                       ffn_dim=_shape(shapes, "blocks.0.ffn.net.0.proj.weight")[0],
                       text_dim=_shape(shapes, "condition_embedder.text_embedder.linear_1.weight")[1],
                       freq_dim=_shape(shapes, "condition_embedder.time_embedder.linear_1.weight")[1],
                       num_layers=_count(set(shapes), r"^blocks\.(\d+)\."), patch_size=(patch[0], patch[1], patch[2]))


# The video models the studio has parts for, by their published dimensions. The 5B model
# works in its VAE's 48 latent channels; the 14B ones in 16, and the image-to-video one takes
# 36 in (the noisy latents, a mask and the start picture's latents). Each 14B model is two
# files of these same dimensions, one per expert.
VIDEO_KINDS: dict[str, VideoConfig] = {
    "5b": VideoConfig(48, 48, 3072, 14336, 4096, 256, 30, (1, 2, 2)),
    "14b-text": VideoConfig(16, 16, 5120, 13824, 4096, 256, 40, (1, 2, 2)),
    "14b-image": VideoConfig(36, 16, 5120, 13824, 4096, 256, 40, (1, 2, 2)),
}
VIDEO_KIND_LABELS = {"5b": "5B video model", "14b-text": "14B text-to-video model (one expert of two)",
                     "14b-image": "14B image-to-video model (one expert of two)"}


def video_kind(config: VideoConfig) -> str:
    """Which video model the studio has parts for a file of these dimensions is; raises SingleFileError
    (before any part is downloaded) for any other size of the family."""
    for kind, known in VIDEO_KINDS.items():
        if config == known:
            return kind
    raise SingleFileError(f"A video model {config.dim} wide with {config.num_layers} blocks and {config.in_channels} "
                          "input channels is not one the studio has the other parts for (the 5B model, or the 14B "
                          "text-to-video or image-to-video model)")


# The video model's configuration keys that the tensor shapes decide, and the ones that must
# be off because they add tensors this layout does not have (an image encoder, extra projections).
VIDEO_SHAPE_KEYS = ("in_channels", "out_channels", "ffn_dim", "text_dim", "freq_dim", "num_layers")
VIDEO_ABSENT_KEYS = ("image_dim", "added_kv_proj_dim", "pos_embed_seq_len")


def check_video_config(config: dict[str, Any], shapes: VideoConfig) -> dict[str, Any]:
    """The model's config.json (diffusers' constructor arguments), checked against the file's shapes."""
    for key in VIDEO_SHAPE_KEYS:
        if config.get(key) != getattr(shapes, key):
            raise SingleFileError(f"The transformer file has {key} = {getattr(shapes, key)}, its config.json says {config.get(key)}")
    if list(config.get("patch_size") or []) != list(shapes.patch_size):
        raise SingleFileError("The transformer file's patch size does not match its config.json")
    heads, head_dim = config.get("num_attention_heads"), config.get("attention_head_dim")
    if not (isinstance(heads, int) and isinstance(head_dim, int) and heads * head_dim == shapes.dim):
        raise SingleFileError(f"The transformer file is {shapes.dim} wide; its config.json says {heads} heads of {head_dim}")
    if config.get("cross_attn_norm") is not True:
        raise SingleFileError("The transformer file has a cross-attention norm; its config.json says it does not")
    for key in VIDEO_ABSENT_KEYS:
        if config.get(key) is not None:
            raise SingleFileError(f"The config.json sets {key}, which this kind of video model does not use")
    return {k: v for k, v in config.items() if not k.startswith("_")}


# ---------------------------------------------------------------- inspection

FAMILIES = ("large", "diffusion", "video")  # image_models.py names the image backends the same way


@dataclass(frozen=True)
class Quantized:
    """One fp8 linear weight: its scale tensor (file name) and the scale's shape."""
    scale: str
    scale_shape: tuple[int, ...]  # () for one scale per tensor, (out,) for one per output row
    rows: tuple[int, int] | None = None  # the rows of a per-row scale that belong to this weight (split weights)


@dataclass
class Inspection:
    family: str  # "large" (12B image), "diffusion" (6B image) or "video"
    config: LargeConfig | SixBConfig | VideoConfig
    prefix: str
    # diffusers name -> file name, for every parameter of the model
    names: dict[str, str]
    # diffusers weight name -> its scale, for the linear weights stored in fp8
    quantized: dict[str, Quantized] = field(default_factory=dict)
    # diffusers name -> rows of the file tensor it is cut from (the 6B model's fused q/k/v)
    rows: dict[str, tuple[int, int]] = field(default_factory=dict)
    # file names of the per-layer quantization tags (read, checked, then not loaded)
    tags: list[str] = field(default_factory=list)
    # file names of tensors the model does not use
    ignored: list[str] = field(default_factory=list)
    dtypes: dict[str, int] = field(default_factory=dict)  # file dtype -> tensor count

    @property
    def storage(self) -> str:
        """"fp8" when the linear weights are fp8, else the dtype most tensors use."""
        if self.quantized:
            return "fp8"
        return max(self.dtypes, key=self.dtypes.get) if self.dtypes else "?"

    @property
    def expected(self) -> dict[str, tuple[int, ...]]:
        if isinstance(self.config, LargeConfig):
            return large_expected_shapes(self.config)
        if isinstance(self.config, VideoConfig):
            return video_expected_shapes(self.config)
        return six_b_expected_shapes(self.config)

    @property
    def parameters(self) -> int:
        return sum(_numel(s) for s in self.expected.values())

    @property
    def video_kind(self) -> str | None:
        """For a video file of a size the studio has parts for, which one ("5b", "14b-text", "14b-image")."""
        if not isinstance(self.config, VideoConfig):
            return None
        try:
            return video_kind(self.config)
        except SingleFileError:
            return None

    @property
    def label(self) -> str:
        if self.family == "video":
            kind = self.video_kind
            return VIDEO_KIND_LABELS[kind] if kind else f"{round(self.parameters / 1e9)}B video model"
        return "12B image model" if self.family == "large" else "6B image model"

    def summary(self) -> dict[str, Any]:
        return {"family": self.family, "label": self.label, "parameters": self.parameters, "storage": self.storage,
                "layers": self.config.num_layers, "hidden_size": self.config.hidden_size,
                "fp8_weights": len(self.quantized), "dtypes": dict(self.dtypes), "video_kind": self.video_kind}


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def decode_tag(raw: bytes, name: str) -> dict[str, Any]:
    if len(raw) > MAX_TAG_BYTES:
        raise SingleFileError(f"Quantization tag {name} is {len(raw)} bytes, more than a tag should be")
    try:
        tag = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SingleFileError(f"Quantization tag {name} is not JSON text") from exc
    if not isinstance(tag, dict):
        raise SingleFileError(f"Quantization tag {name} is not a JSON object")
    unknown = sorted(set(tag) - TAG_KEYS)
    if unknown:
        raise SingleFileError(f"Quantization tag {name} has settings the engine does not know: {', '.join(unknown)}")
    if tag.get("format") not in TAG_FORMATS:
        raise SingleFileError(f"Quantization tag {name} names format {tag.get('format')!r}; "
                              f"supported: {', '.join(sorted(TAG_FORMATS))}")
    for key in TAG_FLAGS & set(tag):
        if not isinstance(tag[key], bool):
            raise SingleFileError(f"Quantization tag {name}: {key} must be true or false, not {tag[key]!r}")
    return tag


def detect_family(stripped: list[str]) -> str:
    """Which supported model the (prefix-free) tensor names belong to."""
    names = set(stripped)
    if any(to_diffusers_name(n) == "text_fusion.projector.weight" for n in names):
        return "large"
    if "layers.0.adaLN_modulation.0.weight" in names and "cap_embedder.1.weight" in names:
        return "diffusion"
    if "patch_embedding.weight" in names and ("head.head.weight" in names or "proj_out.weight" in names):
        return "video"
    raise SingleFileError("This is not a model the engine supports: its tensors follow neither the 6B nor "
                          "the 12B image model's layout, nor the video model's")


@dataclass
class _Mapping:
    params: dict[str, str] = field(default_factory=dict)            # diffusers name -> file name
    rows: dict[str, tuple[int, int]] = field(default_factory=dict)   # diffusers name -> rows of the file tensor
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)  # diffusers name -> shape once mapped
    ignored: list[str] = field(default_factory=list)


def _add(m: _Mapping, new: str, old: str, shape: tuple[int, ...], rows: tuple[int, int] | None = None) -> None:
    if new in m.params:
        raise SingleFileError(f"Two tensors in the file map to {new}")
    m.params[new] = old
    m.shapes[new] = shape
    if rows is not None:
        m.rows[new] = rows


def _map_large(weights: dict[str, str], tensors: dict[str, TensorInfo]) -> _Mapping:
    m = _Mapping()
    for stripped, old in weights.items():
        new = to_diffusers_name(stripped)
        shape = tensors[old].shape
        # The original layout keeps each block's six modulation vectors in one flat row.
        if new.startswith("transformer_blocks.") and new.endswith(".scale_shift_table"):
            n = _numel(shape)
            if n % 6:
                raise SingleFileError(f"{old} does not hold six modulation vectors")
            shape = (6, n // 6)
        _add(m, new, old, shape)
    return m


def _map_six_b(weights: dict[str, str], tensors: dict[str, TensorInfo]) -> _Mapping:
    m = _Mapping()
    diffusers_layout = any(s.startswith("all_x_embedder.") for s in weights)
    for stripped, old in weights.items():
        shape = tensors[old].shape
        if stripped in SIX_B_IGNORED:
            m.ignored.append(old)
            continue
        if diffusers_layout:
            _add(m, stripped, old, shape)
        elif stripped.endswith(SIX_B_FUSED):
            if len(shape) != 2 or shape[0] % 3:
                raise SingleFileError(f"{old} is not three equal q/k/v blocks")
            part = shape[0] // 3
            base = six_b_diffusers_name(stripped[: -len(SIX_B_FUSED)])
            for i, suffix in enumerate(SIX_B_SPLIT):
                _add(m, base + suffix, old, (part, shape[1]), (i * part, (i + 1) * part))
        else:
            _add(m, six_b_diffusers_name(stripped), old, shape)
    return m


def _map_video(weights: dict[str, str], tensors: dict[str, TensorInfo]) -> _Mapping:
    m = _Mapping()
    diffusers_layout = any(s.startswith("condition_embedder.") for s in weights)
    for stripped, old in weights.items():
        _add(m, stripped if diffusers_layout else video_diffusers_name(stripped), old, tensors[old].shape)
    return m


def inspect(path: Path, header: Header | None = None) -> Inspection:
    """Everything the engine needs to know before loading the file. Raises SingleFileError."""
    header = header or read_header(path)
    names = list(header.tensors)
    prefix = strip_prefix(names)
    tensors = header.tensors

    weights: dict[str, str] = {}   # prefix-free name -> file name, for the model's own tensors
    scales: dict[str, str] = {}    # file name of an fp8 weight -> file name of its scale
    input_scales: dict[str, str] = {}  # the same, for the older layout's input scales
    tags: list[str] = []
    markers: list[str] = []
    for name in names:
        if name.endswith(QUANT_TAG):
            tags.append(name)
        elif name.endswith(SCALE_SUFFIXES):
            weight = name[: -len(next(s for s in SCALE_SUFFIXES if name.endswith(s)))] + ".weight"
            if weight in scales:
                raise SingleFileError(f"{weight} has two scales")
            scales[weight] = name
        elif name.endswith(INPUT_SCALE):
            input_scales[name[: -len(INPUT_SCALE)] + ".weight"] = name
        elif name == prefix + SCALED_MARKER:
            markers.append(name)
        else:
            weights[name[len(prefix):]] = name
    family = detect_family(list(weights))
    config: LargeConfig | SixBConfig | VideoConfig
    if family == "large":
        mapping = _map_large(weights, tensors)
        config = infer_large_config(mapping.shapes)
        expected = large_expected_shapes(config)
    elif family == "video":
        mapping = _map_video(weights, tensors)
        config = infer_video_config(mapping.shapes)
        expected = video_expected_shapes(config)
    else:
        mapping = _map_six_b(weights, tensors)
        config = infer_six_b_config(mapping.shapes)
        expected = six_b_expected_shapes(config)

    missing = sorted(set(expected) - set(mapping.shapes))
    extra = sorted(set(mapping.shapes) - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} (first: {missing[0]})")
        if extra:
            details.append(f"{len(extra)} the model does not have (first: {mapping.params[extra[0]]})")
        raise SingleFileError("The file does not match the model's structure: " + "; ".join(details))
    for new, shape in expected.items():
        if mapping.shapes[new] != shape:
            raise SingleFileError(f"{mapping.params[new]} has shape {list(mapping.shapes[new])}, expected {list(shape)}")

    quantized: dict[str, Quantized] = {}
    used_scales: set[str] = set()
    for new, old in mapping.params.items():
        info = tensors[old]
        if info.dtype == FP8_DTYPE:
            if not (new.endswith(".weight") and len(expected[new]) == 2):
                raise SingleFileError(f"{old} is fp8, but only linear weights may be")
            scale = scales.get(old)
            if scale is None:
                raise SingleFileError(f"{old} is fp8 but has no weight_scale next to it")
            s = tensors[scale]
            if s.dtype not in FLOAT_DTYPES or s.shape not in ((), (1,), (info.shape[0],)):
                raise SingleFileError(f"{scale} must be one float per tensor or per output row, "
                                      f"not {s.dtype} {list(s.shape)}")
            per_row = s.shape not in ((), (1,))
            rows = mapping.rows.get(new) if per_row else None
            quantized[new] = Quantized(scale, (rows[1] - rows[0],) if rows else s.shape if per_row else (), rows)
            used_scales.add(scale)
        elif info.dtype not in FLOAT_DTYPES:
            raise SingleFileError(f"{old} is stored as {info.dtype}; expected bf16, fp16, fp32 or fp8")
    unused = sorted(set(scales.values()) - used_scales)
    if unused:
        raise SingleFileError(f"Scale without an fp8 weight: {unused[0]}")

    fp8_files = {mapping.params[new] for new in quantized}
    for weight, name in input_scales.items():
        s = tensors[name]
        if weight not in fp8_files:
            raise SingleFileError(f"Input scale without an fp8 weight: {name}")
        if s.dtype not in FLOAT_DTYPES or s.shape not in ((), (1,)):
            raise SingleFileError(f"{name} must be one float, not {s.dtype} {list(s.shape)}")
    for name in markers:
        if tensors[name].dtype != FP8_DTYPE or tensors[name].numel > 2:
            raise SingleFileError(f"{name} is not the empty fp8 marker it should be")
        if not quantized:
            raise SingleFileError(f"{name} marks the file as fp8, but it has no fp8 weights")

    for tag in tags:
        info = tensors[tag]
        if info.dtype != "U8" or len(info.shape) != 1:
            raise SingleFileError(f"Quantization tag {tag} is not a byte string")
        if tag[: -len(QUANT_TAG)] + ".weight" not in fp8_files:
            raise SingleFileError(f"Quantization tag {tag} belongs to no fp8 weight")
        decode_tag(read_bytes(path, header, tag), tag)

    dtypes: dict[str, int] = {}
    for old in set(mapping.params.values()):
        dtypes[tensors[old].dtype] = dtypes.get(tensors[old].dtype, 0) + 1
    return Inspection(family=family, config=config, prefix=prefix, names=mapping.params, quantized=quantized,
                      rows=mapping.rows, tags=tags, ignored=[*mapping.ignored, *sorted(input_scales.values()), *markers],
                      dtypes=dtypes)
