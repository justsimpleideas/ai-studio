"""Tiny stand-ins for single transformer files (the 12B and 6B image models, the video model), written byte by byte.

Pure Python, so the plain test run can use them. The names follow the authors' layout
(single_file.LARGE_*_RULES in reverse), optionally behind the loader prefix, with the
linear weights of the blocks and the text fusion stage in fp8 next to a scale and a
quantization tag, the way the published fp8 file stores them.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from image_engine import single_file
from image_engine.single_file import DTYPE_SIZES, LargeConfig

PREFIX = "model.diffusion_model."
TAG = b'{"format": "float8_e4m3fn"}'

# The smallest shape the architecture allows: the head size is fixed at 128 by the rotary split.
TINY = LargeConfig(in_channels=16, num_layers=2, attention_head_dim=128, num_attention_heads=2, num_key_value_heads=1,
                   intermediate_size=64, timestep_embed_dim=32, text_hidden_dim=32, num_text_layers=3,
                   text_num_attention_heads=2, text_num_key_value_heads=2, text_intermediate_size=48,
                   num_layerwise_text_blocks=1, num_refiner_text_blocks=1)


def to_original_name(name: str) -> str:
    """diffusers name -> the authors' name (the inverse of single_file.to_diffusers_name)."""
    for old, new in single_file.LARGE_PREFIX_RULES:
        if name.startswith(new):
            name = old + name[len(new):]
            break
    for old, new in single_file.LARGE_INNER_RULES:
        name = name.replace(new, old)
    return name


def write_safetensors(path: Path, entries: dict[str, tuple[str, tuple[int, ...], bytes | None]],
                      metadata: dict[str, str] | None = None, trailing: bytes = b"") -> None:
    """entries: name -> (dtype, shape, data); data None means zeros of the right length."""
    header: dict = {"__metadata__": metadata} if metadata is not None else {}
    blobs, offset = [], 0
    for name, (dtype, shape, data) in entries.items():
        n = DTYPE_SIZES[dtype]
        for d in shape:
            n *= d
        data = bytes(n) if data is None else data
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs) + trailing)


def is_quantized(diffusers_name: str, shape: tuple[int, ...]) -> bool:
    return (diffusers_name.endswith(".weight") and len(shape) == 2
            and diffusers_name.startswith(("transformer_blocks.", "text_fusion.")) and "projector" not in diffusers_name)


def large_entries(config: LargeConfig = TINY, *, prefix: str = PREFIX, fp8: bool = True, per_row: bool = False,
                  tag: bytes | None = TAG) -> dict[str, tuple[str, tuple[int, ...], bytes | None]]:
    out: dict[str, tuple[str, tuple[int, ...], bytes | None]] = {}
    for name, shape in single_file.large_expected_shapes(config).items():
        original = to_original_name(name)
        if name.startswith("transformer_blocks.") and name.endswith(".scale_shift_table"):
            shape = (shape[0] * shape[1],)  # stored flat
        if fp8 and is_quantized(name, shape):
            base = original[: -len(".weight")]
            out[prefix + original] = ("F8_E4M3", shape, None)
            scale = struct.pack("<f", 0.5)
            out[prefix + base + ".weight_scale"] = ("F32", (shape[0],), scale * shape[0]) if per_row else ("F32", (), scale)
            if tag is not None:
                out[prefix + base + ".comfy_quant"] = ("U8", (len(tag),), tag)
        else:
            out[prefix + original] = ("BF16", shape, None)
    return out


def write_large_file(path: Path, **kwargs) -> Path:
    write_safetensors(path, large_entries(**kwargs))
    return path


# ---------------------------------------------------------------- the 6B model

from image_engine.single_file import SixBConfig  # noqa: E402

# The smallest 6B-shaped model: heads of 32, 2 layers, 1 refiner of each kind.
TINY_6B = SixBConfig(in_channels=4, dim=64, n_layers=2, n_refiner_layers=1, n_heads=2, n_kv_heads=2, cap_feat_dim=32)


def to_six_b_original(name: str) -> str:
    """diffusers name -> the authors' name (the inverse of single_file.six_b_diffusers_name), q/k/v not fused."""
    for old, new in single_file.SIX_B_RULES:
        name = name.replace(new, old)
    return name


def six_b_entries(config: SixBConfig = TINY_6B, *, prefix: str = PREFIX, fp8: bool = True, per_row: bool = False,
                  tag: bytes | None = TAG, diffusers_names: bool = False,
                  extra_norm: bool = False) -> dict[str, tuple[str, tuple[int, ...], bytes | None]]:
    """The 6B model's file: by default the authors' layout, q/k/v fused into one weight per block,
    linear weights of the blocks in fp8."""
    shapes = single_file.six_b_expected_shapes(config)
    out: dict[str, tuple[str, tuple[int, ...], bytes | None]] = {}

    def put(name: str, shape: tuple[int, ...], quantize: bool) -> None:
        if fp8 and quantize:
            base = name[: -len(".weight")]
            out[prefix + name] = ("F8_E4M3", shape, None)
            scale = struct.pack("<f", 0.5)
            out[prefix + base + ".weight_scale"] = ("F32", (shape[0],), scale * shape[0]) if per_row else ("F32", (), scale)
            if tag is not None:
                out[prefix + base + ".comfy_quant"] = ("U8", (len(tag),), tag)
        else:
            out[prefix + name] = ("BF16", shape, None)

    for name, shape in shapes.items():
        quantize = name.endswith(".weight") and len(shape) == 2 and name.startswith(("layers.", "noise_refiner.", "context_refiner."))
        if diffusers_names:
            put(name, shape, quantize)
        elif name.endswith(".attention.to_q.weight"):
            base = to_six_b_original(name[: -len(".attention.to_q.weight")])
            put(base + ".attention.qkv.weight", (3 * shape[0], shape[1]), quantize)
        elif name.endswith((".attention.to_k.weight", ".attention.to_v.weight")):
            continue
        else:
            put(to_six_b_original(name), shape, quantize)
    if extra_norm:
        out[prefix + "norm_final.weight"] = ("BF16", (config.dim,), None)
    return out


def six_b_config_json(config: SixBConfig = TINY_6B) -> dict:
    """The transformer's config.json, as the downloaded 6B folder has it, for these shapes."""
    return {"_class_name": "ZImageTransformer2DModel", **config.to_kwargs(), "axes_dims": [8, 12, 12],
            "axes_lens": [256, 128, 128], "norm_eps": 1e-05, "qk_norm": True, "rope_theta": 256.0, "t_scale": 1000.0}


# ---------------------------------------------------------------- the video model

from image_engine.single_file import VideoConfig  # noqa: E402

# The video model's shape where it matters (48 latent channels, 1x2x2 patches), a few kilobytes
# wide: 2 heads of 16, 2 blocks. tests/test_video_backend_tiny.py builds the same model.
TINY_VIDEO = VideoConfig(in_channels=48, out_channels=48, dim=32, ffn_dim=64, text_dim=32, freq_dim=256,
                         num_layers=2, patch_size=(1, 2, 2))
TINY_VIDEO_HEADS = 2
# The same for the 14B models' experts: 16 latent channels, and 36 in for image to video.
TINY_14B_TEXT = VideoConfig(16, 16, 32, 64, 32, 256, 2, (1, 2, 2))
TINY_14B_IMAGE = VideoConfig(36, 16, 32, 64, 32, 256, 2, (1, 2, 2))


def tiny_video_kinds():
    """While active, the tiny shapes stand in for the published ones as the video models the studio has parts for."""
    from unittest import mock

    return mock.patch.dict(single_file.VIDEO_KINDS, {"5b": TINY_VIDEO, "14b-text": TINY_14B_TEXT,
                                                     "14b-image": TINY_14B_IMAGE}, clear=True)

# diffusers name -> the authors' name, in order (the inverse of single_file.VIDEO_RULES).
_TO_VIDEO_ORIGINAL = (
    ("condition_embedder.time_embedder.linear_1", "time_embedding.0"),
    ("condition_embedder.time_embedder.linear_2", "time_embedding.2"),
    ("condition_embedder.text_embedder.linear_1", "text_embedding.0"),
    ("condition_embedder.text_embedder.linear_2", "text_embedding.2"),
    ("condition_embedder.time_proj", "time_projection.1"),
    (".attn1.", ".self_attn."),
    (".attn2.", ".cross_attn."),
    (".to_out.0.", ".o."),
    (".to_q.", ".q."),
    (".to_k.", ".k."),
    (".to_v.", ".v."),
    (".scale_shift_table", ".modulation"),
    ("proj_out", "head.head"),
    ("ffn.net.0.proj", "ffn.0"),
    ("ffn.net.2", "ffn.2"),
    (".norm2.", ".norm3."),
)


def to_video_original(name: str) -> str:
    if name == "scale_shift_table":
        return "head.modulation"
    for new, old in _TO_VIDEO_ORIGINAL:
        name = name.replace(new, old)
    return name


def video_entries(config: VideoConfig = TINY_VIDEO, *, prefix: str = PREFIX, fp8: bool = False, per_row: bool = False,
                  tag: bytes | None = TAG, diffusers_names: bool = False, older_fp8: bool = False,
                  extra: dict[str, tuple[int, ...]] | None = None) -> dict[str, tuple[str, tuple[int, ...], bytes | None]]:
    """The video model's file: by default the authors' layout in bf16; with `fp8`, the blocks'
    linear weights in fp8 next to a scale (and a quantization tag). `older_fp8`: the older fp8
    layout the published 14B files use: every linear weight in fp8 (in the blocks and outside),
    each with a `scale_weight` and a `scale_input`, an empty `scaled_fp8` marker, no prefix, fp16."""
    out: dict[str, tuple[str, tuple[int, ...], bytes | None]] = {}
    scale = struct.pack("<f", 0.5)
    for name, shape in single_file.video_expected_shapes(config).items():
        stored = prefix + (name if diffusers_names else to_video_original(name))
        linear = name.endswith(".weight") and len(shape) == 2
        if older_fp8 and linear:
            base = stored[: -len(".weight")]
            out[stored] = ("F8_E4M3", shape, None)
            out[base + ".scale_weight"] = ("F32", (), scale)
            out[base + ".scale_input"] = ("F32", (), scale)
        elif fp8 and linear and name.startswith("blocks."):
            base = stored[: -len(".weight")]
            out[stored] = ("F8_E4M3", shape, None)
            out[base + ".weight_scale"] = ("F32", (shape[0],), scale * shape[0]) if per_row else ("F32", (), scale)
            if tag is not None:
                out[base + ".comfy_quant"] = ("U8", (len(tag),), tag)
        else:
            out[stored] = ("F16" if older_fp8 else "BF16", shape, None)
    if older_fp8:
        out[prefix + single_file.SCALED_MARKER] = ("F8_E4M3", (0,), None)
    for name, shape in (extra or {}).items():
        out[prefix + name] = ("BF16", shape, None)
    return out


def video_config_json(config: VideoConfig = TINY_VIDEO, heads: int = TINY_VIDEO_HEADS) -> dict:
    """The transformer's config.json, as the downloaded video folder has it, for these shapes."""
    return {"_class_name": "WanTransformer3DModel", "added_kv_proj_dim": None, "attention_head_dim": config.dim // heads,
            "cross_attn_norm": True, "eps": 1e-06, "ffn_dim": config.ffn_dim, "freq_dim": config.freq_dim,
            "image_dim": None, "in_channels": config.in_channels, "num_attention_heads": heads,
            "num_layers": config.num_layers, "out_channels": config.out_channels, "patch_size": list(config.patch_size),
            "pos_embed_seq_len": None, "qk_norm": "rms_norm_across_heads", "rope_max_seq_len": 1024,
            "text_dim": config.text_dim}
