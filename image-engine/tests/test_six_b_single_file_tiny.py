"""The 6B model from a single transformer file, on a tiny random-weight model.

The real architecture classes at a few megabytes: the transformer written as one fp8
file in the authors' layout (fused q/k/v, loader prefix, scales and quantization tags),
the rest of the folder saved by the pipeline itself. This checks the engine's own code:
the name mapping against diffusers' converter, the expected shapes against the model's
parameters, the fp8 weights against the ones they stand for, a LoRA on an fp8 layer, and
the backend loading such a folder through its memory plans. It says nothing about
picture quality, speed or real memory use.

Needs PyTorch and diffusers (the backend tests also CUDA), so it is skipped in a plain
Python. Run it with the image engine's environment:

    <venv>/python -m unittest tests.test_six_b_single_file_tiny
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.large_fixture import PREFIX, TAG, TINY_6B, six_b_config_json, to_six_b_original
from tests.test_large_backend_tiny import HAS_CUDA, HAS_TORCH

CONFIG = {k: v for k, v in six_b_config_json().items() if not k.startswith("_")}
BLOCKS = ("layers.", "noise_refiner.", "context_refiner.")


def tiny_transformer():
    import torch
    from diffusers import ZImageTransformer2DModel

    torch.manual_seed(0)
    model = ZImageTransformer2DModel(**CONFIG)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 1:
                p.normal_(1.0, 0.1)
    return model.eval()


def original_state(model) -> dict:
    """The model's weights in the authors' layout: q/k/v fused, names before diffusers' renaming."""
    import torch

    state = model.state_dict()
    out = {}
    for name, tensor in state.items():
        if name.endswith(".attention.to_q.weight"):
            base = name[: -len(".attention.to_q.weight")]
            out[to_six_b_original(base) + ".attention.qkv.weight"] = torch.cat(
                [state[f"{base}.attention.to_{p}.weight"] for p in "qkv"])
        elif not name.endswith((".attention.to_k.weight", ".attention.to_v.weight")):
            out[to_six_b_original(name)] = tensor
    return out


def write_transformer(path: Path, model, fp8: bool = True) -> None:
    import torch
    from safetensors.torch import save_file

    out = {}
    for name, tensor in original_state(model).items():
        if fp8 and name.startswith(BLOCKS) and name.endswith(".weight") and tensor.ndim == 2:
            scale = tensor.abs().amax().float() / 448
            out[PREFIX + name] = (tensor.float() / scale).to(torch.float8_e4m3fn)
            out[PREFIX + name[: -len(".weight")] + ".weight_scale"] = scale.reshape(())
            out[PREFIX + name[: -len(".weight")] + ".comfy_quant"] = torch.tensor(list(TAG), dtype=torch.uint8)
        else:
            out[PREFIX + name] = tensor.to(torch.bfloat16)
    save_file(out, str(path))


def build_tiny_folder(folder: Path) -> None:
    """A folder as import_model.py leaves it for the 6B model: the pipeline's own parts, the
    transformer's config.json, and the transformer as one fp8 file that model_index.json names."""
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

    words = ["<unk>", "<|endoftext|>", "a", "fox", "in", "the", "snow", "blurry", "user", "assistant"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<|endoftext|>", unk_token="<unk>",
                                        chat_template="{% for m in messages %}{{ m['content'] }}{% endfor %}")
    text_encoder = Qwen3Model(Qwen3Config(vocab_size=len(words), hidden_size=TINY_6B.cap_feat_dim, intermediate_size=64,
                                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16))
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=TINY_6B.in_channels, block_out_channels=[16, 16, 16],
                        down_block_types=["DownEncoderBlock2D"] * 3, up_block_types=["UpDecoderBlock2D"] * 3,
                        norm_num_groups=8, layers_per_block=1, scaling_factor=0.5, shift_factor=0.1)
    transformer = tiny_transformer()
    ZImagePipeline(scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0), vae=vae, text_encoder=text_encoder,
                   tokenizer=tokenizer, transformer=transformer).save_pretrained(folder, safe_serialization=True)
    for weights in (folder / "transformer").glob("diffusion_pytorch_model*"):
        weights.unlink()
    write_transformer(folder / "transformer" / "tiny-fp8.safetensors", transformer)
    index = json.loads((folder / "model_index.json").read_text())
    (folder / "model_index.json").write_text(json.dumps({**index, "is_distilled": True, "transformer_file": "tiny-fp8.safetensors"}))
    files = [{"path": p.relative_to(folder).as_posix(), "size": p.stat().st_size}
             for p in sorted(folder.rglob("*")) if p.is_file()]
    (folder / "manifest.json").write_text(json.dumps({"repo": "local/tiny-fp8.safetensors", "revision": "0" * 64, "files": files}))


@unittest.skipUnless(HAS_TORCH, "needs PyTorch and diffusers (run with the image engine's environment)")
class SixBLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_expected_shapes_are_the_models_parameters(self):
        from image_engine import single_file

        actual = {k: tuple(v.shape) for k, v in tiny_transformer().state_dict().items()}
        self.assertEqual(actual, single_file.six_b_expected_shapes(TINY_6B))

    def test_the_name_mapping_agrees_with_diffusers_converter(self):
        from diffusers.loaders.single_file_utils import convert_z_image_transformer_checkpoint_to_diffusers

        from image_engine import single_file

        write_transformer(self.path, tiny_transformer(), fp8=False)
        inspection = single_file.inspect(self.path)
        original = {PREFIX + k: v for k, v in original_state(tiny_transformer()).items()}
        theirs = convert_z_image_transformer_checkpoint_to_diffusers(dict(original))
        self.assertEqual(set(theirs), set(inspection.names))
        for name in ("layers.0.attention.to_k.weight", "context_refiner.0.attention.to_v.weight"):
            rows = inspection.rows[name]
            self.assertTrue(original[inspection.names[name]][rows[0]:rows[1]].equal(theirs[name]))

    def test_fp8_weights_load_as_the_weights_they_stand_for_and_take_a_lora(self):
        import torch

        from image_engine import fp8, lora, single_file
        from image_engine.transformer_file import build_six_b

        reference = tiny_transformer()
        write_transformer(self.path, reference)
        model = build_six_b(self.path, single_file.inspect(self.path), six_b_config_json())
        for name in ("layers.1.attention.to_v", "noise_refiner.0.feed_forward.w2"):
            layer = model.get_submodule(name)
            self.assertIsInstance(layer, fp8.ScaledFP8Linear)
            original = reference.get_submodule(name).weight
            error = (layer.dequantized(torch.float32) - original).abs().max() / original.abs().max()
            self.assertLess(error.item(), 0.07)
        self.assertNotIsInstance(model.cap_embedder[1], fp8.ScaledFP8Linear)
        self.assertTrue(model.cap_embedder[1].weight.equal(reference.cap_embedder[1].weight.bfloat16()))

        params = lora.add_trainable(model, "check", 2)  # a LoRA wraps the fp8 layers like linear ones
        self.assertTrue(params)
        wrapped = model.get_submodule("layers.0.attention.to_q")
        self.assertIsInstance(wrapped, lora.LoRALinear)
        self.assertEqual(wrapped(torch.randn(3, TINY_6B.dim, dtype=torch.bfloat16)).shape, (3, TINY_6B.dim))


@unittest.skipUnless(HAS_CUDA, "needs PyTorch with CUDA and diffusers (run with the image engine's environment)")
class SixBBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name) / "my-6b"
        build_tiny_folder(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_memory_plan_makes_an_image_from_the_fp8_file(self):
        from image_engine.backends.image import ImageBackend
        from image_engine.config import Settings
        from image_engine.params import parse_request

        for mode in ("quality", "balanced", "low"):
            with self.subTest(mode=mode):
                backend = ImageBackend(lambda: Settings(model_dir=str(self.folder), memory_mode=mode))
                messages: list[str] = []
                backend.load(messages.append)
                self.addCleanup(backend.unload)
                self.assertIn("Checking the transformer file", messages)
                self.assertEqual(backend.model_id, "turbo-6b")
                req = parse_request({"prompt": "a fox in the snow", "width": 256, "height": 256, "steps": 2, "seed": 1})
                cond = backend.encode(req.prompt, req.negative_prompt, req.guidance)
                image = backend.decode(backend.sample(cond, req, req.seed, lambda d, t: None))
                self.assertEqual(image.size, (256, 256))
                self.assertEqual(backend.provenance()["weights"], "fp8")
                backend.unload()
