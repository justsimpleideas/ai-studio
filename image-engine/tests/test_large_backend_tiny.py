"""The 12B model's loading code and backend, run on a tiny random-weight model.

The model is built here: the real architecture classes at a few megabytes, its
transformer written as one fp8 file in the authors' layout behind the loader prefix,
the way the published file is, next to a tiny text encoder (saved as the full
vision-language model, like the real one) and a tiny VAE. So this checks the engine's
own code: the name mapping against diffusers' converter, the fp8 layers against the
weights they stand for, and the backend through every memory plan, streaming
included. It says nothing about picture quality, speed or real memory use.

Needs PyTorch and diffusers (the backend tests also CUDA), so it is skipped in a plain
Python. Run it with the image engine's environment:

    <venv>/python -m unittest tests.test_large_backend_tiny
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tests.large_fixture import PREFIX, TAG, TINY, is_quantized, to_original_name

HAS_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "diffusers", "transformers", "tokenizers", "safetensors"))


def _has_cuda() -> bool:
    if not HAS_TORCH:
        return False
    import torch

    return torch.cuda.is_available()


HAS_CUDA = _has_cuda()
TEXT_LAYERS = [1, 2, 3]  # TINY reads 3 text encoder layers; the tiny encoder has 4


def tiny_transformer():
    import torch
    from diffusers import Krea2Transformer2DModel

    torch.manual_seed(0)
    model = Krea2Transformer2DModel(**TINY.to_kwargs())
    with torch.no_grad():  # the norms and modulation tables start at zero; make them matter
        for name, p in model.named_parameters():
            if p.ndim == 1 or "scale_shift_table" in name:
                p.normal_(0, 0.1)
    return model.eval()


def write_transformer(path: Path, model, fp8: bool = True) -> None:
    """The model's weights as one file in the authors' layout, fp8 with one scale per tensor."""
    import torch
    from safetensors.torch import save_file

    out = {}
    for name, tensor in model.state_dict().items():
        original = PREFIX + to_original_name(name)
        if name.startswith("transformer_blocks.") and name.endswith(".scale_shift_table"):
            tensor = tensor.reshape(-1)
        if fp8 and is_quantized(name, tuple(tensor.shape)):
            scale = tensor.abs().amax().float() / 448
            out[original] = (tensor.float() / scale).to(torch.float8_e4m3fn)
            base = original[: -len(".weight")]
            out[base + ".weight_scale"] = scale.reshape(())
            out[base + ".comfy_quant"] = torch.tensor(list(TAG), dtype=torch.uint8)
        else:
            out[original] = tensor.to(torch.bfloat16)
    save_file(out, str(path))


def build_tiny_model(folder: Path, distilled: bool = False) -> None:
    import torch
    from diffusers import AutoencoderKLQwenImage
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    (folder / "transformer").mkdir(parents=True)
    write_transformer(folder / "transformer" / "tiny-fp8.safetensors", tiny_transformer())

    words = ["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "a", "fox", "in", "the", "snow", "blurry",
             "system", "user", "assistant", "Describe", "image"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<|endoftext|>", unk_token="<unk>").save_pretrained(
        folder / "tokenizer")

    torch.manual_seed(1)
    config = Qwen3VLConfig(
        text_config={"vocab_size": len(words), "hidden_size": TINY.text_hidden_dim, "intermediate_size": 64,
                     "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True}},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                       "out_hidden_size": TINY.text_hidden_dim, "deepstack_visual_indexes": []})
    # Saved as the full vision-language model, with its "model." prefix, like the real checkpoint.
    Qwen3VLForConditionalGeneration(config).save_pretrained(folder / "text_encoder", safe_serialization=True)

    AutoencoderKLQwenImage(base_dim=16, z_dim=TINY.in_channels // 4, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
                           attn_scales=[], temperal_downsample=[False, True, True],
                           latents_mean=[0.1] * 4, latents_std=[2.0] * 4).save_pretrained(
        folder / "vae", safe_serialization=True)

    (folder / "model_index.json").write_text(json.dumps(
        {"_class_name": "Krea2Pipeline", "is_distilled": distilled, "text_encoder_select_layers": TEXT_LAYERS}))
    files = [{"path": p.relative_to(folder).as_posix(), "size": p.stat().st_size}
             for p in sorted(folder.rglob("*")) if p.is_file()]
    (folder / "manifest.json").write_text(json.dumps({"repo": "local/tiny-fp8.safetensors", "revision": "0" * 64,
                                                      "files": files}))


@unittest.skipUnless(HAS_TORCH, "needs PyTorch and diffusers (run with the image engine's environment)")
class LoaderTests(unittest.TestCase):
    """CPU only: the names, shapes and fp8 weights the loader produces."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_expected_shapes_are_the_models_parameters(self):
        from image_engine import single_file

        model = tiny_transformer()
        actual = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        self.assertEqual(actual, single_file.large_expected_shapes(TINY))

    def test_the_name_mapping_agrees_with_diffusers_converter(self):
        from diffusers.loaders.single_file_utils import convert_krea2_transformer_checkpoint_to_diffusers

        from image_engine import single_file

        model = tiny_transformer()
        original = {}
        for name, tensor in model.state_dict().items():
            if name.startswith("transformer_blocks.") and name.endswith(".scale_shift_table"):
                tensor = tensor.reshape(-1)
            original[PREFIX + to_original_name(name)] = tensor
        theirs = convert_krea2_transformer_checkpoint_to_diffusers(dict(original))
        ours = {single_file.to_diffusers_name(k, PREFIX): k for k in original}
        self.assertEqual(set(theirs), set(ours))
        self.assertEqual(theirs["transformer_blocks.1.scale_shift_table"].shape, (6, 256))

    def test_fp8_weights_load_as_the_weights_they_stand_for(self):
        import torch

        from image_engine import fp8, single_file
        from image_engine.backends.large import load_transformer

        reference = tiny_transformer()
        write_transformer(self.path, reference)
        inspection = single_file.inspect(self.path)
        model, pinned = load_transformer(self.path, inspection, pin=False)
        self.assertFalse(pinned)

        layer = model.transformer_blocks[0].ff.down
        self.assertIsInstance(layer, fp8.ScaledFP8Linear)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.weight_scale.shape, ())
        original = reference.transformer_blocks[0].ff.down.weight
        error = (layer.dequantized(torch.float32) - original).abs().max() / original.abs().max()
        self.assertLess(error.item(), 0.07)  # fp8 e4m3 keeps 3 mantissa bits
        self.assertNotIsInstance(model.img_in, fp8.ScaledFP8Linear)
        self.assertEqual(model.img_in.weight.dtype, torch.bfloat16)
        self.assertEqual(model.transformer_blocks[0].norm1.weight.dtype, torch.float32)  # kept as diffusers keeps it

        torch.manual_seed(2)
        latents = torch.randn(1, 16, TINY.in_channels)
        text = torch.randn(1, 5, TINY.num_text_layers, TINY.text_hidden_dim)
        ids = torch.zeros(5 + 16, 3)
        ids[5:, 1] = torch.arange(16) // 4
        ids[5:, 2] = torch.arange(16) % 4
        t = torch.tensor([0.5])
        with torch.no_grad():
            want = reference(latents, text, t, ids, return_dict=False)[0]
            got = model(latents.bfloat16(), text.bfloat16(), t, ids, return_dict=False)[0].float()
        cosine = torch.nn.functional.cosine_similarity(want.flatten(), got.flatten(), dim=0)
        self.assertGreater(cosine.item(), 0.97)

    def test_a_bf16_file_is_quantized_per_row_as_it_loads(self):
        import torch

        from image_engine import fp8, single_file
        from image_engine.backends.large import load_transformer

        write_transformer(self.path, tiny_transformer(), fp8=False)
        model, _ = load_transformer(self.path, single_file.inspect(self.path), pin=False)
        layer = model.text_fusion.layerwise_blocks[0].attn.to_q
        self.assertIsInstance(layer, fp8.ScaledFP8Linear)
        self.assertEqual(layer.weight_scale.shape, (TINY.text_hidden_dim,))
        self.assertNotIsInstance(model.final_layer.linear, fp8.ScaledFP8Linear)
        self.assertFalse(any(p.is_meta for p in model.parameters()))
        self.assertEqual(model.time_embed.linear_1.weight.dtype, torch.bfloat16)


@unittest.skipUnless(HAS_CUDA, "needs PyTorch with CUDA and diffusers (run with the image engine's environment)")
class TinyLargeBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name) / "large-model"
        build_tiny_model(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def backend(self, mode: str):
        from image_engine.backends.large import LargeImageBackend
        from image_engine.config import Settings

        backend = LargeImageBackend(lambda: Settings(model_dir=str(self.folder), memory_mode=mode))
        messages: list[str] = []
        backend.load(messages.append)
        self.addCleanup(backend.unload)
        self.assertIn("Checking the transformer file", messages)
        return backend

    def run_job(self, backend, **fields):
        from image_engine.params import parse_request

        req = parse_request({"prompt": "a fox in the snow", "width": 256, "height": 320, "steps": 2, "seed": 1, **fields})
        steps: list[tuple[int, int]] = []
        cond = backend.encode(req.prompt, req.negative_prompt, req.guidance)
        latents = backend.sample(cond, req, req.seed, lambda d, t: steps.append((d, t)))
        image = backend.decode(latents)
        self.assertEqual(steps, [(1, 2), (2, 2)])
        return cond, image

    def test_every_memory_plan_makes_an_image(self):
        for mode in ("full", "quality", "balanced", "low"):
            with self.subTest(mode=mode):
                backend = self.backend(mode)
                _, image = self.run_job(backend)
                self.assertEqual(image.size, (256, 320))
                self.assertEqual(backend.stats()["memory_plan"], backend.device_info()["memory_plan"])
                self.assertEqual(backend.provenance()["weights"], "fp8")
                backend.unload()

    def test_streaming_leaves_the_weights_in_pinned_ram_and_the_text_encoder_off_the_gpu(self):
        backend = self.backend("balanced")
        self.run_job(backend)
        block = backend._pipe.transformer.transformer_blocks[0].ff.down
        self.assertEqual(block.weight.device.type, "cpu")
        self.assertTrue(block.weight.is_pinned())
        self.assertEqual(next(backend._pipe.text_encoder.parameters()).device.type, "cpu")
        self.assertIsNone(backend._pipe.text_encoder.visual)

    def test_streaming_works_with_no_ram_left_to_lock(self):
        from unittest import mock

        with mock.patch("image_engine.transformer_file.pin_budget", return_value=0):
            backend = self.backend("balanced")
        _, image = self.run_job(backend)
        self.assertEqual(image.size, (256, 320))
        self.assertFalse(backend._pipe.transformer.transformer_blocks[0].ff.down.weight.is_pinned())

    def test_the_arena_locks_what_it_holds_and_stops_at_its_budget(self):
        import torch

        from image_engine.transformer_file import PinnedArena

        arena = PinnedArena(budget=2048, block=1024)
        a = torch.randn(100)                                   # 400 bytes
        b = torch.randn(3, 5).to(torch.float8_e4m3fn)          # 15 bytes, then 64-byte alignment
        c = torch.tensor(2.5)                                  # a scalar scale
        kept = [arena.keep(t) for t in (a, b, c)]
        for src, out in zip((a, b, c), kept):
            self.assertTrue(out.is_pinned())
            self.assertEqual((out.dtype, out.shape), (src.dtype, src.shape))
            self.assertTrue(torch.equal(out.float(), src.float()))
        self.assertEqual(arena.locked, 1024)                   # one block, not one per tensor
        for _ in range(3):
            arena.keep(torch.randn(200))                       # 800 bytes: a second block, then over budget
        self.assertEqual(arena.locked, 2048)
        self.assertFalse(arena.complete)
        self.assertFalse(arena.keep(torch.randn(100)).is_pinned())

        refused = PinnedArena(budget=2**40, block=2**36)       # more than the system will lock
        self.assertFalse(refused.keep(torch.randn(10)).is_pinned())
        self.assertEqual((refused.locked, refused.budget, refused.complete), (0, 0, False))
        # The refusal is not reported again by the next GPU operation.
        self.assertEqual((torch.ones(2, device="cuda") + 1).sum().item(), 4.0)

    def test_a_refused_lock_is_told_apart_from_a_full_gpu(self):
        import torch

        from image_engine.backends.large import host_memory_refused
        from image_engine.transformer_file import clear_refused_lock

        try:
            torch.empty(2**36, dtype=torch.uint8, pin_memory=True)
        except RuntimeError as exc:
            clear_refused_lock()
            self.assertTrue(host_memory_refused(exc))
        else:
            self.skipTest("this machine locked 64 GiB")
        self.assertFalse(host_memory_refused(torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")))
        self.assertFalse(host_memory_refused(ValueError("CUDA error: out of memory")))

    def test_guidance_brings_the_negative_prompt(self):
        backend = self.backend("balanced")
        cond, _ = self.run_job(backend, guidance=0.0)
        self.assertIsNone(cond["negative"])
        cond, _ = self.run_job(backend, guidance=4.5, negative_prompt="blurry")
        self.assertIsNotNone(cond["negative"])
        self.assertEqual(cond["negative"][0].shape, cond["embeds"].shape)

    def test_the_same_seed_gives_the_same_image(self):
        backend = self.backend("quality")
        _, a = self.run_job(backend)
        _, b = self.run_job(backend)
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_the_profile_follows_the_folder(self):
        backend = self.backend("low")
        self.assertEqual((backend.profile.name, backend.model_id), ("large", "dit-12b"))
        self.assertFalse(backend._pipe.config.is_distilled)
        self.assertEqual(list(backend._pipe.text_encoder_select_layers), TEXT_LAYERS)

    def test_the_engines_image_slot_runs_it_and_refuses_loras(self):
        from image_engine.backends import BackendError
        from image_engine.backends.image import ImageBackend
        from image_engine.config import Settings

        router = ImageBackend(lambda: Settings(model_dir=str(self.folder), memory_mode="balanced"))
        router.load(lambda _m: None)
        self.addCleanup(router.unload)
        self.assertEqual(router.model_id, "dit-12b")
        _, image = self.run_job(router)
        self.assertEqual(image.size, (256, 320))
        with self.assertRaises(BackendError):
            router.set_loras([("0" * 16, "x.safetensors", 1.0)])

    def test_a_changed_transformer_is_refused_before_anything_loads(self):
        import shutil

        from image_engine.backends import BackendError
        from image_engine.backends.large import LargeImageBackend
        from image_engine.config import Settings

        copy = Path(self.tmp.name) / "changed"
        shutil.copytree(self.folder, copy)
        target = copy / "transformer" / "tiny-fp8.safetensors"
        target.write_bytes(target.read_bytes() + b"\0")
        backend = LargeImageBackend(lambda: Settings(model_dir=str(copy)))
        with self.assertRaises(BackendError) as ctx:
            backend.load(lambda _m: None)
        self.assertIn("changed since it was verified", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
