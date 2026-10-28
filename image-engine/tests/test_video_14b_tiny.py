"""The 14B video models (two experts) from single files, on tiny random-weight models.

The real architecture classes at a few megabytes: two experts written as single files
(bf16 in the authors' layout, or the older fp8 layout the published files use), the rest
of the folder saved by the pipeline itself, as import_model.py leaves it. This checks the
engine's own code: the switch from the first expert to the second at the boundary, an
expert dropped and read again when both do not fit in RAM, the few-step schedule, image
to video, and fp8 weights outside the blocks widened as they load. It says nothing about
video quality, speed or real memory use.

Needs PyTorch with CUDA and diffusers (the loader test only PyTorch), so it is skipped in a
plain Python. Run it with the image engine's environment:

    <venv>/python -m unittest tests.test_video_14b_tiny
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.large_fixture import TINY_14B_IMAGE, TINY_14B_TEXT, tiny_video_kinds, to_video_original, video_config_json
from tests.test_large_backend_tiny import HAS_CUDA, HAS_TORCH


def tiny_expert(config, seed: int):
    import torch
    from diffusers import WanTransformer3DModel

    torch.manual_seed(seed)
    model = WanTransformer3DModel(**{k: v for k, v in video_config_json(config).items() if not k.startswith("_")})
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 1:
                p.normal_(1.0, 0.1)
    return model.eval()


def write_expert(path: Path, model, older_fp8: bool = False) -> None:
    """The expert in the authors' layout: bf16, or the older fp8 layout of the published files
    (every linear weight in fp8 with a scale_weight and a scale_input, the rest fp16, a marker)."""
    import torch
    from safetensors.torch import save_file

    out = {}
    for name, tensor in model.state_dict().items():
        original = to_video_original(name)
        if older_fp8 and name.endswith(".weight") and tensor.ndim == 2:
            scale = tensor.abs().amax().float() / 448
            base = original[: -len(".weight")]
            out[original] = (tensor.float() / scale).to(torch.float8_e4m3fn)
            out[base + ".scale_weight"] = scale.reshape(())
            out[base + ".scale_input"] = torch.tensor(1.0)
        else:
            out[original] = tensor.to(torch.float16 if older_fp8 else torch.bfloat16)
    if older_fp8:
        out["scaled_fp8"] = torch.empty(0, dtype=torch.float8_e4m3fn)
    save_file(out, str(path))


def build_folder(folder: Path, image: bool = False, older_fp8: bool = False, few_step: dict | None = None) -> None:
    """A folder as import_model.py leaves it for a 14B model: the pipeline's own parts, both
    experts' config.json, and the experts as single files that model_index.json names."""
    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanPipeline
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, UMT5Config, UMT5EncoderModel

    torch.manual_seed(0)
    words = ["<pad>", "</s>", "<unk>", "a", "wave", "at", "night", "blurry", "slowly", "camera", "pans"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    text_encoder = UMT5EncoderModel(UMT5Config(vocab_size=len(words), d_model=32, d_kv=8, d_ff=64, num_layers=2,
                                               num_heads=4, relative_attention_num_buckets=8))
    # The 14B models' VAE: 16 latent channels, 8x in space, 4x in time.
    vae = AutoencoderKLWan(base_dim=16, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
                           temperal_downsample=[False, True, True], latents_mean=[0.0] * 16, latents_std=[1.0] * 16)
    scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
    config = TINY_14B_IMAGE if image else TINY_14B_TEXT
    high, low = tiny_expert(config, 1), tiny_expert(config, 2)
    parts = {"tokenizer": tokenizer, "text_encoder": text_encoder, "vae": vae, "scheduler": scheduler,
             "transformer": high, "transformer_2": low, "boundary_ratio": 0.9 if image else 0.875}
    pipe = (WanImageToVideoPipeline(**parts, image_encoder=None, image_processor=None) if image
            else WanPipeline(**parts))
    pipe.save_pretrained(folder, safe_serialization=True)
    for sub, model, name in (("transformer", high, "high_noise.safetensors"), ("transformer_2", low, "low_noise.safetensors")):
        for weights in (folder / sub).glob("diffusion_pytorch_model*"):
            weights.unlink()
        write_expert(folder / sub / name, model, older_fp8)
    index = json.loads((folder / "model_index.json").read_text())
    index.update(transformer_file="high_noise.safetensors", transformer_2_file="low_noise.safetensors")
    if few_step:
        index["few_step"] = few_step
    (folder / "model_index.json").write_text(json.dumps(index))
    files = [{"path": p.relative_to(folder).as_posix(), "size": p.stat().st_size}
             for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    (folder / "manifest.json").write_text(json.dumps({"repo": "local/high_noise.safetensors", "revision": "0" * 64,
                                                      "files": files}))


@unittest.skipUnless(HAS_TORCH, "needs PyTorch and diffusers (run with the image engine's environment)")
class OlderFp8LoaderTests(unittest.TestCase):
    def test_fp8_outside_the_blocks_is_widened_and_inside_kept(self):
        import torch
        from torch import nn

        from image_engine import fp8, single_file
        from image_engine.transformer_file import build_video

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "expert.safetensors"
            reference = tiny_expert(TINY_14B_TEXT, 1)
            write_expert(path, reference, older_fp8=True)
            model = build_video(path, single_file.inspect(path), video_config_json(TINY_14B_TEXT), quantize=True)
        embed = model.condition_embedder.time_embedder.linear_1
        # Not an fp8 layer: diffusers casts the timestep to this layer's dtype, which must stay fp32.
        self.assertIs(type(embed), nn.Linear)
        self.assertEqual(embed.weight.dtype, torch.float32)
        error = (embed.weight - reference.condition_embedder.time_embedder.linear_1.weight).abs().max()
        self.assertLess(error.item() / reference.condition_embedder.time_embedder.linear_1.weight.abs().max().item(), 0.07)
        self.assertIsInstance(model.blocks[0].attn1.to_q, fp8.ScaledFP8Linear)
        self.assertIsInstance(model.blocks[1].ffn.net[2], fp8.ScaledFP8Linear)


@unittest.skipUnless(HAS_CUDA, "needs PyTorch with CUDA and diffusers (run with the image engine's environment)")
class TwoExpertBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinds = tiny_video_kinds()
        cls.kinds.start()
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.text, cls.image = root / "text-14b", root / "image-14b"
        cls.older, cls.few = root / "text-14b-fp8", root / "text-14b-few"
        build_folder(cls.text)
        build_folder(cls.image, image=True)
        build_folder(cls.older, older_fp8=True)
        build_folder(cls.few, few_step={"steps": 2, "shift": 5.0})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        cls.kinds.stop()

    def backend(self, folder: Path, mode: str = "balanced", ram_gib: int = 32):
        from image_engine.backends.video import VideoBackend
        from image_engine.config import Settings

        backend = VideoBackend(lambda: Settings(video_model_dir=str(folder), video_memory_mode=mode))
        messages: list[str] = []
        with mock.patch("image_engine.backends.video._total_ram", return_value=ram_gib * 2**30), \
                mock.patch("image_engine.backends.video.TEXT_ENCODER_BYTES", 0):
            backend.load(messages.append)
        self.addCleanup(backend.unload)
        self.assertIn("Checking the second expert's file", messages)
        return backend

    def make_video(self, backend, steps: int = 4, start=None):
        from image_engine.params import parse_video_request

        fields = {"prompt": "a wave at night", "width": 256, "height": 288, "frames": 9, "steps": steps, "seed": 1}
        if start is not None:
            fields["start_image"] = "0123456789abcdef"
        req = parse_video_request(fields, backend.profile)
        seen: list[tuple[int, int]] = []
        cond = backend.encode(req.prompt, req.negative_prompt, req.guidance)
        frames = backend.decode(backend.sample(cond, req, req.seed, lambda d, t: seen.append((d, t)), start_frame=start))
        self.assertEqual(seen[-1], (steps, steps))
        self.assertEqual((len(frames), frames[0].size), (9, (256, 288)))
        return frames

    def test_one_expert_at_a_time_when_both_do_not_fit(self):
        # Tiny experts and a "RAM" of 0 GiB: as on a machine without room for both.
        backend = self.backend(self.text, ram_gib=0)
        experts = backend._experts
        self.assertFalse(experts.keep_both)
        self.make_video(backend)
        self.assertEqual(experts.builds, 2)  # the second expert read at the boundary
        self.assertIsNone(experts.models["high"])  # and the first dropped for it
        self.assertIsNone(backend._pipe.transformer)
        self.make_video(backend)
        self.assertEqual(experts.builds, 4)  # each clip reads both again
        self.assertEqual(backend.model_id, "text-14b")

    def test_both_experts_stay_loaded_with_room_for_them(self):
        backend = self.backend(self.text, ram_gib=1024)
        self.assertTrue(backend._experts.keep_both)
        self.make_video(backend)
        self.make_video(backend)
        self.assertEqual(backend._experts.builds, 2)

    def test_the_resident_plans_move_the_experts_instead_of_streaming(self):
        import torch

        from image_engine import fp8

        for mode in ("quality", "full"):
            with self.subTest(mode=mode):
                backend = self.backend(self.older, mode=mode, ram_gib=1024)
                self.make_video(backend)
                high, low = backend._experts.models["high"], backend._experts.models["low"]
                self.assertEqual(next(low.parameters()).device.type, "cuda")   # the one that ran last
                self.assertEqual(next(high.parameters()).device.type, "cpu")
                layer = low.blocks[0].attn1.to_q
                self.assertIsInstance(layer, fp8.ScaledFP8Linear)
                self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(backend.provenance()["weights"], "fp8")
                backend.unload()

    def test_a_few_step_model_switches_experts_on_its_shifted_schedule(self):
        backend = self.backend(self.few, ram_gib=0)
        self.assertEqual((backend.profile.name, backend.profile.steps), ("text-14b-few", 2))
        self.assertEqual(type(backend._pipe.scheduler).__name__, "FlowMatchEulerDiscreteScheduler")
        self.make_video(backend, steps=2)
        self.assertEqual(backend._experts.builds, 2)

    def test_the_image_model_starts_from_a_picture_and_only_from_one(self):
        from PIL import Image

        from image_engine.backends import BackendError

        backend = self.backend(self.image, ram_gib=0)
        self.make_video(backend, start=Image.new("RGB", (256, 288), (200, 30, 30)))
        with self.assertRaises(BackendError):
            backend.sample({"embeds": None, "negative": None}, mock.Mock(steps=2, guidance=3.5), 1, lambda d, t: None)

    def test_a_folder_without_both_files_is_refused(self):
        from image_engine.backends import BackendError
        from image_engine.backends.video import VideoBackend
        from image_engine.config import Settings

        folder = Path(self.tmp.name) / "half"
        build_folder(folder)
        index = json.loads((folder / "model_index.json").read_text())
        del index["transformer_2_file"]
        (folder / "model_index.json").write_text(json.dumps(index))
        manifest = json.loads((folder / "manifest.json").read_text())
        for entry in manifest["files"]:
            if entry["path"] == "model_index.json":
                entry["size"] = (folder / "model_index.json").stat().st_size
        (folder / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(BackendError) as ctx:
            VideoBackend(lambda: Settings(video_model_dir=str(folder))).load(lambda _m: None)
        self.assertIn("transformer_2_file", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
