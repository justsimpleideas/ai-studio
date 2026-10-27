"""The real VideoBackend, run through the real diffusers pipelines on a tiny model.

The model is built here from random weights: the same architecture classes and
the same folder layout as the published 5B model, but a few megabytes in size.
So this checks the backend's own code (component loading, every memory plan,
image to video, the latent normalisation in decode) on a GPU without downloading
anything. It says nothing about picture quality, speed or real memory use.

Needs PyTorch with CUDA and diffusers, so it is skipped in a plain Python. Run it
with the image engine's environment:

    <venv>/python -m unittest tests.test_video_backend_tiny
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_engine.backends import BackendError
from image_engine.config import Settings
from image_engine.params import parse_video_request


def _has_cuda() -> bool:
    if not all(importlib.util.find_spec(m) for m in ("torch", "diffusers", "transformers", "tokenizers")):
        return False
    import torch

    return torch.cuda.is_available()


HAS_CUDA = _has_cuda()


def _has_diffusers() -> bool:
    return all(importlib.util.find_spec(m) for m in ("torch", "diffusers"))


def build_tiny_model(folder: Path, distilled: bool = False) -> None:
    """`distilled`: label the folder the way the 3-step distilled model's folder is labelled."""
    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, UMT5Config, UMT5EncoderModel

    torch.manual_seed(0)
    words = ["<pad>", "</s>", "<unk>", "a", "wave", "at", "night", "blurry", "slowly", "camera", "pans"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")

    text_encoder = UMT5EncoderModel(UMT5Config(vocab_size=len(words), d_model=32, d_kv=8, d_ff=64, num_layers=2,
                                               num_heads=4, relative_attention_num_buckets=8))
    # The published model's shape where it matters: 48 latent channels, 2x2 patches, 16x VAE.
    transformer = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16,
                                        in_channels=48, out_channels=48, text_dim=32, freq_dim=256, ffn_dim=64,
                                        num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                                        eps=1e-6, image_dim=None, rope_max_seq_len=1024)
    vae = AutoencoderKLWan(base_dim=16, decoder_base_dim=16, z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
                           temperal_downsample=[False, True, True], is_residual=True, in_channels=12,
                           out_channels=12, patch_size=2, scale_factor_temporal=4, scale_factor_spatial=16,
                           latents_mean=[0.0] * 48, latents_std=[1.0] * 48)
    scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0)
    pipe = WanPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler,
                       transformer=transformer, expand_timesteps=True)
    pipe.save_pretrained(folder, safe_serialization=True)
    if distilled:
        index = json.loads((folder / "model_index.json").read_text(encoding="utf-8"))
        index["_class_name"] = "WanDMDPipeline"
        (folder / "model_index.json").write_text(json.dumps(index), encoding="utf-8")

    # What fetch_model.py leaves behind, so the backend's pre-load check passes.
    files = [{"path": p.relative_to(folder).as_posix(), "size": p.stat().st_size}
             for p in sorted(folder.rglob("*")) if p.is_file()]
    (folder / "manifest.json").write_text(json.dumps({"repo": "example/tiny-video", "revision": "0" * 40,
                                                      "files": files}), encoding="utf-8")


@unittest.skipUnless(HAS_CUDA, "needs PyTorch with CUDA and diffusers (run with the image engine's environment)")
class TinyVideoBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name) / "video-model"
        build_tiny_model(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def backend(self, mode: str):
        from image_engine.backends.video import VideoBackend

        backend = VideoBackend(lambda: Settings(video_model_dir=str(self.folder), video_memory_mode=mode))
        messages: list[str] = []
        backend.load(messages.append)
        self.addCleanup(backend.unload)
        self.assertIn("Checking model files", messages)
        return backend

    def run_job(self, backend, start_frame=None, **fields):
        req = parse_video_request({"prompt": "a wave at night", "width": 256, "height": 288, "frames": 9,
                                   "steps": 2, "seed": 1, **fields})
        steps: list[tuple[int, int]] = []
        cond = backend.encode(req.prompt, req.negative_prompt, req.guidance)
        latents = backend.sample(cond, req, req.seed, lambda d, t: steps.append((d, t)), start_frame=start_frame)
        frames = backend.decode(latents)
        self.assertEqual(steps, [(1, 2), (2, 2)])
        return req, cond, frames

    def test_every_memory_plan_makes_a_video(self):
        for mode in ("full", "quality", "balanced", "low"):
            with self.subTest(mode=mode):
                backend = self.backend(mode)
                _, _, frames = self.run_job(backend)
                self.assertEqual(len(frames), 9)
                self.assertTrue(all(isinstance(f, Image.Image) and f.size == (256, 288) for f in frames))
                self.assertEqual(backend.stats()["memory_plan"], backend.device_info()["memory_plan"])
                self.assertGreater(backend.stats()["peak_vram_bytes"], 0)
                backend.unload()

    def test_the_balanced_plan_stores_fp8_and_puts_the_transformer_back_after_decoding(self):
        import torch

        backend = self.backend("balanced")
        transformer = backend._pipe.transformer
        self.run_job(backend)
        weights = [p for n, p in transformer.named_parameters() if "blocks" in n and n.endswith("weight") and p.ndim == 2]
        self.assertTrue(weights)
        self.assertTrue(all(p.dtype == torch.float8_e4m3fn for p in weights))
        self.assertEqual(weights[0].device.type, "cuda")
        self.assertEqual(backend._pipe.vae.device.type, "cpu")      # back in RAM after decoding
        self.assertEqual(backend._pipe.text_encoder.device.type, "cpu")

    def test_guidance_one_or_less_skips_the_negative_prompt(self):
        backend = self.backend("balanced")
        _, cond, _ = self.run_job(backend, guidance=1.0)
        self.assertIsNone(cond["negative"])
        _, cond, _ = self.run_job(backend, guidance=5.0, negative_prompt="blurry")
        self.assertIsNotNone(cond["negative"])

    def test_image_to_video_runs_on_the_loaded_components(self):
        backend = self.backend("balanced")
        start = Image.new("RGB", (256, 288), (200, 30, 30))
        _, _, frames = self.run_job(backend, start_frame=start)
        self.assertEqual(len(frames), 9)
        i2v = backend._i2v
        self.assertIs(i2v.transformer, backend._pipe.transformer)  # shared, nothing new loaded
        self.assertTrue(i2v.config.expand_timesteps)

    def test_the_same_seed_gives_the_same_video(self):
        backend = self.backend("quality")
        _, _, a = self.run_job(backend)
        _, _, b = self.run_job(backend)
        self.assertEqual(a[4].tobytes(), b[4].tobytes())

    def test_an_unverified_folder_is_refused_before_anything_loads(self):
        from image_engine.backends.video import VideoBackend

        backend = VideoBackend(lambda: Settings(video_model_dir=str(self.folder.parent / "missing")))
        with self.assertRaises(BackendError) as ctx:
            backend.load(lambda _m: None)
        self.assertIn("--model video", str(ctx.exception))

    def test_the_original_model_keeps_its_own_scheduler(self):
        backend = self.backend("balanced")
        self.assertEqual(backend.profile.name, "standard")
        self.assertEqual(backend.model_id, "ti2v-5b")
        self.assertEqual(type(backend._pipe.scheduler).__name__, "UniPCMultistepScheduler")


@unittest.skipUnless(_has_diffusers(), "needs PyTorch and diffusers (run with the image engine's environment)")
class DmdSchedulerTests(unittest.TestCase):
    """The few-step schedule on the CPU: no model needed."""

    def scheduler(self):
        from image_engine.backends.video import make_dmd_scheduler

        s = make_dmd_scheduler((1.0, 0.757, 0.522))
        s.set_timesteps(50, device="cpu")  # the step count asked for is ignored
        s.set_begin_index(0)
        return s

    def test_it_runs_the_fixed_levels_whatever_step_count_is_asked(self):
        import torch

        s = self.scheduler()
        self.assertEqual([round(float(t)) for t in s.timesteps], [1000, 757, 522])
        self.assertTrue(torch.allclose(s.sigmas, torch.tensor([1.0, 0.757, 0.522, 0.0])))

    def test_each_step_noises_the_clean_estimate_again_and_the_last_returns_it(self):
        import torch

        s = self.scheduler()
        x = torch.randn(1, 4, 2, 4, 4, generator=torch.Generator().manual_seed(1))
        v = torch.randn(1, 4, 2, 4, 4, generator=torch.Generator().manual_seed(2))
        s.generator = torch.Generator().manual_seed(3)
        x0 = x - 1.0 * v
        out = s.step(v, s.timesteps[0], x, return_dict=False)[0]
        noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.allclose(out, (1 - 0.757) * x0 + 0.757 * noise, atol=1e-5))
        out = s.step(v, s.timesteps[1], out, return_dict=False)[0]
        last_in = out
        out = s.step(v, s.timesteps[2], last_in, return_dict=False)[0]
        self.assertTrue(torch.allclose(out, last_in - 0.522 * v, atol=1e-5))  # sigma 0: the clean estimate itself

    def test_the_same_generator_seed_gives_the_same_noise(self):
        import torch

        x, v = torch.ones(1, 4, 2, 4, 4), torch.zeros(1, 4, 2, 4, 4)
        results = []
        for _ in range(2):
            s = self.scheduler()
            s.generator = torch.Generator().manual_seed(7)
            results.append(s.step(v, s.timesteps[0], x, return_dict=False)[0])
        self.assertTrue(torch.equal(results[0], results[1]))


@unittest.skipUnless(HAS_CUDA, "needs PyTorch with CUDA and diffusers (run with the image engine's environment)")
class TinyDistilledVideoBackendTests(unittest.TestCase):
    """The same tiny model, labelled as the 3-step distilled one."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name) / "video-model-fast"
        build_tiny_model(cls.folder, distilled=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def backend(self, mode: str = "balanced"):
        from image_engine.backends.video import VideoBackend

        backend = VideoBackend(lambda: Settings(video_model_dir=str(self.folder), video_memory_mode=mode))
        backend.load(lambda _m: None)
        self.addCleanup(backend.unload)
        return backend

    def run_job(self, backend, start_frame=None, **fields):
        # Asks for 50 steps with guidance, as a form left on the original model's settings would.
        req = parse_video_request({"prompt": "a wave at night", "width": 256, "height": 288, "frames": 9,
                                   "steps": 50, "guidance": 5.0, "seed": 1, **fields})
        steps: list[tuple[int, int]] = []
        cond = backend.encode(req.prompt, req.negative_prompt, 1.0)
        latents = backend.sample(cond, req, req.seed, lambda d, t: steps.append((d, t)), start_frame=start_frame)
        frames = backend.decode(latents)
        self.assertEqual(steps, [(1, 3), (2, 3), (3, 3)])  # three steps, whatever was asked
        return frames

    def test_the_folder_is_recognised_and_gets_the_few_step_schedule(self):
        backend = self.backend()
        self.assertEqual((backend.profile.name, backend.model_id), ("fast", "ti2v-5b-fast"))
        self.assertEqual(type(backend._pipe.scheduler).__name__, "DmdScheduler")
        self.assertEqual(type(backend._pipe).__name__, "WanPipeline")  # the folder's label is never imported

    def test_it_makes_a_video_on_every_memory_plan(self):
        for mode in ("quality", "balanced", "low"):
            with self.subTest(mode=mode):
                frames = self.run_job(self.backend(mode))
                self.assertEqual(len(frames), 9)
                self.assertTrue(all(f.size == (256, 288) for f in frames))

    def test_the_same_seed_gives_the_same_video_and_another_seed_does_not(self):
        backend = self.backend("quality")
        a, b, c = self.run_job(backend), self.run_job(backend), self.run_job(backend, seed=2)
        self.assertEqual(a[4].tobytes(), b[4].tobytes())
        self.assertNotEqual(a[4].tobytes(), c[4].tobytes())

    def test_image_to_video_shares_the_schedule(self):
        backend = self.backend()
        frames = self.run_job(backend, start_frame=Image.new("RGB", (256, 288), (200, 30, 30)))
        self.assertEqual(len(frames), 9)
        self.assertIs(backend._i2v.scheduler, backend._pipe.scheduler)

    def test_an_unknown_model_is_refused(self):
        from image_engine.backends.video import VideoBackend

        other = Path(self.tmp.name) / "other-model"
        other.mkdir()
        (other / "model_index.json").write_text(json.dumps({"_class_name": "SomethingElsePipeline"}), encoding="utf-8")
        (other / "manifest.json").write_text(json.dumps({"files": [{"path": "model_index.json",
                                                                    "size": (other / "model_index.json").stat().st_size}]}),
                                             encoding="utf-8")
        backend = VideoBackend(lambda: Settings(video_model_dir=str(other)))
        with self.assertRaises(BackendError) as ctx:
            backend.load(lambda _m: None)
        self.assertIn("SomethingElsePipeline", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
