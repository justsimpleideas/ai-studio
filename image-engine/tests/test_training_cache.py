"""The latent cache, with a fake encoder (and a tiny random-weight VAE on the CPU).

Needs torch and safetensors, so it is skipped in a plain Python. Run it with the
image engine's environment:

    <venv>/python -m unittest tests.test_training_cache
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_engine.training.subject import CARS
from image_engine.training.dataset import scan
from tests.test_training_dataset import photo

HAS_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "safetensors"))
HAS_DIFFUSERS = HAS_TORCH and importlib.util.find_spec("diffusers") is not None


@unittest.skipUnless(HAS_TORCH, "needs torch and safetensors")
class LatentCacheTests(unittest.TestCase):
    def setUp(self):
        from image_engine.training import cache

        self.cache = cache
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "photos"
        self.cache_dir = base / "cache"
        photo(self.root / "ex" / "m" / "wide.jpg", size=(1600, 900), seed=1)
        photo(self.root / "ex" / "m" / "tall.jpg", size=(600, 1000), seed=2)
        photo(self.root / "ex" / "m" / "tiny.jpg", size=(100, 100), seed=3)
        self.dataset = scan(self.root, CARS, holdout_percent=0)
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def encoder(self, image):
        import torch

        self.calls.append(image.size)
        return torch.full((16, image.height // 8, image.width // 8), float(len(self.calls)))

    def build(self, **kwargs):
        return self.cache.build(self.dataset, self.cache_dir, self.encoder, kwargs.pop("key", "vae@1"),
                                target_side=512, **kwargs)

    def test_encodes_usable_photos_into_their_buckets(self):
        entries = self.build()
        self.assertEqual(len(entries), 2)  # tiny.jpg is too small
        tall, wide = entries  # dataset order is path order
        ids = {i.path: i.id for i in self.dataset.items}
        self.assertEqual((tall.item_id, wide.item_id), (ids["ex/m/tall.jpg"], ids["ex/m/wide.jpg"]))
        self.assertGreater(wide.bucket[0], wide.bucket[1])
        self.assertGreater(tall.bucket[1], tall.bucket[0])
        self.assertEqual(self.calls, [tall.bucket, wide.bucket])
        latent = self.cache.load_latent(self.cache_dir, wide)
        self.assertEqual(tuple(latent.shape), (16, wide.bucket[1] // 8, wide.bucket[0] // 8))
        self.assertEqual(self.cache.read_index(self.cache_dir, "vae@1"), entries)

    def test_rebuild_skips_cached_entries_and_a_new_vae_gets_new_ones(self):
        self.build()
        self.build()
        self.assertEqual(len(self.calls), 2)
        self.build(key="vae@2")
        self.assertEqual(len(self.calls), 4)
        with self.assertRaises(self.cache.CacheError):
            self.cache.read_index(self.cache_dir, "vae@1")  # the index now belongs to vae@2

    def test_holdout_photos_are_left_out_unless_asked(self):
        self.dataset.items[0].holdout = True
        self.assertEqual(len(self.build()), 1)
        self.assertEqual(len(self.build(include_holdout=True)), 2)

    def test_stopping_keeps_finished_work_and_resumes(self):
        stops = iter([False, True])
        with self.assertRaises(self.cache.Stopped):
            self.build(should_stop=lambda: next(stops))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.cache.read_index(self.cache_dir, "vae@1")), 1)
        self.build()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(list(self.cache_dir.rglob("*.tmp")), [])

    def test_prepare_image_is_upright_and_exactly_the_bucket(self):
        path = photo(self.root / "rot.jpg", size=(800, 600), exif_orientation=6)
        img = self.cache.prepare_image(path, (448, 576))
        self.assertEqual(img.size, (448, 576))
        self.assertEqual(img.mode, "RGB")

    def test_progress(self):
        seen = []
        self.build(on_progress=lambda d, t: seen.append((d, t)))
        self.assertEqual(seen, [(1, 2), (2, 2)])


@unittest.skipUnless(HAS_DIFFUSERS, "needs diffusers")
class VaeEncoderTests(unittest.TestCase):
    def test_encoder_is_the_inverse_of_the_backend_decode_normalisation(self):
        import torch
        from diffusers import AutoencoderKL

        from image_engine.training.cache import vae_encoder

        torch.manual_seed(0)
        vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                            down_block_types=("DownEncoderBlock2D",) * 2,
                            up_block_types=("UpDecoderBlock2D",) * 2,
                            block_out_channels=(8, 16), layers_per_block=1, norm_num_groups=4,
                            scaling_factor=0.37, shift_factor=0.11).eval()
        image = Image.new("RGB", (64, 32), (200, 40, 90))
        latent = vae_encoder(vae)(image)
        self.assertEqual(tuple(latent.shape), (4, 16, 32))  # two down blocks: 2x smaller
        with torch.no_grad():
            pixels = torch.tensor([200, 40, 90], dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, 32, 64)
            mean = vae.encode(pixels / 127.5 - 1.0).latent_dist.mode()[0]
        # The backend's decode does latents / scaling_factor + shift_factor.
        torch.testing.assert_close(latent / 0.37 + 0.11, mean)
        # Deterministic: the mean, not a sample.
        torch.testing.assert_close(vae_encoder(vae)(image), latent)


if __name__ == "__main__":
    unittest.main()
