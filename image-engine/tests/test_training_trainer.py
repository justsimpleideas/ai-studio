"""The training loop on a tiny random-weight Z-Image transformer, on the CPU.

Checks the loop's own code: the objective's conventions, that loss falls,
exact resume, stopping, caption dropout, the frozen adapter, and the files it
writes. It says nothing about training quality on a real model.

Needs torch and diffusers, so it is skipped in a plain Python. Run it with the
image engine's environment:

    <venv>/python -m unittest tests.test_training_trainer
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAS_DIFFUSERS = all(importlib.util.find_spec(m) for m in ("torch", "safetensors", "diffusers"))


def tiny_transformer():
    import torch
    from diffusers import ZImageTransformer2DModel

    torch.manual_seed(0)
    return ZImageTransformer2DModel(all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4, dim=64,
                                    n_layers=2, n_refiner_layers=1, n_heads=2, n_kv_heads=2, cap_feat_dim=32,
                                    axes_dims=[8, 12, 12], axes_lens=[64, 32, 32])


@unittest.skipUnless(HAS_DIFFUSERS, "needs torch and diffusers")
class TrainerTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import save_file

        from image_engine.training import trainer as tr

        self.torch, self.tr = torch, tr
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        g = torch.Generator().manual_seed(7)
        self.samples = []
        for i in range(3):
            lat, emb = base / f"lat{i}.safetensors", base / f"emb{i}.safetensors"
            save_file({"latent": torch.randn(4, 8, 8, generator=g)}, str(lat))
            save_file({"embedding": torch.randn(5, 32, generator=g)}, str(emb))
            self.samples.append(tr.Sample(lat, emb))
        self.empty = base / "empty.safetensors"
        save_file({"embedding": torch.zeros(1, 32)}, str(self.empty))
        self.base = base

    def tearDown(self):
        self.tmp.cleanup()

    def trainer(self, out="run", model=None, **config):
        defaults = dict(rank=4, learning_rate=3e-3, steps=6, checkpoint_every=3, caption_dropout=0.0,
                        cpu_threads=2, gradient_checkpointing=True)
        cfg = self.tr.TrainConfig(**{**defaults, **config})
        return self.tr.Trainer(model or tiny_transformer(), self.samples, self.empty, self.base / out, cfg,
                               device="cpu", dtype=self.torch.float32)

    def lora_file(self, path):
        from safetensors.torch import load_file

        return load_file(str(path))

    def test_conventions_match_the_pipeline(self):
        # timestep given to the model is 1 - sigma; the target is x0 - noise; sigmas follow the shift.
        self.assertAlmostEqual(self.tr.sample_sigma(0.5, 3.0), 0.75)
        self.assertAlmostEqual(self.tr.sample_sigma(0.5, 1.0), 0.5)
        t = self.trainer()
        t._prepare()
        calls = []
        real = t.model.forward

        def spy(x, timestep, cap, return_dict=True):
            calls.append((x[0].shape, float(timestep[0]), cap[0].shape))
            return real(x, timestep, cap, return_dict=return_dict)

        t.model.forward = spy
        _, info = t.loss_at(0)
        (shape, timestep, cap_shape), = calls
        self.assertEqual(tuple(shape), (4, 1, 8, 8))  # one latent per entry, with a frame axis
        self.assertAlmostEqual(timestep, 1 - info["sigma"], places=3)
        self.assertEqual(tuple(cap_shape), (5, 32))

    def test_loss_falls_on_the_same_draws(self):
        t = self.trainer(steps=40, checkpoint_every=40, learning_rate=1e-2)
        t._prepare()
        with self.torch.no_grad():
            before = sum(float(t.loss_at(s)[0]) for s in range(6))
        result = t.run()
        with self.torch.no_grad():
            after = sum(float(t.loss_at(s)[0]) for s in range(6))
        self.assertEqual(result.status, "finished")
        self.assertEqual(result.step, 40)
        self.assertLess(after, before * 0.9, (before, after))

    def test_only_the_lora_changes(self):
        model = tiny_transformer()
        frozen = {k: v.clone() for k, v in model.state_dict().items()}
        self.trainer(model=model).run()
        from image_engine import lora

        lora.remove(model, self.tr.TRAINED)
        for k, v in model.state_dict().items():
            self.torch.testing.assert_close(v, frozen[k], msg=k)

    def test_resume_gives_exactly_the_same_result(self):
        straight = self.trainer(out="a").run()
        stops = iter([False, False, False, False, True])
        first = self.trainer(out="b").run(should_stop=lambda: next(stops, True))
        self.assertEqual((first.status, first.step), ("stopped", 4))
        second = self.trainer(out="b").run()
        self.assertEqual((second.status, second.step), ("finished", 6))
        a = self.lora_file(self.base / "a" / "checkpoints" / straight.checkpoint)
        b = self.lora_file(self.base / "b" / "checkpoints" / second.checkpoint)
        self.assertEqual(a.keys(), b.keys())
        for k in a:
            self.torch.testing.assert_close(a[k], b[k], rtol=0, atol=0, msg=k)
        self.assertEqual([e["loss"] for e in straight.log], [e["loss"] for e in second.log])

    def test_a_run_can_be_extended_but_not_changed(self):
        self.trainer(out="c").run()
        longer = self.trainer(out="c", steps=8).run()
        self.assertEqual(longer.step, 8)
        with self.assertRaises(self.tr.TrainError):
            self.trainer(out="c", steps=10, learning_rate=1e-3).run()

    def test_checkpoints_and_files_written(self):
        result = self.trainer(out="d", steps=7, checkpoint_every=3).run()
        names = sorted(p.name for p in (self.base / "d" / "checkpoints").iterdir())
        self.assertEqual(names, ["step-000003.safetensors", "step-000006.safetensors", "step-000007.safetensors"])
        self.assertEqual(result.checkpoint, "step-000007.safetensors")
        every = sorted(p.suffix for p in (self.base / "d").rglob("*") if p.is_file())
        self.assertEqual(set(every), {".safetensors", ".json"})  # nothing pickled, no temp files left
        info = json.loads((self.base / "d" / "state" / "trainer.json").read_text())
        self.assertEqual(info["step"], 7)
        self.assertEqual(len(info["log"]), 7)

    def test_checkpoint_loads_into_a_fresh_model_with_the_trained_behaviour(self):
        from image_engine import lora

        model = tiny_transformer()
        t = self.trainer(model=model)
        result = t.run()
        model.eval()
        x = [self.torch.randn(4, 1, 8, 8, generator=self.torch.Generator().manual_seed(3))]
        cap = [self.torch.randn(5, 32, generator=self.torch.Generator().manual_seed(4))]
        ts = self.torch.tensor([0.4])
        with self.torch.no_grad():
            expected = model(x, ts, cap, return_dict=False)[0][0]
        fresh = tiny_transformer().eval()
        from safetensors import safe_open

        with safe_open(str(self.base / "run" / "checkpoints" / result.checkpoint), "pt") as fh:
            tensors = {k: fh.get_tensor(k) for k in fh.keys()}
            meta = fh.metadata()
        lora.load(fresh, "car", tensors, meta, dtype=self.torch.float32)
        with self.torch.no_grad():
            got = fresh(x, ts, cap, return_dict=False)[0][0]
        self.torch.testing.assert_close(got, expected)

    def test_caption_dropout_uses_the_empty_caption(self):
        t = self.trainer(caption_dropout=0.5, steps=40)
        drops = [t._draws(s)[2] for s in range(40)]
        self.assertTrue(8 <= sum(drops) <= 32, sum(drops))
        self.assertEqual(drops, [self.trainer(caption_dropout=0.5)._draws(s)[2] for s in range(40)])
        self.assertFalse(any(self.trainer()._draws(s)[2] for s in range(40)))
        # Every photo is used once per epoch.
        self.assertEqual(sorted(t._draws(s)[0] for s in range(3)), [0, 1, 2])

    def test_frozen_adapter_is_used_but_not_trained_or_saved(self):
        from image_engine import lora

        donor = tiny_transformer()
        lora.add_trainable(donor, "x", rank=2)
        with self.torch.no_grad():
            for m in donor.modules():
                if isinstance(m, lora.LoRALinear):
                    m.lora["x_B"].normal_(std=0.05)
        adapter = lora.state_dict(donor, "x")
        model = tiny_transformer()
        cfg = self.tr.TrainConfig(rank=4, learning_rate=3e-3, steps=4, checkpoint_every=4, cpu_threads=2)
        t = self.tr.Trainer(model, self.samples, self.empty, self.base / "e", cfg, device="cpu",
                            dtype=self.torch.float32, adapter=adapter)
        result = t.run()
        layer = model.layers[0].attention.to_q
        self.assertEqual(set(layer.slots), {"adapter", "subject"})
        self.torch.testing.assert_close(layer.lora["adapter_B"].detach(),
                                        adapter[0]["transformer.layers.0.attention.to_q.lora_B.weight"])
        saved = self.lora_file(self.base / "e" / "checkpoints" / result.checkpoint)
        self.assertTrue(all(v.shape[0] == 4 or v.shape[1] == 4 for v in saved.values()))  # rank 4 only
        # A different adapter is a different run.
        other = (dict(adapter[0]), adapter[1])
        key = "transformer.layers.0.attention.to_q.lora_B.weight"
        other[0][key] = other[0][key] + 1
        with self.assertRaises(self.tr.TrainError):
            self.tr.Trainer(tiny_transformer(), self.samples, self.empty, self.base / "e", cfg, device="cpu",
                            dtype=self.torch.float32, adapter=other).run()

    def test_bad_config_and_no_samples(self):
        for bad in (dict(rank=0), dict(learning_rate=0), dict(steps=0), dict(caption_dropout=1.0), dict(shift=0)):
            with self.assertRaises(self.tr.TrainError, msg=bad):
                self.trainer(**bad)
        with self.assertRaises(self.tr.TrainError):
            self.tr.Trainer(tiny_transformer(), [], self.empty, self.base / "f", self.tr.TrainConfig(), device="cpu")

    def test_cpu_threads_are_capped_during_the_run_and_restored(self):
        before = self.torch.get_num_threads()
        seen = []
        self.trainer(cpu_threads=1, steps=2).run(on_progress=lambda *a: seen.append(self.torch.get_num_threads()))
        self.assertEqual(seen, [1, 1])
        self.assertEqual(self.torch.get_num_threads(), before)


@unittest.skipUnless(HAS_DIFFUSERS, "needs torch and diffusers")
class EmbeddingCacheTests(unittest.TestCase):
    def test_build_skips_cached_and_includes_the_empty_caption(self):
        import torch

        from image_engine.training import embeddings

        calls = []

        def encoder(text):
            calls.append(text)
            return torch.ones(max(1, len(text.split())), 32)

        with tempfile.TemporaryDirectory() as tmp:
            files = embeddings.build(["a red coupe", "a red coupe", "a van"], Path(tmp), encoder, "te@1")
            self.assertEqual(set(files), {"", "a red coupe", "a van"})
            self.assertEqual(calls, ["", "a red coupe", "a van"])
            embeddings.build(["a van"], Path(tmp), encoder, "te@1")
            self.assertEqual(len(calls), 3)
            embeddings.build(["a van"], Path(tmp), encoder, "te@2")
            self.assertEqual(len(calls), 5)
            with self.assertRaises(ValueError):
                embeddings.build(["x"], Path(tmp), lambda t: torch.ones(3), "te@3")


if __name__ == "__main__":
    unittest.main()
