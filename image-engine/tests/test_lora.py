"""The in-house LoRA (image_engine/lora.py): maths, file layouts, and a tiny real Z-Image transformer.

Needs torch (and diffusers for the last class), so it is skipped in a plain
Python. Run it with the image engine's environment:

    <venv>/python -m unittest tests.test_lora
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None
HAS_DIFFUSERS = HAS_TORCH and importlib.util.find_spec("diffusers") is not None

if HAS_TORCH:
    import torch
    from torch import nn

    from image_engine import lora
    from image_engine.lora import LoRAError, LoRALinear

    class Attention(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_k = nn.Linear(dim, dim, bias=False)
            self.to_v = nn.Linear(dim, dim, bias=False)
            self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])

        def forward(self, x):
            return self.to_out[0](self.to_q(x) + self.to_k(x) * 0.5 + self.to_v(x) * 0.25)

    class FeedForward(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.w1, self.w3 = nn.Linear(dim, 2 * dim, bias=False), nn.Linear(dim, 2 * dim, bias=False)
            self.w2 = nn.Linear(2 * dim, dim, bias=False)

        def forward(self, x):
            return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

    class Block(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attention, self.feed_forward = Attention(dim), FeedForward(dim)
            self.adaLN_modulation = nn.Sequential(nn.Linear(dim, dim))

        def forward(self, x):
            x = x + self.attention(x)
            return x + self.feed_forward(x) + self.adaLN_modulation(x)

    class Toy(nn.Module):
        """Same module names as the Z-Image transformer, a few hundred weights."""

        def __init__(self, dim=8, n=2):
            super().__init__()
            self.layers = nn.ModuleList([Block(dim) for _ in range(n)])
            self.final = nn.Linear(dim, dim)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.final(x)


def toy(seed=0):
    torch.manual_seed(seed)
    return Toy().eval()


@unittest.skipUnless(HAS_TORCH, "needs torch")
class LoRAMathTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(3, 8, generator=torch.Generator().manual_seed(1))

    def test_default_targets_are_attention_and_feed_forward_of_each_block(self):
        names = lora.target_names(toy())
        self.assertEqual(len(names), 14)
        self.assertIn("layers.0.attention.to_out.0", names)
        self.assertIn("layers.1.feed_forward.w2", names)
        self.assertFalse(any("adaLN" in n or n == "final" for n in names))

    def test_new_lora_is_a_no_op_until_trained(self):
        model = toy()
        before = model(self.x)
        params = lora.add_trainable(model, "car", rank=4)
        self.assertEqual(len(params), 28)
        self.assertTrue(all(p.requires_grad and p.dtype == torch.float32 for p in params))
        torch.testing.assert_close(model(self.x), before)
        self.assertIsInstance(model.layers[0].attention.to_out[0], LoRALinear)
        self.assertNotEqual(float(params[0].detach().abs().sum()), 0.0)  # A is random, B is zero

    def test_only_the_lora_learns_when_the_base_is_frozen(self):
        model = toy()
        for p in model.parameters():
            p.requires_grad_(False)
        params = lora.add_trainable(model, "car", rank=2)
        model(self.x).square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in params[1::2]))  # B gets a gradient at B = 0
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if ".lora." not in n))

    def test_output_is_base_plus_scaled_low_rank_terms(self):
        torch.manual_seed(3)
        layer = nn.Linear(8, 6)
        wrapped = LoRALinear(layer)
        a1, b1 = torch.randn(2, 8), torch.randn(6, 2)
        a2, b2 = torch.randn(4, 8), torch.randn(6, 4)
        wrapped.add("one", a1, b1, alpha=1.0, trainable=False)
        wrapped.add("two", a2, b2, alpha=8.0, trainable=False)
        wrapped.slots["one"].strength = 0.5
        expected = layer(self.x) + 0.5 * (1.0 / 2) * (self.x @ a1.T @ b1.T) + 1.0 * (8.0 / 4) * (self.x @ a2.T @ b2.T)
        torch.testing.assert_close(wrapped(self.x), expected)
        wrapped.slots["two"].strength = 0.0
        torch.testing.assert_close(wrapped(self.x), layer(self.x) + 0.25 * (self.x @ a1.T @ b1.T))

    def test_adapter_removed_trained_lora_stays(self):
        model = toy()
        base = model(self.x)
        lora.add_trainable(model, "adapter", rank=2)
        lora.add_trainable(model, "car", rank=2)
        for p in model.parameters():
            if p.ndim == 2 and p.shape[1] == 2:
                torch.nn.init.normal_(p)  # give every B some weight
        both = model(self.x)
        lora.remove(model, "adapter")
        only_car = model(self.x)
        self.assertEqual(set(lora.adapters(model)), {"car"})
        self.assertFalse(torch.allclose(both, only_car))
        self.assertFalse(torch.allclose(only_car, base))
        lora.remove(model, "car")
        torch.testing.assert_close(model(self.x), base)
        self.assertIs(type(model.layers[0].attention.to_q), nn.Linear)  # the original layer is back

    def test_strength_and_listing(self):
        model = toy()
        lora.add_trainable(model, "car", rank=4, alpha=2)
        lora.set_strength(model, "car", 0.7)
        self.assertEqual(lora.adapters(model)["car"],
                         {"rank": 4, "alpha": 2.0, "strength": 0.7, "trainable": True, "layers": 14})
        with self.assertRaises(LoRAError):
            lora.set_strength(model, "nope", 1)
        with self.assertRaises(LoRAError):
            lora.remove(model, "nope")

    def test_bad_names_and_ranks(self):
        model = toy()
        for bad in ("Car", "a.b", "", "x" * 41):
            with self.assertRaises(LoRAError):
                lora.add_trainable(model, bad, rank=2)
        with self.assertRaises(LoRAError):
            lora.add_trainable(model, "car", rank=0)
        with self.assertRaises(LoRAError):
            lora.add_trainable(model, "car", rank=2, targets=r"^nothing$")
        lora.add_trainable(model, "car", rank=2)
        with self.assertRaises(LoRAError):
            lora.add_trainable(model, "car", rank=2)


def trained(seed=5, rank=3, alpha=6.0):
    model = toy()
    lora.add_trainable(model, "car", rank=rank, alpha=alpha)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora["car_B"].normal_(generator=g)
    return model


@unittest.skipUnless(HAS_TORCH, "needs torch")
class LoRAFileTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(3, 8, generator=torch.Generator().manual_seed(1))
        self.source = trained()
        self.tensors, self.meta = lora.state_dict(self.source, "car")

    def test_file_layout_is_diffusers(self):
        self.assertIn("transformer.layers.0.attention.to_out.0.lora_A.weight", self.tensors)
        self.assertEqual(len(self.tensors), 28)
        meta = json.loads(self.meta[lora.METADATA_KEY])
        self.assertEqual((meta["transformer.r"], meta["transformer.lora_alpha"]), (3, 6.0))
        self.assertEqual(len(meta["transformer.target_modules"]), 14)

    def test_safetensors_round_trip_reproduces_the_output(self):
        from safetensors import safe_open
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "car.safetensors"
            save_file(self.tensors, str(path), metadata=self.meta)
            with safe_open(str(path), "pt") as fh:
                tensors = {k: fh.get_tensor(k) for k in fh.keys()}
                meta = fh.metadata()
        fresh = toy()
        self.assertEqual(lora.load(fresh, "car", tensors, meta, dtype=torch.float32), 14)
        torch.testing.assert_close(fresh(self.x), self.source(self.x))
        self.assertFalse(lora.adapters(fresh)["car"]["trainable"])

    def test_alpha_defaults_to_rank_without_metadata(self):
        fresh = toy()
        lora.load(fresh, "car", self.tensors, None, dtype=torch.float32)
        self.assertEqual(lora.adapters(fresh)["car"]["alpha"], 3.0)

    def test_other_layouts_load_to_the_same_result(self):
        layouts = {
            "comfy": lambda k: k.replace("transformer.", "diffusion_model.", 1),
            "peft": lambda k: k.replace("transformer.", "base_model.model.", 1).replace(".weight", ".default.weight"),
            "bare": lambda k: k[len("transformer."):],
        }
        for label, rename in layouts.items():
            fresh = toy()
            lora.load(fresh, "car", {rename(k): v for k, v in self.tensors.items()}, self.meta, dtype=torch.float32)
            torch.testing.assert_close(fresh(self.x), self.source(self.x), msg=label)

    def test_kohya_layout_with_per_layer_alpha(self):
        kohya = {}
        for key, value in self.tensors.items():
            module = key[len("transformer."):].rsplit(".lora_", 1)[0]
            stem = "lora_unet_" + module.replace(".", "_")
            kohya[f"{stem}.lora_down.weight" if ".lora_A." in key else f"{stem}.lora_up.weight"] = value
            kohya[f"{stem}.alpha"] = torch.tensor(6.0)
        fresh = toy()
        lora.load(fresh, "car", kohya, None, dtype=torch.float32)
        torch.testing.assert_close(fresh(self.x), self.source(self.x))

    def test_bad_files_attach_nothing(self):
        cases = {
            "unplaced": {**self.tensors, "transformer.layers.9.attention.to_q.lora_A.weight": torch.zeros(3, 8)},
            "foreign key": {**self.tensors, "text_encoder.something.weight": torch.zeros(1)},
            "half": {k: v for k, v in self.tensors.items() if not k.endswith("to_q.lora_B.weight")},
            "shape": {k: (torch.zeros(3, 9) if k.endswith("layers.0.attention.to_q.lora_A.weight") else v)
                      for k, v in self.tensors.items()},
            "empty": {},
        }
        for label, tensors in cases.items():
            fresh = toy()
            with self.assertRaises(LoRAError, msg=label):
                lora.load(fresh, "car", tensors, self.meta)
            self.assertEqual(lora.adapters(fresh), {}, label)
            self.assertIs(type(fresh.layers[0].attention.to_q), nn.Linear, label)

    def test_loaded_adapter_next_to_a_trainable_lora(self):
        model = toy()
        lora.load(model, "adapter", self.tensors, self.meta, dtype=torch.float32)
        params = lora.add_trainable(model, "car", rank=2)
        torch.testing.assert_close(model(self.x), self.source(self.x))  # new LoRA starts at zero
        self.assertEqual(len(params), 28)
        out, _ = lora.state_dict(model, "car")
        self.assertEqual(len(out), 28)  # the saved file holds only the trained LoRA
        self.assertTrue(all(v.shape[0] == 2 or v.shape[1] == 2 for v in out.values()))


@unittest.skipUnless(HAS_DIFFUSERS, "needs diffusers")
class ZImageTransformerTests(unittest.TestCase):
    """A real ZImageTransformer2DModel with random weights, a few hundred kB, on the CPU."""

    def build(self):
        from diffusers import ZImageTransformer2DModel

        torch.manual_seed(0)
        return ZImageTransformer2DModel(all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4, dim=64,
                                        n_layers=2, n_refiner_layers=1, n_heads=2, n_kv_heads=2,
                                        cap_feat_dim=32, axes_dims=[8, 12, 12], axes_lens=[64, 32, 32]).eval()

    def run_model(self, model, dtype=None):
        dtype = dtype or torch.float32
        g = torch.Generator().manual_seed(2)
        latents = [torch.randn(4, 1, 8, 8, generator=g).to(dtype)]
        caption = [torch.randn(5, 32, generator=g).to(dtype)]
        with torch.no_grad():
            return model(latents, torch.tensor([0.5]), caption, return_dict=False)[0][0].float()

    def test_default_targets_match_the_real_layer_names(self):
        model = self.build()
        names = lora.target_names(model)
        self.assertEqual(len(names), 2 * 7)
        self.assertIn("layers.1.attention.to_out.0", names)
        self.assertIn("layers.0.feed_forward.w3", names)

    def test_lora_changes_the_output_and_removes_cleanly(self):
        model = self.build()
        base = self.run_model(model)
        lora.add_trainable(model, "car", rank=4)
        torch.testing.assert_close(self.run_model(model), base)
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, LoRALinear):
                    m.lora["car_B"].normal_(std=0.5)
        self.assertGreater(float((self.run_model(model) - base).abs().max()), 1e-3)
        lora.remove(model, "car")
        torch.testing.assert_close(self.run_model(model), base)

    def test_works_on_fp8_stored_weights_and_lora_stays_full_precision(self):
        model = self.build().to(torch.bfloat16)  # as the backend does before enabling fp8 storage
        model.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        base = self.run_model(model, torch.bfloat16)
        params = lora.add_trainable(model, "car", rank=4)
        self.assertTrue(all(p.dtype == torch.float32 for p in params))
        self.assertEqual(model.layers[0].attention.to_q.base.weight.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(self.run_model(model, torch.bfloat16), base)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        g = torch.Generator().manual_seed(2)
        out = model([torch.randn(4, 1, 8, 8, generator=g).bfloat16()], torch.tensor([0.5]),
                    [torch.randn(5, 32, generator=g).bfloat16()], return_dict=False)[0][0]
        out.float().square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in params[1::2]))


if __name__ == "__main__":
    unittest.main()
