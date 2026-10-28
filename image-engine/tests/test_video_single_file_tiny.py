"""The video model from a single transformer file, on a tiny random-weight model.

The real architecture classes at a few megabytes: the transformer written as one file in
the authors' layout (behind the loader prefix, bf16 or with fp8 linear weights, scales and
quantization tags), the rest of the folder saved by the pipeline itself. This checks the
engine's own code: the name mapping against diffusers' converter, the expected shapes
against the model's parameters, the weights the file loads into, and the backend loading
such a folder through its memory plans. It says nothing about video quality, speed or
real memory use.

Needs PyTorch and diffusers (the backend tests also CUDA), so it is skipped in a plain
Python. Run it with the image engine's environment:

    <venv>/python -m unittest tests.test_video_single_file_tiny
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.large_fixture import PREFIX, TAG, TINY_VIDEO, to_video_original, video_config_json
from tests.test_large_backend_tiny import HAS_CUDA, HAS_TORCH

CONFIG = {k: v for k, v in video_config_json().items() if not k.startswith("_")}


def tiny_transformer():
    import torch
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    model = WanTransformer3DModel(**CONFIG)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 1:
                p.normal_(1.0, 0.1)
    return model.eval()


def original_state(model) -> dict:
    """The model's weights under the authors' names, before diffusers' renaming."""
    return {to_video_original(name): tensor for name, tensor in model.state_dict().items()}


def write_transformer(path: Path, model, fp8: bool = False) -> None:
    import torch
    from safetensors.torch import save_file

    out = {}
    for name, tensor in original_state(model).items():
        if fp8 and name.startswith("blocks.") and name.endswith(".weight") and tensor.ndim == 2:
            scale = tensor.abs().amax().float() / 448
            out[PREFIX + name] = (tensor.float() / scale).to(torch.float8_e4m3fn)
            out[PREFIX + name[: -len(".weight")] + ".weight_scale"] = scale.reshape(())
            out[PREFIX + name[: -len(".weight")] + ".comfy_quant"] = torch.tensor(list(TAG), dtype=torch.uint8)
        else:
            out[PREFIX + name] = tensor.to(torch.bfloat16)
    save_file(out, str(path))


def build_imported_folder(folder: Path, fp8: bool, distilled: bool = False) -> None:
    """A folder as import_model.py leaves it for the video model: the pipeline's own parts, the
    transformer's config.json, and the transformer as one file that model_index.json names."""
    from tests.test_video_backend_tiny import build_tiny_model

    build_tiny_model(folder, distilled=distilled)
    for weights in (folder / "transformer").glob("diffusion_pytorch_model*"):
        weights.unlink()
    name = f"tiny-{'fp8' if fp8 else 'bf16'}.safetensors"
    write_transformer(folder / "transformer" / name, tiny_transformer(), fp8=fp8)
    index = json.loads((folder / "model_index.json").read_text())
    (folder / "model_index.json").write_text(json.dumps({**index, "transformer_file": name}))
    files = [{"path": p.relative_to(folder).as_posix(), "size": p.stat().st_size}
             for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    (folder / "manifest.json").write_text(json.dumps({"repo": f"local/{name}", "revision": "0" * 64, "files": files}))


@unittest.skipUnless(HAS_TORCH, "needs PyTorch and diffusers (run with the image engine's environment)")
class VideoLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_expected_shapes_are_the_models_parameters(self):
        from image_engine import single_file

        actual = {k: tuple(v.shape) for k, v in tiny_transformer().state_dict().items()}
        self.assertEqual(actual, single_file.video_expected_shapes(TINY_VIDEO))

    def test_the_name_mapping_agrees_with_diffusers_converter(self):
        from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers

        from image_engine import single_file

        write_transformer(self.path, tiny_transformer())
        inspection = single_file.inspect(self.path)
        original = {PREFIX + k: v for k, v in original_state(tiny_transformer()).items()}
        theirs = convert_wan_transformer_to_diffusers(dict(original))
        self.assertEqual(set(theirs), set(inspection.names))
        for name, file_name in inspection.names.items():
            self.assertTrue(original[file_name].equal(theirs[name]), name)

    def test_a_bf16_file_loads_the_weights_it_holds(self):
        import torch

        from image_engine import single_file
        from image_engine.transformer_file import build_video

        reference = tiny_transformer()
        write_transformer(self.path, reference)
        model = build_video(self.path, single_file.inspect(self.path), video_config_json())
        ours, theirs = model.state_dict(), reference.state_dict()
        self.assertEqual(set(ours), set(theirs))
        for name in ("blocks.1.attn2.to_out.0.weight", "blocks.0.norm2.bias", "proj_out.weight"):
            self.assertTrue(ours[name].equal(theirs[name].bfloat16()), name)
        # diffusers keeps these in fp32; so does the loader (from the bf16 the file stores).
        for name in ("scale_shift_table", "blocks.0.scale_shift_table", "condition_embedder.time_embedder.linear_1.weight"):
            self.assertEqual(ours[name].dtype, torch.float32, name)
            self.assertTrue(ours[name].equal(theirs[name].bfloat16().float()), name)

    def test_fp8_weights_load_as_the_weights_they_stand_for(self):
        import torch

        from image_engine import fp8, single_file
        from image_engine.transformer_file import build_video

        reference = tiny_transformer()
        write_transformer(self.path, reference, fp8=True)
        model = build_video(self.path, single_file.inspect(self.path), video_config_json())
        for name in ("blocks.0.attn1.to_q", "blocks.1.ffn.net.2"):
            layer = model.get_submodule(name)
            self.assertIsInstance(layer, fp8.ScaledFP8Linear)
            original = reference.get_submodule(name).weight
            error = (layer.dequantized(torch.float32) - original).abs().max() / original.abs().max()
            self.assertLess(error.item(), 0.07)
        self.assertNotIsInstance(model.proj_out, fp8.ScaledFP8Linear)

    def test_a_config_that_does_not_fit_the_file_is_refused(self):
        from image_engine import single_file
        from image_engine.transformer_file import build_video

        write_transformer(self.path, tiny_transformer())
        with self.assertRaises(single_file.SingleFileError):
            build_video(self.path, single_file.inspect(self.path), {**video_config_json(), "num_attention_heads": 4})


@unittest.skipUnless(HAS_CUDA, "needs PyTorch with CUDA and diffusers (run with the image engine's environment)")
class VideoBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.bf16, cls.fp8, cls.fast = root / "my-video", root / "my-video-fp8", root / "my-video-fast"
        build_imported_folder(cls.bf16, fp8=False)
        build_imported_folder(cls.fp8, fp8=True)
        build_imported_folder(cls.fast, fp8=False, distilled=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def backend(self, folder: Path, mode: str):
        from image_engine.backends.video import VideoBackend
        from image_engine.config import Settings

        backend = VideoBackend(lambda: Settings(video_model_dir=str(folder), video_memory_mode=mode))
        messages: list[str] = []
        backend.load(messages.append)
        self.addCleanup(backend.unload)
        self.assertIn("Checking the transformer file", messages)
        return backend

    def make_video(self, backend, steps: int = 2):
        from image_engine.params import parse_video_request

        req = parse_video_request({"prompt": "a wave at night", "width": 256, "height": 288, "frames": 9,
                                   "steps": steps, "seed": 1})
        cond = backend.encode(req.prompt, req.negative_prompt, req.guidance)
        return backend.decode(backend.sample(cond, req, req.seed, lambda d, t: None))

    def test_every_memory_plan_makes_a_video_from_the_single_file(self):
        for folder, modes, storage in ((self.bf16, ("full", "quality", "balanced", "low"), "BF16"),
                                       (self.fp8, ("quality", "balanced", "low"), "fp8")):
            for mode in modes:
                with self.subTest(folder=folder.name, mode=mode):
                    backend = self.backend(folder, mode)
                    frames = self.make_video(backend)
                    self.assertEqual((len(frames), frames[0].size), (9, (256, 288)))
                    self.assertEqual(backend.provenance()["weights"], storage)
                    backend.unload()

    def test_fp8_weights_stay_fp8_on_the_gpu(self):
        import torch

        from image_engine import fp8

        backend = self.backend(self.fp8, "quality")
        layer = backend._pipe.transformer.get_submodule("blocks.0.attn1.to_q")
        self.assertIsInstance(layer, fp8.ScaledFP8Linear)
        self.assertEqual((layer.weight.dtype, layer.weight.device.type), (torch.float8_e4m3fn, "cuda"))

    def test_a_distilled_import_runs_the_fixed_schedule(self):
        backend = self.backend(self.fast, "balanced")
        self.assertEqual((backend.profile.name, backend.model_id), ("fast", "ti2v-5b-fast"))
        self.assertEqual(len(self.make_video(backend, steps=3)), 9)

    def test_a_folder_naming_an_image_model_file_is_refused(self):
        from image_engine.backends import BackendError
        from tests.large_fixture import six_b_entries, write_safetensors

        folder = Path(self.tmp.name) / "wrong-file"
        build_imported_folder(folder, fp8=False)
        target = next((folder / "transformer").glob("tiny-*.safetensors"))
        write_safetensors(target, six_b_entries())
        manifest = json.loads((folder / "manifest.json").read_text())
        for entry in manifest["files"]:
            if entry["path"] == f"transformer/{target.name}":
                entry["size"] = target.stat().st_size
        (folder / "manifest.json").write_text(json.dumps(manifest))
        from image_engine.backends.video import VideoBackend
        from image_engine.config import Settings

        backend = VideoBackend(lambda: Settings(video_model_dir=str(folder)))
        with self.assertRaises(BackendError) as ctx:
            backend.load(lambda _m: None)
        self.assertIn("not a video model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
