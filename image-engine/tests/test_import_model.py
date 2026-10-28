import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from image_engine import video_models
from image_engine.image_models import BASE_6B, LARGE, LARGE_FAST, TURBO, profile_of
from image_engine.model_check import check_model_folder
from tests.large_fixture import (TINY, TINY_14B_IMAGE, TINY_14B_TEXT, six_b_config_json, six_b_entries,
                                 tiny_video_kinds, video_config_json, video_entries, write_large_file,
                                 write_safetensors)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_model  # noqa: E402
import import_model  # noqa: E402


def verified_folder(folder: Path, repo: str, revision: str, files: dict[str, bytes]) -> Path:
    """A folder as fetch_model.py leaves it: the files plus a manifest naming repo@revision."""
    entries = []
    for name, data in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (folder / "manifest.json").write_text(json.dumps({"repo": repo, "revision": revision, "files": entries}))
    return folder


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "downloads").mkdir()
        self.transformer = write_large_file(root / "downloads" / "model-fp8.safetensors")
        config = json.dumps({"text_config": {"hidden_size": TINY.text_hidden_dim, "num_hidden_layers": 36}}).encode()
        text_files = {name: (config if name == "config.json" else name.encode()) for name in import_model.TEXT_ENCODER_FILES}
        self.text = verified_folder(root / "caption-model", fetch_model.CAPTION_REPO, fetch_model.CAPTION_REVISION, text_files)
        self.vae = verified_folder(root / "large-vae", fetch_model.LARGE_VAE_REPO, fetch_model.LARGE_VAE_REVISION,
                                   {name: name.encode() for name in import_model.VAE_FILES})
        self.dest = root / "models" / "large-model"

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *extra, variant="base"):
        args = ["--transformer", str(self.transformer), "--variant", variant, "--text-encoder-from", str(self.text),
                "--vae-from", str(self.vae), "--dest", str(self.dest), *extra]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = import_model.main(args)
        return code, out.getvalue() + err.getvalue()

    def verify(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            return import_model.main(["--dest", str(self.dest), "--verify-only"]), out.getvalue()

    def test_an_import_makes_a_folder_the_engine_loads(self):
        digest = hashlib.sha256(self.transformer.read_bytes()).hexdigest()
        code, log = self.run_script("--sha256", digest.upper())
        self.assertEqual(code, 0, log)
        self.assertEqual(profile_of(self.dest), LARGE)
        manifest = check_model_folder(self.dest)  # what the engine checks before every load
        self.assertEqual(manifest["revision"], digest)
        self.assertEqual(manifest["sources"]["text_encoder"], f"{fetch_model.CAPTION_REPO}@{fetch_model.CAPTION_REVISION}")
        self.assertTrue(manifest["sources"]["transformer"]["published_sha256_checked"])
        self.assertEqual(manifest["transformer"]["storage"], "fp8")
        for path in ("transformer/model-fp8.safetensors", "text_encoder/model-00001-of-00002.safetensors",
                     "tokenizer/tokenizer.json", "vae/diffusion_pytorch_model.safetensors"):
            self.assertTrue((self.dest / path).is_file(), path)
        self.assertEqual(self.verify()[0], 0)

        # Running it again keeps what is there; a distilled import is labelled as such.
        code, log = self.run_script(variant="distilled")
        self.assertEqual(code, 0, log)
        self.assertIn("kept", log)
        self.assertEqual(profile_of(self.dest), LARGE_FAST)

    def test_a_changed_file_fails_verification(self):
        self.assertEqual(self.run_script()[0], 0)
        target = self.dest / "vae" / "config.json"
        target.unlink()  # a hard link: replace it rather than write through to the source
        target.write_bytes(b"vae/config.jsoX")
        code, log = self.verify()
        self.assertEqual(code, 1)
        self.assertIn("hash mismatch", log)

    def test_a_wrong_published_hash_is_refused_before_anything_is_written(self):
        code, log = self.run_script("--sha256", "0" * 64)
        self.assertEqual(code, 1)
        self.assertIn("SHA-256 mismatch", log)
        self.assertFalse(self.dest.exists())

    def test_sources_must_be_the_pinned_downloads(self):
        manifest = json.loads((self.vae / "manifest.json").read_text())
        manifest["revision"] = "f" * 40
        (self.vae / "manifest.json").write_text(json.dumps(manifest))
        code, log = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("not the VAE (--model large-vae) this model needs", log)

    def test_a_source_changed_since_it_was_verified_is_refused(self):
        (self.text / "tokenizer.json").write_bytes(b"something else entirely")
        code, log = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("changed since it was verified", log)

    def test_a_malformed_transformer_is_refused(self):
        self.transformer.write_bytes(b"PK\x03\x04" + bytes(64))
        code, log = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("zip archive", log)

    def test_strangers_in_the_destination_are_refused(self):
        (self.dest / "vae").mkdir(parents=True)
        (self.dest / "vae" / "extra.json").write_text("{}")
        code, log = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("did not put there", log)


class SixBImportTests(unittest.TestCase):
    """A single file of the 6B model, put together with the rest of the downloaded 6B folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "downloads").mkdir()
        self.transformer = root / "downloads" / "six-b-fp8.safetensors"
        write_safetensors(self.transformer, six_b_entries())
        parts = {}
        for name in fetch_model.MODEL_FILES["image-parts"]:
            if name == "transformer/config.json":
                parts[name] = json.dumps(six_b_config_json()).encode()
            elif name == "model_index.json":
                parts[name] = json.dumps({"_class_name": "ZImagePipeline", "transformer": ["diffusers", "ZImageTransformer2DModel"]}).encode()
            else:
                parts[name] = name.encode()
        self.parts = verified_folder(root / "image-parts", fetch_model.DEFAULT_REPO, fetch_model.DEFAULT_REVISION, parts)
        self.dest = root / "models" / "my-6b"

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = import_model.main([str(a) for a in args])
        return code, out.getvalue()

    def test_a_distilled_import_is_the_6b_model_with_its_parts(self):
        code, log = self.run_script("--transformer", self.transformer, "--variant", "distilled",
                                    "--parts-from", self.parts, "--dest", self.dest)
        self.assertEqual(code, 0, log)
        self.assertIs(profile_of(self.dest), TURBO)
        manifest = check_model_folder(self.dest)
        self.assertEqual(manifest["family"], "diffusion")
        index = json.loads((self.dest / "model_index.json").read_text())
        self.assertEqual((index["_class_name"], index["transformer_file"], index["is_distilled"]),
                         ("ZImagePipeline", "six-b-fp8.safetensors", True))
        self.assertEqual(index["transformer"], ["diffusers", "ZImageTransformer2DModel"])  # the rest is the download's
        for path in ("transformer/config.json", "text_encoder/model-00002-of-00003.safetensors", "scheduler/scheduler_config.json"):
            self.assertTrue((self.dest / path).is_file(), path)
        verify = io.StringIO()
        with contextlib.redirect_stdout(verify), contextlib.redirect_stderr(verify):
            self.assertEqual(import_model.main(["--dest", str(self.dest), "--verify-only"]), 0, verify.getvalue())

    def test_a_base_import_and_the_full_download_as_parts(self):
        code, log = self.run_script("--transformer", self.transformer, "--variant", "base",
                                    "--parts-from", self.parts, "--dest", self.dest)
        self.assertEqual(code, 0, log)
        self.assertIs(profile_of(self.dest), BASE_6B)

    def test_parts_whose_config_does_not_fit_are_refused(self):
        config = self.parts / "transformer" / "config.json"
        config.write_text(json.dumps({**six_b_config_json(), "dim": 3840}))
        manifest = json.loads((self.parts / "manifest.json").read_text())
        for entry in manifest["files"]:
            if entry["path"] == "transformer/config.json":
                entry["size"] = config.stat().st_size
        (self.parts / "manifest.json").write_text(json.dumps(manifest))
        code, log = self.run_script("--transformer", self.transformer, "--variant", "distilled",
                                    "--parts-from", self.parts, "--dest", self.dest)
        self.assertEqual(code, 1)
        self.assertIn("dim", log)

    def test_the_wrong_parts_for_the_file_are_named(self):
        code, log = self.run_script("--transformer", self.transformer, "--variant", "distilled",
                                    "--text-encoder-from", self.parts, "--vae-from", self.parts, "--dest", self.dest)
        self.assertEqual(code, 1)
        self.assertIn("--parts-from", log)

    def test_probe_says_what_the_file_is(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(import_model.main(["--transformer", str(self.transformer), "--probe", "--json"]), 0)
        events = [json.loads(line) for line in out.getvalue().splitlines()]
        probe = next(e for e in events if e["event"] == "probe")
        self.assertEqual((probe["family"], probe["label"], probe["storage"], probe["layers"]),
                         ("diffusion", "6B image model", "fp8", 2))
        out = io.StringIO()
        self.transformer.write_bytes(bytes([0x80, 0x04]) + b"not a model")
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(import_model.main(["--transformer", str(self.transformer), "--probe", "--json"]), 1)
        self.assertEqual(json.loads(out.getvalue().splitlines()[-1])["event"], "error")


class VideoImportTests(unittest.TestCase):
    """A single file of the video model, put together with the rest of the downloaded video folder."""

    def setUp(self):
        kinds = tiny_video_kinds()
        kinds.start()
        self.addCleanup(kinds.stop)
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "downloads").mkdir()
        self.transformer = root / "downloads" / "video-bf16.safetensors"
        write_safetensors(self.transformer, video_entries())
        parts = {}
        for name in fetch_model.MODEL_FILES["video-parts"]:
            if name == "transformer/config.json":
                parts[name] = json.dumps(video_config_json()).encode()
            elif name == "model_index.json":
                parts[name] = json.dumps({"_class_name": "WanPipeline", "expand_timesteps": True,
                                          "transformer": ["diffusers", "WanTransformer3DModel"]}).encode()
            else:
                parts[name] = name.encode()
        self.parts = verified_folder(root / "video-parts", fetch_model.VIDEO_REPO, fetch_model.VIDEO_REVISION, parts)
        self.dest = root / "models" / "my-video"

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = import_model.main([str(a) for a in args])
        return code, out.getvalue()

    def import_as(self, variant):
        return self.run_script("--transformer", self.transformer, "--variant", variant,
                               "--parts-from", self.parts, "--dest", self.dest)

    def test_a_base_import_is_the_video_model_with_its_parts(self):
        code, log = self.import_as("base")
        self.assertEqual(code, 0, log)
        self.assertIn("as the video model folder", log)
        self.assertIs(video_models.profile_of(self.dest), video_models.STANDARD)
        manifest = check_model_folder(self.dest)
        self.assertEqual((manifest["family"], manifest["sources"]["parts"]),
                         ("video", f"{fetch_model.VIDEO_REPO}@{fetch_model.VIDEO_REVISION}"))
        index = json.loads((self.dest / "model_index.json").read_text())
        self.assertEqual((index["transformer_file"], index["expand_timesteps"]), ("video-bf16.safetensors", True))
        for path in ("transformer/config.json", "tokenizer/spiece.model", "text_encoder/model-00003-of-00003.safetensors",
                     "vae/diffusion_pytorch_model.safetensors"):
            self.assertTrue((self.dest / path).is_file(), path)
        verify = io.StringIO()
        with contextlib.redirect_stdout(verify), contextlib.redirect_stderr(verify):
            self.assertEqual(import_model.main(["--dest", str(self.dest), "--verify-only"]), 0, verify.getvalue())

    def test_a_distilled_import_runs_the_fixed_three_step_schedule(self):
        code, log = self.import_as("distilled")
        self.assertEqual(code, 0, log)
        self.assertIs(video_models.profile_of(self.dest), video_models.FAST)

    def test_parts_of_another_model_or_a_config_that_does_not_fit_are_refused(self):
        image_parts = verified_folder(Path(self.tmp.name) / "image-parts", fetch_model.DEFAULT_REPO,
                                      fetch_model.DEFAULT_REVISION, {"model_index.json": b"{}"})
        code, log = self.run_script("--transformer", self.transformer, "--variant", "base",
                                    "--parts-from", image_parts, "--dest", self.dest)
        self.assertEqual(code, 1)
        self.assertIn("not the video model's parts", log)

        config = self.parts / "transformer" / "config.json"
        config.write_text(json.dumps({**video_config_json(), "num_layers": 30}))
        manifest = json.loads((self.parts / "manifest.json").read_text())
        for entry in manifest["files"]:
            if entry["path"] == "transformer/config.json":
                entry["size"] = config.stat().st_size
        (self.parts / "manifest.json").write_text(json.dumps(manifest))
        code, log = self.import_as("base")
        self.assertEqual(code, 1)
        self.assertIn("num_layers", log)
        self.assertFalse(self.dest.exists())

    def test_probe_says_it_is_a_video_model(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(import_model.main(["--transformer", str(self.transformer), "--probe", "--json"]), 0)
        probe = next(e for e in map(json.loads, out.getvalue().splitlines()) if e["event"] == "probe")
        self.assertEqual((probe["family"], probe["storage"], probe["layers"]), ("video", "BF16", 2))

    def test_a_video_model_of_another_size_is_refused_before_any_part_is_needed(self):
        from image_engine.single_file import VideoConfig

        # Another size of the family (wider, in 16 latent channels): no parts for it.
        other = VideoConfig(16, 16, 48, 64, 32, 256, 2, (1, 2, 2))
        write_safetensors(self.transformer, video_entries(other))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(import_model.main(["--transformer", str(self.transformer), "--probe", "--json"]), 1)
        error = json.loads(out.getvalue().splitlines()[-1])
        self.assertEqual(error["event"], "error")
        self.assertIn("not one the studio has the other parts for", error["message"])

    def test_one_file_of_the_5b_model_takes_no_second_expert(self):
        second = Path(self.tmp.name) / "downloads" / "other.safetensors"
        write_safetensors(second, video_entries())
        code, log = self.run_script("--transformer", self.transformer, "--transformer-2", second, "--variant", "base",
                                    "--parts-from", self.parts, "--dest", self.dest)
        self.assertEqual(code, 1)
        self.assertIn("one file", log)


class FourteenBImportTests(unittest.TestCase):
    """A 14B video model: two expert files, put together with the parts of the 14B download."""

    def setUp(self):
        kinds = tiny_video_kinds()
        kinds.start()
        self.addCleanup(kinds.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "downloads").mkdir()
        self.dest = self.root / "models" / "my-14b"

    def tearDown(self):
        self.tmp.cleanup()

    def experts(self, config=TINY_14B_TEXT, **kwargs):
        paths = []
        for which in ("high_noise", "low_noise"):
            path = self.root / "downloads" / f"{which}.safetensors"
            write_safetensors(path, video_entries(config, **kwargs))
            paths.append(path)
        return paths

    def parts(self, kind="14b-text", config=TINY_14B_TEXT):
        repo, revision, parts_id, pipeline = import_model.VIDEO_SOURCES[kind]
        files = {}
        for name in fetch_model.MODEL_FILES[parts_id]:
            if name in ("transformer/config.json", "transformer_2/config.json"):
                files[name] = json.dumps(video_config_json(config)).encode()
            elif name == "model_index.json":
                boundary = 0.875 if kind == "14b-text" else 0.9
                files[name] = json.dumps({"_class_name": pipeline, "boundary_ratio": boundary}).encode()
            else:
                files[name] = name.encode()
        return verified_folder(self.root / parts_id, repo, revision, files)

    def run_script(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = import_model.main([str(a) for a in args])
        return code, out.getvalue()

    def test_two_experts_of_the_text_model_in_the_older_fp8_layout(self):
        high, low = self.experts(older_fp8=True, prefix="")
        code, log = self.run_script("--transformer", high, "--transformer-2", low, "--variant", "base",
                                    "--parts-from", self.parts(), "--dest", self.dest)
        self.assertEqual(code, 0, log)
        self.assertIs(video_models.profile_of(self.dest), video_models.TEXT_14B)
        manifest = check_model_folder(self.dest)
        self.assertEqual(set(manifest["sources"]), {"transformer", "transformer_2", "parts"})
        self.assertEqual(manifest["transformer_2"]["storage"], "fp8")
        index = json.loads((self.dest / "model_index.json").read_text())
        self.assertEqual((index["transformer_file"], index["transformer_2_file"]), ("high_noise.safetensors", "low_noise.safetensors"))
        self.assertTrue((self.dest / "transformer_2" / "low_noise.safetensors").is_file())
        verify = io.StringIO()
        with contextlib.redirect_stdout(verify), contextlib.redirect_stderr(verify):
            self.assertEqual(import_model.main(["--dest", str(self.dest), "--verify-only"]), 0, verify.getvalue())

    def test_a_distilled_image_model_records_its_steps_and_shift(self):
        high, low = self.experts(TINY_14B_IMAGE)
        parts = self.parts("14b-image", TINY_14B_IMAGE)
        args = ["--transformer", high, "--transformer-2", low, "--variant", "distilled", "--parts-from", parts,
                "--dest", self.dest]
        code, log = self.run_script(*args)
        self.assertEqual(code, 1)
        self.assertIn("--steps and --shift", log)
        code, log = self.run_script(*args, "--steps", "4", "--shift", "5")
        self.assertEqual(code, 0, log)
        profile = video_models.profile_of(self.dest)
        self.assertEqual((profile.name, profile.steps, profile.shift, profile.start, profile.fixed),
                         ("image-14b-few", 4, 5.0, "image", True))

    def test_mismatched_or_missing_experts_are_refused(self):
        high, _ = self.experts()
        image_low = self.root / "downloads" / "image_low.safetensors"
        write_safetensors(image_low, video_entries(TINY_14B_IMAGE))
        for extra, fragment in ((["--transformer-2", image_low], "not experts of the same model"),
                                ([], "two files, one per expert"),
                                (["--transformer-2", high], "two different files")):
            with self.subTest(fragment=fragment):
                code, log = self.run_script("--transformer", high, *extra, "--variant", "base",
                                            "--parts-from", self.parts(), "--dest", self.dest)
                self.assertEqual(code, 1)
                self.assertIn(fragment, log)

    def test_the_parts_of_the_other_14b_model_are_refused(self):
        high, low = self.experts()
        code, log = self.run_script("--transformer", high, "--transformer-2", low, "--variant", "base",
                                    "--parts-from", self.parts("14b-image", TINY_14B_TEXT), "--dest", self.dest)
        self.assertEqual(code, 1)
        self.assertIn("video-14b-text-parts", log)


if __name__ == "__main__":
    unittest.main()
