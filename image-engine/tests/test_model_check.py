import json
import tempfile
import unittest
from pathlib import Path

from image_engine.model_check import ModelFolderError, check_model_folder


class ModelCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "model"
        (self.root / "vae").mkdir(parents=True)
        (self.root / "model_index.json").write_text("{}")
        (self.root / "vae" / "w.safetensors").write_bytes(b"x" * 10)
        self.write_manifest()

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self, files=None):
        files = files or [{"path": "model_index.json", "size": 2}, {"path": "vae/w.safetensors", "size": 10}]
        (self.root / "manifest.json").write_text(json.dumps({"files": files}))

    def assertRefused(self, fragment):
        with self.assertRaises(ModelFolderError) as ctx:
            check_model_folder(self.root)
        self.assertIn(fragment, str(ctx.exception))

    def test_verified_folder_passes(self):
        self.assertEqual(len(check_model_folder(self.root)["files"]), 2)

    def test_missing_folder_explains_how_to_fetch(self):
        with self.assertRaises(ModelFolderError) as ctx:
            check_model_folder(self.root / "nope")
        self.assertIn("fetch_model.py", str(ctx.exception))

    def test_missing_manifest(self):
        (self.root / "manifest.json").unlink()
        self.assertRefused("no manifest.json")

    def test_changed_file(self):
        (self.root / "vae" / "w.safetensors").write_bytes(b"x" * 11)
        self.assertRefused("changed since it was verified")

    def test_missing_file(self):
        (self.root / "vae" / "w.safetensors").unlink()
        self.assertRefused("Missing model file")

    def test_pickle_file_is_refused(self):
        (self.root / "vae" / "extra.bin").write_bytes(b"\x80\x04")
        self.assertRefused("executable or pickle")

    def test_python_file_is_refused(self):
        (self.root / "modeling_evil.py").write_text("import os")
        self.assertRefused("executable or pickle")

    def test_unlisted_config_is_refused(self):
        (self.root / "vae" / "config.json").write_text('{"auto_map": {}}')
        self.assertRefused("Unverified file")


if __name__ == "__main__":
    unittest.main()
