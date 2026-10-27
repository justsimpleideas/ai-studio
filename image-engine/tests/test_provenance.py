import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from image_engine.__main__ import APP_VERSION_ENV, app_label
from image_engine.backends.diffusion import _provenance


class ProvenanceTests(unittest.TestCase):
    def test_model_revision_comes_from_the_download_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "manifest.json").write_text(json.dumps({"repo": "org/model", "revision": "a" * 40}))
            self.assertEqual(_provenance(Path(tmp))["model_revision"], f"org/model@{'a' * 40}")

    def test_a_folder_without_a_manifest_records_no_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotIn("model_revision", _provenance(Path(tmp)))
            (Path(tmp) / "manifest.json").write_text("not json")
            self.assertNotIn("model_revision", _provenance(Path(tmp)))

    def test_app_label_uses_the_studio_version_when_given_a_plain_one(self):
        with mock.patch.dict(os.environ, {APP_VERSION_ENV: "0.1.0"}):
            self.assertEqual(app_label(), "Local AI Studio 0.1.0")
        with mock.patch.dict(os.environ, {APP_VERSION_ENV: "0.1.0\nInjected: yes"}):
            self.assertTrue(app_label().startswith("image engine "))


if __name__ == "__main__":
    unittest.main()
