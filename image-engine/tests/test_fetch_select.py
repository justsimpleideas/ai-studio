import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

# scripts/ is not a package; load the script by path. Registered first, as dataclasses require.
_spec = importlib.util.spec_from_file_location(
    "fetch_model", Path(__file__).resolve().parents[1] / "scripts" / "fetch_model.py")
fetch_model = importlib.util.module_from_spec(_spec)
sys.modules["fetch_model"] = fetch_model
_spec.loader.exec_module(fetch_model)


def remote(path, lfs=True):
    return fetch_model.RemoteFile(path=path, size=1, sha256="a" * 64 if lfs else None, git_sha1="b" * 40)


BASE = [remote("model_index.json", lfs=False), remote("transformer/config.json", lfs=False),
        remote("transformer/diffusion_pytorch_model.safetensors")]


class SelectFilesTests(unittest.TestCase):
    def test_a_sentencepiece_vocabulary_is_accepted_in_the_tokenizer_folder(self):
        chosen = fetch_model.select_files([*BASE, remote("tokenizer/spiece.model")])
        self.assertIn("tokenizer/spiece.model", [f.path for f in chosen])

    def test_but_nowhere_else(self):
        with self.assertRaises(fetch_model.VerificationError):
            fetch_model.select_files([*BASE, remote("transformer/weights.model")])

    def test_samples_and_code_are_left_out_or_refused(self):
        chosen = fetch_model.select_files([*BASE, remote("assets/logo.png"), remote("examples/i2v_input.JPG")])
        self.assertEqual(len(chosen), len(BASE))
        with self.assertRaises(fetch_model.VerificationError):
            fetch_model.select_files([*BASE, remote("tokenizer/tokenization.py", lfs=False)])

    def test_stray_files_at_the_top_are_left_out(self):
        chosen = fetch_model.select_files([*BASE, remote("test.txt", lfs=False), remote("README.md", lfs=False)])
        self.assertEqual(len(chosen), len(BASE))

    def test_every_model_is_pinned_to_a_commit(self):
        self.assertEqual(set(fetch_model.MODELS), {"image", "video", "video-fast"})
        for repo, revision in fetch_model.MODELS.values():
            self.assertEqual(len(revision), 40)
            self.assertTrue(all(c in "0123456789abcdef" for c in revision))


class ReuseTests(unittest.TestCase):
    """--reuse-from: a file shared with another model folder is linked, then checked like any other."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.source = Path(self.tmp.name) / "video-model"
        self.dest = Path(self.tmp.name) / "video-model-fast"
        (self.source / "vae").mkdir(parents=True)
        self.data = b"same weights in both models"
        (self.source / "vae" / "weights.safetensors").write_bytes(self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def remote_for(self, data: bytes) -> "fetch_model.RemoteFile":
        return fetch_model.RemoteFile(path="vae/weights.safetensors", size=len(data),
                                      sha256=hashlib.sha256(data).hexdigest(), git_sha1="b" * 40)

    def test_a_file_of_the_same_size_is_linked_and_passes_the_published_hash(self):
        remote = self.remote_for(self.data)
        local = self.dest / remote.path
        self.assertTrue(fetch_model.reuse(self.source, local, remote))
        self.assertEqual(local.read_bytes(), self.data)
        fetch_model.verify_hash(local, remote)  # no error

    def test_a_different_file_of_the_same_size_fails_the_hash_so_it_is_downloaded_instead(self):
        other = b"other weights, same length!"
        self.assertEqual(len(other), len(self.data))
        remote = self.remote_for(other)
        local = self.dest / remote.path
        self.assertTrue(fetch_model.reuse(self.source, local, remote))
        with self.assertRaises(fetch_model.VerificationError):
            fetch_model.verify_hash(local, remote)

    def test_a_missing_or_differently_sized_file_is_not_reused(self):
        remote = self.remote_for(self.data + b"!")
        self.assertFalse(fetch_model.reuse(self.source, self.dest / remote.path, remote))
        missing = fetch_model.RemoteFile(path="text_encoder/x.safetensors", size=1, sha256="a" * 64, git_sha1="b" * 40)
        self.assertFalse(fetch_model.reuse(self.source, self.dest / missing.path, missing))
        self.assertFalse(self.dest.exists())

    def test_reuse_from_the_destination_itself_is_refused(self):
        with self.assertRaises(SystemExit):
            fetch_model.main(["--model", "video-fast", "--dest", str(self.source), "--reuse-from", str(self.source)])


if __name__ == "__main__":
    unittest.main()
