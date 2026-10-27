"""--json progress events, read by the studio's download button. No network: the
listing and the download are replaced by fakes."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from tests.test_fetch_select import fetch_model


def safetensors_bytes() -> bytes:
    header = json.dumps({"w": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}).encode()
    return len(header).to_bytes(8, "little") + header + b"\x01\x02\x03\x04"


def git_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


INDEX = b'{"_class_name": "Pipeline"}'
WEIGHTS = safetensors_bytes()
FILES = [
    fetch_model.RemoteFile(path="model_index.json", size=len(INDEX), sha256=None, git_sha1=git_sha1(INDEX)),
    fetch_model.RemoteFile(path="vae/w.safetensors", size=len(WEIGHTS),
                           sha256=hashlib.sha256(WEIGHTS).hexdigest(), git_sha1=git_sha1(WEIGHTS)),
]
CONTENT = {"model_index.json": INDEX, "vae/w.safetensors": WEIGHTS}


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_urlopen(req, timeout=None):
    name = req.full_url.split("/resolve/")[1].split("/", 1)[1]
    return FakeResponse(CONTENT[name])


class JsonProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "model"

    def tearDown(self):
        self.tmp.cleanup()
        fetch_model._json_events = False

    def run_main(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(fetch_model, "list_remote", return_value=FILES), \
                mock.patch.object(fetch_model.urllib.request, "urlopen", side_effect=fake_urlopen), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = fetch_model.main(["--dest", str(self.dest), "--json", *extra])
        events = [json.loads(line) for line in out.getvalue().splitlines()]
        return code, events, err.getvalue()

    def test_stdout_is_only_events_and_the_messages_go_to_stderr(self):
        code, events, err = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(events[0], {"event": "plan", "repo": fetch_model.DEFAULT_REPO,
                                     "revision": fetch_model.DEFAULT_REVISION, "files": 2,
                                     "bytes": len(INDEX) + len(WEIGHTS)})
        self.assertEqual(events[-1], {"event": "done", "files": 2, "bytes": len(INDEX) + len(WEIGHTS)})
        self.assertIn("Verified 2 files", err)

    def test_each_file_is_fetched_reported_with_its_bytes_and_then_marked_ok(self):
        _, events, _ = self.run_main()
        weights = [e for e in events if e.get("path") == "vae/w.safetensors"]
        self.assertEqual([e["event"] for e in weights], ["file", "progress", "verify", "file"])
        self.assertEqual(weights[0]["action"], "fetch")
        self.assertEqual(weights[-1]["action"], "ok")
        self.assertEqual((self.dest / "vae/w.safetensors").read_bytes(), WEIGHTS)

    def test_files_already_present_are_verified_without_downloading(self):
        self.run_main()
        with mock.patch.object(fetch_model.urllib.request, "urlopen", side_effect=AssertionError("no download")):
            code, events, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertNotIn("progress", [e["event"] for e in events])
        self.assertEqual([e["action"] for e in events if e["event"] == "file"], ["ok", "ok"])

    def test_a_failed_listing_ends_with_an_error_event_not_a_traceback(self):
        out = io.StringIO()
        with mock.patch.object(fetch_model, "list_remote", side_effect=urllib.error.URLError("offline")), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = fetch_model.main(["--dest", str(self.dest), "--json"])
        self.assertEqual(code, 1)
        last = json.loads(out.getvalue().splitlines()[-1])
        self.assertEqual(last["event"], "error")
        self.assertIn("offline", last["message"])

    def test_without_json_nothing_but_text_is_printed(self):
        out = io.StringIO()
        with mock.patch.object(fetch_model, "list_remote", return_value=FILES), \
                mock.patch.object(fetch_model.urllib.request, "urlopen", side_effect=fake_urlopen), \
                contextlib.redirect_stdout(out):
            self.assertEqual(fetch_model.main(["--dest", str(self.dest)]), 0)
        self.assertNotIn('"event"', out.getvalue())
        self.assertIn("Verified 2 files", out.getvalue())


if __name__ == "__main__":
    unittest.main()
