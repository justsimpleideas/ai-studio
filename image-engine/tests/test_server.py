import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from image_engine.config import SettingsStore
from image_engine.events import EventHub
from image_engine.gallery import Gallery
from image_engine.jobs import JobQueue
from image_engine.server import App, serve

from .fakes import FakeBackend

TOKEN = "t" * 40


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "secret.txt").write_text("do not serve", encoding="utf-8")

        cls.backend = FakeBackend()
        cls.events = EventHub()
        cls.gallery = Gallery(root / "outputs")
        cls.jobs = JobQueue(cls.backend, cls.gallery, cls.events)
        cls.settings = SettingsStore(root / "data")
        cls.app = App(settings=cls.settings, backend=cls.backend, gallery=cls.gallery, events=cls.events,
                      jobs=cls.jobs, port=0, token=TOKEN)
        cls.httpd = serve(cls.app, "127.0.0.1")
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.jobs.start()

    @classmethod
    def tearDownClass(cls):
        cls.app.stopping.set()
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.jobs.stop()
        cls.gallery.close()
        cls.tmp.cleanup()

    # ------------------------------------------------------------ helpers

    def request(self, method, path, body=None, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        hdrs = {"Host": f"127.0.0.1:{self.port}", "X-Studio-Token": TOKEN}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            hdrs["Content-Type"] = "application/json"
        if raw is not None:
            data = raw
        hdrs.update(headers or {})
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        payload = resp.read()
        conn.close()
        ctype = resp.getheader("Content-Type", "")
        parsed = json.loads(payload) if ctype.startswith("application/json") else payload
        return resp.status, parsed, resp

    def generate(self, **fields):
        status, job, _ = self.request("POST", "/api/jobs", {"prompt": "a cat", "width": 256, "height": 256,
                                                            "steps": 2, **fields})
        self.assertEqual(status, 200, job)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            _, job, _ = self.request("GET", f"/api/jobs/{job['id']}")
            if job["status"] in ("done", "failed", "cancelled"):
                return job
            time.sleep(0.02)
        self.fail("job did not finish")

    # ------------------------------------------------------------ happy path

    def test_status(self):
        status, body, resp = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(body["model"]["id"], "fake-model")
        self.assertEqual(body["defaults"]["steps"], 8)
        self.assertEqual(resp.getheader("Cross-Origin-Resource-Policy"), "same-origin")
        self.assertIsNone(resp.getheader("Access-Control-Allow-Origin"))

    def test_generate_list_fetch_and_delete(self):
        job = self.generate(count=2)
        self.assertEqual(job["status"], "done")
        status, listing, _ = self.request("GET", "/api/gallery?limit=10&q=cat")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(listing["total"], 2)
        image_id = job["images"][0]
        status, png, resp = self.request("GET", f"/images/{image_id}.png")
        self.assertEqual(status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "image/png")
        self.assertTrue(png.startswith(b"\x89PNG"))
        status, webp, _ = self.request("GET", f"/images/{image_id}.webp")
        self.assertEqual(status, 200)
        self.assertEqual(webp[8:12], b"WEBP")
        status, rec, _ = self.request("PATCH", f"/api/gallery/{image_id}", {"favorite": True})
        self.assertTrue(rec["favorite"])
        status, _, _ = self.request("DELETE", f"/api/gallery/{image_id}")
        self.assertEqual(status, 200)
        status, _, _ = self.request("GET", f"/images/{image_id}.png")
        self.assertEqual(status, 404)

    def test_invalid_request_gets_readable_error(self):
        status, body, _ = self.request("POST", "/api/jobs", {"prompt": "x", "width": 1000})
        self.assertEqual(status, 400)
        self.assertIn("multiple of 16", body["error"])

    def test_cancel_unknown_job(self):
        status, _, _ = self.request("POST", "/api/jobs/0123456789abcdef/cancel", {})
        self.assertEqual(status, 404)

    def test_settings_roundtrip_and_validation(self):
        status, body, _ = self.request("PUT", "/api/settings", {"memory_mode": "balanced"})
        self.assertEqual(status, 200)
        self.assertEqual(body["memory_mode"], "balanced")
        status, body, _ = self.request("PUT", "/api/settings", {"memory_mode": "turbo"})
        self.assertEqual(status, 400)
        status, body, _ = self.request("PUT", "/api/settings", {"evil": 1})
        self.assertEqual(status, 400)
        _, body, _ = self.request("GET", "/api/settings")
        self.assertEqual(body["memory_mode"], "balanced")

    def test_events_stream_starts_with_hello(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/api/events", headers={"Host": f"127.0.0.1:{self.port}", "X-Studio-Token": TOKEN})
        resp = conn.getresponse()
        self.assertEqual(resp.getheader("Content-Type"), "text/event-stream; charset=utf-8")
        line = resp.fp.readline().decode()
        self.assertTrue(line.startswith("data: "))
        self.assertEqual(json.loads(line[6:])["type"], "hello")
        conn.close()

    def test_serves_no_web_ui(self):
        # The studio renders the UI itself; the engine exposes only its API and images.
        for path in ("/", "/index.html", "/assets/app.js", "/gallery"):
            with self.subTest(path=path):
                status, _, _ = self.request("GET", path)
                self.assertEqual(status, 404)

    # ------------------------------------------------------------ guards

    def test_rejects_foreign_host_header(self):
        for host in ("evil.example", f"evil.example:{self.port}", "127.0.0.1:1", "192.168.0.10:5198"):
            with self.subTest(host=host):
                status, _, _ = self.request("GET", "/api/status", headers={"Host": host})
                self.assertEqual(status, 403)

    def test_rejects_cross_origin_writes(self):
        status, _, _ = self.request("POST", "/api/jobs", {"prompt": "x"},
                                    headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, _, _ = self.request("POST", "/api/jobs", {"prompt": "x"},
                                    headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(status, 403)

    def test_refuses_any_browser_origin(self):
        # Only the studio's main process talks to the engine, and it sends no Origin.
        for origin in ("http://localhost:5197", "http://127.0.0.1:5198", "null", "file://"):
            with self.subTest(origin=origin):
                status, _, _ = self.request("GET", "/api/status", headers={"Origin": origin})
                self.assertEqual(status, 403)
        status, _, _ = self.request("GET", "/api/status", headers={"Sec-Fetch-Site": "same-origin"})
        self.assertEqual(status, 403)

    def test_token_is_required_everywhere(self):
        image_id = self.generate()["images"][0]
        for path in ("/api/status", "/api/gallery", f"/images/{image_id}.png", "/api/events"):
            for token in (None, "", "wrong" * 8, TOKEN[:-1]):
                with self.subTest(path=path, token=token):
                    headers = {"X-Studio-Token": token} if token is not None else {}
                    conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                    conn.request("GET", path, headers={"Host": f"127.0.0.1:{self.port}", **headers})
                    resp = conn.getresponse()
                    self.assertEqual(resp.status, 401)
                    conn.close()
        status, _, _ = self.request("POST", "/api/jobs", {"prompt": "x"}, headers={"X-Studio-Token": "nope"})
        self.assertEqual(status, 401)

    def test_rejects_form_posts(self):
        # A cross-site <form> can only send these content types; none may trigger work.
        for ctype in ("application/x-www-form-urlencoded", "text/plain", "multipart/form-data"):
            with self.subTest(ctype=ctype):
                status, _, _ = self.request("POST", "/api/jobs", raw=b'{"prompt":"x"}',
                                            headers={"Content-Type": ctype})
                self.assertEqual(status, 415)

    def test_rejects_oversized_and_malformed_bodies(self):
        status, _, _ = self.request("POST", "/api/jobs", raw=b"x" * (65 * 1024),
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(status, 413)
        status, _, _ = self.request("POST", "/api/jobs", raw=b"{not json",
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_unread_body_closes_keepalive_connection(self):
        # Regression: an unread oversized body used to stay on the socket and be parsed as the
        # next request (the server answered "414 URI Too Long" to bytes from the old body).
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("POST", "/api/jobs", body=b"x" * (65 * 1024),
                     headers={"Host": f"127.0.0.1:{self.port}", "X-Studio-Token": TOKEN,
                              "Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        self.assertEqual(resp.status, 413)
        self.assertEqual(resp.getheader("Connection"), "close")
        conn.close()

    def test_no_path_traversal(self):
        for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/../../secret.txt",
                     "/images/../secret.txt", "/images/0123456789abcdef.png"):
            with self.subTest(path=path):
                status, body, _ = self.request("GET", path)
                self.assertNotEqual(body, b"do not serve")
                self.assertNotIn(b"do not serve", body if isinstance(body, bytes) else b"")

    def test_unknown_api_path_and_wrong_method(self):
        status, _, _ = self.request("GET", "/api/nope")
        self.assertEqual(status, 404)
        status, _, _ = self.request("DELETE", "/api/status")
        self.assertEqual(status, 405)


if __name__ == "__main__":
    unittest.main()
