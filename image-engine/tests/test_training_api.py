"""The Train tab's engine side through its HTTP API: projects, scan, captions, training, the LoRA
library, LoRAs in generation, and checkpoint comparison.

The parts that need no model run everywhere. Training itself uses a tiny random-weight
transformer on the CPU (tests/fakes.py), so those tests need torch and diffusers and are
skipped in a plain Python.
"""

import importlib.util
import tempfile
import threading
import time
import unittest
from pathlib import Path

from image_engine.config import SettingsStore
from image_engine.events import EventHub
from image_engine.gallery import Gallery
from image_engine.jobs import JobQueue
from image_engine.lora_library import LoraLibrary
from image_engine.params import RequestError, parse_request
from image_engine.server import App, serve
from image_engine.training.service import TrainingService, register

from .fakes import FakeTrainingBackend
from .test_server import TOKEN, ServerTests
from .test_training_dataset import photo

HAS_TRAINING = all(importlib.util.find_spec(m) for m in ("torch", "safetensors", "diffusers"))


class LoraRequestTests(unittest.TestCase):
    BASE = {"prompt": "a car", "width": 512, "height": 512}

    def test_loras_are_optional_and_validated(self):
        self.assertEqual(parse_request(self.BASE).loras, ())
        self.assertNotIn("loras", parse_request(self.BASE).to_dict())
        req = parse_request({**self.BASE, "loras": [{"id": "a" * 16, "strength": 0.8}]})
        self.assertEqual((req.loras[0].id, req.loras[0].strength), ("a" * 16, 0.8))
        self.assertEqual(req.to_dict()["loras"], [{"id": "a" * 16, "strength": 0.8}])
        bad = [
            {"id": "a" * 16},
            [{"id": "a" * 16, "strength": 3}],
            [{"id": "xyz", "strength": 1}],
            [{"id": "a" * 16, "strength": True}],
            [{"id": "a" * 16, "strength": 1}, {"id": "a" * 16, "strength": 1}],
            [{"id": f"{i:016x}", "strength": 1} for i in range(4)],
            [{"id": "a" * 16, "strength": 1, "extra": 1}],
        ]
        for loras in bad:
            with self.assertRaises(RequestError, msg=loras):
                parse_request({**self.BASE, "loras": loras})


class TrainingApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.photos = root / "photos"
        for i in range(6):
            photo(cls.photos / "Example_Motors" / "Coupe" / f"{i}.jpg", size=(640, 512), seed=i)
        photo(cls.photos / "Example_Motors" / "Coupe" / "tiny.jpg", size=(100, 100), seed=40)
        cls.backend = FakeTrainingBackend()
        cls.events = EventHub()
        cls.gallery = Gallery(root / "outputs")
        cls.loras = LoraLibrary(root / "loras")
        cls.jobs = JobQueue(cls.backend, cls.gallery, cls.events, loras=cls.loras)
        cls.training = TrainingService(root / "training", cls.loras, cls.gallery)
        register(cls.jobs, cls.training)
        cls.app = App(settings=SettingsStore(root / "data"), backend=cls.backend, gallery=cls.gallery,
                      events=cls.events, jobs=cls.jobs, port=0, token=TOKEN, training=cls.training, loras=cls.loras)
        cls.httpd = serve(cls.app, "127.0.0.1")
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.jobs.start()

    @classmethod
    def tearDownClass(cls):
        cls.app.stopping.set()
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.jobs.stop()
        cls.gallery.close()
        cls.tmp.cleanup()

    request = ServerTests.request

    def ok(self, method, path, body=None):
        status, payload, _ = self.request(method, path, body)
        self.assertEqual(status, 200, payload)
        return payload

    def wait(self, job, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            current = self.ok("GET", f"/api/jobs/{job['id']}")
            if current["status"] in ("done", "failed", "cancelled"):
                return current
            time.sleep(0.05)
        self.fail(f"job {job['id']} did not finish")

    def project(self, name="Coupe", subject="cars"):
        body = {"name": name, "photo_root": str(self.photos)}
        if subject is not None:
            body["subject"] = self.preset(subject) if isinstance(subject, str) else subject
        created = self.ok("POST", "/api/training/projects", body)
        done = self.wait(self.ok("POST", f"/api/training/projects/{created['id']}/scan", {}))
        self.assertEqual(done["status"], "done", done.get("error"))
        return self.ok("GET", f"/api/training/projects/{created['id']}")

    # ------------------------------------------------------------ no model needed

    def preset(self, name):
        return next(p for p in self.ok("GET", "/api/training/subjects")["items"] if p["preset"] == name)

    def test_status_says_training_is_available(self):
        self.assertTrue(self.ok("GET", "/api/status")["training"])

    def test_create_validates_name_and_folder(self):
        for body in ({"name": "", "photo_root": str(self.photos)}, {"name": "x", "photo_root": "relative/dir"},
                     {"name": "x", "photo_root": str(self.photos / "missing")}, {"name": "x" * 61,
                                                                                 "photo_root": str(self.photos)}):
            status, payload, _ = self.request("POST", "/api/training/projects", body)
            self.assertEqual(status, 400, body)

    def test_scan_indexes_photos_and_creates_fact_captions(self):
        project = self.project()
        self.assertEqual(project["dataset"]["total"], 7)
        self.assertEqual(project["dataset"]["usable"], 6)
        self.assertEqual(project["captions"]["total"], 6)
        page = self.ok("GET", f"/api/training/projects/{project['id']}/captions?limit=10")
        self.assertEqual(page["total"], 6)
        first = page["items"][0]
        self.assertEqual(first["facts"], {"make": "Example Motors", "model": "Coupe"})
        self.assertEqual(first["caption"], "Example Motors Coupe, car")
        self.assertEqual(project["subject"]["preset"], "cars")
        self.assertEqual(first["status"], "auto")

    def test_review_exclude_and_edit(self):
        project = self.project()
        item = self.ok("GET", f"/api/training/projects/{project['id']}/captions")["items"][0]["item"]
        url = f"/api/training/projects/{project['id']}/captions/{item}"
        edited = self.ok("POST", url, {"action": "edit", "edits": {"view": "side", "colour": "red"}})
        self.assertEqual(edited["caption"], "Example Motors Coupe, car, side view, red")
        self.assertEqual(edited["status"], "edited")
        status, payload, _ = self.request("POST", url, {"action": "edit", "edits": {"view": "sideways"}})
        self.assertEqual(status, 400)
        self.ok("POST", url, {"action": "exclude"})
        excluded = self.ok("GET", f"/api/training/projects/{project['id']}/captions?view=excluded")
        self.assertEqual([i["item"] for i in excluded["items"]], [item])
        status, _, _ = self.request("POST", f"/api/training/projects/{project['id']}/captions/{'0' * 16}",
                                    {"action": "accept"})
        self.assertEqual(status, 404)

    def test_a_project_is_about_whatever_the_user_defines(self):
        presets = self.ok("GET", "/api/training/subjects")["items"]
        self.assertEqual([p["preset"] for p in presets], ["general", "cars"])
        project = self.project("Anything", subject=None)  # no subject given: the general one
        self.assertEqual(project["subject"]["preset"], "general")
        first = self.ok("GET", f"/api/training/projects/{project['id']}/captions")["items"][0]
        self.assertEqual(first["facts"], {"name": "Example Motors", "variant": "Coupe"})
        self.assertEqual(first["caption"], "Example Motors Coupe")

        watches = {"preset": "custom", "noun": "wristwatch", "trigger": "mywatch",
                   "levels": [{"label": "Brand", "required": True}, {"label": "Model", "required": True}],
                   "fields": [{"label": "View", "question": "which side of the {noun} faces the camera",
                               "choices": ["front", "back"], "phrase": "{value} view"}]}
        url = f"/api/training/projects/{project['id']}"
        changed = self.ok("PUT", f"{url}/subject", watches)
        self.assertEqual(changed["subject"]["levels"][0]["key"], "brand")
        self.assertEqual(self.wait(self.ok("POST", f"{url}/scan", {}))["status"], "done")
        item = self.ok("GET", f"{url}/captions")["items"][0]
        self.assertEqual(item["facts"], {"brand": "Example Motors", "model": "Coupe"})
        self.assertEqual(item["description"], {"view": "unknown"})
        edited = self.ok("POST", f"{url}/captions/{item['item']}", {"action": "edit", "edits": {"view": "back"}})
        self.assertEqual(edited["caption"], "mywatch, Example Motors Coupe, wristwatch, back view")

        for bad in ({**watches, "levels": []}, {**watches, "trigger": "two words"}, ["not", "an", "object"]):
            status, payload, _ = self.request("PUT", f"{url}/subject", bad)
            self.assertEqual(status, 400, payload)
        status, payload, _ = self.request("POST", "/api/training/projects",
                                          {"name": "x", "photo_root": str(self.photos), "subject": {"noun": 3}})
        self.assertEqual(status, 400, payload)
        self.assertEqual(self.request("PUT", f"/api/training/projects/{'0' * 16}/subject", watches)[0], 404)

    def test_photo_thumbnails_only_for_indexed_photos(self):
        project = self.project()
        item = self.ok("GET", f"/api/training/projects/{project['id']}/captions")["items"][0]["item"]
        status, body, resp = self.request("GET", f"/training/{project['id']}/{item}.webp")
        self.assertEqual(status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "image/webp")
        self.assertEqual(body[:4], b"RIFF")
        for path in (f"/training/{project['id']}/{'0' * 16}.webp", f"/training/{'0' * 16}/{item}.webp",
                     f"/training/{project['id']}/../secret.webp"):
            self.assertEqual(self.request("GET", path)[0], 404, path)

    def test_bad_training_settings_are_refused_before_queueing(self):
        project = self.project()
        for body in ({"rank": 0}, {"learning_rate": 1}, {"steps": "10"}, {"target_side": 640}, {"bogus": 1},
                     {"run": "../x"}):
            status, payload, _ = self.request("POST", f"/api/training/projects/{project['id']}/train", body)
            self.assertEqual(status, 400, body)

    def test_unknown_lora_in_a_generation_request_is_refused(self):
        status, payload, _ = self.request("POST", "/api/jobs", {"prompt": "a car", "width": 512, "height": 512,
                                                                "loras": [{"id": "f" * 16, "strength": 1}]})
        self.assertEqual(status, 400)
        self.assertIn("not in the library", payload["error"])

    def test_delete_project_keeps_the_photos(self):
        project = self.project("Throwaway")
        self.ok("DELETE", f"/api/training/projects/{project['id']}")
        self.assertEqual(self.request("GET", f"/api/training/projects/{project['id']}")[0], 404)
        self.assertEqual(len(list(self.photos.rglob("*.jpg"))), 7)

    # ------------------------------------------------------------ training (tiny model on the CPU)

    @unittest.skipUnless(HAS_TRAINING, "needs torch and diffusers")
    def test_train_use_generate_and_compare(self):
        project = self.project("Full")
        pid = project["id"]
        job = self.ok("POST", f"/api/training/projects/{pid}/train",
                      {"rank": 4, "steps": 4, "checkpoint_every": 2, "learning_rate": 0.001, "target_side": 512})
        status, payload, _ = self.request("POST", f"/api/training/projects/{pid}/scan", {})
        self.assertEqual(status, 409)  # one task per project at a time
        done = self.wait(job)
        self.assertEqual(done["status"], "done", done.get("error"))
        result = done["result"]
        self.assertEqual((result["status"], result["step"]), ("finished", 4))
        self.assertIn(("training_session",), self.backend.calls)
        self.assertIn(("training_session_end",), self.backend.calls)
        run = self.ok("GET", f"/api/training/projects/{pid}")["runs"][0]
        self.assertEqual((run["status"], run["step"], run["checkpoints"]), ("finished", 4, [2, 4]))
        self.assertEqual(run["photos"], sum(1 for _ in range(6)) - project["dataset"]["holdout"])
        log = self.ok("GET", f"/api/training/projects/{pid}/runs/{run['id']}/log")["items"]
        self.assertEqual(len(log), 4)

        # Extend the same run: it resumes from step 4.
        more = self.wait(self.ok("POST", f"/api/training/projects/{pid}/train",
                                 {"rank": 4, "steps": 6, "checkpoint_every": 2, "learning_rate": 0.001,
                                  "target_side": 512, "run": run["id"]}))
        self.assertEqual(more["result"]["step"], 6, more.get("error"))

        lora = self.ok("POST", f"/api/training/projects/{pid}/runs/{run['id']}/use", {"step": 6, "name": "Coupe v1"})
        self.assertEqual(lora["rank"], 4)
        self.assertEqual(lora["origin"]["step"], 6)
        self.assertEqual([i["id"] for i in self.ok("GET", "/api/loras")["items"]], [lora["id"]])
        status, payload, _ = self.request("POST", f"/api/training/projects/{pid}/runs/{run['id']}/use",
                                          {"step": 6, "name": "again"})
        self.assertEqual(status, 400)  # same file twice
        self.assertIn("already in the library", payload["error"])

        self.backend.calls.clear()
        gen = self.wait(self.ok("POST", "/api/jobs", {"prompt": "a coupe", "width": 256, "height": 256, "steps": 2,
                                                       "loras": [{"id": lora["id"], "strength": 0.7}]}))
        self.assertEqual(gen["status"], "done", gen.get("error"))
        self.assertIn(("set_loras", [(lora["id"], 0.7)]), self.backend.calls)
        record = self.ok("GET", f"/api/gallery/{gen['images'][0]}")
        self.assertEqual(record["provenance"]["loras"][0]["sha256"], lora["sha256"])
        from PIL import Image

        from image_engine.gallery import PNG_METADATA_KEY

        with Image.open(self.gallery.path_of(record["id"], "image")) as png:
            meta = __import__("json").loads(png.text[PNG_METADATA_KEY])
        self.assertEqual(meta["loras"], [{"id": lora["id"], "strength": 0.7}])
        plain = self.wait(self.ok("POST", "/api/jobs", {"prompt": "a coupe", "width": 256, "height": 256, "steps": 2}))
        self.assertEqual(plain["status"], "done")
        # The plain image ran with the LoRA taken off again.
        self.assertEqual([c for c in self.backend.calls if c[0] == "set_loras"][-1], ("set_loras", []))

        self.backend.calls.clear()
        cmp_job = self.wait(self.ok("POST", f"/api/training/projects/{pid}/compare",
                                    {"run": run["id"], "steps": [2, 6], "size": 512, "seed": 3}))
        self.assertEqual(cmp_job["status"], "done", cmp_job.get("error"))
        grid = cmp_job["result"]
        self.assertEqual(grid["columns"], ["base", "2", "6"])
        self.assertTrue(1 <= len(grid["prompts"]) <= 3)
        self.assertTrue(all(len(row) == 3 for row in grid["images"]))
        self.assertEqual([c for c in self.backend.calls if c[0] == "set_loras"][-1], ("set_loras", []))
        saved = self.ok("GET", f"/api/training/projects/{pid}")["runs"][0]["compare"]
        self.assertEqual(saved["images"], grid["images"])
        folders = self.ok("GET", "/api/folders")["items"]
        self.assertTrue(any(f["name"] == "Training: Full" and f["count"] == 3 * len(grid["prompts"]) for f in folders))

        status, _, _ = self.request("POST", f"/api/training/projects/{pid}/compare", {"run": run["id"], "steps": [5]})
        self.assertEqual(status, 200)  # accepted; the task itself reports the missing checkpoint
        self.ok("DELETE", f"/api/loras/{lora['id']}")
        self.assertEqual(self.ok("GET", "/api/loras")["items"], [])

    @unittest.skipUnless(HAS_TRAINING, "needs torch and diffusers")
    def test_stopping_a_run_keeps_a_checkpoint(self):
        project = self.project("Stop")
        job = self.ok("POST", f"/api/training/projects/{project['id']}/train",
                      {"rank": 2, "steps": 2000, "checkpoint_every": 1000, "target_side": 512})
        deadline = time.time() + 60
        while time.time() < deadline:
            current = self.ok("GET", f"/api/jobs/{job['id']}")
            if current["stage"] == "training" and current["step"] >= 2:
                break
            time.sleep(0.05)
        self.ok("POST", f"/api/jobs/{job['id']}/cancel", {})
        done = self.wait(job)
        self.assertEqual(done["status"], "done", done.get("error"))
        self.assertEqual(done["result"]["status"], "stopped")
        run = self.ok("GET", f"/api/training/projects/{project['id']}")["runs"][0]
        self.assertEqual(run["status"], "stopped")
        self.assertEqual(run["checkpoints"], [done["result"]["step"]])


class TrainWithoutTrainingSupportTests(unittest.TestCase):
    def test_a_backend_without_training_fails_the_task_with_a_message(self):
        from image_engine.jobs import TaskRequest

        from .fakes import FakeBackend

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(3):
                photo(root / "p" / "Ex" / "M" / f"{i}.jpg", seed=i)
            gallery = Gallery(root / "out")
            jobs = JobQueue(FakeBackend(), gallery, EventHub())
            service = TrainingService(root / "t", LoraLibrary(root / "l"), gallery)
            register(jobs, service)
            jobs.start()
            try:
                pid = service.create("x", str(root / "p"))["id"]
                scan = jobs.submit(TaskRequest(task="training.scan", project=pid))
                train = jobs.submit(TaskRequest(task="training.train", project=pid, params={"steps": 1}))
                deadline = time.time() + 30
                while train.status not in ("done", "failed") and time.time() < deadline:
                    time.sleep(0.02)
                self.assertEqual(scan.status, "done", scan.error)
                self.assertEqual(train.status, "failed")
                self.assertIn("cannot be trained", train.error)
            finally:
                jobs.stop()
                gallery.close()


if __name__ == "__main__":
    unittest.main()
