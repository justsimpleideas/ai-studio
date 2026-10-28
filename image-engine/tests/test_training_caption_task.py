"""Describing training photos with the captioning model (a fake one), through the engine API."""

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
from image_engine.server import App, serve
from image_engine.training.captions import (ABSENT, MAIN, RELEVANCE, REAL_PART, SMALL, TOY, CaptionRecord, judge,
                                            prompts)
from image_engine.training.service import TrainingService, register
from image_engine.training.subject import CARS, GENERAL, SubjectError, parse_subject

from .fakes import FakeBackend, FakeCaptionBackend
from .test_server import TOKEN, ServerTests
from .test_training_dataset import photo


def reply(**fields):
    import json
    return json.dumps(fields)


class RelevanceTests(unittest.TestCase):
    SPECS = RELEVANCE + CARS.fields

    def verdict(self, shown, size=MAIN):
        return judge("a", (reply(subject_shown=shown, subject_size=size),) * 2, self.SPECS, {})

    def test_every_photo_is_asked_whether_it_shows_the_subject(self):
        first, second = prompts(parse_subject({**GENERAL.to_json(), "noun": "wristwatch"}))
        for p in (first, second):
            self.assertIn('"subject_shown"', p)
            self.assertIn('"subject_size"', p)
            self.assertIn("a toy, model or replica of it", p)
            self.assertIn("no wristwatch at all", p)

    def test_toys_pictures_absent_and_small_are_off_subject(self):
        for shown, size in ((TOY, MAIN), ("a picture of it", MAIN), (ABSENT, MAIN), ("the real thing, whole", SMALL)):
            rec = self.verdict(shown, size)
            self.assertIs(rec.relevant, False, (shown, size))
            self.assertTrue(any(f.startswith("off_subject:") for f in rec.flags))
            self.assertTrue(rec.excluded)

    def test_a_part_or_the_inside_is_kept_but_flagged(self):
        for shown in (REAL_PART, "the inside of it"):
            rec = self.verdict(shown)
            self.assertIs(rec.relevant, True)
            self.assertIn("partial:subject_shown", rec.flags)
            self.assertFalse(rec.excluded)
        self.assertEqual(self.verdict("the real thing, whole").flags.count("partial:subject_shown"), 0)

    def test_the_user_decides_in_the_end(self):
        rec = self.verdict(TOY)
        rec.status = "accepted"  # "Include"
        self.assertFalse(rec.excluded)
        rec = CaptionRecord("b", {}, [], status="excluded", relevant=True)
        self.assertTrue(rec.excluded)
        self.assertFalse(CaptionRecord("c", {}, []).excluded)  # not looked at yet

    def test_the_studios_own_keys_cannot_be_field_names(self):
        for label in ("Subject shown", "subject_size", "Unknown"):
            with self.assertRaises(SubjectError, msg=label):
                parse_subject({**GENERAL.to_json(), "fields": [{"label": label, "question": "q"}]})


class LicenceCountTests(unittest.TestCase):
    def test_runs_know_which_licences_their_collected_photos_came_under(self):
        import json
        from image_engine.training.service import photo_licences
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(photo_licences(root, ["a.jpg"]), {})  # the user's own photos
            (root / "ATTRIBUTION.json").write_text(json.dumps([
                {"file": "A/a.jpg", "license": "cc-by-sa"}, {"file": "A/b.jpg", "license": "cc0"},
                {"file": "A/c.jpg", "license": "cc-by-sa"}]))
            self.assertEqual(photo_licences(root, ["A/a.jpg", "A/b.jpg", "A/mine.jpg"]), {"cc-by-sa": 1, "cc0": 1})
            (root / "ATTRIBUTION.json").write_text("{broken")
            self.assertEqual(photo_licences(root, ["A/a.jpg"]), {})


class CaptionTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.photos = root / "photos"
        for i in range(4):
            photo(cls.photos / "Example_Motors" / "Coupe" / f"{i}.jpg", size=(640, 512), seed=i)
        # The one portrait photo is "a toy" to the fake model.
        photo(cls.photos / "Example_Motors" / "Coupe" / "toy.jpg", size=(512, 640), seed=9)
        cls.captioner = FakeCaptionBackend(
            answer=lambda image, keys: {"subject_shown": TOY} if image.size == (512, 640) else {})
        cls.image = FakeBackend()
        cls.events = EventHub()
        cls.gallery = Gallery(root / "outputs")
        cls.loras = LoraLibrary(root / "loras")
        cls.jobs = JobQueue({"image": cls.image, "caption": cls.captioner}, cls.gallery, cls.events, loras=cls.loras)
        cls.training = TrainingService(root / "training", cls.loras, cls.gallery)
        register(cls.jobs, cls.training)
        cls.app = App(settings=SettingsStore(root / "data"), backend=cls.image, gallery=cls.gallery,
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

    def wait(self, job, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            current = self.ok("GET", f"/api/jobs/{job['id']}")
            if current["status"] in ("done", "failed", "cancelled"):
                return current
            time.sleep(0.05)
        self.fail("job did not finish")

    def test_describe_judge_and_leave_out_what_is_off_subject(self):
        self.assertTrue(self.ok("GET", "/api/status")["captioning"])
        project = self.ok("POST", "/api/training/projects",
                          {"name": "Coupes", "photo_root": str(self.photos), "subject": CARS.to_json()})
        url = f"/api/training/projects/{project['id']}"
        self.assertEqual(self.wait(self.ok("POST", f"{url}/scan", {}))["status"], "done")

        done = self.wait(self.ok("POST", f"{url}/caption", {}))
        self.assertEqual(done["status"], "done", done.get("error"))
        self.assertEqual((done["result"]["status"], done["result"]["described"], done["result"]["off_subject"]),
                         ("finished", 5, 1))
        self.assertEqual(sum(1 for c in self.captioner.calls if c[0] == "describe"), 10)  # two prompts each

        summary = self.ok("GET", url)
        self.assertEqual((summary["captions"]["described"], summary["captions"]["off_subject"],
                          summary["captions"]["excluded"]), (5, 1, 1))
        self.assertTrue(summary["captioner"].startswith("example/fake-vlm"))
        items = self.ok("GET", f"{url}/captions?limit=10")["items"]
        toy = next(i for i in items if i["path"].endswith("toy.jpg"))
        self.assertTrue(toy["excluded"])
        self.assertEqual(toy["relevance"]["subject_shown"], TOY)
        self.assertIn("off_subject:subject_shown", toy["flags"])
        real = next(i for i in items if not i["path"].endswith("toy.jpg"))
        self.assertEqual(real["caption"], "Example Motors Coupe, car, front view, grey, sedan, grey, daylight, "
                                          "whole car in frame, not hidden")
        self.assertNotIn("subject_shown", real["description"])  # relevance answers never reach captions
        excluded = self.ok("GET", f"{url}/captions?view=excluded")["items"]
        self.assertEqual([i["item"] for i in excluded], [toy["item"]])
        review = self.ok("GET", f"{url}/captions?view=review")["items"]
        self.assertEqual(review[0]["item"], toy["item"])  # off subject comes first

        included = self.ok("POST", f"{url}/captions/{toy['item']}", {"action": "accept"})
        self.assertFalse(included["excluded"])

        # Running again describes nothing new; changing the subject's questions describes all again.
        again = self.wait(self.ok("POST", f"{url}/caption", {}))
        self.assertEqual(again["result"]["described"], 0)
        fewer = {**CARS.to_json(), "fields": CARS.to_json()["fields"][:2]}
        self.ok("PUT", f"{url}/subject", fewer)
        self.assertEqual(self.wait(self.ok("POST", f"{url}/caption", {}))["result"]["described"], 5)
        self.assertEqual(self.ok("GET", f"{url}/captions?limit=10")["items"][0]["status"] in ("auto", "accepted"), True)

    def test_refused_without_an_installed_model(self):
        project = self.ok("POST", "/api/training/projects", {"name": "None", "photo_root": str(self.photos)})
        self.captioner._available = False
        try:
            self.assertFalse(self.ok("GET", "/api/status")["captioning"])
            status, payload, _ = self.request("POST", f"/api/training/projects/{project['id']}/caption", {})
            self.assertEqual(status, 409, payload)
        finally:
            self.captioner._available = True


if __name__ == "__main__":
    unittest.main()
