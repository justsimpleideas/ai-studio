import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_engine.events import EventHub
from image_engine.gallery import Gallery
from image_engine.jobs import DONE, FAILED, JobQueue
from image_engine.params import parse_job

from .fakes import FakeBackend, FakeVideoBackend
from .test_jobs import wait_for


class VideoJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.events = EventHub()
        self.feed = self.events.subscribe()
        self.gallery = Gallery(Path(self.tmp.name))
        self.image = FakeBackend()
        self.video = FakeVideoBackend()
        self.jobs = JobQueue({"image": self.image, "video": self.video}, self.gallery, self.events,
                             app="Local AI Studio 9.9")
        self.jobs.start()

    def tearDown(self):
        self.jobs.stop()
        self.gallery.close()
        self.tmp.cleanup()

    def video_job(self, **fields):
        return self.jobs.submit(parse_job({"kind": "video", "prompt": "waves", "width": 256, "height": 256,
                                           "frames": 9, "steps": 3, **fields}))

    def image_job(self):
        return self.jobs.submit(parse_job({"prompt": "a still", "width": 256, "height": 256, "steps": 2}))

    def events_of(self, kind):
        out = []
        while not self.feed.empty():
            event = self.feed.get_nowait()
            if event["type"] == kind:
                out.append(event)
        return out

    def test_a_video_job_saves_a_video(self):
        job = self.video_job(seed=5, count=2)
        self.assertTrue(wait_for(lambda: job.status == DONE), job.error)
        records = [self.gallery.get(i) for i in job.images]
        self.assertEqual([r["media"] for r in records], ["video", "video"])
        self.assertEqual(sorted(r["seed"] for r in records), [5, 6])
        self.assertEqual(records[0]["frames"], 9)
        self.assertEqual(records[0]["model"], "fake-video")
        self.assertEqual(records[0]["provenance"]["app"], "Local AI Studio 9.9")
        self.assertEqual(job.to_dict()["kind"], "video")
        stages = [e["job"]["stage"] for e in self.events_of("job")]
        self.assertIn("encoding video", stages)

    def test_only_one_model_is_loaded_at_a_time(self):
        first = self.image_job()
        self.assertTrue(wait_for(lambda: first.status == DONE))
        self.assertTrue(self.image.loaded)
        second = self.video_job()
        self.assertTrue(wait_for(lambda: second.status == DONE), second.error)
        self.assertFalse(self.image.loaded)
        self.assertTrue(self.video.loaded)
        model_events = [(e["kind"], e["state"]) for e in self.events_of("model")]
        self.assertLess(model_events.index(("image", "unloaded")), model_events.index(("video", "loading")))
        self.assertEqual(self.jobs.models["image"]["state"], "unloaded")
        self.assertEqual(self.jobs.models["video"]["state"], "ready")

    def test_unloading_without_a_kind_frees_the_gpu_entirely(self):
        job = self.video_job()
        self.assertTrue(wait_for(lambda: job.status == DONE))
        self.jobs.unload_model()
        self.assertFalse(self.video.loaded or self.image.loaded)

    def test_the_start_picture_is_cropped_to_the_video_not_squashed(self):
        wide = Image.new("RGB", (800, 200), "black")
        wide.paste(Image.new("RGB", (200, 200), "white"), (300, 0))  # a white square in the middle
        input_id = self.gallery.save_input(wide)
        job = self.video_job(start_image=input_id)
        self.assertTrue(wait_for(lambda: job.status == DONE), job.error)
        start = self.video.start_frames[-1]
        self.assertEqual(start.size, (256, 256))
        self.assertEqual(start.getpixel((128, 128)), (255, 255, 255))  # still a square, centered
        self.assertEqual(self.gallery.get(job.images[0])["start_image"], input_id)

    def test_a_missing_start_picture_fails_the_job_with_a_reason(self):
        job = self.video_job(start_image="0" * 16)
        self.assertTrue(wait_for(lambda: job.status == FAILED))
        self.assertIn("start picture", job.error)

    def test_prompt_encodings_are_cached_per_model(self):
        a = self.image_job()
        self.assertTrue(wait_for(lambda: a.status == DONE))
        b = self.jobs.submit(parse_job({"kind": "video", "prompt": "a still", "width": 256, "height": 256,
                                        "frames": 5, "steps": 1}))
        self.assertTrue(wait_for(lambda: b.status == DONE))
        # Same prompt text, but the video model must encode it itself.
        self.assertIn(("encode", "a still"), self.video.calls)

    def test_an_engine_without_video_says_so(self):
        jobs = JobQueue(FakeBackend(), self.gallery, self.events)
        jobs.start()
        try:
            job = jobs.submit(parse_job({"kind": "video", "prompt": "x", "width": 256, "height": 256, "frames": 5}))
            self.assertTrue(wait_for(lambda: job.status == FAILED))
            self.assertIn("no video model", job.error)
        finally:
            jobs.stop()


if __name__ == "__main__":
    unittest.main()
