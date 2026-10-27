import tempfile
import threading
import time
import unittest
from pathlib import Path

from image_engine.events import EventHub
from image_engine.gallery import Gallery
from image_engine.jobs import CANCELLED, DONE, FAILED, JobQueue
from image_engine.params import parse_request

from .fakes import FakeBackend


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class JobQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.events = EventHub()
        self.feed = self.events.subscribe()
        self.gallery = Gallery(Path(self.tmp.name))
        self.backend = FakeBackend()
        self.jobs = JobQueue(self.backend, self.gallery, self.events)
        self.jobs.start()

    def tearDown(self):
        self.jobs.stop()
        self.gallery.close()
        self.tmp.cleanup()

    def submit(self, **fields):
        return self.jobs.submit(parse_request({"prompt": "a cat", "width": 256, "height": 256, **fields}))

    def drain(self):
        out = []
        while not self.feed.empty():
            out.append(self.feed.get_nowait())
        return out

    def test_batch_runs_to_completion_and_saves_each_image(self):
        job = self.submit(count=3, seed=10, steps=4)
        self.assertTrue(wait_for(lambda: job.status == DONE))
        self.assertEqual(len(job.images), 3)
        self.assertEqual(self.gallery.list()["total"], 3)
        seeds = sorted(self.gallery.get(i)["seed"] for i in job.images)
        self.assertEqual(seeds, [10, 11, 12])
        self.assertEqual(self.jobs.model_state, "ready")

    def test_each_image_records_the_app_and_the_backend_provenance(self):
        self.jobs.app = "Local AI Studio 9.9"
        job = self.submit()
        self.assertTrue(wait_for(lambda: job.status == DONE))
        provenance = self.gallery.get(job.images[0])["provenance"]
        self.assertEqual(provenance["app"], "Local AI Studio 9.9")
        self.assertTrue(provenance["model_revision"].startswith("example/fake-model@"))

    def test_model_loads_once_and_prompt_is_encoded_once_per_batch(self):
        job = self.submit(count=2)
        self.assertTrue(wait_for(lambda: job.status == DONE))
        job2 = self.submit(count=1)
        self.assertTrue(wait_for(lambda: job2.status == DONE))
        names = [c[0] for c in self.backend.calls]
        self.assertEqual(names.count("load"), 1)
        self.assertEqual(names.count("encode"), 1)  # same prompt: conditioning served from cache
        self.assertEqual(names.count("sample"), 3)

    def test_turning_on_guidance_reencodes_the_prompt(self):
        # The pipeline needs negative embeddings for any guidance above 0, so a cached
        # no-guidance encoding must not be reused once guidance is switched on (even to 0.5).
        job = self.submit(guidance=0.0)
        self.assertTrue(wait_for(lambda: job.status == DONE))
        job2 = self.submit(guidance=0.5)
        self.assertTrue(wait_for(lambda: job2.status == DONE))
        self.assertEqual([c[0] for c in self.backend.calls].count("encode"), 2)

    def test_progress_events_report_every_step(self):
        job = self.submit(steps=5)
        self.assertTrue(wait_for(lambda: job.status == DONE))
        events = self.drain()
        steps = [e["job"]["step"] for e in events if e["type"] == "job" and e["job"]["stage"] == "sampling"]
        self.assertEqual(sorted(set(steps)), [0, 1, 2, 3, 4, 5])
        stages = [e["job"]["stage"] for e in events if e["type"] == "job"]
        for expected in ("queued", "loading model", "encoding prompt", "sampling", "decoding", "done"):
            self.assertIn(expected, stages)
        self.assertEqual(sum(1 for e in events if e["type"] == "image"), 1)

    def test_cancel_while_running_stops_between_steps(self):
        self.backend.step_gate = threading.Event()
        job = self.submit(steps=20, count=4)
        self.assertTrue(wait_for(lambda: job.stage == "sampling"))
        self.jobs.cancel(job.id)
        self.backend.step_gate.set()
        self.assertTrue(wait_for(lambda: job.status == CANCELLED))
        self.assertEqual(job.images, [])
        self.assertLess(job.step, 20)

    def test_cancel_queued_job_never_runs(self):
        self.backend.step_gate = threading.Event()
        first = self.submit(steps=3)
        second = self.submit(prompt="a dog")
        self.assertTrue(wait_for(lambda: first.stage == "sampling"))
        self.jobs.cancel(second.id)
        self.assertEqual(second.status, CANCELLED)
        self.backend.step_gate.set()
        self.assertTrue(wait_for(lambda: first.status == DONE))
        self.assertNotIn(("encode", "a dog"), self.backend.calls)

    def test_jobs_run_in_submission_order(self):
        jobs = [self.submit(prompt=f"p{i}") for i in range(3)]
        self.assertTrue(wait_for(lambda: all(j.status == DONE for j in jobs)))
        encoded = [c[1] for c in self.backend.calls if c[0] == "encode"]
        self.assertEqual(encoded, ["p0", "p1", "p2"])

    def test_load_failure_fails_job_and_reports_model_error(self):
        self.backend.fail_load = "Model folder not found"
        job = self.submit()
        self.assertTrue(wait_for(lambda: job.status == FAILED))
        self.assertEqual(job.error, "Model folder not found")
        self.assertEqual(self.jobs.model_state, "error")
        # worker is still alive and recovers once the cause is fixed
        self.backend.fail_load = None
        job2 = self.submit()
        self.assertTrue(wait_for(lambda: job2.status == DONE))

    def test_sample_failure_does_not_kill_worker(self):
        self.backend.fail_sample = "CUDA out of memory"
        job = self.submit()
        self.assertTrue(wait_for(lambda: job.status == FAILED))
        self.assertIn("out of memory", job.error)
        self.backend.fail_sample = None
        job2 = self.submit()
        self.assertTrue(wait_for(lambda: job2.status == DONE))

    def test_unload_clears_prompt_cache(self):
        job = self.submit()
        self.assertTrue(wait_for(lambda: job.status == DONE))
        self.jobs.unload_model()
        job2 = self.submit()
        self.assertTrue(wait_for(lambda: job2.status == DONE))
        names = [c[0] for c in self.backend.calls]
        self.assertEqual(names.count("load"), 2)
        self.assertEqual(names.count("encode"), 2)


if __name__ == "__main__":
    unittest.main()
