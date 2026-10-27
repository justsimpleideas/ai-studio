import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from PIL import Image

from image_engine.config import SettingsStore
from image_engine.events import EventHub
from image_engine.gallery import Gallery
from image_engine.jobs import JobQueue
from image_engine.server import App, serve

from . import test_server as server_tests  # the module, so its TestCase is not collected twice
from .fakes import FakeBackend, FakeVideoBackend

TOKEN = server_tests.TOKEN


def png_bytes(size=(320, 200), color="orange", fmt="PNG", **save):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt, **save)
    return buffer.getvalue()


class VideoServerTests(unittest.TestCase):
    request = server_tests.ServerTests.request
    generate = server_tests.ServerTests.generate

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.image = FakeBackend()
        cls.video = FakeVideoBackend()
        cls.events = EventHub()
        cls.gallery = Gallery(root / "outputs")
        cls.jobs = JobQueue({"image": cls.image, "video": cls.video}, cls.gallery, cls.events)
        cls.settings = SettingsStore(root / "data")
        cls.app = App(settings=cls.settings, backend=cls.image, gallery=cls.gallery, events=cls.events,
                      jobs=cls.jobs, port=0, token=TOKEN)
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

    def make_video(self, **fields):
        status, job, _ = self.request("POST", "/api/jobs", {"kind": "video", "prompt": "waves", "width": 256,
                                                            "height": 256, "frames": 9, "steps": 2, **fields})
        self.assertEqual(status, 200, job)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            _, job, _ = self.request("GET", f"/api/jobs/{job['id']}")
            if job["status"] in ("done", "failed", "cancelled"):
                self.assertEqual(job["status"], "done", job["error"])
                return self.gallery.get(job["images"][0])
            time.sleep(0.02)
        self.fail("job did not finish")

    def upload(self, raw, ctype="image/png"):
        return self.request("POST", "/api/inputs", raw=raw, headers={"Content-Type": ctype})

    # ------------------------------------------------------------ status and settings

    def test_status_describes_the_video_side(self):
        status, body, _ = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(set(body["models"]), {"image", "video"})
        self.assertEqual(body["models"]["video"]["id"], "fake-video")
        self.assertEqual(body["video"]["limits"]["max_frames"], 121)
        self.assertEqual(body["video"]["defaults"]["width"], 1280)
        self.assertIn(body["video"]["encoder"], ("mp4", "webp"))
        self.assertEqual(body["model"]["id"], "fake-model")  # the image model, as before

    def test_video_settings_round_trip(self):
        status, body, _ = self.request("PUT", "/api/settings", {"video_memory_mode": "low"})
        self.assertEqual((status, body["video_memory_mode"]), (200, "low"))
        status, _, _ = self.request("PUT", "/api/settings", {"video_memory_mode": "huge"})
        self.assertEqual(status, 400)
        self.request("PUT", "/api/settings", {"video_memory_mode": "auto"})

    def use_video_folder(self, class_name: str) -> None:
        """Point the settings at a folder holding the given kind of video model."""
        folder = Path(self.tmp.name) / f"video-{class_name}"
        folder.mkdir(exist_ok=True)
        (folder / "model_index.json").write_text(json.dumps({"_class_name": class_name}), encoding="utf-8")
        before = self.request("GET", "/api/settings")[1]["video_model_dir"]
        status, body, _ = self.request("PUT", "/api/settings", {"video_model_dir": str(folder)})
        self.assertEqual(status, 200, body)
        self.addCleanup(lambda: self.assertEqual(
            self.request("PUT", "/api/settings", {"video_model_dir": before})[0], 200))

    def test_status_says_which_video_model_the_folder_holds(self):
        _, body, _ = self.request("GET", "/api/status")
        self.assertIsNone(body["video"]["profile"])  # no model folder yet
        self.use_video_folder("WanDMDPipeline")
        _, body, _ = self.request("GET", "/api/status")
        self.assertEqual(body["video"]["profile"]["name"], "fast")
        self.assertTrue(body["video"]["profile"]["fixed"])
        self.assertEqual((body["video"]["defaults"]["steps"], body["video"]["defaults"]["guidance"]), (3, 1.0))
        self.use_video_folder("WanPipeline")
        _, body, _ = self.request("GET", "/api/status")
        self.assertEqual(body["video"]["profile"]["name"], "standard")
        self.assertEqual(body["video"]["defaults"]["steps"], 50)

    def test_a_distilled_model_runs_and_records_its_own_steps_and_no_guidance(self):
        self.use_video_folder("WanDMDPipeline")
        rec = self.make_video(steps=40, guidance=5.0, negative_prompt="blurry")
        self.assertEqual((rec["steps"], rec["guidance"], rec["negative_prompt"]), (3, 1.0, ""))

    def test_the_original_model_keeps_the_steps_asked_for(self):
        self.use_video_folder("WanPipeline")
        rec = self.make_video(steps=4, guidance=5.0, negative_prompt="blurry")
        self.assertEqual((rec["steps"], rec["guidance"], rec["negative_prompt"]), (4, 5.0, "blurry"))

    def test_loading_a_kind_that_does_not_exist_is_refused(self):
        status, _, _ = self.request("POST", "/api/model/load", {"kind": "audio"})
        self.assertEqual(status, 400)

    # ------------------------------------------------------------ jobs

    def test_a_video_job_and_its_files(self):
        rec = self.make_video()
        status, body, resp = self.request("GET", rec["video_url"])
        self.assertEqual(status, 200)
        self.assertIn(resp.getheader("Content-Type"), ("video/mp4", "image/webp"))
        self.assertEqual(resp.getheader("Accept-Ranges"), "bytes")
        self.assertGreater(len(body), 100)
        status, _, resp = self.request("GET", rec["last_frame_url"])
        self.assertEqual((status, resp.getheader("Content-Type")), (200, "image/png"))
        status, _, _ = self.request("GET", rec["thumb_url"])
        self.assertEqual(status, 200)
        # A video has no full-size PNG, and an image has no video.
        self.assertEqual(self.request("GET", f"/images/{rec['id']}.png")[0], 404)

    def test_videos_can_be_read_in_byte_ranges(self):
        rec = self.make_video()
        _, whole, _ = self.request("GET", rec["video_url"])
        status, part, resp = self.request("GET", rec["video_url"], headers={"Range": "bytes=10-19"})
        self.assertEqual(status, 206)
        self.assertEqual(part, whole[10:20])
        self.assertEqual(resp.getheader("Content-Range"), f"bytes 10-19/{len(whole)}")
        status, tail, _ = self.request("GET", rec["video_url"], headers={"Range": "bytes=-5"})
        self.assertEqual((status, tail), (206, whole[-5:]))
        status, rest, _ = self.request("GET", rec["video_url"], headers={"Range": f"bytes={len(whole) - 3}-"})
        self.assertEqual((status, rest), (206, whole[-3:]))
        status, _, resp = self.request("GET", rec["video_url"], headers={"Range": f"bytes={len(whole)}-"})
        self.assertEqual(status, 416)
        self.assertEqual(resp.getheader("Content-Range"), f"bytes */{len(whole)}")

    def test_bad_video_requests_are_refused(self):
        for body in ({"kind": "video", "prompt": "x", "frames": 10},
                     {"kind": "video", "prompt": "x", "width": 1000},
                     {"kind": "video", "prompt": "x", "start_image": "0" * 16}):  # not uploaded
            with self.subTest(body=body):
                status, _, _ = self.request("POST", "/api/jobs", body)
                self.assertEqual(status, 400)

    def test_a_video_can_be_exported_as_a_gif(self):
        rec = self.make_video(prompt="a secret lighthouse prompt")
        status, body, resp = self.request("POST", f"/api/gallery/{rec['id']}/gif", {"width": 128, "fps": 12})
        self.assertEqual(status, 200, body)
        self.assertEqual(resp.getheader("Content-Type"), "image/gif")
        self.assertEqual(body[:6], b"GIF89a")
        with Image.open(io.BytesIO(body)) as gif:
            self.assertEqual(gif.size, (128, 128))
        self.assertNotIn(b"lighthouse", body)
        # Nothing is left behind in the gallery: the GIF is made, sent and dropped.
        self.assertEqual([p for p in Path(self.gallery.root).rglob("*") if p.suffix in (".gif", ".part")], [])

    def test_bad_gif_requests_are_refused(self):
        rec = self.make_video()
        for body in ({"width": 128}, {"width": 128, "fps": 12, "extra": 1}, {"width": 10, "fps": 12},
                     {"width": 128, "fps": 60}, {"width": "128", "fps": 12}, {"width": True, "fps": 12}, []):
            with self.subTest(body=body):
                self.assertEqual(self.request("POST", f"/api/gallery/{rec['id']}/gif", body)[0], 400)
        self.assertEqual(self.request("POST", f"/api/gallery/{'0' * 16}/gif", {"width": 128, "fps": 12})[0], 404)
        image_job = self.generate()
        self.assertEqual(image_job["status"], "done", image_job["error"])
        image_id = image_job["images"][0]  # an image is not a video
        self.assertEqual(self.request("POST", f"/api/gallery/{image_id}/gif", {"width": 128, "fps": 12})[0], 404)
        status, _, _ = self.request("POST", f"/api/gallery/{rec['id']}/gif", raw=b"{}",
                                    headers={"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_the_gallery_lists_one_media_on_request(self):
        self.make_video()
        _, page, _ = self.request("GET", "/api/gallery?media=video")
        self.assertTrue(page["items"])
        self.assertTrue(all(i["media"] == "video" for i in page["items"]))
        self.assertEqual(self.request("GET", "/api/gallery?media=audio")[0], 400)

    # ------------------------------------------------------------ start pictures

    def test_uploading_a_start_picture_and_making_a_video_from_it(self):
        status, body, _ = self.upload(png_bytes())
        self.assertEqual(status, 200, body)
        self.assertEqual((body["width"], body["height"]), (320, 200))
        status, served, resp = self.request("GET", f"/inputs/{body['id']}.png")
        self.assertEqual((status, resp.getheader("Content-Type")), (200, "image/png"))
        rec = self.make_video(start_image=body["id"])
        self.assertEqual(rec["start_image"], body["id"])

    def test_jpeg_and_webp_are_accepted_and_phone_rotation_is_applied(self):
        exif = Image.Exif()
        exif[0x0112] = 6  # "rotate 90 degrees clockwise to view"
        status, body, _ = self.upload(png_bytes((300, 100), fmt="JPEG", exif=exif.tobytes()), "image/jpeg")
        self.assertEqual(status, 200, body)
        self.assertEqual((body["width"], body["height"]), (100, 300))
        status, body, _ = self.upload(png_bytes(fmt="WEBP"), "image/webp")
        self.assertEqual(status, 200, body)

    def test_uploads_that_are_not_pictures_are_refused(self):
        self.assertEqual(self.upload(b"not an image")[0], 400)
        self.assertEqual(self.upload(png_bytes(), "image/jpeg")[0], 400)  # says JPEG, is PNG
        self.assertEqual(self.upload(png_bytes(), "image/gif")[0], 415)
        self.assertEqual(self.upload(png_bytes(), "application/octet-stream")[0], 415)
        # The JSON endpoints still refuse anything but JSON.
        status, _, _ = self.request("POST", "/api/jobs", raw=png_bytes(), headers={"Content-Type": "image/png"})
        self.assertEqual(status, 415)

    def test_uploads_still_need_the_token(self):
        status, _, _ = self.request("POST", "/api/inputs", raw=png_bytes(),
                                    headers={"Content-Type": "image/png", "X-Studio-Token": "wrong"})
        self.assertEqual(status, 401)

    def test_start_pictures_from_the_gallery(self):
        rec = self.make_video()
        status, body, _ = self.request("POST", "/api/inputs/from-gallery", {"id": rec["id"], "frame": "last"})
        self.assertEqual(status, 200, body)
        self.assertEqual((body["width"], body["height"]), (256, 256))
        status, _, _ = self.request("POST", "/api/inputs/from-gallery", {"id": rec["id"], "frame": "image"})
        self.assertEqual(status, 404)  # a video is not an image
        status, _, _ = self.request("POST", "/api/inputs/from-gallery", {"id": "../x", "frame": "image"})
        self.assertEqual(status, 400)

    def test_inputs_outside_the_store_are_not_served(self):
        self.assertEqual(self.request("GET", "/inputs/..%2Fgallery.db")[0], 404)
        self.assertEqual(self.request("GET", f"/inputs/{'0' * 16}.png")[0], 404)


if __name__ == "__main__":
    unittest.main()
