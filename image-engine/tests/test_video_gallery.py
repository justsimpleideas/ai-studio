import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_engine.gallery import Gallery

VIDEO_REQUEST = {"kind": "video", "prompt": "waves at night", "negative_prompt": "", "width": 64, "height": 32,
                 "frames": 9, "fps": 24, "steps": 4, "guidance": 5.0, "seed": 7, "count": 1, "start_image": None}
IMAGE_REQUEST = {"prompt": "a still", "negative_prompt": "", "width": 64, "height": 32, "steps": 8,
                 "guidance": 0.0, "seed": 1, "count": 1}


def clip(count=9):
    return [Image.new("RGB", (64, 32), (i * 20, 40, 90)) for i in range(count)]


class VideoGalleryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gallery = Gallery(Path(self.tmp.name))

    def tearDown(self):
        self.gallery.close()
        self.tmp.cleanup()

    def save_video(self, **overrides):
        request = {**VIDEO_REQUEST, **overrides}
        return self.gallery.save_video(clip(request["frames"]), job_id="j" * 16, model="fake-video",
                                       request=request, seed=request["seed"], duration_ms=5, ffmpeg=None,
                                       stats={"peak_vram_bytes": 1}, provenance={"app": "x"})

    def save_image(self):
        return self.gallery.save(Image.new("RGB", (64, 32), "white"), job_id="j" * 16, model="m",
                                 request=IMAGE_REQUEST, seed=1, duration_ms=1)

    def test_a_video_record_says_what_it_is(self):
        rec = self.save_video()
        self.assertEqual(rec["media"], "video")
        self.assertEqual((rec["frames"], rec["fps"], rec["width"], rec["height"]), (9, 24, 64, 32))
        self.assertEqual(rec["mime"], "image/webp")  # no ffmpeg given
        self.assertIn("animated WebP", rec["stats"]["encoder"])
        self.assertEqual(rec["stats"]["peak_vram_bytes"], 1)
        self.assertEqual(rec["video_url"], f"/videos/{rec['id']}")
        self.assertEqual(rec["last_frame_url"], f"/frames/{rec['id']}-last.png")
        self.assertEqual(rec["image_url"], rec["thumb_url"])
        self.assertIsNone(rec["start_image"])
        self.assertNotIn("file", rec)
        self.assertNotIn("last_frame", rec)

    def test_the_last_frame_is_kept_as_a_png(self):
        rec = self.save_video()
        with Image.open(self.gallery.path_of(rec["id"], "last")) as last:
            self.assertEqual(last.getpixel((0, 0)), (8 * 20, 40, 90))

    def test_each_kind_of_file_resolves_only_for_its_media(self):
        video, image = self.save_video(), self.save_image()
        self.assertIsNone(self.gallery.path_of(video["id"], "image"))
        self.assertIsNone(self.gallery.path_of(image["id"], "video"))
        self.assertIsNone(self.gallery.path_of(image["id"], "last"))
        self.assertTrue(self.gallery.path_of(video["id"], "video").is_file())
        self.assertTrue(self.gallery.path_of(video["id"], "thumb").is_file())

    def test_listing_can_be_narrowed_to_one_media(self):
        self.save_video()
        self.save_image()
        self.assertEqual(self.gallery.list()["total"], 2)
        self.assertEqual([i["media"] for i in self.gallery.list(media="video")["items"]], ["video"])
        self.assertEqual([i["media"] for i in self.gallery.list(media="image")["items"]], ["image"])

    def test_delete_removes_the_video_its_thumbnail_and_its_last_frame(self):
        rec = self.save_video()
        files = [self.gallery.path_of(rec["id"], k) for k in ("video", "thumb", "last")]
        self.assertTrue(self.gallery.delete(rec["id"]))
        self.assertFalse(any(f.exists() for f in files))

    def test_a_start_picture_is_stored_once_by_content(self):
        a = self.gallery.save_input(Image.new("RGB", (40, 30), "red"))
        b = self.gallery.save_input(Image.new("RGBA", (40, 30), (255, 0, 0, 255)))
        c = self.gallery.save_input(Image.new("RGB", (40, 30), "blue"))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(list(self.gallery.inputs_dir.glob("*.png"))), 2)
        self.assertEqual(self.gallery.load_input(a).size, (40, 30))

    def test_huge_start_pictures_are_scaled_down(self):
        big = self.gallery.save_input(Image.new("RGB", (8000, 1000), "gray"))
        self.assertEqual(max(self.gallery.load_input(big).size), 4096)

    def test_start_pictures_from_the_gallery(self):
        image, video = self.save_image(), self.save_video()
        from_image = self.gallery.input_from(image["id"], "image")
        from_video = self.gallery.input_from(video["id"], "last")
        self.assertEqual(self.gallery.load_input(from_image).getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(self.gallery.load_input(from_video).getpixel((0, 0)), (160, 40, 90))
        self.assertIsNone(self.gallery.input_from(image["id"], "last"))  # an image has no last frame
        self.assertIsNone(self.gallery.input_from("0" * 16, "image"))

    def test_input_ids_are_checked(self):
        self.assertIsNone(self.gallery.input_path("../gallery"))
        self.assertIsNone(self.gallery.input_path("0" * 16))
        self.assertIsNone(self.gallery.load_input("not-an-id"))

    def test_the_start_picture_is_recorded_with_the_video(self):
        input_id = self.gallery.save_input(Image.new("RGB", (64, 32), "green"))
        self.assertEqual(self.save_video(start_image=input_id)["start_image"], input_id)


if __name__ == "__main__":
    unittest.main()
