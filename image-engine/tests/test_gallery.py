import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_engine.gallery import PNG_METADATA_KEY, Gallery

REQUEST = {"prompt": "a lighthouse at dusk", "negative_prompt": "", "width": 64, "height": 32,
           "steps": 9, "guidance": 0.0, "seed": 7, "count": 1}


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.gallery = Gallery(self.root)

    def tearDown(self):
        self.gallery.close()
        self.tmp.cleanup()

    def save(self, prompt="a lighthouse at dusk", seed=7):
        return self.gallery.save(Image.new("RGB", (64, 32), "navy"), job_id="j" * 16, model="m",
                                 request={**REQUEST, "prompt": prompt}, seed=seed, duration_ms=1500,
                                 stats={"peak_vram_bytes": 10})

    def test_png_carries_full_request(self):
        rec = self.save()
        path = self.gallery.path_of(rec["id"], "image")
        with Image.open(path) as img:
            meta = json.loads(img.text[PNG_METADATA_KEY])
        self.assertEqual(meta["prompt"], "a lighthouse at dusk")
        self.assertEqual(meta["seed"], 7)
        self.assertEqual(meta["model"], "m")
        self.assertEqual(meta["format"], 1)

    def test_provenance_is_recorded_but_never_overrides_the_request(self):
        rec = self.gallery.save(Image.new("RGB", (8, 8)), job_id="j" * 16, model="m", request=REQUEST,
                                seed=7, duration_ms=1,
                                provenance={"app": "Local AI Studio 9.9", "model_revision": "org/model@abc",
                                            "libraries": {"diffusers": "1.0"}, "prompt": "not this"})
        with Image.open(self.gallery.path_of(rec["id"], "image")) as img:
            meta = json.loads(img.text[PNG_METADATA_KEY])
        self.assertEqual(meta["prompt"], "a lighthouse at dusk")
        self.assertEqual(meta["app"], "Local AI Studio 9.9")
        self.assertEqual(meta["model_revision"], "org/model@abc")
        self.assertEqual(self.gallery.get(rec["id"])["provenance"]["libraries"], {"diffusers": "1.0"})

    def test_a_gallery_from_before_provenance_still_opens(self):
        self.gallery.close()
        old_root = self.root / "old"
        old_root.mkdir()
        import sqlite3
        db = sqlite3.connect(old_root / "gallery.db")
        db.executescript(
            "CREATE TABLE images (id TEXT PRIMARY KEY, created REAL NOT NULL, file TEXT NOT NULL,"
            " thumb TEXT NOT NULL, job_id TEXT NOT NULL, model TEXT NOT NULL, prompt TEXT NOT NULL,"
            " negative_prompt TEXT NOT NULL DEFAULT '', width INTEGER NOT NULL, height INTEGER NOT NULL,"
            " steps INTEGER NOT NULL, guidance REAL NOT NULL, seed INTEGER NOT NULL,"
            " duration_ms INTEGER NOT NULL, favorite INTEGER NOT NULL DEFAULT 0, stats TEXT NOT NULL DEFAULT '{}');"
            "INSERT INTO images VALUES ('0123456789abcdef', 1.0, 'images/a.png', 'thumbs/a.webp', 'job', 'm',"
            " 'old prompt', '', 8, 8, 8, 0.0, 1, 10, 0, '{}');")
        db.commit()
        db.close()
        self.gallery = Gallery(old_root)
        self.assertEqual(self.gallery.get("0123456789abcdef")["provenance"], {})
        new = self.gallery.save(Image.new("RGB", (8, 8)), job_id="j" * 16, model="m", request=REQUEST,
                                seed=1, duration_ms=1, provenance={"app": "x"})
        self.assertEqual(self.gallery.get(new["id"])["provenance"], {"app": "x"})

    def test_thumbnail_is_small_webp(self):
        big = self.gallery.save(Image.new("RGB", (1024, 512)), job_id="j" * 16, model="m",
                                request=REQUEST, seed=1, duration_ms=1)
        with Image.open(self.gallery.path_of(big["id"], "thumb")) as thumb:
            self.assertEqual(thumb.format, "WEBP")
            self.assertEqual(thumb.size, (384, 192))

    def test_record_hides_paths_and_exposes_urls(self):
        rec = self.save()
        self.assertNotIn("file", rec)
        self.assertEqual(rec["image_url"], f"/images/{rec['id']}.png")
        self.assertEqual(rec["stats"], {"peak_vram_bytes": 10})
        self.assertFalse(rec["favorite"])

    def test_list_newest_first_with_search_and_paging(self):
        a = self.save("red apple", 1)
        b = self.save("green apple", 2)
        c = self.save("blue sky", 3)
        listing = self.gallery.list()
        self.assertEqual(listing["total"], 3)
        self.assertEqual([i["id"] for i in listing["items"]], [c["id"], b["id"], a["id"]])
        self.assertEqual(self.gallery.list(query="apple")["total"], 2)
        self.assertEqual(len(self.gallery.list(limit=1, offset=1)["items"]), 1)

    def test_search_treats_wildcards_literally(self):
        self.save("100% cotton")
        self.save("1000 cats")
        self.assertEqual(self.gallery.list(query="100%")["total"], 1)
        self.assertEqual(self.gallery.list(query="_")["total"], 0)

    def test_favorite_and_filter(self):
        a = self.save("one")
        self.save("two")
        self.assertTrue(self.gallery.set_favorite(a["id"], True)["favorite"])
        self.assertEqual([i["id"] for i in self.gallery.list(favorites_only=True)["items"]], [a["id"]])

    def test_delete_removes_files(self):
        rec = self.save()
        image, thumb = self.gallery.path_of(rec["id"], "image"), self.gallery.path_of(rec["id"], "thumb")
        self.assertTrue(self.gallery.delete(rec["id"]))
        self.assertFalse(image.exists() or thumb.exists())
        self.assertIsNone(self.gallery.get(rec["id"]))
        self.assertFalse(self.gallery.delete(rec["id"]))

    def test_index_survives_reopen(self):
        rec = self.save()
        self.gallery.close()
        self.gallery = Gallery(self.root)
        self.assertEqual(self.gallery.get(rec["id"])["prompt"], REQUEST["prompt"])


if __name__ == "__main__":
    unittest.main()
