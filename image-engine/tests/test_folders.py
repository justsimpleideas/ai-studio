import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from PIL import Image

from image_engine.config import SettingsStore
from image_engine.events import EventHub
from image_engine.gallery import FolderError, Gallery
from image_engine.jobs import JobQueue
from image_engine.server import App, serve

from . import test_server as server_tests
from .fakes import FakeBackend

REQUEST = {"prompt": "a still", "negative_prompt": "", "width": 8, "height": 8, "steps": 8,
           "guidance": 0.0, "seed": 1, "count": 1}


class FolderGalleryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gallery = Gallery(Path(self.tmp.name))

    def tearDown(self):
        self.gallery.close()
        self.tmp.cleanup()

    def save(self, prompt="a still"):
        return self.gallery.save(Image.new("RGB", (8, 8)), job_id="j" * 16, model="m",
                                 request={**REQUEST, "prompt": prompt}, seed=1, duration_ms=1)

    def test_create_rename_and_list_folders_by_name(self):
        b = self.gallery.create_folder("  Beach   trip ")
        a = self.gallery.create_folder("animals")
        self.assertEqual(b["name"], "Beach trip")  # spaces tidied
        self.assertEqual([f["name"] for f in self.gallery.list_folders()], ["animals", "Beach trip"])
        renamed = self.gallery.rename_folder(a["id"], "Zoo")
        self.assertEqual(renamed["name"], "Zoo")
        self.assertIsNone(self.gallery.rename_folder("0" * 16, "x"))

    def test_names_are_checked(self):
        self.gallery.create_folder("Work")
        for bad in ("", "   ", "x" * 61, 5, None, "a\u0007b", "work", "WORK"):
            with self.subTest(bad=bad), self.assertRaises(FolderError):
                self.gallery.create_folder(bad)
        other = self.gallery.create_folder("Other")
        with self.assertRaises(FolderError):
            self.gallery.rename_folder(other["id"], "work")
        self.assertEqual(self.gallery.rename_folder(other["id"], "other")["name"], "other")  # its own name, recased

    def test_items_move_between_folders_and_counts_follow(self):
        folder = self.gallery.create_folder("Keep")
        a, b = self.save("one"), self.save("two")
        self.assertIsNone(a["folder_id"])
        self.assertEqual(self.gallery.set_folder(a["id"], folder["id"])["folder_id"], folder["id"])
        self.assertEqual(self.gallery.list_folders()[0]["count"], 1)
        self.assertEqual([i["id"] for i in self.gallery.list(folder=folder["id"])["items"]], [a["id"]])
        self.assertEqual(self.gallery.list()["total"], 2)  # "All" still shows everything
        self.gallery.set_folder(a["id"], None)
        self.assertEqual(self.gallery.list(folder=folder["id"])["total"], 0)
        with self.assertRaises(FolderError):
            self.gallery.set_folder(b["id"], "0" * 16)

    def test_favourites_are_shown_whatever_folder_they_are_in(self):
        x, y = self.gallery.create_folder("X"), self.gallery.create_folder("Y")
        a, b, c = self.save("a"), self.save("b"), self.save("c")
        self.gallery.set_folder(a["id"], x["id"])
        self.gallery.set_folder(b["id"], y["id"])
        for item in (a, b, c):
            self.gallery.set_favorite(item["id"], True)
        self.assertEqual(self.gallery.list(favorites_only=True)["total"], 3)
        self.assertEqual(self.gallery.list(favorites_only=True, folder=x["id"])["total"], 1)

    def test_deleting_a_folder_keeps_what_was_in_it(self):
        folder = self.gallery.create_folder("Temp")
        a = self.save()
        self.gallery.set_folder(a["id"], folder["id"])
        self.assertEqual(self.gallery.delete_folder(folder["id"]), 1)
        self.assertEqual(self.gallery.list_folders(), [])
        kept = self.gallery.get(a["id"])
        self.assertIsNotNone(kept)
        self.assertIsNone(kept["folder_id"])
        self.assertTrue(self.gallery.path_of(a["id"], "image").is_file())
        self.assertIsNone(self.gallery.delete_folder(folder["id"]))

    def test_a_gallery_from_before_folders_opens_with_everything_in_no_folder(self):
        self.save()
        self.gallery.close()
        db = sqlite3.connect(Path(self.tmp.name) / "gallery.db")
        db.execute("DROP TABLE folders")
        db.execute("ALTER TABLE images DROP COLUMN folder_id")
        db.commit()
        db.close()
        self.gallery = Gallery(Path(self.tmp.name))
        self.assertIsNone(self.gallery.list()["items"][0]["folder_id"])
        self.assertEqual(self.gallery.list_folders(), [])


class MoveManyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gallery = Gallery(Path(self.tmp.name))

    def tearDown(self):
        self.gallery.close()
        self.tmp.cleanup()

    def save(self):
        return self.gallery.save(Image.new("RGB", (8, 8)), job_id="j" * 16, model="m", request=REQUEST,
                                 seed=1, duration_ms=1)

    def test_moves_several_at_once_and_skips_what_is_gone_or_out_of_view(self):
        trips = self.gallery.create_folder("Trips")
        locked = self.gallery.create_folder("Private")
        a, b, c, hidden = self.save(), self.save(), self.save(), self.save()
        self.gallery.set_folder(hidden["id"], locked["id"])
        self.gallery.set_pin(locked["id"], "123456")
        self.gallery.lock_again(locked["id"])
        moved = self.gallery.set_folder_many([a["id"], b["id"], a["id"], "0" * 16, hidden["id"]], trips["id"])
        self.assertEqual([r["id"] for r in moved], [a["id"], b["id"]])
        self.assertEqual(self.gallery.get(c["id"])["folder_id"], None)
        self.assertEqual(self.gallery.list(folder=trips["id"])["total"], 2)
        out = self.gallery.set_folder_many([a["id"]], None)
        self.assertIsNone(out[0]["folder_id"])
        with self.assertRaises(FolderError):
            self.gallery.set_folder_many([a["id"]], "0" * 16)


class FolderOrderTests(unittest.TestCase):
    """The order the user drags the folders into (Gallery.move_folder)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.gallery = Gallery(self.root)

    def tearDown(self):
        self.gallery.close()
        self.tmp.cleanup()

    def names(self):
        return [f["name"] for f in self.gallery.list_folders()]

    def make(self, *names):
        return {name: self.gallery.create_folder(name)["id"] for name in names}

    def test_folders_start_alphabetical(self):
        self.make("Zoo", "animals", "Beach")
        self.assertEqual(self.names(), ["animals", "Beach", "Zoo"])

    def test_a_folder_can_be_moved_in_front_of_another(self):
        ids = self.make("animals", "Beach", "Zoo")
        self.gallery.move_folder(ids["Zoo"], ids["Beach"])
        self.assertEqual(self.names(), ["animals", "Zoo", "Beach"])

    def test_a_folder_can_be_moved_to_the_end(self):
        ids = self.make("animals", "Beach", "Zoo")
        self.gallery.move_folder(ids["animals"], None)
        self.assertEqual(self.names(), ["Beach", "Zoo", "animals"])

    def test_the_order_outlives_the_gallery_being_closed(self):
        ids = self.make("animals", "Beach", "Zoo")
        self.gallery.move_folder(ids["Zoo"], ids["animals"])
        self.gallery.close()
        self.gallery = Gallery(self.root)
        self.assertEqual(self.names(), ["Zoo", "animals", "Beach"])

    def test_a_new_folder_is_alphabetical_until_there_is_an_order_then_last(self):
        ids = self.make("Beach", "Zoo")
        self.gallery.create_folder("animals")
        self.assertEqual(self.names(), ["animals", "Beach", "Zoo"])
        self.gallery.move_folder(ids["Zoo"], ids["Beach"])
        self.gallery.create_folder("Alps")
        self.assertEqual(self.names(), ["animals", "Zoo", "Beach", "Alps"])

    def test_a_hidden_folder_keeps_its_place_among_the_ones_in_view(self):
        ids = self.make("animals", "Beach", "Zoo")
        self.gallery.set_hidden(ids["Beach"], True)
        self.assertEqual(self.names(), ["animals", "Zoo"])
        # Moved by name the session can see; the hidden one is still between them.
        self.gallery.move_folder(ids["Zoo"], ids["animals"])
        self.gallery.set_show_hidden(True)
        self.assertEqual(self.names(), ["Zoo", "animals", "Beach"])

    def test_nothing_moves_anywhere_on_disk_and_no_item_changes_folder(self):
        ids = self.make("animals", "Beach")
        item = self.gallery.save(Image.new("RGB", (8, 8)), job_id="j" * 16, model="m", request=REQUEST,
                                 seed=1, duration_ms=1)
        self.gallery.set_folder(item["id"], ids["Beach"])
        before = self.gallery.get(item["id"])["file_name"]
        files = sorted(p.name for p in self.root.rglob("*.png"))
        self.gallery.move_folder(ids["Beach"], ids["animals"])
        after = self.gallery.get(item["id"])
        self.assertEqual(after["folder_id"], ids["Beach"])
        self.assertEqual(after["file_name"], before)
        self.assertEqual(sorted(p.name for p in self.root.rglob("*.png")), files)

    def test_an_impossible_move_is_refused(self):
        ids = self.make("animals", "Beach")
        with self.assertRaises(FolderError):
            self.gallery.move_folder(ids["animals"], ids["animals"])
        with self.assertRaises(FolderError):
            self.gallery.move_folder("0" * 16, None)
        with self.assertRaises(FolderError):
            self.gallery.move_folder(ids["animals"], "0" * 16)
        with self.assertRaises(FolderError):
            self.gallery.move_folder(ids["animals"], 5)
        self.assertEqual(self.names(), ["animals", "Beach"])


class FolderServerTests(unittest.TestCase):
    request = server_tests.ServerTests.request

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.backend = FakeBackend()
        cls.events = EventHub()
        cls.gallery = Gallery(root / "outputs")
        cls.jobs = JobQueue(cls.backend, cls.gallery, cls.events)
        cls.app = App(settings=SettingsStore(root / "data"), backend=cls.backend, gallery=cls.gallery,
                      events=cls.events, jobs=cls.jobs, port=0, token=server_tests.TOKEN)
        cls.httpd = serve(cls.app, "127.0.0.1")
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.app.stopping.set()
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.gallery.close()
        cls.tmp.cleanup()

    def item(self):
        return self.gallery.save(Image.new("RGB", (8, 8)), job_id="j" * 16, model="m", request=REQUEST,
                                 seed=1, duration_ms=1)

    def test_folder_lifecycle_over_http(self):
        status, folder, _ = self.request("POST", "/api/folders", {"name": "Trips"})
        self.assertEqual(status, 200, folder)
        status, _, _ = self.request("POST", "/api/folders", {"name": "trips"})
        self.assertEqual(status, 400)  # duplicate, ignoring case
        item = self.item()
        status, moved, _ = self.request("PATCH", f"/api/gallery/{item['id']}", {"folder": folder["id"]})
        self.assertEqual((status, moved["folder_id"]), (200, folder["id"]))
        _, page, _ = self.request("GET", f"/api/gallery?folder={folder['id']}")
        self.assertEqual([i["id"] for i in page["items"]], [item["id"]])
        status, renamed, _ = self.request("PATCH", f"/api/folders/{folder['id']}", {"name": "Holidays"})
        self.assertEqual((status, renamed["name"], renamed["count"]), (200, "Holidays", 1))
        _, listing, _ = self.request("GET", "/api/folders")
        self.assertIn("Holidays", [f["name"] for f in listing["items"]])
        status, gone, _ = self.request("DELETE", f"/api/folders/{folder['id']}")
        self.assertEqual((status, gone["moved"]), (200, 1))
        self.assertIsNone(self.gallery.get(item["id"])["folder_id"])
        self.assertEqual(self.request("DELETE", f"/api/folders/{folder['id']}")[0], 404)

    def test_move_many_over_http(self):
        _, folder, _ = self.request("POST", "/api/folders", {"name": "Bulk"})
        a, b = self.item(), self.item()
        status, body, _ = self.request("POST", "/api/gallery/folder", {"ids": [a["id"], b["id"]], "folder": folder["id"]})
        self.assertEqual(status, 200, body)
        self.assertEqual({i["folder_id"] for i in body["items"]}, {folder["id"]})
        for bad in ({"ids": [], "folder": None}, {"ids": ["../x"], "folder": None}, {"ids": [a["id"]]},
                    {"ids": [a["id"]], "folder": 5}, {"ids": [a["id"]] * 2001, "folder": None}):
            with self.subTest(bad=str(bad)[:60]):
                self.assertEqual(self.request("POST", "/api/gallery/folder", bad)[0], 400)
        self.assertEqual(self.request("POST", "/api/gallery/folder", {"ids": [a["id"]], "folder": "0" * 16})[0], 400)

    def test_bad_folder_requests_are_refused(self):
        item = self.item()
        for body in ({"folder": "../x"}, {"folder": 5}, {"folder": "0" * 16}, {"colour": "red"}, {}):
            with self.subTest(body=body):
                self.assertEqual(self.request("PATCH", f"/api/gallery/{item['id']}", body)[0], 400)
        self.assertEqual(self.request("PATCH", f"/api/gallery/{'0' * 16}", {"folder": None})[0], 404)
        self.assertEqual(self.request("GET", "/api/gallery?folder=../../x")[0], 400)
        self.assertEqual(self.request("POST", "/api/folders", {"name": ""})[0], 400)
        self.assertEqual(self.request("PATCH", f"/api/folders/{'0' * 16}", {"name": "x"})[0], 404)

    def test_the_order_can_be_changed_over_http(self):
        made = [self.request("POST", "/api/folders", {"name": name})[1] for name in ("Ants", "Bees", "Cats")]
        ids = {f["name"]: f["id"] for f in made}
        def mine(listing):
            """Only this test's folders: the class shares its gallery with the others."""
            return [f["name"] for f in listing["items"] if f["name"] in ids]

        status, listing, _ = self.request("POST", f"/api/folders/{ids['Cats']}/order", {"before": ids["Bees"]})
        self.assertEqual(status, 200, listing)
        self.assertEqual(mine(listing), ["Ants", "Cats", "Bees"])
        _, listing, _ = self.request("GET", "/api/folders")
        self.assertEqual(mine(listing), ["Ants", "Cats", "Bees"])
        _, listing, _ = self.request("POST", f"/api/folders/{ids['Ants']}/order", {"before": None})
        self.assertEqual(mine(listing), ["Cats", "Bees", "Ants"])
        self.assertEqual(self.request("POST", f"/api/folders/{'0' * 16}/order", {"before": None})[0], 400)
        self.assertEqual(self.request("POST", f"/api/folders/{ids['Ants']}/order", {"before": "../x"})[0], 400)
        for name in ("Ants", "Bees", "Cats"):
            self.request("DELETE", f"/api/folders/{ids[name]}")

    def test_favourite_and_folder_together(self):
        _, folder, _ = self.request("POST", "/api/folders", {"name": "Both"})
        item = self.item()
        status, rec, _ = self.request("PATCH", f"/api/gallery/{item['id']}", {"favorite": True, "folder": folder["id"]})
        self.assertEqual((status, rec["favorite"], rec["folder_id"]), (200, True, folder["id"]))


if __name__ == "__main__":
    unittest.main()
