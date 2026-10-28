"""Hidden and locked gallery folders (gallery.py): what each session sees, PINs, wrong tries."""

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from image_engine import gallery as gallery_module
from image_engine.config import SettingsStore
from image_engine.events import EventHub
from image_engine.gallery import FolderError, FolderLocked, Gallery, TooManyTries
from image_engine.jobs import JobQueue
from image_engine.server import App, serve

from . import test_server as server_tests
from .fakes import FakeBackend

REQUEST = {"prompt": "a still", "negative_prompt": "", "width": 8, "height": 8, "steps": 8,
           "guidance": 0.0, "seed": 1, "count": 1}
PIN = "246810"


class PrivacyGalleryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gallery = Gallery(Path(self.tmp.name))

    def tearDown(self):
        self.gallery.close()
        self.tmp.cleanup()

    def save(self, prompt="a still", folder=None):
        record = self.gallery.save(Image.new("RGB", (8, 8)), job_id="j" * 16, model="m",
                                   request={**REQUEST, "prompt": prompt}, seed=1, duration_ms=1)
        if folder:
            self.gallery.set_folder(record["id"], folder["id"])
        return record

    def visible(self, **query):
        return {i["id"] for i in self.gallery.list(**query)["items"]}

    def assert_out_of_view(self, item, folder):
        self.assertNotIn(item["id"], self.visible())
        self.assertNotIn(item["id"], self.visible(favorites_only=True))
        self.assertNotIn(item["id"], self.visible(query="secret"))
        self.assertIsNone(self.gallery.get(item["id"]))
        for kind in ("image", "thumb"):
            self.assertIsNone(self.gallery.path_of(item["id"], kind))
        self.assertIsNone(self.gallery.input_from(item["id"], "image"))
        self.assertFalse(self.gallery.delete(item["id"]))
        self.assertIn(folder["id"], self.gallery.blocked_folders())

    def assert_in_view(self, item):
        self.assertIn(item["id"], self.visible())
        self.assertIsNotNone(self.gallery.get(item["id"]))
        self.assertTrue(self.gallery.path_of(item["id"], "image").is_file())

    # ------------------------------------------------------------ hidden

    def test_a_hidden_folder_and_its_items_are_out_of_view_until_shown(self):
        folder = self.gallery.create_folder("Private")
        item = self.save("secret", folder)
        self.gallery.set_favorite(item["id"], True)
        loose = self.save("loose")
        self.assertTrue(self.gallery.set_hidden(folder["id"], True)["hidden"])

        self.assertEqual(self.gallery.list_folders(), [])
        self.assert_out_of_view(item, folder)
        self.assertIn(loose["id"], self.visible())
        with self.assertRaises(FolderError):
            self.gallery.list(folder=folder["id"])
        self.assertIsNone(self.gallery.rename_folder(folder["id"], "x"))  # as if it did not exist

        self.gallery.set_show_hidden(True)
        self.assertEqual([(f["name"], f["hidden"]) for f in self.gallery.list_folders()], [("Private", True)])
        self.assert_in_view(item)
        self.assertEqual(self.visible(folder=folder["id"]), {item["id"]})

        self.gallery.reset_privacy()  # the studio window closed
        self.assertFalse(self.gallery.show_hidden)
        self.assertEqual(self.gallery.list_folders(), [])
        self.assert_out_of_view(item, folder)

        self.gallery.set_show_hidden(True)
        self.assertFalse(self.gallery.set_hidden(folder["id"], False)["hidden"])
        self.gallery.reset_privacy()
        self.assert_in_view(item)

    def test_items_go_into_a_hidden_folder_only_while_it_is_shown(self):
        folder = self.gallery.create_folder("Private")
        self.gallery.set_hidden(folder["id"], True)
        item = self.save()
        with self.assertRaises(FolderError):
            self.gallery.set_folder(item["id"], folder["id"])
        self.gallery.set_folder(item["id"], folder["id"], everything=True)  # the engine's own filing
        self.assertIsNone(self.gallery.get(item["id"]))

    def test_a_hidden_name_does_not_block_a_new_folder(self):
        hidden = self.gallery.create_folder("Diary")
        self.gallery.set_hidden(hidden["id"], True)
        self.gallery.create_folder("diary")  # refusing it would tell that the hidden one exists
        self.gallery.set_show_hidden(True)
        with self.assertRaises(FolderError):
            self.gallery.create_folder("DIARY")
        names = [f["name"] for f in self.gallery.list_folders(everything=True)]
        self.assertEqual(sorted(names), ["Diary", "diary"])

    # ------------------------------------------------------------ locked

    def test_a_locked_folder_keeps_its_items_out_of_view_until_unlocked(self):
        folder = self.gallery.create_folder("Vault")
        item = self.save("secret", folder)
        self.gallery.set_favorite(item["id"], True)
        listed = self.gallery.set_pin(folder["id"], PIN)
        self.assertEqual((listed["locked"], listed["unlocked"]), (True, True))  # set by this session
        self.assert_in_view(item)

        self.gallery.reset_privacy()
        [row] = self.gallery.list_folders()
        self.assertEqual((row["name"], row["locked"], row["unlocked"], row["count"]), ("Vault", True, False, None))
        self.assert_out_of_view(item, folder)
        with self.assertRaises(FolderLocked):
            self.gallery.list(folder=folder["id"])
        for change in (lambda: self.gallery.rename_folder(folder["id"], "Open"),
                       lambda: self.gallery.delete_folder(folder["id"])):
            with self.assertRaises(FolderLocked):
                change()

        with self.assertRaises(FolderLocked):
            self.gallery.unlock(folder["id"], "135790")
        self.assertTrue(self.gallery.unlock(folder["id"], PIN)["unlocked"])
        self.assert_in_view(item)
        self.assertEqual(self.gallery.list_folders()[0]["count"], 1)

        self.assertFalse(self.gallery.lock_again(folder["id"])["unlocked"])
        self.assert_out_of_view(item, folder)

    def test_a_locked_folder_takes_new_items_without_its_pin(self):
        folder = self.gallery.create_folder("Vault")
        self.gallery.set_pin(folder["id"], PIN)
        self.gallery.reset_privacy()
        item = self.save()
        self.gallery.set_folder(item["id"], folder["id"])
        self.assertIsNone(self.gallery.get(item["id"]))

    def test_changing_or_removing_a_pin_needs_the_current_one_even_unlocked(self):
        folder = self.gallery.create_folder("Vault")
        self.gallery.set_pin(folder["id"], PIN)
        for current in (None, "000000"):
            with self.subTest(current=current), self.assertRaises(FolderLocked):
                self.gallery.set_pin(folder["id"], "111111", current)
        self.gallery.set_pin(folder["id"], "111111", PIN)
        self.gallery.reset_privacy()
        with self.assertRaises(FolderLocked):
            self.gallery.unlock(folder["id"], PIN)
        self.gallery.unlock(folder["id"], "111111")
        removed = self.gallery.set_pin(folder["id"], None, "111111")
        self.assertEqual((removed["locked"], removed["unlocked"]), (False, False))
        self.gallery.reset_privacy()
        self.assertEqual(self.gallery.blocked_folders(), [])

    def test_pins_are_six_digits(self):
        folder = self.gallery.create_folder("Vault")
        for bad in ("12345", "1234567", "12345a", "123456\n", " 123456", "١٢٣٤٥٦", 123456, ""):
            with self.subTest(bad=bad), self.assertRaises(FolderError):
                self.gallery.set_pin(folder["id"], bad)
        self.gallery.set_pin(folder["id"], "000000")
        with self.assertRaises(FolderError):
            self.gallery.unlock(self.gallery.create_folder("Plain")["id"], "000000")  # no PIN to unlock

    def test_the_pin_is_stored_as_a_salted_hash_only(self):
        folder = self.gallery.create_folder("Vault")
        self.gallery.set_pin(folder["id"], PIN)
        other = self.gallery.create_folder("Vault 2")
        self.gallery.set_pin(other["id"], PIN)
        db = sqlite3.connect(Path(self.tmp.name) / "gallery.db")
        stored = [row[0] for row in db.execute("SELECT pin FROM folders ORDER BY name")]
        db.close()
        for value in stored:
            self.assertTrue(value.startswith("scrypt$"))
            self.assertNotIn(PIN, value)
        self.assertNotEqual(stored[0], stored[1])  # salted

    def test_wrong_pins_make_the_next_try_wait(self):
        folder = self.gallery.create_folder("Vault")
        self.gallery.set_pin(folder["id"], PIN)
        self.gallery.reset_privacy()
        now = [1000.0]
        with mock.patch.object(gallery_module.time, "monotonic", lambda: now[0]):
            for _ in range(gallery_module.PIN_FREE_TRIES):
                with self.assertRaises(FolderLocked):
                    self.gallery.unlock(folder["id"], "999999")
            with self.assertRaises(TooManyTries) as waiting:
                self.gallery.unlock(folder["id"], PIN)  # even the right one, while waiting
            self.assertAlmostEqual(waiting.exception.wait, gallery_module.PIN_FIRST_WAIT)
            now[0] += gallery_module.PIN_FIRST_WAIT + 1
            with self.assertRaises(FolderLocked):
                self.gallery.unlock(folder["id"], "999999")
            with self.assertRaises(TooManyTries) as longer:
                self.gallery.unlock(folder["id"], PIN)
            self.assertAlmostEqual(longer.exception.wait, 2 * gallery_module.PIN_FIRST_WAIT)
            now[0] += gallery_module.PIN_MAX_WAIT
            self.assertTrue(self.gallery.unlock(folder["id"], PIN)["unlocked"])
            self.gallery.lock_again(folder["id"])
            with self.assertRaises(FolderLocked):  # the count started over
                self.gallery.unlock(folder["id"], "999999")
            self.assertTrue(self.gallery.unlock(folder["id"], PIN)["unlocked"])

    def test_hidden_and_locked_together(self):
        folder = self.gallery.create_folder("Both")
        item = self.save("secret", folder)
        self.gallery.set_pin(folder["id"], PIN)
        self.gallery.set_hidden(folder["id"], True)
        self.gallery.reset_privacy()
        self.assert_out_of_view(item, folder)
        self.assertIsNone(self.gallery.unlock(folder["id"], PIN))  # cannot be found while hidden
        self.gallery.set_show_hidden(True)
        self.assert_out_of_view(item, folder)  # shown, still locked
        self.gallery.unlock(folder["id"], PIN)
        self.assert_in_view(item)

    def test_a_gallery_from_before_privacy_opens_with_nothing_hidden_or_locked(self):
        folder = self.gallery.create_folder("Old")
        item = self.save(folder=folder)
        self.gallery.close()
        db = sqlite3.connect(Path(self.tmp.name) / "gallery.db")
        db.execute("ALTER TABLE folders DROP COLUMN hidden")
        db.execute("ALTER TABLE folders DROP COLUMN pin")
        db.commit()
        db.close()
        self.gallery = Gallery(Path(self.tmp.name))
        [row] = self.gallery.list_folders()
        self.assertEqual((row["hidden"], row["locked"], row["count"]), (False, False, 1))
        self.assert_in_view(item)


class PrivacyServerTests(unittest.TestCase):
    request = server_tests.ServerTests.request

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        backend = FakeBackend()
        self.events = EventHub()
        self.gallery = Gallery(root / "outputs")
        self.app = App(settings=SettingsStore(root / "data"), backend=backend, gallery=self.gallery,
                       events=self.events, jobs=JobQueue(backend, self.gallery, self.events),
                       port=0, token=server_tests.TOKEN)
        self.httpd = serve(self.app, "127.0.0.1")
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.app.stopping.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.gallery.close()
        self.tmp.cleanup()

    def item_in(self, folder):
        record = self.gallery.save(Image.new("RGB", (8, 8)), job_id="j" * 16, model="m", request=REQUEST,
                                   seed=1, duration_ms=1)
        self.gallery.set_folder(record["id"], folder["id"])
        return record

    def test_lock_and_unlock_over_http(self):
        _, folder, _ = self.request("POST", "/api/folders", {"name": "Vault"})
        item = self.item_in(folder)
        status, listed, _ = self.request("PUT", f"/api/folders/{folder['id']}/pin", {"pin": PIN})
        self.assertEqual((status, listed["locked"]), (200, True))
        self.assertEqual(self.request("POST", "/api/gallery/privacy/reset", {})[0], 200)

        self.assertEqual(self.request("GET", f"/api/gallery/{item['id']}")[0], 404)
        self.assertEqual(self.request("GET", f"/images/{item['id']}.png")[0], 404)
        self.assertEqual(self.request("GET", f"/api/gallery?folder={folder['id']}")[0], 423)
        self.assertEqual(self.request("DELETE", f"/api/folders/{folder['id']}")[0], 423)
        self.assertEqual(self.request("PATCH", f"/api/folders/{folder['id']}", {"name": "x"})[0], 423)
        self.assertEqual(self.request("POST", f"/api/folders/{folder['id']}/unlock", {"pin": "111111"})[0], 423)
        self.assertEqual(self.request("POST", f"/api/folders/{folder['id']}/unlock", {"pin": "12"})[0], 400)
        self.assertEqual(self.request("PUT", f"/api/folders/{folder['id']}/pin", {"pin": None})[0], 423)

        status, opened, _ = self.request("POST", f"/api/folders/{folder['id']}/unlock", {"pin": PIN})
        self.assertEqual((status, opened["unlocked"], opened["count"]), (200, True, 1))
        self.assertEqual(self.request("GET", f"/images/{item['id']}.png")[0], 200)
        self.assertEqual(self.request("POST", f"/api/folders/{folder['id']}/lock", {})[0], 200)
        self.assertEqual(self.request("GET", f"/api/gallery/{item['id']}")[0], 404)

    def test_too_many_wrong_pins_answer_429(self):
        _, folder, _ = self.request("POST", "/api/folders", {"name": "Vault"})
        self.request("PUT", f"/api/folders/{folder['id']}/pin", {"pin": PIN})
        self.request("POST", "/api/gallery/privacy/reset", {})
        for _ in range(5):
            self.request("POST", f"/api/folders/{folder['id']}/unlock", {"pin": "000000"})
        status, body, _ = self.request("POST", f"/api/folders/{folder['id']}/unlock", {"pin": PIN})
        self.assertEqual(status, 429, body)

    def test_hide_and_show_over_http(self):
        _, folder, _ = self.request("POST", "/api/folders", {"name": "Hidden"})
        item = self.item_in(folder)
        status, listed, _ = self.request("PATCH", f"/api/folders/{folder['id']}", {"hidden": True})
        self.assertEqual((status, listed["hidden"]), (200, True))
        _, listing, _ = self.request("GET", "/api/folders")
        self.assertEqual((listing["items"], listing["show_hidden"]), ([], False))
        self.assertEqual(self.request("GET", f"/api/gallery?folder={folder['id']}")[0], 404)
        _, page, _ = self.request("GET", "/api/gallery")
        self.assertNotIn(item["id"], [i["id"] for i in page["items"]])

        status, privacy, _ = self.request("PUT", "/api/gallery/privacy", {"show_hidden": True})
        self.assertEqual((status, privacy["show_hidden"]), (200, True))
        _, listing, _ = self.request("GET", "/api/folders")
        self.assertEqual([f["id"] for f in listing["items"]], [folder["id"]])
        self.assertEqual(self.request("GET", f"/api/gallery/{item['id']}")[0], 200)

        self.request("POST", "/api/gallery/privacy/reset", {})
        self.assertEqual(self.request("GET", f"/api/gallery/{item['id']}")[0], 404)
        for body in ({"show_hidden": "yes"}, {}, {"show_hidden": True, "x": 1}):
            with self.subTest(body=body):
                self.assertEqual(self.request("PUT", "/api/gallery/privacy", body)[0], 400)


if __name__ == "__main__":
    unittest.main()
