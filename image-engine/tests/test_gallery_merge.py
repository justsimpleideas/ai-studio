"""Merging other galleries into this one (gallery_merge.py): nothing lost, nothing overwritten."""

import contextlib
import hashlib
import http.client
import json
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from image_engine import gallery_merge
from image_engine.config import SettingsStore
from image_engine.events import EventHub
from image_engine.gallery import Gallery
from image_engine.gallery_merge import MergeError, describe, open_source, plan_merge, run_merge
from image_engine.jobs import JobQueue
from image_engine.server import App, serve

from .fakes import FakeBackend

REQUEST = {"prompt": "a still", "negative_prompt": "blurry", "width": 8, "height": 8, "steps": 8,
           "guidance": 1.5, "seed": 1, "count": 1}
VIDEO_REQUEST = {"kind": "video", "prompt": "waves", "negative_prompt": "", "width": 16, "height": 16,
                 "frames": 5, "fps": 24, "steps": 4, "guidance": 5.0, "seed": 7, "count": 1, "start_image": None}
PIN = "246810"
TOKEN = "t" * 40


@contextlib.contextmanager
def closing_db(path: Path):
    """Change a gallery database behind the gallery's back. sqlite3's own "with" commits but
    does not close, and Windows keeps an open database file locked."""
    db = sqlite3.connect(path)
    try:
        yield db
        db.commit()
    finally:
        db.close()


def snapshot(root: Path) -> dict[str, str]:
    """Every file under `root` and its hash: to show a folder was not changed."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


class MergeTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.here = Gallery(self.base / "outputs")
        self.others: list[Gallery] = []

    def tearDown(self):
        self.here.close()
        for g in self.others:
            g.close()
        self.tmp.cleanup()

    def gallery(self, name: str) -> Gallery:
        g = Gallery(self.base / name / "outputs")
        self.others.append(g)
        return g

    @staticmethod
    def save(g: Gallery, prompt="a still", colour=(10, 20, 30), seed=1, folder=None, favorite=False):
        record = g.save(Image.new("RGB", (8, 8), colour), job_id="j" * 16, model="m",
                        request={**REQUEST, "prompt": prompt, "seed": seed}, seed=seed, duration_ms=5,
                        stats={"peak": 1}, provenance={"app": "test", "model_revision": "abc"})
        if folder:
            g.set_folder(record["id"], folder["id"], everything=True)
        if favorite:
            g.set_favorite(record["id"], True)
        return record

    def merge(self, *galleries: Gallery | Path, renames=None, lora_ids=None):
        sources = [open_source(g if isinstance(g, Path) else g.root) for g in galleries]
        plan = plan_merge(self.here, sources, renames, lora_ids)
        return plan, run_merge(self.here, plan)

    def all_here(self):
        self.here.set_show_hidden(True)
        for f in self.here.list_folders(everything=True):
            if f["locked"]:
                self.here.unlock(f["id"], PIN)
        return self.here.list(limit=1000)["items"]


class MergeTests(MergeTestCase):
    def test_everything_comes_across_and_the_sources_are_untouched(self):
        a = self.gallery("pc1")
        trips = a.create_folder("Trips")
        vault = a.create_folder("Vault")
        a.set_pin(vault["id"], PIN)
        a.set_hidden(vault["id"], True)
        one = self.save(a, "a red boat", (200, 0, 0), seed=11, folder=trips, favorite=True)
        two = self.save(a, "a secret", (0, 200, 0), seed=12, folder=vault)
        start = a.save_input(Image.new("RGB", (16, 16), (1, 2, 3)))
        video = a.save_video([Image.new("RGB", (16, 16), (i * 30, 0, 0)) for i in range(5)], job_id="v" * 16,
                             model="fake-video", request={**VIDEO_REQUEST, "start_image": start}, seed=7,
                             duration_ms=9, ffmpeg=None)
        mine = self.save(self.here, "already here", (5, 5, 5))
        before = snapshot(a.root)

        plan, result = self.merge(a)

        self.assertEqual(snapshot(a.root), before)  # read-only: not one byte changed
        self.assertEqual((result["added"], result["images"], result["videos"], result["favorites"]), (3, 2, 1, 1))
        items = {i["id"]: i for i in self.all_here()}
        self.assertEqual(set(items), {one["id"], two["id"], video["id"], mine["id"]})
        boat = items[one["id"]]
        self.assertEqual((boat["prompt"], boat["negative_prompt"], boat["seed"], boat["guidance"]), ("a red boat", "blurry", 11, 1.5))
        self.assertEqual(boat["provenance"], {"app": "test", "model_revision": "abc"})
        self.assertEqual(boat["created"], one["created"])
        self.assertTrue(boat["favorite"])
        folders = {f["name"]: f for f in self.here.list_folders(everything=True)}
        self.assertEqual(boat["folder_id"], folders["Trips"]["id"])
        self.assertTrue(folders["Vault"]["hidden"] and folders["Vault"]["locked"])
        self.assertEqual(items[two["id"]]["folder_id"], folders["Vault"]["id"])
        # The same PIN opens it here; the files are the same bytes.
        self.assertIsNotNone(self.here.unlock(folders["Vault"]["id"], PIN))
        a.set_show_hidden(True)
        a.unlock(vault["id"], PIN)
        for item in (one, two, video):
            self.assertEqual(self.here.path_of(item["id"], "thumb").read_bytes(), a.path_of(item["id"], "thumb").read_bytes())
        self.assertEqual(self.here.path_of(video["id"], "video").read_bytes(), a.path_of(video["id"], "video").read_bytes())
        self.assertEqual(items[video["id"]]["start_image"], start)
        self.assertIsNotNone(self.here.load_input(start))
        # Embedded PNG settings travel with the file.
        with Image.open(self.here.path_of(one["id"], "image")) as png:
            self.assertEqual(json.loads(png.text["local-ai-studio"])["prompt"], "a red boat")

    def test_a_copy_of_this_gallery_adds_nothing_and_says_so(self):
        trips = self.here.create_folder("Trips")
        self.here.set_pin(trips["id"], PIN)
        item = self.save(self.here, folder=trips)
        self.save(self.here, "loose", (9, 9, 9))
        backup = self.base / "backup" / "outputs"
        shutil.copytree(self.here.root, backup)
        # Changed in the copy afterwards: its PIN is gone and the item is a favourite there.
        with closing_db(backup / "gallery.db") as db:
            db.execute("UPDATE folders SET pin = ''")
            db.execute("UPDATE images SET favorite = 1 WHERE id = ?", (item["id"],))
        before = snapshot(self.here.root)

        plan = plan_merge(self.here, [open_source(backup)])
        self.assertEqual((len(plan.items), plan.duplicates, len(plan.new_folders), plan.conflicts), (0, 2, 0, []))
        self.assertEqual(plan.differing, 1)
        self.assertEqual(plan.joined, [{"name": "Trips", "source": "backup", "settings_differ": True}])
        result = run_merge(self.here, plan)
        self.assertEqual((result["added"], result["duplicates"]), (0, 2))
        self.assertEqual(snapshot(self.here.root), before)  # nothing here changed: this gallery's settings stay
        self.assertTrue(self.here.list_folders()[0]["locked"])

    def test_the_same_id_with_a_different_file_keeps_both(self):
        mine = self.save(self.here, "here", (1, 1, 1))
        other = self.gallery("pc2")
        theirs = self.save(other, "there", (250, 250, 250))
        with closing_db(other.root / "gallery.db") as db:  # an id clash, however unlikely
            db.execute("UPDATE images SET id = ? WHERE id = ?", (mine["id"], theirs["id"]))
        plan, result = self.merge(other)
        self.assertEqual((result["added"], result["duplicates"]), (1, 0))
        items = self.all_here()
        self.assertEqual(sorted(i["prompt"] for i in items), ["here", "there"])
        self.assertEqual(len({i["id"] for i in items}), 2)
        self.assertEqual(self.here.path_of(mine["id"], "image").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_the_same_file_with_a_different_record_keeps_both(self):
        mine = self.save(self.here, "the prompt here", (1, 1, 1))
        backup = self.base / "backup" / "outputs"
        shutil.copytree(self.here.root, backup)
        with closing_db(backup / "gallery.db") as db:  # the same id and bytes, but another prompt on record
            db.execute("UPDATE images SET prompt = 'another prompt' WHERE id = ?", (mine["id"],))
        plan, result = self.merge(backup)
        self.assertEqual((result["added"], result["duplicates"]), (1, 0))
        self.assertEqual(sorted(i["prompt"] for i in self.all_here()), ["another prompt", "the prompt here"])

    def test_a_folder_is_the_same_only_with_the_same_id_and_exactly_the_same_name(self):
        mine = self.here.create_folder("Trips")
        self.save(self.here, folder=mine)
        backup = self.base / "backup" / "outputs"
        shutil.copytree(self.here.root, backup)
        # The same id and name, holding something else besides: one folder, both items in it.
        other = Gallery(backup)
        self.others.append(other)
        self.save(other, "only in the backup", (99, 0, 0), seed=9, folder=other.list_folders()[0])
        plan, result = self.merge(backup)
        self.assertEqual((result["added"], result["folders"], len(plan.joined)), (1, 0, 1))
        self.assertEqual(self.here.list_folders()[0]["count"], 2)
        # The same id, but the name differs in case: not the same folder, so a clash to resolve.
        with closing_db(backup / "gallery.db") as db:
            db.execute("UPDATE folders SET name = 'trips'")
        plan = plan_merge(self.here, [open_source(backup)])
        self.assertEqual((len(plan.joined), len(plan.conflicts)), (0, 1))

    def test_no_file_here_is_ever_replaced(self):
        mine = self.save(self.here, "here", (1, 1, 1))
        other = self.gallery("pc2")
        theirs = self.save(other, "there", (250, 250, 250))
        # Their file sits at the very path this gallery's own file has.
        mine_rel = self.here.root.joinpath(*Path(mine["file_name"]).parts)
        rel = next(self.here.images_dir.rglob(mine["file_name"])).relative_to(self.here.root).as_posix()
        target = other.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(other.path_of(theirs["id"], "image"), target)
        with closing_db(other.root / "gallery.db") as db:
            db.execute("UPDATE images SET file = ? WHERE id = ?", (rel, theirs["id"]))
        before = (self.here.root / rel).read_bytes()
        self.merge(other)
        self.assertEqual((self.here.root / rel).read_bytes(), before)
        moved = self.here.path_of(theirs["id"], "image")
        self.assertNotEqual(moved, self.here.root / rel)
        self.assertEqual(moved.read_bytes(), target.read_bytes())
        del mine_rel

    def test_two_sources_sharing_items_bring_them_once(self):
        a = self.gallery("pc1")
        shared = self.save(a, "shared", (7, 7, 7), favorite=True)
        b_root = self.base / "pc1-backup" / "outputs"
        shutil.copytree(a.root, b_root)
        plan, result = self.merge(a, b_root)
        self.assertEqual((result["added"], result["duplicates"]), (1, 1))
        self.assertEqual([i["id"] for i in self.all_here()], [shared["id"]])


class FolderNameTests(MergeTestCase):
    def test_names_that_differ_only_in_case_clash_and_suggestions_resolve_them(self):
        self.here.create_folder("Trips")
        a, b = self.gallery("pc1"), self.gallery("pc2")
        self.save(a, folder=a.create_folder("trips"))
        self.save(b, "b", (3, 3, 3), folder=b.create_folder("TRIPS"))
        sources = [open_source(a.root), open_source(b.root)]
        sources[0].label, sources[1].label = "pc1", "pc2"
        plan = plan_merge(self.here, sources)
        self.assertEqual(len(plan.conflicts), 1)
        entries = plan.conflicts[0]["entries"]
        self.assertEqual([e["where"] for e in entries], ["This gallery", "pc1", "pc2"])
        self.assertEqual([e["suggested"] for e in entries], ["Trips", "trips (pc1)", "TRIPS (pc2)"])
        with self.assertRaises(MergeError):
            run_merge(self.here, plan)  # never two folders of one name
        renames = {e["key"]: e["suggested"] for e in entries}
        plan = plan_merge(self.here, sources, renames)
        self.assertEqual(plan.conflicts, [])
        run_merge(self.here, plan)
        self.assertEqual(sorted((f["name"] for f in self.here.list_folders()), key=str.casefold),
                         ["Trips", "trips (pc1)", "TRIPS (pc2)"])

    def test_this_gallerys_folder_can_be_the_one_renamed(self):
        mine = self.here.create_folder("Trips")
        a = self.gallery("pc1")
        self.save(a, folder=a.create_folder("Trips"))
        plan = plan_merge(self.here, [open_source(a.root)], {f"target:{mine['id']}": "Old trips"})
        self.assertEqual(plan.conflicts, [])
        run_merge(self.here, plan)
        self.assertEqual(sorted(f["name"] for f in self.here.list_folders()), ["Old trips", "Trips"])

    def test_suggestions_stay_within_the_name_limit(self):
        long = "x" * 60
        self.here.create_folder(long)
        a = self.gallery("pc1")
        self.save(a, folder=a.create_folder(long))
        entries = plan_merge(self.here, [open_source(a.root)]).conflicts[0]["entries"]
        self.assertLessEqual(len(entries[1]["suggested"]), 60)
        self.assertTrue(entries[1]["suggested"].endswith("(pc1)"))

    def test_a_hidden_folder_is_not_told_by_a_clash_unless_hidden_folders_are_shown(self):
        secret = self.here.create_folder("Secret")
        self.here.set_hidden(secret["id"], True)
        a = self.gallery("pc1")
        self.save(a, folder=a.create_folder("Secret"))
        source = open_source(a.root)
        self.assertEqual(plan_merge(self.here, [source]).conflicts, [])
        self.here.set_show_hidden(True)
        self.assertEqual(len(plan_merge(self.here, [source]).conflicts), 1)

    def test_hidden_folder_names_from_a_source_are_not_shown_while_hidden_folders_are_hidden(self):
        a = self.gallery("pc1")
        secret = a.create_folder("Diary")
        a.set_hidden(secret["id"], True)
        self.save(a, folder=secret)
        plan = plan_merge(self.here, [open_source(a.root)])
        shown = describe(plan, show_hidden=False)
        self.assertEqual(shown["sources"][0]["folders"], [{"name": None, "hidden": True, "locked": False}])
        self.assertNotIn("Diary", json.dumps(shown))
        self.assertIn("Diary", json.dumps(describe(plan, show_hidden=True)))


class UntrustedSourceTests(MergeTestCase):
    def test_rows_that_point_outside_their_gallery_or_at_nothing_are_left_out_and_counted(self):
        (self.base / "secret.png").write_bytes(b"\x89PNG not yours")
        a = self.gallery("pc1")
        good = self.save(a, "good")
        bad = [self.save(a, f"bad{i}", (i, i, i), seed=20 + i) for i in range(4)]
        with closing_db(a.root / "gallery.db") as db:
            db.execute("UPDATE images SET file = '../../secret.png' WHERE id = ?", (bad[0]["id"],))
            db.execute("UPDATE images SET file = ? WHERE id = ?", (str(self.base / "secret.png"), bad[1]["id"]))
            db.execute("UPDATE images SET file = 'images/2020-01-01/missing.png' WHERE id = ?", (bad[2]["id"],))
            db.execute("UPDATE images SET width = 'wide' WHERE id = ?", (bad[3]["id"],))
        source = open_source(a.root)
        self.assertEqual(([i["id"] for i in source.items], source.unreadable), ([good["id"]], 4))

    def test_a_missing_thumbnail_is_made_again_rather_than_the_item_left_out(self):
        a = self.gallery("pc1")
        item = self.save(a)
        a.path_of(item["id"], "thumb").unlink()
        plan, result = self.merge(a)
        self.assertEqual((result["added"], result["new_thumbs"]), (1, 1))
        with Image.open(self.here.path_of(item["id"], "thumb")) as thumb:
            self.assertEqual(thumb.format, "WEBP")

    def test_an_older_gallery_without_later_columns_merges_with_their_defaults(self):
        old = self.base / "old" / "outputs"
        (old / "images" / "2024-01-01").mkdir(parents=True)
        (old / "thumbs").mkdir()
        Image.new("RGB", (8, 8), (9, 8, 7)).save(old / "images" / "2024-01-01" / "first.png")
        Image.new("RGB", (8, 8)).save(old / "thumbs" / "0123456789abcdef.webp", format="WEBP")
        with closing_db(old / "gallery.db") as db:  # the first release's schema: no folders, no video
            db.executescript(gallery_merge.Gallery.__init__.__globals__["SCHEMA"].split("CREATE TABLE IF NOT EXISTS folders")[0])
            db.execute("INSERT INTO images (id, created, file, thumb, job_id, model, prompt, width, height, steps,"
                       " guidance, seed, duration_ms, favorite) VALUES ('0123456789abcdef', 1700000000.0,"
                       " 'images/2024-01-01/first.png', 'thumbs/0123456789abcdef.webp', 'job', 'm', 'old times',"
                       " 8, 8, 8, 0.0, 3, 10, 1)")
        source = open_source(old)
        self.assertEqual((len(source.items), source.unreadable, source.folders), (1, 0, []))
        plan, result = self.merge(old)
        item = self.all_here()[0]
        self.assertEqual((item["prompt"], item["media"], item["mime"], item["favorite"], item["folder_id"]),
                         ("old times", "image", "image/png", True, None))

    def test_columns_from_a_newer_version_are_named_in_the_report(self):
        a = self.gallery("pc1")
        self.save(a)
        with closing_db(a.root / "gallery.db") as db:
            db.execute("ALTER TABLE images ADD COLUMN rating INTEGER NOT NULL DEFAULT 0")
            db.execute("ALTER TABLE folders ADD COLUMN colour TEXT NOT NULL DEFAULT ''")
        source = open_source(a.root)
        self.assertEqual(source.unknown_columns, ["folders.colour", "rating"])
        self.assertEqual(describe(plan_merge(self.here, [source]), False)["sources"][0]["unknown_columns"],
                         ["folders.colour", "rating"])

    def test_a_pin_this_version_cannot_check_keeps_its_folder_locked(self):
        a = self.gallery("pc1")
        folder = a.create_folder("Future")
        self.save(a, folder=folder)
        with closing_db(a.root / "gallery.db") as db:
            db.execute("UPDATE folders SET pin = 'argon2id$v=19$future'")
        self.merge(a)
        mine = self.here.list_folders()[0]
        self.assertTrue(mine["locked"] and not mine["unlocked"])
        self.assertEqual(self.here.list()["total"], 0)  # still out of view

    def test_a_folder_that_is_not_a_gallery_is_refused(self):
        (self.base / "photos").mkdir()
        with self.assertRaises(MergeError):
            open_source(self.base / "photos")


class SafetyTests(MergeTestCase):
    def test_a_failure_part_way_leaves_this_gallery_as_it_was(self):
        self.save(self.here, "mine")
        a = self.gallery("pc1")
        for i in range(4):
            self.save(a, f"a{i}", (i * 40, 0, 0), seed=i)
        a.create_folder("Trips")
        before = snapshot(self.here.root)
        plan = plan_merge(self.here, [open_source(a.root)])
        real = gallery_merge._copy
        calls = {"n": 0}

        def flaky(src, dst, cancelled):
            calls["n"] += 1
            if calls["n"] == 5:
                raise OSError("disk went away")
            return real(src, dst, cancelled)

        with mock.patch.object(gallery_merge, "_copy", flaky):
            with self.assertRaises(OSError):
                run_merge(self.here, plan)
        self.assertEqual(snapshot(self.here.root), before)
        self.assertEqual(self.here.list()["total"], 1)
        self.assertEqual(self.here.list_folders(), [])

    def test_cancelling_removes_what_was_copied_and_adds_nothing(self):
        self.save(self.here, "mine")
        a = self.gallery("pc1")
        for i in range(3):
            self.save(a, f"a{i}", (i * 40, 0, 0), seed=i)
        before = snapshot(self.here.root)
        plan = plan_merge(self.here, [open_source(a.root)])
        cancelled = threading.Event()

        def after_two(done, total):
            if done == 2:
                cancelled.set()

        with self.assertRaises(MergeError) as caught:
            run_merge(self.here, plan, after_two, cancelled)
        self.assertIn("cancelled", str(caught.exception))
        self.assertEqual(snapshot(self.here.root), before)
        self.assertEqual(self.here.list()["total"], 1)

    def test_hashes_are_remembered_until_a_file_changes(self):
        a = self.gallery("pc1")
        item = self.save(a, "hashed")
        path = a.path_of(item["id"], "image")
        gallery_merge._HASHES.clear()
        first = gallery_merge._sha256(path)
        with mock.patch.object(gallery_merge.hashlib, "sha256", side_effect=AssertionError("read again")):
            self.assertEqual(gallery_merge._sha256(path), first)  # remembered: not read again
        path.write_bytes(path.read_bytes() + b"\x00")
        self.assertNotEqual(gallery_merge._sha256(path), first)  # changed: read again

    def test_not_enough_space_stops_it_before_anything_is_copied(self):
        a = self.gallery("pc1")
        self.save(a)
        usage = shutil.disk_usage(self.base)._replace(free=10)
        with mock.patch.object(gallery_merge.shutil, "disk_usage", return_value=usage):
            plan = plan_merge(self.here, [open_source(a.root)])
        self.assertTrue(any("space" in p for p in plan.problems))
        with self.assertRaises(MergeError):
            run_merge(self.here, plan)
        self.assertEqual(self.here.list()["total"], 0)

    def test_items_made_with_a_lora_this_studio_lacks_are_counted(self):
        a = self.gallery("pc1")
        record = a.save(Image.new("RGB", (8, 8)), job_id="j" * 16, model="m", request=REQUEST, seed=1,
                        duration_ms=1, provenance={"loras": [{"id": "abcdef0123456789", "strength": 0.8}]})
        plan = plan_merge(self.here, [open_source(a.root)], lora_ids=set())
        self.assertEqual(plan.missing_loras, 1)
        self.assertEqual(plan_merge(self.here, [open_source(a.root)], lora_ids={"abcdef0123456789"}).missing_loras, 0)
        del record


class MergeServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.events = EventHub()
        self.gallery = Gallery(root / "outputs")
        self.backend = FakeBackend()
        self.jobs = JobQueue(self.backend, self.gallery, self.events)
        self.app = App(settings=SettingsStore(root / "data"), backend=self.backend, gallery=self.gallery,
                       events=self.events, jobs=self.jobs, port=0, token=TOKEN)
        self.httpd = serve(self.app, "127.0.0.1")
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.other = Gallery(root / "pc1" / "outputs")
        self.other.save(Image.new("RGB", (8, 8)), job_id="j" * 16, model="m", request=REQUEST, seed=1, duration_ms=1)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.other.close()
        self.gallery.close()
        self.tmp.cleanup()

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers={"Host": f"127.0.0.1:{self.port}", "X-Studio-Token": TOKEN,
                                                       "Content-Type": "application/json"})
        response = conn.getresponse()
        payload = json.loads(response.read() or b"null")
        conn.close()
        return response.status, payload

    def test_inspect_then_merge(self):
        status, plan = self.request("POST", "/api/gallery/merge/inspect", {"sources": [str(self.other.root)]})
        self.assertEqual((status, plan["new_items"], plan["problems"]), (200, 1, []))
        status, started = self.request("POST", "/api/gallery/merge", {"sources": [str(self.other.root)]})
        self.assertEqual((status, started["started"]), (200, True))
        deadline = time.time() + 10
        while self.request("GET", "/api/gallery/merge")[1]["state"] != "done" and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.request("GET", "/api/gallery/merge")[1]["result"]["added"], 1)
        self.assertEqual(self.gallery.list()["total"], 1)

    def test_a_merge_can_be_cancelled_from_the_api(self):
        status, payload = self.request("POST", "/api/gallery/merge/cancel", {})
        self.assertEqual((status, payload), (200, {"cancelling": False}))  # nothing running: nothing to stop

    def test_this_gallery_and_relative_or_repeated_paths_are_refused(self):
        for sources in ([str(self.gallery.root)], [str(self.gallery.root.parent)], ["pc1/outputs"],
                        [str(self.other.root), str(self.other.root)], []):
            status, _ = self.request("POST", "/api/gallery/merge/inspect", {"sources": sources})
            self.assertEqual(status, 400, sources)


if __name__ == "__main__":
    unittest.main()
