"""scripts/collect_photos.py against a local stand-in for Commons and Openverse (no internet)."""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from image_engine.training.dataset import scan
from image_engine.training.subject import CARS, parse_subject

from .fake_photo_sources import FakeSources, _page

_spec = importlib.util.spec_from_file_location(
    "collect_photos", Path(__file__).resolve().parents[1] / "scripts" / "collect_photos.py")
cp = importlib.util.module_from_spec(_spec)
sys.modules["collect_photos"] = cp
_spec.loader.exec_module(cp)
cp.REQUEST_GAP = 0.0

CAR_LEVELS = [{"key": lv.key, "required": lv.required} for lv in CARS.levels]
COUPE = {"values": ["Example Motors", "Coupe GT", "Mk2", "2019-2023"], "category": "Example Coupe"}
TWO = [{"key": "brand", "required": True}, {"key": "model", "required": False}]


class LicenceTests(unittest.TestCase):
    def test_cc0_public_domain_cc_by_and_share_alike_are_recognised(self):
        accepted = {("cc0", ""): "cc0", ("cc-by-4.0", ""): "cc-by", ("cc-by-2.0", ""): "cc-by",
                    ("pd", ""): "pd", ("pd-old-70", ""): "pd", ("", "Public domain"): "pd", ("", "CC BY 3.0"): "cc-by",
                    ("by", ""): "cc-by", ("pdm", ""): "pd", ("", "CC0"): "cc0"}
        for (code, short), want in accepted.items():
            self.assertEqual(cp.normalise_license(code, short), want, (code, short))
        for code in ("cc-by-sa-4.0", "by-sa", "cc-by-sa-2.0", "CC BY-SA 3.0"):
            self.assertEqual(cp.normalise_license(code), "cc-by-sa", code)
        for code in ("cc-by-nc-2.0", "cc-by-nd-4.0", "cc-by-nc-sa-4.0", "by-nc", "gfdl", "fair use", "", "attribution"):
            self.assertIsNone(cp.normalise_license(code), code)

    def test_share_alike_only_when_the_plan_asks(self):
        plan = cp.parse_plan({"levels": TWO, "targets": [{"values": ["A", "B"]}]})
        self.assertNotIn("cc-by-sa", plan.licences)
        plan = cp.parse_plan({"levels": TWO, "targets": [{"values": ["A", "B"]}], "share_alike": True})
        self.assertIn("cc-by-sa", plan.licences)
        with self.assertRaises(cp.PlanError):
            cp.parse_plan({"levels": TWO, "targets": [{"values": ["A", "B"]}], "share_alike": "yes"})

    def test_the_user_agent_names_the_app_and_a_contact(self):
        self.assertEqual(cp.Http().user_agent,
                         "LocalAIStudio-collect/1.0 (https://github.com/justsimpleideas/ai-studio; "
                         "dataset tool for personal training)")
        self.assertIn("(you@example.com; ", cp.Http("you@example.com").user_agent)
        plan = cp.parse_plan({"levels": TWO, "targets": [{"values": ["A", "B"]}], "contact": " you@example.com "})
        self.assertEqual(plan.contact, "you@example.com")
        for bad in ("a\r\nX-Evil: 1", "x" * 101, "a;b", "a(b)", 5):
            with self.assertRaises(cp.PlanError, msg=repr(bad)):
                cp.parse_plan({"levels": TWO, "targets": [{"values": ["A", "B"]}], "contact": bad})

    def test_html_is_stripped_from_authors(self):
        self.assertEqual(cp.strip_html('<a href="x">Jane &amp; Co</a> <span>(talk)</span>'), "Jane & Co (talk)")


class NameTests(unittest.TestCase):
    def test_folder_names_are_safe(self):
        self.assertEqual(cp.folder_name("Example Motors"), "Example_Motors")
        self.assertEqual(cp.folder_name("F-150"), "F-150")
        self.assertEqual(cp.folder_name("../../etc"), "etc")
        self.assertEqual(cp.folder_name("CON"), "_CON")
        self.assertEqual(cp.folder_name("a/b\\c:d*e"), "abcde")
        self.assertEqual(cp.folder_name("..."), "unnamed")

    def test_file_names_come_from_titles_but_cannot_escape(self):
        self.assertEqual(cp.file_stem("File:My Car (front).JPG", "101"), "my-car-front-101")
        self.assertEqual(cp.file_stem("File:../../evil.jpg", "../x"), "evil-x")
        self.assertEqual(cp.file_stem("File:車.jpg", "7"), "photo-7")


class SpreadTests(unittest.TestCase):
    def test_interleave_takes_turns_and_lets_skip_reasons_through(self):
        a = iter(["licence", 11, 12, 13])  # numbers stand in for candidates; text is a skip reason
        b = iter([21])
        c = iter(["too small"])
        self.assertEqual(list(cp.interleave([a, b, c])), ["licence", 11, 21, "too small", 12, 13])

    def test_names_that_hold_copies_or_pictures_of_the_subject_are_skipped(self):
        skip = ["Porsche 911 model cars", "Die-cast Porsche 911", "Lego Porsche 911", "Porsche 911 in art",
                "Ford Mustang in popular culture", "Seiko advertisements", "Corolla badges", "Scale models of ships"]
        keep = ["Porsche 911 (996)", "Porsche 911 interiors", "Red Porsche 911", "Ford Mustang (1994)",
                "Toyota Corolla police cars", "Shiba Inu puppies"]
        for name in skip:
            self.assertTrue(cp.SKIP_CATEGORY.search(name), name)
        for name in keep:
            self.assertFalse(cp.SKIP_CATEGORY.search(name), name)

    def test_retry_once_after_a_timeout(self):
        http = cp.Http()
        calls = []

        def flaky(url, limit):
            calls.append(url)
            if len(calls) == 1:
                raise TimeoutError("timed out")
            return b"ok", "text/plain"

        with mock.patch.object(http, "_get", flaky), mock.patch.object(cp.time, "sleep"):
            self.assertEqual(http.get("https://example.org/x", 10), (b"ok", "text/plain"))
            self.assertEqual(len(calls), 2)

        def down(url, limit):
            raise TimeoutError("timed out")

        with mock.patch.object(http, "_get", down), mock.patch.object(cp.time, "sleep"),                 self.assertRaises(TimeoutError):
            http.get("https://example.org/x", 10)


class PlanTests(unittest.TestCase):
    def test_valid_plan_and_defaults(self):
        plan = cp.parse_plan({"levels": TWO, "targets": [{"values": ["A", "B"], "category": "X"}]})
        self.assertEqual(plan.targets[0].values, ("A", "B"))
        self.assertEqual(plan.targets[0].count, 200)
        self.assertEqual(plan.targets[0].category, "Category:X")
        self.assertEqual(plan.max_total_bytes, 2 * 1024**3)
        self.assertEqual(plan.sources, ["commons", "openverse"])

    def test_bad_plans(self):
        ok = [{"values": ["A", "B"]}]
        bad = [None, {}, {"targets": ok}, {"levels": TWO, "targets": []},
               {"levels": TWO, "targets": [{"values": ["", "B"]}]},  # a required level left out
               {"levels": TWO, "targets": [{"values": ["A"]}]},  # one name short
               {"levels": TWO, "targets": ok * 2},
               {"levels": TWO, "targets": [{"values": ["A", "B"], "count": 0}]},
               {"levels": TWO, "targets": ok, "sources": ["google"]},
               {"levels": TWO, "targets": ok, "max_total_bytes": 1},
               {"levels": TWO, "targets": [{"values": ["A" * 61, "B"]}]},
               {"levels": TWO, "targets": [{"values": ["A", f"M{i}"]} for i in range(11)]},
               {"levels": [{"key": "brand", "required": False}], "targets": [{"values": ["A"]}]},
               {"levels": [{"key": "../x", "required": True}], "targets": [{"values": ["A"]}]},
               {"levels": [{"key": "a", "required": True}] * 2, "targets": [{"values": ["A", "B"]}]},
               {"levels": [{"key": f"l{i}", "required": True} for i in range(5)], "targets": [{"values": ["A"] * 5}]}]
        for plan in bad:
            with self.assertRaises(cp.PlanError, msg=plan):
                cp.parse_plan(plan)


class CollectTests(unittest.TestCase):
    def setUp(self):
        cp._json_events = False  # main() turns it on; other tests should stay quiet
        self.addCleanup(setattr, cp, "_json_events", False)
        self.fake = FakeSources().start()
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "photos"

    def tearDown(self):
        self.fake.stop()
        self.tmp.cleanup()

    def run_plan(self, raw):
        raw = {"levels": CAR_LEVELS, **raw}
        return cp.collect(cp.parse_plan(raw), raw, self.dest, commons_api=self.fake.commons_api,
                          openverse_api=self.fake.openverse_api)

    def test_keeps_only_allowed_photos_and_credits_them(self):
        manifest = self.run_plan({"targets": [{**COUPE, "count": 10}], "min_side": 512})
        titles = sorted(i.title for i in manifest.items)
        # The category's own files, then its subcategories' (two levels); the logos subcategory is left out.
        self.assertEqual(titles, ["Coupe at a show", "coupe-front.jpg", "coupe-mk2-red.jpg", "coupe-mk2.jpg",
                                  "coupe-rear.jpg", "coupe-side.jpg"])
        self.assertEqual({i.license for i in manifest.items}, {"cc-by", "cc0", "pd"})
        self.assertEqual(manifest.skipped, {"licence": 2, "personality rights": 1, "not a photo": 1,
                                            "already from Commons": 1})
        folder = self.dest / "Example_Motors" / "Coupe_GT" / "Mk2" / "2019-2023"
        self.assertEqual(len(list(folder.glob("*.jpg"))), 6)
        self.assertEqual(json.loads((folder / "labels.json").read_text()),
                         {"make": "Example Motors", "model": "Coupe GT", "generation": "Mk2", "years": "2019-2023"})
        credits = json.loads((self.dest / "ATTRIBUTION.json").read_text())
        front = next(c for c in credits if c["title"] == "coupe-front.jpg")
        self.assertEqual((front["author"], front["license_text"]), ("Photographer 101", "CC BY 4.0"))
        self.assertTrue(front["page"].endswith("/wiki/File:coupe-front.jpg"))
        md = (self.dest / "CREDITS.md").read_text()
        self.assertIn("| Photographer 101 | CC BY 4.0 |", md)

    def test_the_folder_is_ready_for_the_train_tab(self):
        self.run_plan({"targets": [{**COUPE, "count": 10}]})
        dataset = scan(self.dest, CARS)
        usable = [i for i in dataset.items if i.usable]
        self.assertEqual(len(usable), 6)
        self.assertEqual(usable[0].facts, {"make": "Example Motors", "model": "Coupe GT", "generation": "Mk2",
                                           "years": "2019-2023"})
        self.assertEqual(dataset.warnings, [])

    def test_a_small_quota_is_spread_over_subcategories(self):
        manifest = self.run_plan({"targets": [{**COUPE, "count": 3}], "sources": ["commons"]})
        # One from the category's own files, one from the subcategory branch, then the category again.
        self.assertEqual([i.title for i in manifest.items], ["coupe-front.jpg", "coupe-mk2.jpg", "coupe-side.jpg"])

    def test_a_walked_tree_is_reused_for_a_day(self):
        cache = cp.TreeCache(Path(self.tmp.name) / "cache")
        plan = cp.parse_plan({"levels": CAR_LEVELS, "targets": [COUPE], "sources": ["commons"]})
        walks = lambda: sum(1 for q in self.fake.queries if q.get("gcmtype") == "subcat")  # noqa: E731
        first = cp.check(plan, commons_api=self.fake.commons_api, cache=cache)
        before = walks()
        self.assertGreater(before, 0)
        cp.collect(plan, {}, self.dest, commons_api=self.fake.commons_api, cache=cache)
        self.assertEqual(walks(), before)  # the collection used the check's tree
        self.assertEqual(cp.check(plan, commons_api=self.fake.commons_api, cache=cache)[0]["commons"],
                         first[0]["commons"])
        for f in (Path(self.tmp.name) / "cache").iterdir():
            os.utime(f, (0, 0))  # a day old
        cp.check(plan, commons_api=self.fake.commons_api, cache=cache)
        self.assertGreater(walks(), before)

    def test_share_alike_photos_when_allowed(self):
        manifest = self.run_plan({"targets": [{**COUPE, "count": 10}], "share_alike": True})
        titles = sorted(i.title for i in manifest.items)
        self.assertIn("coupe-sa.jpg", titles)  # CC BY-SA on Commons
        self.assertIn("Share-alike coupe", titles)  # BY-SA on Openverse
        sa = next(i for i in manifest.items if i.title == "coupe-sa.jpg")
        self.assertEqual((sa.license, sa.license_text), ("cc-by-sa", "CC BY-SA 4.0"))
        self.assertEqual({q.get("license") for q in self.fake.queries if "q" in q}, {"by,by-sa,cc0,pdm"})

    def test_count_per_target_and_commons_first(self):
        manifest = self.run_plan({"targets": [{**COUPE, "count": 2}]})
        self.assertEqual(len(manifest.items), 2)
        self.assertTrue(all(i.source == "commons" for i in manifest.items))
        self.assertFalse(any(h.startswith("/openverse") for h in self.fake.hits))  # not needed

    def test_running_again_resumes_without_downloading_again(self):
        self.run_plan({"targets": [{**COUPE, "count": 2}]})
        images_before = sum(1 for h in self.fake.hits if h.startswith("/img/"))
        manifest = self.run_plan({"targets": [{**COUPE, "count": 3}]})
        images_after = sum(1 for h in self.fake.hits if h.startswith("/img/"))
        self.assertEqual(len(manifest.items), 3)
        self.assertEqual(images_after - images_before, 1)

    def test_limits_per_file_and_in_total(self):
        self.fake.extra_commons = [_page(201, "big-coupe.jpg", "cc0", "CC0")]
        manifest = self.run_plan({"targets": [{**COUPE, "count": 10}], "max_file_bytes": 1024 * 1024,
                                  "sources": ["commons"]})
        self.assertEqual(manifest.skipped.get("file too large"), 1)
        self.assertNotIn("big-coupe.jpg", [i.title for i in manifest.items])

        self.tmp.cleanup()
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "photos"
        one = len(self.fake.hits)
        cap = 10 * 1024 * 1024 + 1  # the plan's minimum total
        self.fake.extra_commons = [_page(300 + i, f"big-{i}.jpg", "cc0", "CC0") for i in range(6)]
        manifest = self.run_plan({"targets": [{**COUPE, "count": 50, "category": ""}], "max_total_bytes": cap,
                                  "max_file_bytes": 4 * 1024 * 1024, "sources": ["commons"]})
        self.assertLessEqual(sum(i.bytes for i in manifest.items), cap)
        self.assertGreater(len(self.fake.hits), one)

    def test_bad_downloads_are_skipped_not_fatal(self):
        self.fake.extra_commons = [_page(401, "broken-a.jpg", "cc0", "CC0"), _page(402, "tiny-b.jpg", "cc0", "CC0"),
                                   _page(403, "small.jpg", "cc0", "CC0", width=300, height=200)]
        manifest = self.run_plan({"targets": [{**COUPE, "count": 20}], "sources": ["commons"]})
        self.assertEqual(manifest.skipped.get("not a readable picture"), 1)
        self.assertEqual(manifest.skipped.get("too small"), 2)  # one by its stated size, one after download
        self.assertEqual(len(manifest.items), 5)

    def test_duplicate_pictures_are_kept_once(self):
        self.fake.extra_commons = [_page(501, "same-a.jpg", "cc0", "CC0"), _page(502, "same-b.jpg", "cc0", "CC0")]
        manifest = self.run_plan({"targets": [{**COUPE, "count": 20}], "sources": ["commons"]})
        self.assertEqual(manifest.skipped.get("duplicate"), 1)
        self.assertEqual(sum(1 for i in manifest.items if i.title.startswith("same-")), 1)

    def test_openverse_throttling_ends_that_source_quietly(self):
        self.fake.openverse_status = 429
        manifest = self.run_plan({"targets": [{**COUPE, "count": 10}]})
        self.assertEqual(len(manifest.items), 5)
        self.assertEqual(manifest.skipped.get("openverse limit reached"), 1)

    def test_search_when_no_category(self):
        target = {k: v for k, v in COUPE.items() if k != "category"}
        manifest = self.run_plan({"targets": [{**target, "count": 1}], "sources": ["commons"]})
        self.assertEqual(len(manifest.items), 1)
        self.assertEqual(self.fake.searches[0], "Example Motors Coupe GT Mk2 2019-2023 filetype:bitmap")

    def test_any_subject_and_own_search_words(self):
        watches = parse_subject({"noun": "wristwatch", "levels": [
            {"label": "Brand", "required": True}, {"label": "Model", "required": True}, {"label": "Reference"}]})
        levels = [{"key": lv.key, "required": lv.required} for lv in watches.levels]
        raw = {"levels": levels, "sources": ["commons"],
               "targets": [{"values": ["Example Watch Co", "Diver", "D-100"], "query": "Example diver watch",
                            "count": 2}]}
        manifest = cp.collect(cp.parse_plan(raw), raw, self.dest, commons_api=self.fake.commons_api,
                              openverse_api=self.fake.openverse_api)
        self.assertEqual(len(manifest.items), 2)
        self.assertEqual(self.fake.searches[0], "Example diver watch filetype:bitmap")  # the user's own words
        folder = self.dest / "Example_Watch_Co" / "Diver" / "D-100"
        self.assertEqual(json.loads((folder / "labels.json").read_text()),
                         {"brand": "Example Watch Co", "model": "Diver", "reference": "D-100"})
        usable = [i for i in scan(self.dest, watches).items if i.usable]
        self.assertEqual(usable[0].facts, {"brand": "Example Watch Co", "model": "Diver", "reference": "D-100"})

    def test_a_level_left_out_ends_the_folders_and_labels_carry_the_rest(self):
        raw = {"levels": CAR_LEVELS, "sources": ["commons"],
               "targets": [{"values": ["Example Motors", "Coupe GT", "", "2019-2023"], "count": 1}]}
        cp.collect(cp.parse_plan(raw), raw, self.dest, commons_api=self.fake.commons_api,
                   openverse_api=self.fake.openverse_api)
        folder = self.dest / "Example_Motors" / "Coupe_GT"
        self.assertEqual(json.loads((folder / "labels.json").read_text()),
                         {"make": "Example Motors", "model": "Coupe GT", "years": "2019-2023"})
        item = next(i for i in scan(self.dest, CARS).items if i.usable)
        self.assertEqual(item.facts, {"make": "Example Motors", "model": "Coupe GT", "years": "2019-2023"})

    def test_only_https_leaves_this_machine(self):
        http = cp.Http()
        for url in ("http://example.org/a.jpg", "ftp://example.org/a.jpg", "file:///C:/secret.jpg"):
            with self.assertRaises(cp.NotHttps, msg=url):
                http.get(url, 1024)
        self.fake.extra_commons = [_page(601, "plain.jpg", "cc0", "CC0")]
        self.fake.absolute_override = {"plain.jpg": "http://example.invalid/plain.jpg"}
        manifest = self.run_plan({"targets": [{**COUPE, "count": 20}], "sources": ["commons"]})
        self.assertEqual(manifest.skipped.get("not https"), 1)
        self.assertNotIn("plain.jpg", [i.title for i in manifest.items])

    def test_check_counts_without_downloading(self):
        plan = cp.parse_plan({"levels": CAR_LEVELS, "targets": [COUPE, {"values": ["Example Motors", "Roadster", "", ""],
                                                                        "category": "Example Roadster"},
                                                                {"values": ["Example Motors", "Van", "", ""]}]})
        results = cp.check(plan, commons_api=self.fake.commons_api, openverse_api=self.fake.openverse_api)
        self.assertFalse(any(h.startswith("/img/") for h in self.fake.hits))  # no photo fetched
        coupe, roadster, van = results
        c = coupe["commons"]
        self.assertEqual((c["mode"], c["categories"], c["found"]), ("category", 3, 8))  # 6 + 1 + 1
        self.assertEqual(c["skipped"], ["Category:Example Coupe logos"])
        self.assertEqual((c["sampled"], c["kept"], c["estimate"]), (8, 5, 5))  # SA, people and the SVG are refused
        o = coupe["openverse"]
        self.assertEqual((o["found"], o["reachable"], o["sampled"], o["kept"], o["estimate"]), (3, 3, 3, 1, 1))
        self.assertTrue(roadster["commons"]["missing"])
        self.assertEqual(van["commons"]["mode"], "search")
        self.assertEqual(van["commons"]["found"], 6)

    def test_main_reports_json_progress(self):
        plan = Path(self.tmp.name) / "plan.json"
        plan.write_text(json.dumps({"levels": CAR_LEVELS, "targets": [{**COUPE, "count": 2}], "sources": ["commons"]}))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cp.main(["--plan", str(plan), "--dest", str(self.dest), "--json",
                            "--commons-api", self.fake.commons_api, "--openverse-api", self.fake.openverse_api])
        events = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(code, 0)
        self.assertEqual(events[0]["event"], "start")
        self.assertEqual([e["have"] for e in events if e["event"] == "saved"], [1, 2])
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["photos"], 2)
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text("{}")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cp.main(["--plan", str(bad), "--dest", str(self.dest), "--json"]), 2)


if __name__ == "__main__":
    unittest.main()
