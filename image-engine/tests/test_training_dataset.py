import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from image_engine.training import dataset as ds
from image_engine.training.dataset import Dataset, DatasetError, dhash, hamming, scan
from image_engine.training.subject import CARS, parse_subject


def photo(path: Path, size=(640, 512), seed=0, fmt=None, exif_orientation=None):
    """A synthetic photo with enough structure that its dHash depends on `seed`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size)
    w, h = size
    stripes = 3 + seed % 7
    img.paste((40 + seed * 37 % 200, 90, 160), (0, 0, w // stripes, h))
    for i in range(stripes):
        shade = (i * 53 + seed * 91) % 256
        img.paste((shade, 255 - shade, (shade * 3) % 256), (i * w // stripes, (i % 2) * h // 2,
                                                             (i + 1) * w // stripes, h))
    kwargs = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif
    img.save(path, fmt, **kwargs)
    return path


class DatasetScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "cars"
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def by_path(self, dataset):
        return {item.path: item for item in dataset.items}

    def test_facts_come_from_folder_levels_in_subject_order(self):
        photo(self.root / "Example_Motors" / "Coupe_GT" / "Mk2" / "a.jpg")
        item = scan(self.root, CARS).items[0]
        self.assertEqual(item.facts, {"make": "Example Motors", "model": "Coupe GT", "generation": "Mk2"})
        self.assertTrue(item.usable)

    def test_hyphens_in_folder_names_are_kept(self):
        photo(self.root / "Example" / "F-150" / "a.jpg")
        self.assertEqual(scan(self.root, CARS).items[0].facts["model"], "F-150")

    def test_labels_json_overrides_folders_nearest_wins(self):
        photo(self.root / "ex" / "coupe" / "a.jpg")
        photo(self.root / "ex" / "coupe" / "late" / "b.jpg", seed=1)
        (self.root / "ex" / "labels.json").write_text(json.dumps({"make": "Example Motors"}))
        (self.root / "ex" / "coupe" / "labels.json").write_text(json.dumps({"years": "2019-2023"}))
        (self.root / "ex" / "coupe" / "late" / "labels.json").write_text(
            json.dumps({"years": "2022-2023", "generation": "Facelift"}))
        items = self.by_path(scan(self.root, CARS))
        self.assertEqual(items["ex/coupe/a.jpg"].facts,
                         {"make": "Example Motors", "model": "coupe", "years": "2019-2023"})
        self.assertEqual(items["ex/coupe/late/b.jpg"].facts,
                         {"make": "Example Motors", "model": "coupe", "generation": "Facelift",
                          "years": "2022-2023"})

    def test_bad_labels_json_is_a_warning_not_a_crash(self):
        photo(self.root / "ex" / "coupe" / "a.jpg")
        (self.root / "ex" / "labels.json").write_text("{not json")
        (self.root / "ex" / "coupe" / "labels.json").write_text(json.dumps({"colour": "red", "make": 3}))
        dataset = scan(self.root, CARS)
        self.assertEqual(dataset.items[0].facts, {"make": "ex", "model": "coupe"})
        self.assertEqual(len(dataset.warnings), 3)
        self.assertIn("not valid JSON", dataset.warnings[0])
        self.assertTrue(any("unknown field 'colour'" in w for w in dataset.warnings))

    def test_missing_required_facts_keep_a_photo_out(self):
        photo(self.root / "loose.jpg")
        photo(self.root / "ex" / "only-make.jpg", seed=1)
        items = self.by_path(scan(self.root, CARS))
        self.assertEqual(items["loose.jpg"].issues, ["missing_facts:make,model"])
        self.assertEqual(items["ex/only-make.jpg"].issues, ["missing_facts:model"])

    def test_small_unreadable_and_non_images(self):
        photo(self.root / "ex" / "m" / "small.png", size=(300, 900))
        (self.root / "ex" / "m" / "broken.jpg").write_bytes(b"not a jpeg at all")
        (self.root / "ex" / "m" / "notes.txt").write_text("ignored")
        (self.root / "ex" / "m" / ".hidden.jpg").write_bytes(b"ignored")
        items = self.by_path(scan(self.root, CARS))
        self.assertEqual(sorted(items), ["ex/m/broken.jpg", "ex/m/small.png"])
        self.assertEqual(items["ex/m/small.png"].issues, ["too_small"])
        self.assertEqual(items["ex/m/broken.jpg"].issues, ["unreadable"])
        self.assertEqual(items["ex/m/broken.jpg"].width, 0)

    def test_exif_rotation_swaps_the_recorded_size(self):
        photo(self.root / "ex" / "m" / "rotated.jpg", size=(800, 600), exif_orientation=6)
        item = scan(self.root, CARS).items[0]
        self.assertEqual((item.width, item.height), (600, 800))

    def test_exact_and_near_duplicates_are_flagged_first_copy_stays(self):
        first = photo(self.root / "ex" / "m" / "a.jpg", seed=3)
        (self.root / "ex" / "m" / "b-copy.jpg").write_bytes(first.read_bytes())
        with Image.open(first) as img:
            img.resize((600, 480)).save(self.root / "ex" / "m" / "c-smaller.png")
        photo(self.root / "ex" / "m" / "d-other.jpg", seed=4)
        items = self.by_path(scan(self.root, CARS))
        original = items["ex/m/a.jpg"]
        self.assertTrue(original.usable)
        self.assertEqual(items["ex/m/b-copy.jpg"].issues, [f"duplicate_of:{original.id}"])
        self.assertEqual(items["ex/m/c-smaller.png"].issues, ["too_small", f"duplicate_of:{original.id}"])
        self.assertTrue(items["ex/m/d-other.jpg"].usable)

    def test_the_largest_usable_copy_is_kept_whatever_its_name(self):
        big = photo(self.root / "ex" / "m" / "z-big.png", size=(1280, 1024), seed=2)
        with Image.open(big) as img:
            img.resize((640, 512)).save(self.root / "ex" / "m" / "a-small.png")
        items = self.by_path(scan(self.root, CARS))
        self.assertTrue(items["ex/m/z-big.png"].usable)
        self.assertEqual(items["ex/m/a-small.png"].issues, [f"duplicate_of:{items['ex/m/z-big.png'].id}"])

    def test_near_duplicate_band_search_matches_brute_force(self):
        # Hashes differing in exactly DUPLICATE_DISTANCE bits, spread across bands, must still be found.
        base = 0x0123456789ABCDEF
        for bits in ([0, 13, 26, 39], [60, 61, 62, 63], [5, 17, 29, 41]):
            other = base
            for b in bits:
                other ^= 1 << b
            a = ds.Item(id="a" * 16, path="a", size=1, mtime_ns=0, sha256="a" * 64, width=600, height=600,
                        dhash=f"{base:016x}", facts={}, holdout=False)
            b_ = ds.Item(id="b" * 16, path="b", size=1, mtime_ns=0, sha256="b" * 64, width=600, height=600,
                         dhash=f"{other:016x}", facts={}, holdout=False)
            ds._mark_duplicates([a, b_])
            self.assertEqual(b_.issues, ["duplicate_of:" + "a" * 16], bits)
        far = ds.Item(id="c" * 16, path="c", size=1, mtime_ns=0, sha256="c" * 64, width=600, height=600,
                      dhash=f"{base ^ 0b11111:016x}", facts={}, holdout=False)
        a = ds.Item(id="a" * 16, path="a", size=1, mtime_ns=0, sha256="a" * 64, width=600, height=600,
                    dhash=f"{base:016x}", facts={}, holdout=False)
        ds._mark_duplicates([a, far])
        self.assertTrue(far.usable)

    def test_holdout_is_deterministic_and_near_the_requested_share(self):
        for i in range(60):
            photo(self.root / "ex" / "m" / f"{i:02}.png", size=(512, 512), seed=i)
        first = scan(self.root, CARS, holdout_percent=20)
        again = scan(self.root, CARS, holdout_percent=20)
        self.assertEqual([i.holdout for i in first.items], [i.holdout for i in again.items])
        held = sum(i.holdout for i in first.items)
        self.assertTrue(3 <= held <= 24, held)
        self.assertEqual(sum(i.holdout for i in scan(self.root, CARS, holdout_percent=0).items), 0)

    def test_rescan_rereads_only_changed_files(self):
        photo(self.root / "ex" / "m" / "a.jpg")
        b = photo(self.root / "ex" / "m" / "b.jpg", seed=1)
        first = scan(self.root, CARS)
        photo(b, seed=2)
        os.utime(b, ns=(b.stat().st_atime_ns, b.stat().st_mtime_ns + 10_000_000))
        with mock.patch.object(ds, "_file_sha256", wraps=ds._file_sha256) as hashed:
            second = scan(self.root, CARS, previous=first)
        self.assertEqual(hashed.call_count, 1)
        self.assertNotEqual(self.by_path(first)["ex/m/b.jpg"].sha256, self.by_path(second)["ex/m/b.jpg"].sha256)

    def test_manifest_round_trip_and_summary(self):
        photo(self.root / "ex" / "m" / "a.jpg")
        photo(self.root / "ex" / "m" / "tiny.jpg", size=(100, 100), seed=1)
        dataset = scan(self.root, CARS, holdout_percent=0)
        path = Path(self.tmp.name) / "dataset.json"
        dataset.save(path)
        loaded = Dataset.load(path)
        self.assertEqual(loaded.to_json(), dataset.to_json())
        self.assertEqual(loaded.summary(), {"total": 2, "usable": 1, "train": 1, "holdout": 0,
                                            "issues": {"too_small": 1}, "warnings": 0})

    def test_scan_never_writes_into_the_photo_folder(self):
        photo(self.root / "ex" / "m" / "a.jpg")
        before = sorted((p.as_posix(), p.stat().st_mtime_ns) for p in self.root.rglob("*"))
        scan(self.root, CARS)
        after = sorted((p.as_posix(), p.stat().st_mtime_ns) for p in self.root.rglob("*"))
        self.assertEqual(before, after)

    def test_errors(self):
        with self.assertRaises(DatasetError):
            scan(self.root / "missing", CARS)
        with self.assertRaises(DatasetError):
            scan(self.root, CARS, holdout_percent=80)
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text(json.dumps({"version": 99}))
        with self.assertRaises(DatasetError):
            Dataset.load(bad)

    def test_progress_is_reported(self):
        for i in range(3):
            photo(self.root / "ex" / "m" / f"{i}.png", seed=i)
        seen = []
        scan(self.root, CARS, on_progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(1, 3), (2, 3), (3, 3)])


class HashTests(unittest.TestCase):
    def test_dhash_survives_resizing_but_not_a_different_picture(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = photo(Path(tmp) / "a.png", size=(800, 600), seed=5)
            b = photo(Path(tmp) / "b.png", size=(800, 600), seed=6)
            with Image.open(a) as ia, Image.open(b) as ib:
                ha, hb = dhash(ia), dhash(ib)
                small = dhash(ia.resize((200, 150)))
        self.assertLessEqual(hamming(ha, small), ds.DUPLICATE_DISTANCE)
        self.assertGreater(hamming(ha, hb), ds.DUPLICATE_DISTANCE)


class SubjectLevelTests(unittest.TestCase):
    """Folder levels and labels.json follow whatever subject the user defined."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.watches = parse_subject({
            "noun": "wristwatch",
            "levels": [{"label": "Brand", "required": True}, {"label": "Model", "required": True},
                       {"label": "Reference"}],
            "fields": [{"label": "Strap", "question": "q", "choices": ["metal", "leather"], "consistent": True},
                       {"label": "Setting", "question": "q"}],
        })

    def tearDown(self):
        self.tmp.cleanup()

    def test_levels_and_labels(self):
        photo(self.root / "Seiko" / "SKX007" / "a.jpg")
        photo(self.root / "Seiko" / "loose.jpg", seed=1)
        (self.root / "Seiko" / "SKX007" / "labels.json").write_text(
            json.dumps({"reference": "SKX007K2", "strap": "metal", "setting": "desk", "make": "x"}))
        dataset = scan(self.root, self.watches)
        items = {i.path: i for i in dataset.items}
        self.assertEqual(items["Seiko/SKX007/a.jpg"].facts,
                         {"brand": "Seiko", "model": "SKX007", "reference": "SKX007K2", "strap": "metal"})
        self.assertEqual(items["Seiko/loose.jpg"].issues, ["missing_facts:model"])
        # Only levels and consistent fields can be set by labels.json; the rest is reported.
        self.assertEqual(sorted(w.split("unknown field ")[1].split()[0] for w in dataset.warnings),
                         ["'make'", "'setting'"])
        self.assertIn("(fields: brand, model, reference, strap)", dataset.warnings[0])


if __name__ == "__main__":
    unittest.main()
