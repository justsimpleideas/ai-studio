import json
import tempfile
import unittest
from pathlib import Path

from image_engine.video_models import FAST, STANDARD, UnknownVideoModel, profile_of, try_profile_of


def model_folder(root: Path, class_name: str | None) -> Path:
    folder = root / (class_name or "empty")
    folder.mkdir()
    if class_name is not None:
        (folder / "model_index.json").write_text(json.dumps({"_class_name": class_name}), encoding="utf-8")
    return folder


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_original_and_the_distilled_model_are_told_apart_by_their_folder(self):
        self.assertIs(profile_of(model_folder(self.root, "WanPipeline")), STANDARD)
        self.assertIs(profile_of(model_folder(self.root, "WanDMDPipeline")), FAST)

    def test_the_distilled_model_has_fixed_steps_and_no_guidance(self):
        self.assertTrue(FAST.fixed)
        self.assertEqual((FAST.steps, FAST.guidance), (3, 1.0))
        self.assertEqual(len(FAST.sigmas), FAST.steps)
        self.assertEqual(list(FAST.sigmas), sorted(FAST.sigmas, reverse=True))
        self.assertFalse(STANDARD.fixed)
        self.assertIsNone(STANDARD.sigmas)

    def test_anything_else_is_refused(self):
        with self.assertRaises(UnknownVideoModel):
            profile_of(model_folder(self.root, "SomeOtherPipeline"))
        with self.assertRaises(UnknownVideoModel):
            profile_of(model_folder(self.root, None))  # no model_index.json
        broken = self.root / "broken"
        broken.mkdir()
        (broken / "model_index.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(UnknownVideoModel):
            profile_of(broken)

    def test_for_display_an_unknown_folder_is_just_none(self):
        self.assertIsNone(try_profile_of(""))
        self.assertIsNone(try_profile_of(self.root / "missing"))
        self.assertIs(try_profile_of(str(model_folder(self.root, "WanDMDPipeline"))), FAST)

    def test_the_window_gets_no_internal_schedule(self):
        shown = FAST.to_dict()
        self.assertNotIn("sigmas", shown)
        self.assertEqual(shown["name"], "fast")
        self.assertTrue(shown["fixed"])


if __name__ == "__main__":
    unittest.main()
