import json
import tempfile
import unittest
from pathlib import Path

from image_engine.video_models import (FAST, IMAGE_14B, STANDARD, TEXT_14B, UnknownVideoModel, profile_of,
                                       try_profile_of)


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
        self.assertEqual((shown["experts"], shown["fps"], shown["start"]), (1, 24, "any"))


class FourteenBProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def folder(self, name: str, index: dict) -> Path:
        folder = self.root / name
        folder.mkdir()
        (folder / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
        return folder

    def test_two_experts_are_told_apart_by_their_pipeline_and_boundary(self):
        text = profile_of(self.folder("t", {"_class_name": "WanPipeline", "boundary_ratio": 0.875}))
        image = profile_of(self.folder("i", {"_class_name": "WanImageToVideoPipeline", "boundary_ratio": 0.9}))
        self.assertIs(text, TEXT_14B)
        self.assertIs(image, IMAGE_14B)
        # The models' own settings: 40 steps, 16 fps, 81 frames, 480p by default, sides in 16s.
        self.assertEqual((text.steps, text.guidance, text.guidance_2, text.start, text.experts), (40, 4.0, 3.0, "text", 2))
        self.assertEqual((image.guidance, image.guidance_2, image.start), (3.5, 3.5, "image"))
        self.assertEqual((text.fps, text.trained_frames, text.width, text.height, text.side_multiple), (16, 81, 832, 480, 16))

    def test_a_few_step_setting_makes_a_fixed_distilled_profile(self):
        few = profile_of(self.folder("f", {"_class_name": "WanPipeline", "boundary_ratio": 0.875,
                                            "few_step": {"steps": 4, "shift": 5.0}}))
        self.assertEqual((few.name, few.steps, few.guidance, few.guidance_2, few.shift, few.fixed),
                         ("text-14b-few", 4, 1.0, 1.0, 5.0, True))
        self.assertEqual(few.boundary, 0.875)

    def test_wrong_boundaries_and_few_step_settings_are_refused(self):
        for name, index in (("b", {"_class_name": "WanPipeline", "boundary_ratio": 0.5}),
                            ("s", {"_class_name": "WanPipeline", "boundary_ratio": 0.875, "few_step": {"steps": 0, "shift": 5}}),
                            ("x", {"_class_name": "WanPipeline", "boundary_ratio": 0.875, "few_step": {"steps": 4}}),
                            ("n", {"_class_name": "WanPipeline", "boundary_ratio": 0.875,
                                   "few_step": {"steps": 4, "shift": float("nan")}}),
                            ("d", {"_class_name": "WanDMDPipeline", "boundary_ratio": 0.875})):
            with self.subTest(name=name), self.assertRaises(UnknownVideoModel):
                profile_of(self.folder(name, index))


if __name__ == "__main__":
    unittest.main()
