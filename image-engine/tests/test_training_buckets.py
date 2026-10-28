import unittest

from image_engine.params import SIDE_MULTIPLE
from image_engine.training.buckets import MAX_RATIO, assign, crop_plan, make_buckets


class BucketTests(unittest.TestCase):
    def test_buckets_respect_pixel_budget_multiple_and_ratio(self):
        for side in (512, 768, 1024):
            buckets = make_buckets(side)
            self.assertIn((side, side), buckets)
            for w, h in buckets:
                self.assertLessEqual(w * h, side * side)
                self.assertGreater(w * h, side * side * 0.85, (w, h))
                self.assertEqual((w % SIDE_MULTIPLE, h % SIDE_MULTIPLE), (0, 0))
                self.assertLessEqual(max(w / h, h / w), MAX_RATIO)
                self.assertIn((h, w), buckets)

    def test_target_side_must_fit_the_multiple(self):
        with self.assertRaises(ValueError):
            make_buckets(770)

    def test_assign_picks_the_closest_shape(self):
        buckets = make_buckets(768)
        self.assertEqual(assign(4000, 4000, buckets), (768, 768))
        w, h = assign(1920, 1080, buckets)  # 16:9 side view
        self.assertAlmostEqual(w / h, 16 / 9, delta=0.05)
        w, h = assign(1080, 1920, buckets)
        self.assertGreater(h, w)
        w, h = assign(5000, 1000, buckets)  # wider than any bucket: the widest one
        self.assertEqual(w / h, max(b[0] / b[1] for b in buckets))

    def test_crop_plan_covers_the_bucket_and_centres(self):
        sw, sh, left, top = crop_plan(1920, 1080, (1024, 576))
        self.assertEqual((sw, sh, left, top), (1024, 576, 0, 0))
        sw, sh, left, top = crop_plan(1000, 1000, (832, 704))
        self.assertGreaterEqual(sw, 832)
        self.assertGreaterEqual(sh, 704)
        self.assertEqual((sw, sh), (832, 832))
        self.assertEqual((left, top), (0, 64))


if __name__ == "__main__":
    unittest.main()
