import unittest

from image_engine.params import MAX_SEED, RequestError, parse_request


class ParseRequestTests(unittest.TestCase):
    def test_defaults_fill_missing_fields(self):
        req = parse_request({"prompt": "  a red fox  "})
        self.assertEqual(req.prompt, "a red fox")
        self.assertEqual((req.width, req.height, req.steps, req.guidance, req.count), (1024, 1024, 8, 0.0, 1))
        self.assertTrue(0 <= req.seed <= MAX_SEED)

    def test_random_seed_when_minus_one(self):
        seeds = {parse_request({"prompt": "x", "seed": -1}).seed for _ in range(20)}
        self.assertGreater(len(seeds), 1)

    def test_fixed_seed_and_batch_seeds_are_consecutive(self):
        req = parse_request({"prompt": "x", "seed": 41, "count": 3})
        self.assertEqual([req.seed_for(i) for i in range(3)], [41, 42, 43])

    def test_batch_seed_wraps_at_max(self):
        req = parse_request({"prompt": "x", "seed": MAX_SEED, "count": 2})
        self.assertEqual(req.seed_for(1), 0)

    def test_rejections(self):
        bad = [
            None, [], "prompt",
            {}, {"prompt": "   "}, {"prompt": 5},
            {"prompt": "x" * 4001},
            {"prompt": "x", "width": 1000},          # not a multiple of 16
            {"prompt": "x", "width": 128},           # too small
            {"prompt": "x", "width": 4096},          # too large
            {"prompt": "x", "width": 1024.0},        # float, not int
            {"prompt": "x", "steps": True},          # bool is not an int here
            {"prompt": "x", "steps": 0},
            {"prompt": "x", "steps": 51},
            {"prompt": "x", "guidance": -1},
            {"prompt": "x", "guidance": "3"},
            {"prompt": "x", "count": 17},
            {"prompt": "x", "seed": 2**32},
            {"prompt": "x", "seed": -2},
            {"prompt": "x", "seed": "7"},
        ]
        for body in bad:
            with self.subTest(body=body if not isinstance(body, dict) or len(str(body)) < 80 else "long"):
                with self.assertRaises(RequestError):
                    parse_request(body)

    def test_accepts_extreme_but_valid_sizes(self):
        req = parse_request({"prompt": "x", "width": 2048, "height": 2048})
        self.assertEqual((req.width, req.height), (2048, 2048))
        req = parse_request({"prompt": "x", "width": 256, "height": 2048})
        self.assertEqual(req.width, 256)


if __name__ == "__main__":
    unittest.main()
