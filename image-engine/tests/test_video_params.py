import unittest

from image_engine.params import (VIDEO_DEFAULTS, VIDEO_FPS, GenerationRequest, RequestError, VideoRequest,
                                 parse_job, parse_video_request)
from image_engine.video_models import IMAGE_14B, TEXT_14B


class ParseVideoRequestTests(unittest.TestCase):
    def test_defaults_are_the_models_own(self):
        req = parse_video_request({"prompt": " a wave breaking "})
        self.assertEqual(req.prompt, "a wave breaking")
        self.assertEqual((req.width, req.height, req.frames, req.steps, req.guidance, req.count),
                         (1280, 704, 121, 50, 5.0, 1))
        self.assertEqual(req.fps, VIDEO_FPS)
        self.assertAlmostEqual(req.seconds, 121 / 24)
        self.assertIsNone(req.start_image)

    def test_frame_counts_must_be_four_k_plus_one(self):
        for frames in (5, 49, 97, 121):
            self.assertEqual(parse_video_request({"prompt": "x", "frames": frames}).frames, frames)
        for frames in (4, 48, 120, 125, 1, 0):
            with self.subTest(frames=frames), self.assertRaises(RequestError):
                parse_video_request({"prompt": "x", "frames": frames})

    def test_sizes_are_multiples_of_32_within_720p(self):
        ok = [(1280, 704), (704, 1280), (960, 544), (832, 480), (928, 928)]
        for w, h in ok:
            req = parse_video_request({"prompt": "x", "width": w, "height": h})
            self.assertEqual((req.width, req.height), (w, h))
        bad = [(1280, 720),   # 720 is not a multiple of 32
               (960, 960),    # more pixels than 1280x704
               (1312, 704),   # wider than 1280
               (224, 704)]    # narrower than 256
        for w, h in bad:
            with self.subTest(size=(w, h)), self.assertRaises(RequestError):
                parse_video_request({"prompt": "x", "width": w, "height": h})

    def test_fps_cannot_be_asked_for(self):
        self.assertEqual(parse_video_request({"prompt": "x", "fps": 60}).fps, VIDEO_FPS)

    def test_start_image_must_be_an_input_id(self):
        self.assertEqual(parse_video_request({"prompt": "x", "start_image": "0123456789abcdef"}).start_image,
                         "0123456789abcdef")
        for bad in ("../../etc/passwd", "0123", 5, "0123456789ABCDEF"):
            with self.subTest(bad=bad), self.assertRaises(RequestError):
                parse_video_request({"prompt": "x", "start_image": bad})

    def test_rejections(self):
        bad = [None, {}, {"prompt": ""}, {"prompt": "x", "count": 5}, {"prompt": "x", "steps": 0},
               {"prompt": "x", "guidance": 11}, {"prompt": "x", "frames": "121"}]
        for data in bad:
            with self.subTest(data=data), self.assertRaises(RequestError):
                parse_video_request(data)

    def test_to_dict_says_what_it_is(self):
        data = parse_video_request({"prompt": "x", "seed": 3}).to_dict()
        self.assertEqual(data["kind"], "video")
        self.assertEqual(data["seed"], 3)
        self.assertEqual(data["frames"], VIDEO_DEFAULTS["frames"])


class FourteenBRequestTests(unittest.TestCase):
    """A request checked against a 14B model: its sizes, frame rate, defaults and start."""

    def test_defaults_sizes_and_frame_rate_are_the_models(self):
        req = parse_video_request({"prompt": "x"}, TEXT_14B)
        self.assertEqual((req.width, req.height, req.frames, req.fps, req.steps, req.guidance),
                         (832, 480, 81, 16, 40, 4.0))
        self.assertEqual((parse_video_request({"prompt": "x", "width": 1280, "height": 720}, TEXT_14B).height), 720)
        for w, h in ((1280, 736), (840, 480)):  # more than 720p's pixels; not a multiple of 16
            with self.subTest(size=(w, h)), self.assertRaises(RequestError):
                parse_video_request({"prompt": "x", "width": w, "height": h}, TEXT_14B)

    def test_each_model_takes_only_its_own_start(self):
        picture = {"prompt": "x", "start_image": "0123456789abcdef"}
        self.assertEqual(parse_video_request(picture, IMAGE_14B).start_image, "0123456789abcdef")
        with self.assertRaises(RequestError) as ctx:
            parse_video_request({"prompt": "x"}, IMAGE_14B)
        self.assertIn("start picture", str(ctx.exception))
        with self.assertRaises(RequestError) as ctx:
            parse_video_request(picture, TEXT_14B)
        self.assertIn("text only", str(ctx.exception))
        self.assertIsInstance(parse_job({"kind": "video", **picture}, IMAGE_14B), VideoRequest)


class ParseJobTests(unittest.TestCase):
    def test_kind_picks_the_parser(self):
        self.assertIsInstance(parse_job({"prompt": "x"}), GenerationRequest)
        self.assertIsInstance(parse_job({"prompt": "x", "kind": "image"}), GenerationRequest)
        self.assertIsInstance(parse_job({"prompt": "x", "kind": "video"}), VideoRequest)

    def test_unknown_kind_is_refused(self):
        with self.assertRaises(RequestError):
            parse_job({"prompt": "x", "kind": "audio"})

    def test_image_requests_are_unchanged(self):
        req = parse_job({"prompt": "x", "seed": 1})
        self.assertNotIn("kind", req.to_dict())
        self.assertEqual(req.kind, "image")


if __name__ == "__main__":
    unittest.main()
