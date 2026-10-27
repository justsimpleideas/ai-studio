import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_engine import video_io


def frames(count=9, size=(64, 32)):
    return [Image.new("RGB", size, (i * 25 % 256, 80, 200 - i * 10)) for i in range(count)]


class FindFfmpegTests(unittest.TestCase):
    def test_a_configured_path_that_is_gone_is_not_replaced_by_another(self):
        self.assertIsNone(video_io.find_ffmpeg("C:/nowhere/ffmpeg.exe"))

    def test_a_configured_file_must_be_named_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "python.exe"
            other.write_bytes(b"")
            self.assertIsNone(video_io.find_ffmpeg(str(other)))
            named = Path(tmp) / ("ffmpeg.exe" if video_io.FFMPEG_NAMES[0].endswith(".exe") else "ffmpeg")
            named.write_bytes(b"")
            self.assertEqual(video_io.find_ffmpeg(str(named)), str(named))


class EncodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_without_ffmpeg_the_frames_become_an_animated_webp(self):
        out = video_io.encode(frames(9), 24, self.root / "clip", None)
        self.assertEqual(out.path.suffix, ".webp")
        self.assertEqual(out.mime, "image/webp")
        self.assertIn("ffmpeg not found", out.encoder)
        with Image.open(out.path) as img:
            self.assertEqual(img.n_frames, 9)
            self.assertEqual(img.size, (64, 32))
        self.assertEqual(list(self.root.glob("*.part")), [])

    def test_a_failing_ffmpeg_falls_back_instead_of_losing_the_result(self):
        out = video_io.encode(frames(5), 24, self.root / "clip", str(self.root / "missing" / "ffmpeg.exe"))
        self.assertEqual(out.mime, "image/webp")
        self.assertIn("ffmpeg failed", out.encoder)

    def test_no_frames_is_an_error(self):
        with self.assertRaises(ValueError):
            video_io.encode([], 24, self.root / "clip", None)

    @unittest.skipUnless(video_io.find_ffmpeg(), "ffmpeg is not installed here")
    def test_with_ffmpeg_an_mp4_without_metadata(self):
        out = video_io.encode(frames(25, (128, 64)), 24, self.root / "clip", video_io.find_ffmpeg())
        self.assertEqual((out.path.suffix, out.mime), (".mp4", "video/mp4"))
        data = out.path.read_bytes()
        self.assertEqual(data[4:8], b"ftyp")
        # faststart puts the index (moov) before the frames (mdat), so playback starts at once.
        self.assertLess(data.find(b"moov"), data.find(b"mdat"))
        self.assertNotIn(b"creation_time", data)
        self.assertEqual(list(self.root.glob("*.part")), [])


class GifTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def gif(self, out: Path) -> tuple[tuple[int, int], int, int]:
        """Size, frame count and per-frame duration (ms) of a GIF."""
        data = out.read_bytes()
        self.assertEqual(data[:6], b"GIF89a")
        with Image.open(out) as img:
            return img.size, img.n_frames, img.info.get("duration", 0)

    def test_the_gif_size_keeps_the_shape_and_never_enlarges(self):
        self.assertEqual(video_io.gif_size(1280, 704, 480), (480, 264))
        self.assertEqual(video_io.gif_size(832, 480, 1280), (832, 480))
        self.assertEqual(video_io.gif_size(704, 1280, 320), (320, 582))
        self.assertEqual(video_io.gif_size(4, 1, 2), (2, 1))  # 0.5 rounds up, as in the window

    def test_a_webp_video_becomes_a_gif_without_ffmpeg(self):
        webp = video_io.encode(frames(24, (128, 64)), 24, self.root / "clip", None).path
        dest = self.root / "out.gif"
        made = video_io.to_gif(webp, "image/webp", 24, (128, 64), dest, width=64, gif_fps=12, ffmpeg=None)
        self.assertEqual(made, "Pillow")
        size, count, duration = self.gif(dest)
        self.assertEqual(size, (64, 32))
        self.assertEqual(count, 12)  # half the frame rate, half the frames
        self.assertAlmostEqual(duration, 1000 / 12, delta=10)
        self.assertEqual([p.name for p in self.root.iterdir() if p.suffix == ".part"], [])

    def test_the_gif_frame_rate_is_never_above_the_videos(self):
        webp = video_io.encode(frames(8, (64, 32)), 8, self.root / "clip", None).path
        dest = self.root / "out.gif"
        video_io.to_gif(webp, "image/webp", 8, (64, 32), dest, width=64, gif_fps=24, ffmpeg=None)
        self.assertEqual(self.gif(dest)[1], 8)

    def test_an_mp4_needs_ffmpeg(self):
        with self.assertRaises(video_io.GifError) as caught:
            video_io.to_gif(self.root / "clip.mp4", "video/mp4", 24, (64, 32), self.root / "out.gif",
                            width=64, gif_fps=12, ffmpeg=None)
        self.assertIn("ffmpeg", str(caught.exception))

    def test_a_broken_source_is_a_gif_error_and_leaves_nothing(self):
        bad = self.root / "clip.webp"
        bad.write_bytes(b"RIFF....WEBPnot really")
        with self.assertRaises(video_io.GifError):
            video_io.to_gif(bad, "image/webp", 24, (64, 32), self.root / "out.gif", width=64, gif_fps=12, ffmpeg=None)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["clip.webp"])

    @unittest.skipUnless(video_io.find_ffmpeg(), "ffmpeg is not installed here")
    def test_with_ffmpeg_an_mp4_becomes_a_gif(self):
        ffmpeg = video_io.find_ffmpeg()
        mp4 = video_io.encode(frames(24, (128, 64)), 24, self.root / "clip", ffmpeg).path
        dest = self.root / "out.gif"
        self.assertEqual(video_io.to_gif(mp4, "video/mp4", 24, (128, 64), dest, width=96, gif_fps=12,
                                         ffmpeg=ffmpeg), "ffmpeg")
        size, count, _ = self.gif(dest)
        self.assertEqual(size, (96, 48))
        self.assertEqual(count, 12)
        self.assertNotIn(b"Lavf", dest.read_bytes())  # no encoder tag or other metadata

    @unittest.skipUnless(video_io.find_ffmpeg(), "ffmpeg is not installed here")
    def test_with_ffmpeg_a_webp_video_goes_through_its_palette_filters(self):
        webp = video_io.encode(frames(12, (128, 64)), 24, self.root / "clip", None).path
        dest = self.root / "out.gif"
        made = video_io.to_gif(webp, "image/webp", 24, (128, 64), dest, width=1280, gif_fps=24,
                               ffmpeg=video_io.find_ffmpeg())
        self.assertEqual(made, "ffmpeg")
        self.assertEqual(self.gif(dest)[:2], ((128, 64), 12))


if __name__ == "__main__":
    unittest.main()
