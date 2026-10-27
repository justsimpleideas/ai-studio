"""Turning decoded frames into a video file.

MP4 (H.264) through ffmpeg when one can be found: the path chosen in settings,
else one on PATH, else where winget installs it on Windows. Without ffmpeg, or if
it fails, the frames are saved as an animated WebP with Pillow instead, so a
finished generation is never lost to a missing tool.

ffmpeg runs as a plain child process (no shell) that reads raw frames on stdin
and writes one file. The engine passes no metadata, and strips any, so the file
carries neither the prompt nor a timestamp; the gallery index keeps the settings
(docs/image-engine-decisions.md, D10).

A stored video can also be exported as a GIF (to_gif): through ffmpeg's palette
filters when it is there, else Pillow for WebP videos. Reading an MP4 needs ffmpeg.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

FFMPEG_TIMEOUT_SECONDS = 600
FFMPEG_NAMES = ("ffmpeg.exe", "ffmpeg") if sys.platform == "win32" else ("ffmpeg",)
# Visually lossless for generated footage; the files stay a few MB per 5 s at 720p.
H264_CRF = "17"


@dataclass(frozen=True)
class EncodedVideo:
    path: Path
    mime: str
    encoder: str  # what made the file, shown with the video


def is_ffmpeg_name(path: str) -> bool:
    return Path(path).name.lower() in FFMPEG_NAMES


def find_ffmpeg(configured: str = "") -> str | None:
    """The ffmpeg to use, or None. A configured path that no longer exists is not silently replaced."""
    if configured:
        return configured if Path(configured).is_file() and is_ffmpeg_name(configured) else None
    found = shutil.which("ffmpeg")
    if found:
        return found
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            # winget adds a link here, or leaves the binary inside its package folder.
            candidates = [str(Path(base) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe")]
            candidates += sorted(glob.glob(str(Path(base) / "Microsoft" / "WinGet" / "Packages" / "*FFmpeg*" / "*" / "bin" / "ffmpeg.exe")))
            for candidate in candidates:
                if Path(candidate).is_file():
                    return candidate
    return None


def encode(frames: list[Image.Image], fps: int, dest_stem: Path, ffmpeg: str | None) -> EncodedVideo:
    """Write `frames` next to `dest_stem` (".mp4" or ".webp" is added) and say how."""
    if not frames:
        raise ValueError("no frames to encode")
    note = "ffmpeg not found"
    if ffmpeg:
        try:
            return _encode_mp4(frames, fps, dest_stem.with_suffix(".mp4"), ffmpeg)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            note = f"ffmpeg failed: {str(exc)[:200]}"
    path = dest_stem.with_suffix(".webp")
    _encode_webp(frames, fps, path)
    return EncodedVideo(path, "image/webp", f"animated WebP ({note})")


def _raw_input(size: tuple[int, int], fps: int) -> list[str]:
    """ffmpeg arguments for RGB frames arriving on stdin."""
    return ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "pipe:0"]


def _run_ffmpeg(ffmpeg: str, args: list[str], part: Path, frames: list[Image.Image] | None = None) -> None:
    """Run ffmpeg with `args` (output last, to `part`), feeding it `frames` on stdin if given.
    Raises RuntimeError with ffmpeg's own last message if it fails; `part` is gone then."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args, str(part)]
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    # stderr goes to a file: a pipe nobody reads while we write frames could fill up and deadlock.
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if frames is not None else subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=err, creationflags=flags)
        try:
            if frames is not None:
                assert proc.stdin is not None
                size = frames[0].size
                try:
                    for frame in frames:
                        if frame.size != size:
                            raise RuntimeError("frames differ in size")
                        proc.stdin.write(frame.convert("RGB").tobytes())
                    proc.stdin.close()
                except BrokenPipeError:
                    pass  # ffmpeg stopped reading; its exit code and message below say why
            code = proc.wait(timeout=FFMPEG_TIMEOUT_SECONDS)
        except BaseException:
            proc.kill()
            proc.wait()
            part.unlink(missing_ok=True)
            raise
        if code != 0:
            err.seek(0)
            message = err.read().decode("utf-8", "replace").strip().splitlines()
            part.unlink(missing_ok=True)
            raise RuntimeError(message[-1] if message else f"exit code {code}")


def _encode_mp4(frames: list[Image.Image], fps: int, path: Path, ffmpeg: str) -> EncodedVideo:
    part = path.with_name(path.name + ".part")
    _run_ffmpeg(ffmpeg, [
        *_raw_input(frames[0].size, fps),
        "-an", "-map_metadata", "-1",
        "-c:v", "libx264", "-preset", "medium", "-crf", H264_CRF, "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-f", "mp4",
    ], part, frames)
    os.replace(part, path)
    return EncodedVideo(path, "video/mp4", "H.264 MP4 (ffmpeg)")


def _encode_webp(frames: list[Image.Image], fps: int, path: Path) -> None:
    part = path.with_name(path.name + ".part")
    first, *rest = [f.convert("RGB") for f in frames]
    first.save(part, format="WEBP", save_all=True, append_images=rest,
               duration=round(1000 / fps), loop=0, quality=90, method=4)
    os.replace(part, path)


# ---------------------------------------------------------------- GIF export
#
# A copy for places that only take GIFs. A GIF holds 256 colours per frame and
# compresses poorly, so the copy is usually smaller and at a lower frame rate than
# the video. Like the video itself it carries no prompt or settings.

GIF_MIN_WIDTH = 64
GIF_MAX_FPS = 24
# One palette per frame, built from that frame and diffused into it. "diff" keeps the
# palette for what moves; "rectangle" re-encodes only the part of a frame that changed.
GIF_PALETTE = ("split[a][b];[a]palettegen=max_colors=256:stats_mode=diff[p];"
               "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle")


class GifError(Exception):
    """The GIF cannot be made; the message says why, for the user."""


def gif_size(width: int, height: int, asked: int) -> tuple[int, int]:
    """The GIF's size for a video of `width` x `height`: `asked` wide at most, never enlarged."""
    out_w = min(asked, width)
    return out_w, max(1, _round_half_up(height * out_w / width))


def _round_half_up(value: float) -> int:
    """Rounding as the window does it (Math.round), so the size it shows is the size made."""
    return int(value + 0.5)


def to_gif(source: Path, mime: str, fps: int, video_size: tuple[int, int], dest: Path, *, width: int,
           gif_fps: int, ffmpeg: str | None) -> str:
    """Write the video at `source` (its mime type, frame rate and size as stored) to `dest` as a
    looping GIF, `width` wide at most and at `gif_fps` (never more than the video's own).
    Returns what made it. Raises GifError if it cannot be done here."""
    gif_fps = max(1, min(gif_fps, fps or gif_fps, GIF_MAX_FPS))
    size = gif_size(*video_size, width)
    part = dest.with_name(dest.name + ".part")
    try:
        if mime == "video/mp4":
            if not ffmpeg:
                raise GifError("Reading an MP4 needs ffmpeg, which was not found. Choose it in Settings.")
            _run_ffmpeg(ffmpeg, [
                "-i", str(source), "-an", "-map_metadata", "-1",
                "-vf", f"fps={gif_fps},scale={size[0]}:{size[1]}:flags=lanczos,{GIF_PALETTE}",
                "-loop", "0", "-f", "gif",
            ], part)
            made = "ffmpeg"
        elif mime == "image/webp":
            frames = _webp_frames(source, fps, gif_fps, size)
            if ffmpeg:
                _run_ffmpeg(ffmpeg, [*_raw_input(frames[0].size, gif_fps), "-map_metadata", "-1",
                                     "-vf", GIF_PALETTE, "-loop", "0", "-f", "gif"], part, frames)
                made = "ffmpeg"
            else:
                _gif_with_pillow(frames, gif_fps, part)
                made = "Pillow"
        else:
            raise GifError(f"Cannot make a GIF from {mime}.")
        os.replace(part, dest)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        part.unlink(missing_ok=True)
        raise GifError(f"Making the GIF failed: {str(exc)[:200]}") from exc
    return made


def _webp_frames(source: Path, fps: int, gif_fps: int, size: tuple[int, int]) -> list[Image.Image]:
    """Every frame the GIF keeps (fewer when its frame rate is lower), resized to `size`."""
    with Image.open(source, formats=["WEBP"]) as img:
        total = getattr(img, "n_frames", 1)
        keep = max(1, _round_half_up(total * gif_fps / (fps or gif_fps)))
        frames = []
        for i in range(keep):
            img.seek(min(total - 1, int(i * total / keep)))
            frame = img.convert("RGB")
            frames.append(frame if frame.size == size else frame.resize(size, Image.Resampling.LANCZOS))
    return frames


def _gif_with_pillow(frames: list[Image.Image], fps: int, part: Path) -> None:
    """The fallback without ffmpeg: a palette per frame, Floyd-Steinberg dithered."""
    first, *rest = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
                    for f in frames]
    first.save(part, format="GIF", save_all=True, append_images=rest,
               duration=round(1000 / fps), loop=0, disposal=1, optimize=False)
