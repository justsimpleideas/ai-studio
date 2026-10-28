"""Synthetic photos for the end-to-end test's Train tab (scripts/e2e-studio.mjs).

Usage: python e2e-training-photos.py <folder>

Writes <folder>/Example_Motors/Coupe_GT/ with four distinct pictures, one exact
copy (a duplicate) and one too small to train on. Blocks of random colour, so
no two of the four look alike to the duplicate check. Needs Pillow.
"""

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

DISTINCT = 4


def picture(seed: int, size=(640, 512)) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", size, tuple(rng.randrange(256) for _ in range(3)))
    draw = ImageDraw.Draw(img)
    for _ in range(24):
        x, y = rng.randrange(size[0]), rng.randrange(size[1])
        w, h = rng.randrange(40, 260), rng.randrange(40, 200)
        draw.rectangle((x, y, x + w, y + h), fill=tuple(rng.randrange(256) for _ in range(3)))
    return img


def main() -> None:
    folder = Path(sys.argv[1]) / "Example_Motors" / "Coupe_GT"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(DISTINCT):
        picture(i).save(folder / f"photo-{i}.jpg", quality=92)
    (folder / "photo-0-copy.jpg").write_bytes((folder / "photo-0.jpg").read_bytes())
    picture(99, (200, 160)).save(folder / "thumbnail.jpg")


if __name__ == "__main__":
    main()
