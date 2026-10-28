"""Aspect-ratio buckets: the sizes photos are trained at.

Photos come in every shape. Squashing them all to one square would teach the
model squashed cars, and cropping them all square would cut off the ends of a
side view. Instead each photo goes to the bucket whose shape is closest to its
own, and is scaled to cover it and centre-cropped by at most a sliver.

Every bucket has at most target_side² pixels, so memory use is the same
whichever bucket a training step draws from.
"""

from __future__ import annotations

import math

from ..params import SIDE_MULTIPLE

DEFAULT_TARGET_SIDE = 768
MAX_RATIO = 2.0


def make_buckets(target_side: int = DEFAULT_TARGET_SIDE, multiple: int = SIDE_MULTIPLE,
                 max_ratio: float = MAX_RATIO) -> list[tuple[int, int]]:
    if target_side % multiple:
        raise ValueError(f"target_side must be a multiple of {multiple}")
    target = target_side * target_side
    buckets = set()
    for width in range(multiple, int(target_side * math.sqrt(max_ratio)) + 1, multiple):
        height = target // width // multiple * multiple
        if height >= multiple and max(width / height, height / width) <= max_ratio:
            buckets.add((width, height))
            buckets.add((height, width))
    return sorted(buckets)


def assign(width: int, height: int, buckets: list[tuple[int, int]]) -> tuple[int, int]:
    """The bucket closest in shape; on a tie, the larger one."""
    ratio = math.log(width / height)
    return min(buckets, key=lambda b: (abs(math.log(b[0] / b[1]) - ratio), -b[0] * b[1]))


def crop_plan(width: int, height: int, bucket: tuple[int, int]) -> tuple[int, int, int, int]:
    """(scaled_width, scaled_height, left, top): scale to cover the bucket, then centre-crop."""
    bw, bh = bucket
    scale = max(bw / width, bh / height)
    sw, sh = max(bw, round(width * scale)), max(bh, round(height * scale))
    return sw, sh, (sw - bw) // 2, (sh - bh) // 2
