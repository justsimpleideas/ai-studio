"""The real image engine, with the test-only fake backends in place of the models.

Accepts exactly the engine's own arguments, so the studio can start it in place
of the real engine for end-to-end tests without a GPU or model weights:

    STUDIO_IMAGE_ENGINE_MODULE=tests.ui_harness  (read by the studio's main process)

It produces flat placeholder gradients (moving ones for video) and is not part of
the application. Set HARNESS_STEP_DELAY (seconds per step, default 0.15) to slow
it down.
"""

from __future__ import annotations

import os
import sys

from image_engine.__main__ import main

from .fakes import FakeBackend, FakeVideoBackend


def _delay() -> float:
    return float(os.environ.get("HARNESS_STEP_DELAY", "0.15"))


def fake_backend(_get_settings):
    return FakeBackend(step_delay=_delay())


def fake_video_backend(_get_settings):
    return FakeVideoBackend(step_delay=_delay())


if __name__ == "__main__":
    sys.exit(main(backend_factory=fake_backend, video_backend_factory=fake_video_backend))
