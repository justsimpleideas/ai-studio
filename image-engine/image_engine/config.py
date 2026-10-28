"""Where things live, and the settings the user can change from the UI."""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_PORT = 5198
APP_DIR_NAME = "LocalAIStudio"
MEMORY_MODES = ("auto", "full", "quality", "balanced", "low")
PATH_KEYS = ("model_dir", "video_model_dir", "ffmpeg_path", "caption_model_dir")


def default_data_dir() -> Path:
    env = os.environ.get("LIS_DATA_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "local-ai-studio"


@dataclass
class Settings:
    model_dir: str = ""
    memory_mode: str = "auto"
    load_on_start: bool = False
    video_model_dir: str = ""
    video_memory_mode: str = "auto"
    # Empty: look for ffmpeg on PATH and where winget puts it (video_io.find_ffmpeg).
    ffmpeg_path: str = ""
    # The vision-language model that describes training photos (backends/captioner.py).
    caption_model_dir: str = ""

    def validate(self) -> None:
        for key in ("memory_mode", "video_memory_mode"):
            if getattr(self, key) not in MEMORY_MODES:
                raise ValueError(f"{key} must be one of {', '.join(MEMORY_MODES)}")
        if not isinstance(self.load_on_start, bool):
            raise ValueError("load_on_start must be true or false")
        for key in PATH_KEYS:
            if not isinstance(getattr(self, key), str):
                raise ValueError(f"{key} must be a string")


class SettingsStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.path = data_dir / "settings.json"
        self._lock = threading.Lock()
        self._settings = self._read()

    def _defaults(self) -> Settings:
        models = self.data_dir / "models"
        return Settings(model_dir=str(models / "image-model"), video_model_dir=str(models / "video-model"),
                        caption_model_dir=str(models / "caption-model"))

    def _read(self) -> Settings:
        settings = self._defaults()
        if self.path.exists():
            try:
                stored = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stored = {}
            known = {f.name for f in fields(Settings)}
            for key, value in stored.items():
                if key in known:
                    setattr(settings, key, value)
            try:
                settings.validate()
            except ValueError:
                # Keep the folders that are still readable; everything else goes back to defaults.
                fallback = self._defaults()
                for key in PATH_KEYS:
                    if isinstance(getattr(settings, key), str):
                        setattr(fallback, key, getattr(settings, key))
                settings = fallback
        return settings

    def get(self) -> Settings:
        with self._lock:
            return Settings(**asdict(self._settings))

    def update(self, changes: dict) -> Settings:
        with self._lock:
            known = {f.name for f in fields(Settings)}
            unknown = set(changes) - known
            if unknown:
                raise ValueError(f"unknown setting: {', '.join(sorted(unknown))}")
            candidate = Settings(**{**asdict(self._settings), **changes})
            candidate.validate()
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(candidate), indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
            self._settings = candidate
            return Settings(**asdict(candidate))
