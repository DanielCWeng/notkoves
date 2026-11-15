"""Locate (and on first run, download) the MediaPipe FaceLandmarker model."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "notkoves"


def model_path(explicit: str | None = None) -> Path:
    """Return a usable model file, downloading the official one if needed.

    Order: --model argument, NOTKOVES_MODEL env var, cached download.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Model file not found: {p}")
        return p
    env = os.environ.get("NOTKOVES_MODEL")
    if env:
        return model_path(env)

    target = cache_dir() / "face_landmarker.task"
    if target.is_file() and target.stat().st_size > 1_000_000:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    print(f"Downloading face landmark model (~3.6 MB) to {target} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Could not download the MediaPipe face landmark model.\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            "and pass it with --model PATH (or set NOTKOVES_MODEL)."
        ) from exc
    tmp.replace(target)
    return target
