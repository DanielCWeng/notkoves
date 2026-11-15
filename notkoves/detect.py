"""Image loading and landmark detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import landmark_ids as L

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}

try:  # optional iPhone HEIC support
    from pillow_heif import register_heif_opener  # type: ignore

    register_heif_opener()
    HEIC_OK = True
except Exception:  # pragma: no cover
    HEIC_OK = False


@dataclass
class Photo:
    path: Path
    rgb: np.ndarray                    # H x W x 3 uint8, EXIF-rotated
    focal_35mm: float | None = None    # EXIF FocalLengthIn35mmFilm, if present
    focal_assumed: bool = False        # True if supplied by --focal-35mm, not EXIF


@dataclass
class Face:
    photo: Photo
    pts: np.ndarray                    # 478 x 2, pixel coordinates
    z: np.ndarray                      # 478 relative depth (may be all zeros)
    iris_ok: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.photo.rgb.shape[:2]
        return w, h


def list_images(folder: Path) -> list[Path]:
    if folder.is_file():
        return [folder]
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))


def load_photo(path: Path, max_side: int = 2000) -> Photo:
    if path.suffix.lower() in {".heic", ".heif"} and not HEIC_OK:
        raise RuntimeError("HEIC needs `pip install pillow-heif` (or export as JPEG).")
    im = Image.open(path)
    focal = None
    try:
        exif = im.getexif()
        sub = exif.get_ifd(0x8769)  # Exif sub-IFD
        f35 = sub.get(0xA405)       # FocalLengthIn35mmFilm
        if f35:
            focal = float(f35)
    except Exception:
        pass
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return Photo(path=path, rgb=np.asarray(im).copy(), focal_35mm=focal)


class Detector:
    """MediaPipe face landmarks.

    backend="tasks"  : FaceLandmarker Tasks API (current MediaPipe; needs a .task model)
    backend="legacy" : mp.solutions.face_mesh (MediaPipe <= 0.10.14; models built in)
    Both return the same 478-point topology.
    """

    def __init__(self, model_file: Path | None = None, backend: str = "tasks"):
        import mediapipe as mp

        self._mp = mp
        self.backend = backend
        if backend == "legacy":
            self._lm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=2,
                refine_landmarks=True, min_detection_confidence=0.5)
            return
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_file)),
            num_faces=2,
            min_face_detection_confidence=0.5,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def close(self) -> None:
        try:
            self._lm.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _raw(self, photo: Photo):
        if self.backend == "legacy":
            res = self._lm.process(photo.rgb)
            return [f.landmark for f in (res.multi_face_landmarks or [])]
        mp = self._mp
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=photo.rgb)
        return self._lm.detect(img).face_landmarks or []

    def detect(self, photo: Photo) -> Face | None:
        faces = self._raw(photo)
        if not faces:
            return None
        h, w = photo.rgb.shape[:2]
        # if several faces, keep the largest
        best = max(faces, key=lambda f: (max(p.x for p in f) - min(p.x for p in f)))
        arr = np.array([[p.x * w, p.y * h, p.z * w] for p in best], dtype=float)
        face = Face(photo=photo, pts=arr[:, :2], z=arr[:, 2])
        if len(faces) > 1:
            face.notes.append("More than one face found; used the largest.")
        face.iris_ok = _iris_valid(face.pts)
        return face


def _iris_valid(pts: np.ndarray) -> bool:
    """Iris points are only trustworthy if each iris centre sits inside its eye
    and the iris ring has a plausible size relative to the eye width."""
    if len(pts) < 478:
        return False
    for centre, ring, inner, outer in (
        (L.R_IRIS_CENTRE, L.R_IRIS_RING, L.R_EN, L.R_EX),
        (L.L_IRIS_CENTRE, L.L_IRIS_RING, L.L_EN, L.L_EX),
    ):
        c = pts[centre]
        a, b = pts[inner], pts[outer]
        eye_w = np.linalg.norm(a - b)
        if eye_w < 5:
            return False
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        if not (lo[0] - 0.05 * eye_w <= c[0] <= hi[0] + 0.05 * eye_w):
            return False
        diam = np.linalg.norm(pts[ring[0]] - pts[ring[2]])
        # visible iris is ~35-45% of palpebral fissure width in adults
        if not (0.25 * eye_w <= diam <= 0.6 * eye_w):
            return False
    return True
