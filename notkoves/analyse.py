"""Run the pipeline over a set of photos and aggregate across them."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import metrics as M
from . import pose as P
from . import symmetry as S
from .detect import Detector, Face, Photo, list_images, load_photo


@dataclass
class PhotoResult:
    path: Path
    face: Face | None
    pose: P.Pose | None = None
    rejected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    symmetry: dict = field(default_factory=dict)
    mm: dict[str, float] = field(default_factory=dict)

    @property
    def used(self) -> bool:
        return self.face is not None and not self.rejected


@dataclass
class Summary:
    mean: float
    sd: float
    n: int
    stability: str


def stability(kind: str, values: list[float]) -> Summary:
    v = np.array([x for x in values if not math.isnan(x)])
    n = len(v)
    if n == 0:
        return Summary(float("nan"), float("nan"), 0, "n/a")
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    if n < 3:
        label = "too few photos"
    elif kind == "angle":
        label = "stable" if sd < 1.0 else "moderate" if sd < 2.5 else "noisy"
    elif kind == "pct":  # small % of eye distance: judge by absolute spread
        label = "stable" if sd < 0.75 else "moderate" if sd < 1.5 else "noisy"
    else:
        cv = sd / abs(mean) * 100 if mean else float("inf")
        label = "stable" if cv < 2 else "moderate" if cv < 5 else "noisy"
    return Summary(mean, sd, n, label)


def run(folder: Path, model_file: Path | None, gate: P.Gate | None = None,
        backend: str = "tasks") -> list[PhotoResult]:
    gate = gate or P.Gate()
    results: list[PhotoResult] = []
    paths = list_images(folder)
    if not paths:
        raise SystemExit(f"No images found in {folder}")
    with Detector(model_file, backend=backend) as det:
        for path in paths:
            try:
                photo = load_photo(path)
            except Exception as exc:
                results.append(PhotoResult(path, None, rejected=[f"could not open: {exc}"]))
                continue
            face = det.detect(photo)
            if face is None:
                results.append(PhotoResult(path, None, rejected=["no face found"]))
                continue
            r = PhotoResult(path, face, warnings=list(face.notes))
            r.pose = P.estimate(face)
            bad, warn = gate.check(r.pose)
            r.rejected += bad
            r.warnings += warn
            # Analyse the mirror image too: averaging cancels any left/right
            # bias in the landmark model itself.
            mphoto = Photo(path, np.ascontiguousarray(photo.rgb[:, ::-1]), photo.focal_35mm)
            mface = det.detect(mphoto)
            r.metrics = M.compute(face)
            if mface is not None:
                mm_ = M.compute(mface)
                swap = {"canthal_tilt_r": "canthal_tilt_l", "canthal_tilt_l": "canthal_tilt_r"}
                r.metrics = {k: (v + mm_[swap.get(k, k)]) / 2 for k, v in r.metrics.items()}
            else:
                r.warnings.append("mirror pass failed; landmark-model bias not cancelled")
            r.symmetry = S.analyse(face, mface)
            r.mm = M.mm_measurements(face)
            results.append(r)
    return results


def aggregate(results: list[PhotoResult]) -> dict:
    used = [r for r in results if r.used]
    out = {"metrics": {}, "symmetry": {}, "regions": {}, "side": {}, "mm": {}}
    for m in M.METRICS:
        out["metrics"][m.key] = stability(m.kind, [r.metrics[m.key] for r in used])
    out["symmetry"] = stability("pct", [r.symmetry["overall"] for r in used])
    out["debiased"] = bool(used) and all(r.symmetry.get("debiased") for r in used)
    if used:
        for reg in used[0].symmetry["regions"]:
            out["regions"][reg] = stability("pct", [r.symmetry["regions"][reg] for r in used])
        for k in used[0].symmetry["side"]:
            out["side"][k] = stability("pct", [r.symmetry["side"][k] for r in used])
        mm_keys = {k for r in used for k in r.mm}
        for k in sorted(mm_keys):
            out["mm"][k] = stability("ratio", [r.mm[k] for r in used if k in r.mm])
    return out
