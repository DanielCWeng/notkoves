"""Facial measurements.

Every metric is a scale-free ratio or an angle, so photo resolution and camera
distance (beyond perspective effects) don't matter. Points are first rotated
so the eye line is horizontal ("roll-corrected"), which makes "vertical" mean
vertical relative to the face, not the photo.

Reference values are classic *canons* or published means, shown for context.
They are not targets: real attractive faces routinely miss the canons
(Farkas et al. found the neoclassical canons rarely hold), and population
norms differ by sex, age and ancestry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import landmark_ids as L
from .detect import Face
from .pose import IRIS_DIAMETER_MM


def normalise(pts: np.ndarray) -> np.ndarray:
    """Rotate so the eye centres are level, origin at their midpoint."""
    re = pts[L.R_EYE_CONTOUR].mean(0)
    le = pts[L.L_EYE_CONTOUR].mean(0)
    ang = math.atan2(le[1] - re[1], le[0] - re[0])
    c, s = math.cos(-ang), math.sin(-ang)
    R = np.array([[c, -s], [s, c]])
    mid = (re + le) / 2
    return (pts - mid) @ R.T


def _d(p, a, b):
    return float(np.linalg.norm(p[a] - p[b]))


def _dy(p, a, b):
    return float(abs(p[a][1] - p[b][1]))


def _stomion(p):
    return (p[L.STOMION_UP] + p[L.STOMION_LO]) / 2


def _eye_width(p):
    return (_d(p, L.R_EX, L.R_EN) + _d(p, L.L_EX, L.L_EN)) / 2


def _iod(p):
    return float(np.linalg.norm(p[L.R_EYE_CONTOUR].mean(0) - p[L.L_EYE_CONTOUR].mean(0)))


def _tilt(p, outer, inner):
    """Canthal tilt in degrees; + means the outer corner sits higher."""
    o, i = p[outer], p[inner]
    return math.degrees(math.atan2(i[1] - o[1], abs(i[0] - o[0])))


@dataclass
class Metric:
    key: str
    name: str
    kind: str                       # "ratio" | "angle"
    fn: Callable[[np.ndarray], float]
    what: str
    reference: str
    group: str


def _midface_lower(p):
    sn = p[L.SUBNASALE]
    return abs(sn[1] - p[L.NASION][1]) / abs(p[L.MENTON][1] - sn[1])


def _lip_chin(p):
    st = _stomion(p)
    return abs(st[1] - p[L.SUBNASALE][1]) / abs(p[L.MENTON][1] - st[1])


def _fwhr(p):
    brow_y = (p[L.R_BROW_PEAK][1] + p[L.L_BROW_PEAK][1]) / 2
    return _d(p, L.R_ZYGION, L.L_ZYGION) / abs(p[L.LABRALE_SUP][1] - brow_y)


def _brow_height(p):
    r = abs(p[L.R_EYE_CONTOUR].mean(0)[1] - p[L.R_BROW_PEAK][1])
    l = abs(p[L.L_EYE_CONTOUR].mean(0)[1] - p[L.L_BROW_PEAK][1])
    return (r + l) / 2 / _iod(p)


def _eye_open(p):
    r = _d(p, L.R_LID_UP, L.R_LID_LO) / _d(p, L.R_EX, L.R_EN)
    l = _d(p, L.L_LID_UP, L.L_LID_LO) / _d(p, L.L_EX, L.L_EN)
    return (r + l) / 2


def _lip_ratio(p):
    up = _d(p, L.LABRALE_SUP, L.STOMION_UP)
    lo = _d(p, L.STOMION_LO, L.LABRALE_INF)
    return up / lo if lo > 0 else float("nan")


METRICS: list[Metric] = [
    # --- vertical proportions ---
    Metric("midface_lower", "Midface : lower face", "ratio", _midface_lower,
           "Nasion→subnasale height divided by subnasale→chin height (the middle and lower facial thirds).",
           "Neoclassical canon = 1.00 (equal thirds). Most adults measure below 1, i.e. a slightly longer lower third.",
           "Vertical proportions"),
    Metric("lip_chin", "Upper lip : lip-to-chin", "ratio", _lip_chin,
           "Subnasale→mouth line divided by mouth line→chin.",
           "Classic orthodontic norm ≈ 0.50 (lower third split 1:2).",
           "Vertical proportions"),
    Metric("facial_index", "Facial index", "ratio",
           lambda p: 100 * _dy(p, L.NASION, L.MENTON) / _d(p, L.R_ZYGION, L.L_ZYGION),
           "Face height (nasion→chin) as a % of face width at the side outline. Overall long vs broad.",
           "Martin's classes (<85 broad, 85–90 medium, >90 long) use cheekbone width. This mesh width sits nearer the ears and runs ~5–10% wider, so values read lower (broader) than Martin's scale. Track it against yourself.",
           "Vertical proportions"),
    # --- horizontal proportions ---
    Metric("intercanthal_eye", "Eye spacing : eye width", "ratio",
           lambda p: _d(p, L.R_EN, L.L_EN) / _eye_width(p),
           "Gap between inner eye corners divided by the width of one eye.",
           "Rule-of-fifths canon = 1.00.",
           "Horizontal proportions"),
    Metric("iod_face", "Eye distance : face width", "ratio",
           lambda p: _iod(p) / _d(p, L.R_ZYGION, L.L_ZYGION),
           "Distance between eye centres divided by face width at the side outline.",
           "Pallett, Link & Lee (2010) found ~0.46 most attractive, measuring face width at the ears, roughly where this mesh width sits, so the comparison is approximately like-for-like.",
           "Horizontal proportions"),
    Metric("nose_intercanthal", "Nose width : eye spacing", "ratio",
           lambda p: _d(p, L.R_ALA, L.L_ALA) / _d(p, L.R_EN, L.L_EN),
           "Width across the nostril wings divided by the gap between inner eye corners.",
           "Neoclassical canon = 1.00. Varies substantially by ancestry; very sensitive to camera distance.",
           "Horizontal proportions"),
    Metric("mouth_nose", "Mouth width : nose width", "ratio",
           lambda p: _d(p, L.R_CHEILION, L.L_CHEILION) / _d(p, L.R_ALA, L.L_ALA),
           "Corner-to-corner mouth width divided by nose width.",
           "Often-quoted ideal ≈ 1.5; weak evidence. Expression changes it a lot.",
           "Horizontal proportions"),
    Metric("jaw_face", "Jaw width : face width", "ratio",
           lambda p: _d(p, L.R_GONION, L.L_GONION) / _d(p, L.R_ZYGION, L.L_ZYGION),
           "Width at the jaw angles divided by face width at the side outline. Higher = squarer lower face.",
           "Sexually dimorphic (higher in men on average). Jaw-angle landmarks are the least reliable in the mesh.",
           "Horizontal proportions"),
    Metric("fwhr", "Facial width-to-height (fWHR)", "ratio", _fwhr,
           "Face width (side outline) divided by brow-to-upper-lip height.",
           "Popular in psychology research; definitions vary, so compare only against your own past scans.",
           "Horizontal proportions"),
    # --- features ---
    Metric("canthal_tilt", "Canthal tilt (mean)", "angle",
           lambda p: (_tilt(p, L.R_EX, L.R_EN) + _tilt(p, L.L_EX, L.L_EN)) / 2,
           "Angle of the line from inner to outer eye corner. Positive = outer corner higher.",
           "Most adults are mildly positive; averages differ by sex and ancestry. Very sensitive to head tilt and squinting.",
           "Eyes & brows"),
    Metric("canthal_tilt_r", "Canthal tilt (your right)", "angle",
           lambda p: _tilt(p, L.R_EX, L.R_EN), "Right eye only.", "Compare with left for asymmetry.",
           "Eyes & brows"),
    Metric("canthal_tilt_l", "Canthal tilt (your left)", "angle",
           lambda p: _tilt(p, L.L_EX, L.L_EN), "Left eye only.", "Compare with right for asymmetry.",
           "Eyes & brows"),
    Metric("eye_open", "Eye openness", "ratio", _eye_open,
           "Lid-to-lid height divided by eye width.",
           "No canon. Strongly affected by expression, lighting and fatigue.",
           "Eyes & brows"),
    Metric("brow_height", "Brow height", "ratio", _brow_height,
           "Height of the brow peak above the eye centre, relative to the distance between the eyes.",
           "No canon. Higher brows read as more feminine; lower, straighter brows as more masculine.",
           "Eyes & brows"),
    Metric("lip_ratio", "Upper : lower lip", "ratio", _lip_ratio,
           "Upper-lip vermilion height divided by lower-lip vermilion height (mouth closed).",
           "Commonly quoted targets range from about 1:1.6 to 1:2 (0.5–0.6); evidence is thin.",
           "Nose & mouth"),
]

METRIC_BY_KEY = {m.key: m for m in METRICS}


def compute(face: Face) -> dict[str, float]:
    p = normalise(face.pts)
    out = {}
    for m in METRICS:
        try:
            v = float(m.fn(p))
        except Exception:
            v = float("nan")
        out[m.key] = v
    return out


def mm_measurements(face: Face, pd_mm: float | None = None,
                    distance_m: float | None = None) -> dict[str, float]:
    """Approximate real-world sizes.

    Scale comes from the user's measured PD if given (best), otherwise from
    the average iris diameter (needs trustworthy iris landmarks).
    """
    from .pose import photo_pd, pupil_px

    p = normalise(face.pts)
    iris_px = (np.mean([_d(p, L.R_IRIS_RING[0], L.R_IRIS_RING[2]),
                        _d(p, L.L_IRIS_RING[0], L.L_IRIS_RING[2])])
               if face.iris_ok else None)
    extra = {}
    if pd_mm:
        k = photo_pd(pd_mm, distance_m) / pupil_px(face)
        if iris_px:
            extra["Iris diameter (implied)"] = iris_px * k
    elif iris_px:
        k = IRIS_DIAMETER_MM / iris_px
    else:
        return {}
    pupil = "Interpupillary distance (in photo)" if pd_mm else "Interpupillary distance"
    out = {
        pupil: pupil_px(face) * k,
        "Face width (side outline)": _d(p, L.R_ZYGION, L.L_ZYGION) * k,
        "Jaw width": _d(p, L.R_GONION, L.L_GONION) * k,
        "Nose width": _d(p, L.R_ALA, L.L_ALA) * k,
        "Mouth width": _d(p, L.R_CHEILION, L.L_CHEILION) * k,
        "Nasion to chin": _dy(p, L.NASION, L.MENTON) * k,
    }
    out.update(extra)
    return out
