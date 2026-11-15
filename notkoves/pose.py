"""Head-pose and camera-distance estimation, used to reject unusable photos.

Head turn (yaw) creates fake asymmetry and squashes widths; head tilt up/down
(pitch) distorts every vertical ratio. Close camera distance enlarges the
centre of the face (nose) relative to the edges. These checks matter more than
any metric downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import landmark_ids as L
from .detect import Face

# MediaPipe canonical face model (468 vertices, cm; x to subject's left,
# y up, z towards the viewer). Apache-2.0, (c) Google LLC; see data/NOTICE.
_CANON = np.loadtxt(Path(__file__).parent / "data" / "canonical_face_model.txt")
# Fit pose on rigid regions only: skip lips/mouth and lower jaw, which move
# with expression and would bias the head-pose estimate.
_MOBILE = {0, 11, 12, 13, 14, 15, 16, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91,
           95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317,
           318, 321, 324, 375, 402, 405, 409, 415,
           # lower face / chin outline
           32, 140, 148, 149, 150, 152, 170, 171, 175, 176, 199, 200, 208, 211, 262,
           369, 377, 378, 379, 395, 396, 400, 421, 428, 431}
_RIGID = np.array([i for i in range(468) if i not in _MOBILE])

AVG_BIZYGOMATIC_MM = 135.0   # rough adult average (men ~140, women ~130)
IRIS_DIAMETER_MM = 11.7      # horizontal visible iris diameter, adult mean (SD ~0.5)


@dataclass
class Pose:
    yaw: float      # degrees, + = face turned toward the subject's right
    pitch: float    # degrees, + = chin up
    roll: float     # degrees, from the eye line
    distance_m: float | None
    distance_basis: str


def estimate(face: Face) -> Pose:
    p = face.pts
    w, h = face.size
    f_px = float(max(w, h))  # assumption when EXIF focal length is absent
    if face.photo.focal_35mm:
        f_px = face.photo.focal_35mm / 36.0 * max(w, h)

    cam = np.array([[f_px, 0, w / 2], [0, f_px, h / 2], [0, 0, 1]], dtype=float)
    img_pts = p[_RIGID].astype(float)
    # camera frame: y down, z away from camera (a 180° turn about x)
    model = _CANON[_RIGID] * np.array([1.0, -1.0, -1.0])
    yaw = pitch = float("nan")
    ok, rvec, tvec = cv2.solvePnP(model, img_pts, cam, None, flags=cv2.SOLVEPNP_SQPNP)
    if ok:
        rvec, tvec = cv2.solvePnPRefineLM(model, img_pts, cam, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        ang = cv2.RQDecomp3x3(R)[0]
        pitch, yaw = _wrap(-ang[0]), _wrap(ang[1])

    re = p[L.R_EYE_CONTOUR].mean(0)
    le = p[L.L_EYE_CONTOUR].mean(0)
    roll = math.degrees(math.atan2(le[1] - re[1], le[0] - re[0]))

    # camera distance from a known physical size
    dist, basis = None, ""
    if face.photo.focal_35mm:
        if face.iris_ok:
            iris_px = np.mean([
                np.linalg.norm(p[L.R_IRIS_RING[0]] - p[L.R_IRIS_RING[2]]),
                np.linalg.norm(p[L.L_IRIS_RING[0]] - p[L.L_IRIS_RING[2]]),
            ])
            dist = f_px * IRIS_DIAMETER_MM / iris_px / 1000
            basis = "iris size + EXIF focal length"
        else:
            zyg_px = np.linalg.norm(p[L.R_ZYGION] - p[L.L_ZYGION])
            dist = f_px * AVG_BIZYGOMATIC_MM / zyg_px / 1000
            basis = "average face width + EXIF focal length (±10%)"
    if basis and face.photo.focal_assumed:
        basis = basis.replace("EXIF focal length", "assumed focal length")
    return Pose(yaw=yaw, pitch=pitch, roll=roll, distance_m=dist, distance_basis=basis)


def _wrap(a: float) -> float:
    while a > 90:
        a -= 180
    while a < -90:
        a += 180
    return a


@dataclass
class Gate:
    max_yaw: float = 8.0
    max_pitch: float = 15.0
    max_roll: float = 20.0
    min_distance_m: float = 0.9

    def check(self, pose: Pose) -> tuple[list[str], list[str]]:
        """Return (reasons to reject, warnings)."""
        bad, warn = [], []
        if not math.isnan(pose.yaw) and abs(pose.yaw) > self.max_yaw:
            bad.append(f"head turned {abs(pose.yaw):.0f}° (limit {self.max_yaw:.0f}°)")
        if not math.isnan(pose.pitch) and abs(pose.pitch) > self.max_pitch:
            bad.append(f"head tilted up/down ~{abs(pose.pitch):.0f}° (limit {self.max_pitch:.0f}°)")
        if abs(pose.roll) > self.max_roll:
            bad.append(f"head tilted sideways {abs(pose.roll):.0f}°")
        if pose.distance_m is not None and pose.distance_m < self.min_distance_m:
            warn.append(
                f"camera ~{pose.distance_m:.2f} m away: close-range perspective enlarges "
                "the nose and shrinks the face edges. Shoot from ~1.5 m with zoom."
            )
        if pose.distance_m is None:
            warn.append("no EXIF focal length, so camera distance can't be checked "
                        "(pass --focal-35mm, e.g. 23 for recent iPhone front cameras)")
        return bad, warn
