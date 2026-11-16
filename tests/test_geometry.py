"""Geometry tests on a synthetic face projected from MediaPipe's canonical model.

No images or ML model needed: we build landmark arrays directly.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from notkoves import landmark_ids as L
from notkoves import metrics as M
from notkoves import pose as P
from notkoves import symmetry as S
from notkoves.detect import Face, Photo

W = H = 1000
F = 1000.0
CAM = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
MODEL = P._CANON * np.array([1.0, -1.0, -1.0])


def project(yaw=0.0, pitch=0.0, dist=60.0, model=MODEL):
    ry, rx = math.radians(yaw), math.radians(pitch)
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    pts, _ = cv2.projectPoints(model, cv2.Rodrigues(Rx @ Ry)[0], np.array([0, 0, dist]), CAM, None)
    pts = pts.reshape(-1, 2)
    # fake iris points at the eye centres so arrays have the full 478 rows
    iris = np.vstack([np.repeat(pts[L.R_EYE_CONTOUR].mean(0, keepdims=True), 5, 0),
                      np.repeat(pts[L.L_EYE_CONTOUR].mean(0, keepdims=True), 5, 0)])
    return np.vstack([pts, iris])


def face(pts):
    return Face(photo=Photo(Path("synthetic"), np.zeros((H, W, 3), np.uint8)),
                pts=pts.copy(), z=np.zeros(len(pts)))


def mirror_pts(pts):
    """What an ideal detector returns on the horizontally flipped image."""
    out = pts.copy()
    flipped = pts.copy()
    flipped[:, 0] = W - flipped[:, 0]
    for i in range(468):
        out[i] = flipped[_partner468(i)]
    out[468:473], out[473:478] = flipped[473:478], flipped[468:473]
    return out


_C = P._CANON
_MIR = _C * np.array([-1, 1, 1])
_PART = np.array([int(np.argmin(np.linalg.norm(_MIR - _C[i], axis=1))) for i in range(468)])


def _partner468(i):
    return _PART[i]


def transform(pts, angle_deg=0.0, scale=1.0, shift=(0.0, 0.0)):
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.array([W / 2, H / 2])
    return (pts - c) @ R.T * scale + c + np.array(shift)


# --------------------------------------------------------------------------

def test_symmetric_face_has_no_asymmetry():
    res = S.analyse(face(project()))
    assert res["overall"] < 0.05
    for v in res["side"].values():
        assert abs(v) < 0.05


def test_metrics_invariant_to_rotation_scale_translation():
    base = M.compute(face(project()))
    moved = M.compute(face(transform(project(), angle_deg=12, scale=0.6, shift=(40, -25))))
    for k, v in base.items():
        assert moved[k] == pytest.approx(v, rel=1e-6, abs=1e-6), k


def test_symmetric_canthal_tilts_match():
    m = M.compute(face(project()))
    assert m["canthal_tilt_r"] == pytest.approx(m["canthal_tilt_l"], abs=1e-6)


def test_raised_left_brow_detected():
    pts = project()
    iod = np.linalg.norm(pts[L.R_EYE_CONTOUR].mean(0) - pts[L.L_EYE_CONTOUR].mean(0))
    pts[L.L_BROW_PEAK, 1] -= 0.05 * iod  # 5% IOD higher (image y is down)
    res = S.analyse(face(pts))
    assert res["side"]["Brow peak height (L − R)"] == pytest.approx(5.0, abs=0.1)
    assert res["regions"]["Brows"] > res["regions"]["Eyes"]


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (10, 0), (-10, 0), (0, 12), (0, -12), (6, -8)])
def test_pose_recovered(yaw, pitch):
    po = P.estimate(face(project(yaw, pitch)))
    assert po.yaw == pytest.approx(yaw, abs=0.5)
    # P.estimate reports pitch with + = chin up; Rx(+) in project() tips the chin down
    assert po.pitch == pytest.approx(-pitch, abs=0.5)


def test_pose_gate_rejects_turned_face():
    bad, _ = P.Gate(max_yaw=8).check(P.estimate(face(project(yaw=15))))
    assert any("turned" in b for b in bad)


def test_mirror_pass_cancels_model_bias():
    true = project()
    iod = np.linalg.norm(true[L.R_EYE_CONTOUR].mean(0) - true[L.L_EYE_CONTOUR].mean(0))
    true[L.L_CHEILION, 1] -= 0.03 * iod  # real asymmetry: left mouth corner 3% higher

    orig = true.copy()
    mirr = mirror_pts(true)
    # a label-tied model bias: the detector always places landmark 172 4% IOD too high
    for arr in (orig, mirr):
        arr[L.R_GONION, 1] -= 0.04 * iod

    naive = S.analyse(face(orig))
    fixed = S.analyse(face(orig), face(mirr))
    assert abs(naive["side"]["Jaw angle height (L − R)"]) > 1.5
    assert fixed["side"]["Jaw angle height (L − R)"] == pytest.approx(0.0, abs=0.15)
    assert fixed["side"]["Mouth corner height (L − R)"] == pytest.approx(3.0, abs=0.15)
    assert fixed["model_bias"]["Jaw angle height (L − R)"] == pytest.approx(-4.0, abs=0.15)


def test_symmetry_pairs_are_true_mirrors():
    for prs in L.SYMMETRY_PAIRS.values():
        for a, b in prs:
            assert _PART[a] == b
    for i in L.MIDLINE:
        assert abs(_C[i, 0]) < 1e-6


def test_pd_scaling_and_distance():
    pts = project(dist=100.0)  # canonical units are cm -> 1 m away
    f = face(pts)
    f.photo.focal_35mm = 36.0  # f_px = max(W, H) = 1000, matching CAM
    # canonical eye-centre distance in mm, treated as the "measured" PD
    eye_c = np.linalg.norm(P._CANON[L.R_EYE_CONTOUR].mean(0) - P._CANON[L.L_EYE_CONTOUR].mean(0)) * 10
    pd = eye_c / (1 - P.EYE_ROTATION_MM / 1000)  # so that photo_pd at 1 m == eye_c
    po = P.estimate(f, pd_mm=pd)
    # distance is to the eyes, which sit in front of the canonical origin
    eye_z_cm = P._CANON[L.R_EYE_CONTOUR + L.L_EYE_CONTOUR, 2].mean()
    assert po.distance_m == pytest.approx((100.0 - eye_z_cm) / 100, rel=0.01)
    mm = M.mm_measurements(f, pd_mm=pd, distance_m=po.distance_m)
    assert mm["Interpupillary distance (in photo)"] == pytest.approx(eye_c, rel=0.01)


def test_photo_pd_vergence():
    assert P.photo_pd(66, 1.0) == pytest.approx(66 - 0.792, abs=1e-6)
    assert P.photo_pd(66, 1.0) < P.photo_pd(66, 2.0) < 66


def test_baseline_matches_frontal_canonical():
    b = M.baseline()
    ref = M.compute(face(project(dist=200.0)))
    for k, v in ref.items():
        assert b[k] == pytest.approx(v, rel=1e-6, abs=1e-6), k
    # mesh 'nasion' sits below anatomical nasion, so the neutral face reads ~0.74, not 1.0
    assert 0.65 < b["midface_lower"] < 0.8
