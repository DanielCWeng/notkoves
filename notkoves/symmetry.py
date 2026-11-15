"""Bilateral symmetry by mirrored Procrustes alignment.

Standard geometric-morphometrics approach: reflect the landmark configuration,
swap left/right labels, align the mirror copy to the original (rotation +
translation), and treat half the residual at each point as that point's
asymmetry. Values are expressed as % of inter-ocular distance (IOD) so they
are scale-free.

Limitation: one face can't separate *fluctuating* asymmetry (random, the kind
linked to attractiveness ratings) from *directional* asymmetry (population-wide
side biases). Neither can a single photo separate true asymmetry from lens,
lighting or a slight head turn, which is why results are averaged over photos.
"""

from __future__ import annotations

import numpy as np

from . import landmark_ids as L
from .detect import Face
from .metrics import normalise

_PAIRS = [pr for prs in L.SYMMETRY_PAIRS.values() for pr in prs]
_IDS = sorted({i for pr in _PAIRS for i in pr} | set(L.MIDLINE))
_POS = {lid: k for k, lid in enumerate(_IDS)}


def _partner(lid: int) -> int:
    for a, b in _PAIRS:
        if lid == a:
            return b
        if lid == b:
            return a
    return lid  # midline


def _align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid (rotation + translation) least-squares fit of src onto dst."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    R = u @ np.diag([1, d]) @ vt
    return a @ R + md


def asymmetry_vectors(pts: np.ndarray) -> tuple[np.ndarray, list[int], float]:
    """Return per-landmark asymmetry vectors (px), their ids, and IOD (px)."""
    p = normalise(pts)
    X = p[_IDS]
    M = np.array([p[_partner(i)] * np.array([-1.0, 1.0]) for i in _IDS])
    M_al = _align(M, X)
    iod = float(np.linalg.norm(p[L.R_EYE_CONTOUR].mean(0) - p[L.L_EYE_CONTOUR].mean(0)))
    return (X - M_al) / 2.0, _IDS, iod


def analyse(face: Face, mirrored: Face | None = None) -> dict:
    """Asymmetry for one photo.

    If `mirrored` (the same photo flipped horizontally) is given, the landmark
    model's own left/right bias is cancelled: true = (orig - mirror)/2,
    model bias = (orig + mirror)/2, after mapping the mirror back.
    """
    res = _analyse_one(face)
    if mirrored is None:
        res["debiased"] = False
        return res
    mres = _analyse_one(mirrored)

    vec, ids, iod = asymmetry_vectors(face.pts)
    mvec, _, miod = asymmetry_vectors(mirrored.pts)
    pos = {lid: k for k, lid in enumerate(ids)}
    back = np.array([mvec[pos[_partner(i)]] * np.array([-1.0, 1.0]) for i in ids])
    true_vec = (vec / iod - back / miod) / 2
    orig_side = res["side"]
    res.update(_summarise(true_vec * 100, ids))
    res["side"] = {k: (orig_side[k] - mres["side"][k]) / 2 for k in orig_side}
    res["model_bias"] = {k: (orig_side[k] + mres["side"][k]) / 2 for k in orig_side}
    res["debiased"] = True
    return res


def _summarise(mag_vecs: np.ndarray, ids: list[int]) -> dict:
    mag = np.linalg.norm(mag_vecs, axis=1)
    by_id = dict(zip(ids, mag))
    regions = {}
    for region, prs in L.SYMMETRY_PAIRS.items():
        regions[region] = float(np.mean([by_id[a] for a, _ in prs] + [by_id[b] for _, b in prs]))
    return {"overall": float(np.sqrt(np.mean(mag ** 2))), "regions": regions}


def _analyse_one(face: Face) -> dict:
    vec, ids, iod = asymmetry_vectors(face.pts)
    mag = np.linalg.norm(vec, axis=1) / iod * 100  # % IOD
    by_id = dict(zip(ids, mag))

    regions = {}
    for region, prs in L.SYMMETRY_PAIRS.items():
        regions[region] = float(np.mean([by_id[a] for a, _ in prs] + [by_id[b] for _, b in prs]))
    overall = float(np.sqrt(np.mean(mag ** 2)))

    p = normalise(face.pts)

    def height_diff(r_ids, l_ids):
        # + means the subject's LEFT feature sits higher (smaller y)
        ry = np.mean([p[i][1] for i in r_ids])
        ly = np.mean([p[i][1] for i in l_ids])
        return float((ry - ly) / iod * 100)

    side = {
        "Brow peak height (L − R)": height_diff([L.R_BROW_PEAK], [L.L_BROW_PEAK]),
        "Mouth corner height (L − R)": height_diff([L.R_CHEILION], [L.L_CHEILION]),
        "Jaw angle height (L − R)": height_diff([L.R_GONION], [L.L_GONION]),
        # + means chin shifted toward the subject's LEFT
        "Chin offset from eye midline (→L)": float(p[L.MENTON][0] / iod * 100),
        "Nose tip offset from eye midline (→L)": float(p[L.PRONASALE][0] / iod * 100),
    }
    # eye centres are levelled by normalisation, so compare eye *openings* instead
    r_open = np.linalg.norm(p[L.R_LID_UP] - p[L.R_LID_LO])
    l_open = np.linalg.norm(p[L.L_LID_UP] - p[L.L_LID_LO])
    side["Eye opening (L − R)"] = float((l_open - r_open) / iod * 100)
    return {"overall": overall, "regions": regions, "side": side}
