"""Self-contained HTML report and JSON export."""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import __version__
from . import landmark_ids as L
from .analyse import PhotoResult, Summary
from .metrics import METRICS

E = html.escape


def _fmt(x: float, kind: str = "ratio") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if kind == "angle":
        return f"{x:+.1f}°"
    if kind == "pct":
        return f"{x:.1f}%"
    if kind == "mm":
        return f"{x:.1f} mm"
    return f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"


def _sd(s: Summary, kind: str) -> str:
    if math.isnan(s.sd):
        return ""
    unit = "°" if kind == "angle" else ""
    return f"± {s.sd:.3f}{unit}" if kind == "ratio" else f"± {s.sd:.1f}{unit}"


def overlay(r: PhotoResult, width: int = 420) -> str:
    """Cropped photo with landmarks and measurement lines, as a data URI."""
    img = r.face.photo.rgb.copy()
    p = r.face.pts
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    m = 0.18 * (x1 - x0)
    h, w = img.shape[:2]
    x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m * 1.4))
    x1, y1 = int(min(w, x1 + m)), int(min(h, y1 + m))
    t = max(1, int((x1 - x0) / 300))

    def pt(i):
        return tuple(int(v) for v in p[i])

    for q in p[:468]:
        cv2.circle(img, (int(q[0]), int(q[1])), t, (200, 200, 200), -1)
    gold, teal, red = (232, 170, 40), (40, 180, 170), (220, 70, 70)
    # vertical thirds markers
    for i in (L.NASION, L.SUBNASALE, L.MENTON):
        y = int(p[i][1])
        cv2.line(img, (int(p[L.R_ZYGION][0]), y), (int(p[L.L_ZYGION][0]), y), gold, t)
    for a, b in ((L.R_ZYGION, L.L_ZYGION), (L.R_GONION, L.L_GONION),
                 (L.R_EN, L.L_EN), (L.R_ALA, L.L_ALA), (L.R_CHEILION, L.L_CHEILION)):
        cv2.line(img, pt(a), pt(b), teal, t)
    for a, b in ((L.R_EX, L.R_EN), (L.L_EN, L.L_EX)):
        cv2.line(img, pt(a), pt(b), red, t + 1)
    crop = Image.fromarray(img[y0:y1, x0:x1])
    crop.thumbnail((width, width * 2), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


CSS = """
:root{--ink:#1d1d1f;--mute:#6e6e73;--line:#e5e5ea;--bg:#fbfbfd;--acc:#0a7c74;--warn:#b25000;--bad:#b3261e}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;color:var(--ink);background:var(--bg)}
main{max-width:1040px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.02em}h2{font-size:20px;margin:44px 0 12px;letter-spacing:-.01em}
h3{font-size:15px;margin:22px 0 6px;color:var(--mute);text-transform:uppercase;letter-spacing:.06em}
.sub{color:var(--mute);margin:0 0 24px}
table{width:100%;border-collapse:collapse;font-size:14px;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:#f5f5f7;font-weight:600;font-size:12px;color:var(--mute);text-transform:uppercase;letter-spacing:.04em}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.what{color:var(--mute);font-size:13px}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600}
.stable{background:#e3f4f1;color:var(--acc)}.moderate{background:#fff2e0;color:var(--warn)}.noisy{background:#fde8e6;color:var(--bad)}
.n\\/a,.too{background:#eee;color:var(--mute)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px;font-size:13px}
.card img{width:100%;border-radius:6px;display:block}.card.rej{opacity:.6}
.kpi{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0 4px}.k{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 16px;min-width:150px}
.k b{display:block;font-size:24px;letter-spacing:-.02em}.k span{color:var(--mute);font-size:12px}
.bar{height:8px;background:#eee;border-radius:4px;min-width:120px}.bar i{display:block;height:8px;background:var(--acc);border-radius:4px}
.note{background:#fff;border-left:3px solid var(--acc);padding:12px 16px;border-radius:0 8px 8px 0;margin:12px 0;font-size:14px}
.warn{border-left-color:var(--warn)}ul{padding-left:20px}footer{color:var(--mute);font-size:12px;margin-top:50px}
"""


def _tag(label: str) -> str:
    cls = label.split()[0]
    return f'<span class="tag {E(cls)}">{E(label)}</span>'


def render(results: list[PhotoResult], agg: dict, out: Path) -> None:
    used = [r for r in results if r.used]
    rej = [r for r in results if not r.used]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>notkoves report</title><style>{CSS}</style></head><body><main>"]
    parts.append("<h1>Face measurement report</h1>")
    parts.append(f"<p class='sub'>notkoves {__version__} · {dt.date.today():%d %b %Y} · "
                 f"{len(used)} of {len(results)} photos used</p>")

    sym = agg["symmetry"]
    stable_n = sum(1 for s in agg["metrics"].values() if s.stability == "stable")
    parts.append("<div class='kpi'>"
                 f"<div class='k'><b>{len(used)}</b><span>photos analysed</span></div>"
                 f"<div class='k'><b>{_fmt(sym.mean, 'pct')}</b><span>overall asymmetry (% eye distance)</span></div>"
                 f"<div class='k'><b>{stable_n}/{len(METRICS)}</b><span>metrics stable across photos</span></div>"
                 "</div>")
    if len(used) < 3:
        parts.append("<div class='note warn'><b>Fewer than 3 usable photos.</b> You can't tell a real "
                     "feature from photo noise with one or two images. Add 5–10 photos (see the "
                     "photo guide at the bottom).</div>")
    dists = [r.pose.distance_m for r in results if r.pose and r.pose.distance_m]
    assumed = any(r.face is not None and r.face.photo.focal_assumed for r in results)
    if assumed and dists:
        # crops/screenshots break the distance estimate; test it against the photos themselves
        idx = [r.metrics.get("facial_index") for r in used if r.pose and r.pose.distance_m]
        ds = [r.pose.distance_m for r in used if r.pose and r.pose.distance_m]
        if len(ds) >= 2 and max(ds) / min(ds) > 1.6 and np.nanstd(idx) / np.nanmean(idx) < 0.04:
            parts.append("<div class='note warn'><b>Distance estimates don't add up.</b> They vary "
                         f"{min(ds):.2f}–{max(ds):.2f} m, but the face's width ratios barely change between "
                         "photos, which wouldn't happen at genuinely different distances. The photos are "
                         "probably cropped, zoomed or screenshots, which breaks distance estimation when "
                         "the lens is assumed. Use the original, uncropped files.</div>")
            dists = []
    if dists and min(dists) < 0.9:
        near = sum(d < 0.9 for d in dists)
        parts.append(f"<div class='note warn'><b>{near} of {len(dists)} photos were taken from closer than "
                     f"0.9 m</b> (closest ≈{min(dists):.2f} m). At that range perspective enlarges the nose "
                     "and centre of the face and shrinks the cheeks and jaw, so nose-, mouth- and "
                     "width-based ratios are biased. Retake from ~1.5 m with the rear camera at 2–3× "
                     "zoom.</div>")
    elif used and not dists and not assumed:
        parts.append("<div class='note warn'><b>Camera distance unknown.</b> These photos carry no lens "
                     "data (messaging apps and screenshots strip it). Re-run with "
                     "<code>--focal-35mm</code> (e.g. 23 for a recent iPhone front camera) or use the "
                     "original files.</div>")
    parts.append("<div class='note'>This report <b>measures</b>; it doesn't <b>judge</b>. There's no "
                 "attractiveness score, on purpose. Reference values are classic canons or published "
                 "means, shown for context, not targets. The <i>stability</i> column tells you whether "
                 "a number held up across your photos: if it's noisy, it's mostly measuring the "
                 "camera, not your face.</div>")

    # --- metrics table
    parts.append("<h2>Proportions</h2>")
    group = None
    for m in METRICS:
        if m.group != group:
            if group is not None:
                parts.append("</table>")
            group = m.group
            parts.append(f"<h3>{E(group)}</h3><table><tr><th style='width:22%'>Metric</th>"
                         "<th style='width:16%'>You</th><th style='width:11%'>Stability</th>"
                         "<th>What it is · reference</th></tr>")
        s = agg["metrics"][m.key]
        parts.append(f"<tr><td><b>{E(m.name)}</b></td>"
                     f"<td class='num'>{_fmt(s.mean, m.kind)} <span class='what'>{_sd(s, m.kind)}</span></td>"
                     f"<td>{_tag(s.stability)}</td>"
                     f"<td>{E(m.what)}<div class='what'>{E(m.reference)}</div></td></tr>")
    parts.append("</table>")

    # --- symmetry
    parts.append("<h2>Symmetry</h2>")
    parts.append("<p class='what'>Mirrored-Procrustes asymmetry: how far each landmark sits from where "
                 "a perfectly mirrored version of your face would put it, as % of the distance between "
                 "your eyes. Everyone is asymmetric; values of a few percent are normal and mostly "
                 "invisible. Research suggests symmetry matters less to attractiveness than averageness "
                 "once both are measured together (Lee et al., 2025).</p>")
    if agg.get("debiased"):
        parts.append("<p class='what'>Each photo was analysed twice, once mirrored, so any left/right "
                     "bias in the landmark model itself cancels out.</p>")
    regs = agg["regions"]
    top = max([s.mean for s in regs.values()] + [1e-6])
    parts.append("<table><tr><th>Region</th><th>Asymmetry</th><th></th><th>Stability</th></tr>")
    for reg, s in regs.items():
        parts.append(f"<tr><td>{E(reg)}</td><td class='num'>{_fmt(s.mean, 'pct')} "
                     f"<span class='what'>{_sd(s, 'pct')}</span></td>"
                     f"<td><div class='bar'><i style='width:{100 * s.mean / top:.0f}%'></i></div></td>"
                     f"<td>{_tag(s.stability)}</td></tr>")
    parts.append("</table>")

    parts.append("<h3>Side-to-side differences</h3><table><tr><th>Measure</th>"
                 "<th>Mean (% eye distance)</th><th>Consistent across photos?</th></tr>")
    scale = agg["mm"].get("Interpupillary distance") or agg["mm"].get("Interpupillary distance (in photo)")
    for k, s in agg["side"].items():
        consistent = "–"
        if s.n >= 3 and not math.isnan(s.sd):
            se = s.sd / math.sqrt(s.n)
            consistent = ("yes, a real tendency" if abs(s.mean) > 2 * se and abs(s.mean) > 1
                          else "no, within photo noise")
        extra = ""
        if scale and not math.isnan(scale.mean):
            extra = f" <span class='what'>(≈{abs(s.mean) / 100 * scale.mean:.1f} mm)</span>"
        parts.append(f"<tr><td>{E(k)}</td><td class='num'>{s.mean:+.1f}%{extra}</td><td>{consistent}</td></tr>")
    parts.append("</table><p class='what'>Positive = your left side higher / shifted to your left. "
                 "For scale: laypeople usually don't notice chin deviations under about 5–6 mm "
                 "(McAvinchey et al., 2014).</p>")

    if agg["mm"]:
        if any("implied" in k for k in agg["mm"]) or any("in photo" in k for k in agg["mm"]):
            basis = ("Scaled from your measured PD, corrected for your eyes converging on the camera "
                     "(pupils sit slightly closer in a photo). Error is mostly landmark placement, "
                     "typically 1–3 mm.")
        else:
            basis = ("Scaled from your iris (adult average 11.7 mm, SD ~0.5 mm), so expect ±5% error "
                     "on top of landmark error. Pass <code>--pd</code> with your measured PD for a "
                     "real scale.")
        parts.append(f"<h2>Approximate sizes</h2><p class='what'>{basis}</p>"
                     "<table><tr><th>Measure</th><th>Size</th><th>Stability</th></tr>")
        for k, s in agg["mm"].items():
            parts.append(f"<tr><td>{E(k)}</td><td class='num'>{_fmt(s.mean, 'mm')}</td><td>{_tag(s.stability)}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<div class='note'>Real-world sizes (mm) are unavailable: the iris landmarks "
                     "weren't reliable in these photos.</div>")

    # --- photos
    parts.append("<h2>Photos</h2><div class='cards'>")
    for r in used + rej:
        img = f"<img src='{overlay(r)}' alt=''>" if r.face is not None else ""
        pose = ""
        if r.pose:
            d = ""
            if r.pose.distance_m:
                d = f", camera ≈{r.pose.distance_m:.2f} m"
                if r.face.photo.focal_assumed:
                    d += " (assumed lens)"
            pose = (f"<div class='what'>turn {r.pose.yaw:+.0f}°, tilt {r.pose.pitch:+.0f}°, "
                    f"roll {r.pose.roll:+.0f}°{d}</div>")
        status = "<b style='color:var(--acc)'>used</b>" if r.used else \
                 "<b style='color:var(--bad)'>rejected</b>: " + E("; ".join(r.rejected))
        warns = "".join(f"<div class='what'>⚠ {E(w)}</div>" for w in r.warnings)
        parts.append(f"<div class='card{'' if r.used else ' rej'}'>{img}<div><b>{E(r.path.name)}</b></div>"
                     f"<div>{status}</div>{pose}{warns}</div>")
    parts.append("</div>")

    parts.append("""
<h2>How to take photos that measure well</h2>
<ul>
<li><b>Distance beats everything.</b> Use the rear camera at 2–3× zoom from about 1.5 m (a timer or a friend). Arm's-length selfies enlarge the nose by roughly 30% (Ward et al., 2018).</li>
<li><b>Face the camera squarely</b>, lens at eye height, chin level. Photos turned more than 8° are rejected automatically.</li>
<li><b>Neutral expression</b>, lips closed and relaxed, eyes open normally. Expression changes mouth, eye and brow metrics a lot.</li>
<li><b>Even, frontal light</b> (facing a window). Hair off the forehead and away from the jaw outline.</li>
<li><b>Take 5–10 shots</b>, ideally on different days. The stability column only means something with several photos.</li>
</ul>
<h2>Limits</h2>
<ul>
<li>Landmarks come from MediaPipe's face mesh: a fitted surface, not hand-placed anthropometric points. Jaw-angle and cheekbone points are the least reliable.</li>
<li>The hairline isn't detected, so the upper facial third isn't measured.</li>
<li>Front-facing photos only; side-profile measures (chin projection, jaw angle, nasal angles) aren't included.</li>
<li>Reference canons are descriptive history, not ideals. Population norms differ by sex, age and ancestry.</li>
<li>No attractiveness score. Any single score would just encode one rater panel's taste.</li>
</ul>""")
    parts.append(f"<footer>Generated locally by notkoves {__version__}. Your photos never left this "
                 "computer.</footer></main></body></html>")
    out.write_text("".join(parts), encoding="utf-8")


def to_json(results: list[PhotoResult], agg: dict, out: Path) -> None:
    def summ(s: Summary):
        return {"mean": _nan(s.mean), "sd": _nan(s.sd), "n": s.n, "stability": s.stability}

    data = {
        "notkoves_version": __version__,
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "metrics": {k: summ(v) for k, v in agg["metrics"].items()},
            "symmetry_overall_pct_iod": summ(agg["symmetry"]),
            "symmetry_regions_pct_iod": {k: summ(v) for k, v in agg["regions"].items()},
            "side_differences_pct_iod": {k: summ(v) for k, v in agg["side"].items()},
            "mm": {k: summ(v) for k, v in agg["mm"].items()},
        },
        "photos": [{
            "file": r.path.name,
            "used": r.used,
            "rejected": r.rejected,
            "warnings": r.warnings,
            "pose": None if not r.pose else {
                "yaw": _nan(r.pose.yaw), "pitch": _nan(r.pose.pitch), "roll": _nan(r.pose.roll),
                "distance_m": r.pose.distance_m},
            "metrics": {k: _nan(v) for k, v in r.metrics.items()},
            "symmetry": r.symmetry,
        } for r in results],
    }
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _nan(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)
