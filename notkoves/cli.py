"""Command line entry point: `notkoves PHOTOS [-o report.html]`."""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

from . import __version__
from . import analyse as A
from . import report as R
from .model import model_path
from .pose import Gate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="notkoves",
        description="Measure your face from front-facing photos. Runs entirely on your machine.",
    )
    ap.add_argument("photos", type=Path, help="a folder of photos (or a single image)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="report path (default: <photos>/notkoves-report.html)")
    ap.add_argument("--json", action="store_true", help="also write a JSON file for tracking over time")
    ap.add_argument("--model", default=None, help="path to face_landmarker.task (auto-downloaded if omitted)")
    ap.add_argument("--max-yaw", type=float, default=8.0, help="reject photos turned more than this (deg)")
    ap.add_argument("--max-pitch", type=float, default=15.0, help="reject photos tilted up/down more than this (deg)")
    ap.add_argument("--backend", choices=["auto", "tasks", "legacy"], default="auto",
                    help="MediaPipe API: 'tasks' (current) or 'legacy' (mediapipe<=0.10.14). Default: auto")
    ap.add_argument("--focal-35mm", type=float, default=None,
                    help="lens (35mm-equivalent focal length) to assume for photos without EXIF, "
                         "e.g. 23 for an iPhone 12-15 front camera, 26 for the main rear camera")
    ap.add_argument("--mirrored", action="store_true",
                    help="photos are mirror images (iPhone Settings > Camera > Mirror Front Camera on); "
                         "flips them so left/right are reported correctly")
    ap.add_argument("--no-open", action="store_true", help="don't open the report in a browser")
    ap.add_argument("--version", action="version", version=f"notkoves {__version__}")
    args = ap.parse_args(argv)

    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if not args.photos.exists():
        ap.error(f"{args.photos} does not exist")
    out = args.out or ((args.photos if args.photos.is_dir() else args.photos.parent) / "notkoves-report.html")

    backend = args.backend
    if backend == "auto":
        import mediapipe as mp
        backend = "legacy" if (args.model is None and hasattr(mp, "solutions")) else "tasks"
    model = model_path(args.model) if backend == "tasks" else None
    results = A.run(args.photos, model, Gate(max_yaw=args.max_yaw, max_pitch=args.max_pitch),
                    backend=backend, focal_35mm=args.focal_35mm, mirrored=args.mirrored)
    agg = A.aggregate(results)
    R.render(results, agg, out)
    used = sum(r.used for r in results)
    print(f"Analysed {len(results)} photos, used {used}.")
    for r in results:
        if not r.used:
            print(f"  rejected {r.path.name}: {'; '.join(r.rejected)}")
    print(f"Report: {out}")
    if args.json:
        jp = out.with_suffix(".json")
        R.to_json(results, agg, jp)
        print(f"JSON:   {jp}")
    if not args.no_open:
        try:
            webbrowser.open(out.resolve().as_uri())
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
