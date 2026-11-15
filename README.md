# notkoves

Measure your face from ordinary photos, on your own machine, with every number explained.

notkoves finds 478 facial landmarks with Google's MediaPipe, measures proportions, angles and symmetry, and writes a self-contained HTML report. It's the transparent opposite of a black-box "facial analysis": every metric has a published definition, the code is right here, and your photos never leave your computer.

**It measures; it doesn't judge.** There is no attractiveness score, on purpose: any single score just encodes one rater panel's taste.

## Quick start

```bash
# Python 3.10+ recommended in a virtual environment
pip install git+https://github.com/DanielCWeng/notkoves
# optional, for iPhone .HEIC photos:
pip install pillow-heif

notkoves path/to/photo-folder
```

The first run downloads MediaPipe's face landmark model (~3.6 MB) to `~/.cache/notkoves/`. The report opens in your browser and is saved as `notkoves-report.html` in the photo folder.

```
notkoves PHOTOS [-o report.html] [--json] [--pd MM] [--focal-35mm MM] [--mirrored]
                [--max-yaw 8] [--max-pitch 15]
                [--model face_landmarker.task] [--backend auto|tasks|legacy] [--no-open]
```

`--json` also writes the full per-photo data, so you can track changes over months.

**Photos without lens data** (sent through WhatsApp/Messenger, screenshots): pass the lens so camera distance and head tilt can still be estimated, e.g. `--focal-35mm 23` for an iPhone 12–15 front camera or `--focal-35mm 26` for the main rear camera. Original files (AirDrop, cable, iCloud export) keep this automatically.

**Know your PD?** Pass it (`--pd 66`, from an optician's measurement). It replaces the average-iris assumption as the scale for every mm figure and the camera-distance estimate, with a correction for your eyes converging on the camera. The report also shows your implied iris diameter.

**Front-camera selfies:** if *Settings → Camera → Mirror Front Camera* is on, your photos are mirror images; add `--mirrored` so left and right are reported the right way round. Front cameras are wide-angle and usually held at arm's length, which is the worst case for perspective distortion. The rear camera with zoom from ~1.5 m is much better.

## Taking photos that measure well

Photo technique matters more than any algorithm.

1. **Distance.** Use the rear camera at 2–3× zoom from about 1.5 m (timer or a friend). Arm's-length selfies make the nose look roughly 30% bigger (Ward et al., 2018, *JAMA Facial Plastic Surgery*). notkoves estimates camera distance from EXIF and warns you when it's too close.
2. **Square on.** Lens at eye height, chin level, facing the camera. Photos turned more than 8° or tilted more than 15° are rejected automatically.
3. **Neutral face.** Lips closed and relaxed, eyes normally open, no smile.
4. **Even light** from the front (face a window). Hair off the forehead and away from the jaw.
5. **5–10 photos**, ideally on different days. A single photo can't separate your face from photo noise.

## What it measures

| Group | Metrics |
|---|---|
| Vertical proportions | midface : lower face, upper lip : lip-to-chin, facial index |
| Horizontal proportions | eye spacing : eye width, eye distance : face width, nose : eye spacing, mouth : nose, jaw : face width, fWHR |
| Eyes & brows | canthal tilt (each eye and mean), eye openness, brow height |
| Mouth | upper : lower lip |
| Symmetry | overall and per-region asymmetry (eyes, brows, nose, mouth, cheeks, jaw); side-to-side brow, mouth-corner and jaw height; chin and nose deviation |
| Sizes (mm) | when iris landmarks are reliable, scaled from the ~11.7 mm human iris |

Each metric comes with its definition and the classic canon or published reference for context. Canons are history, not targets: Farkas's anthropometric work showed they rarely hold even in attractive faces.

## How it works, and why you can trust the numbers more than most tools

- **Rotation-, scale- and resolution-free.** Every metric is a ratio or angle computed after levelling the eye line.
- **Pose gating.** Head pose is estimated by fitting MediaPipe's 468-point canonical 3D face to the landmarks (rigid regions only, so a smile doesn't fool it). Turned or tilted photos are rejected, since a few degrees of turn fakes asymmetry.
- **Mirror debiasing.** Each photo is analysed twice, once flipped. MediaPipe's mesh has a small built-in left/right bias (about 1–2% of eye distance at the jaw and chin in testing). Averaging the two passes cancels it: true asymmetry = (original − mirror)/2.
- **Standard symmetry method.** Mirrored Procrustes alignment over 19 verified bilateral landmark pairs (checked against the canonical model in the tests).
- **Reliability shown, not hidden.** Every number is reported as mean ± SD across your photos, with a stability label. Side-to-side differences are only called "real" when they exceed twice their standard error and 1% of eye distance.

## Limits

- MediaPipe's mesh is a fitted surface, not hand-placed anthropometric points. Jaw-angle and cheekbone points are the weakest.
- No hairline, so no upper facial third.
- Front view only: no profile measures (chin projection, nasal angles, gonial angle).
- One face can't separate *fluctuating* asymmetry (random, the kind linked to attractiveness ratings) from *directional* asymmetry (population-wide side biases).
- Population norms vary by sex, age and ancestry; v0.1 compares you only with canons and with yourself.

## Roadmap

- **v0.2: reference comparisons.** Averageness (Procrustes distance from a sex-specific mean face) and sexual dimorphism (position on the male–female shape axis), against the Chicago Face Database / Face Research Lab London Set. You request the datasets yourself; nothing is redistributed.
- Progress tracking across dated JSON exports.
- Optional profile-photo module with manual point correction.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests run on synthetic faces projected from the canonical model (known pose, known asymmetry, injected model bias), so they need no photos.

Two MediaPipe backends are supported: `tasks` (current MediaPipe, uses the downloaded `.task` model) and `legacy` (`mediapipe<=0.10.14`, models built in). `--backend auto` picks the right one.

## Licence

MIT. `notkoves/data/canonical_face_model.txt` is derived from MediaPipe (Apache-2.0, © Google LLC); see `notkoves/data/NOTICE`.
