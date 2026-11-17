# ISS UAP Detector — working brief

## What this project is

A detector that watches the ISS live feed for solid objects against the black of space that match nothing prosaic. V2 (branch `v2.0`) rebuilds it as a self-improving system. The design and milestones are in `docs/V2_PLAN.md`; read it before changing architecture.

Three rules from the plan that govern every change:

1. **We train a rejector, not a finder.** Real anomalies are almost never seen. The system learns what prosaic things look like (flare, ice, panel edge, debris, artifact) and surfaces what is left. Labelled data is a hard-negative stockpile.
2. **Decisions are temporal.** Candidates become tracks, tracks become events; events are what get classified, saved and labelled.
3. **Nothing is promoted without the gate.** A new model or parameter profile must reject more labelled junk without losing planted synthetic objects.

## Layout

- `uapd/` — the V2 package. This is where new work goes.
  - `config.py` — typed, immutable config; loaded once and passed explicitly
  - `pipeline.py` — the `Pipeline` interface and result types
  - `sources/` — frame sources (video file, frame folder)
  - `regions/` — region models; `rcnn.py` runs the V1 Faster R-CNN weights; `schedule.py` runs them in step (eval) or in a background thread (live)
  - `framestate.py` — whether a frame is searched (dark, no feed, no space)
  - `candidates.py` — full-resolution candidate generator; returns rejects with their gates
  - `tracking.py` — links detections into tracks; a track must persist for `tracking.confirm_frames` frames before it counts
  - `pipelines/` — runnable pipelines; `core.py` is V2, `legacy_v1.py` wraps the V1 detector
  - `events.py` — the data file written beside each recording
  - `runtime/live.py` — the live app: V2 detection behind the legacy window, recorder and restream
  - `eval/headless.py` — the headless harness
  - `cli.py` — the `uapd` command
- `legacy/` — the V1 live app, with the July 2026 patch set applied (scale fix, 3-point contour minimum, working lens-flare guard, truthful `area`, dead-code cleanup). It is the app that runs day to day until V2 has a live runtime, and the baseline `uapd eval --pipeline legacy-v1` compares against. The original unpatched V1 is on `dev`. New work still goes in `uapd/`.
- `tests/` — pytest suite
- `docs/V2_PLAN.md` — architecture, data model, milestones, open decisions

## Commands

```bash
uv venv && uv pip install -e '.[torch,dev]'
pytest
ruff check .
uapd eval path/to/clip.mp4 --weights best_map_model.pth --out reports/clip.core.json
uapd eval path/to/clip.mp4 --pipeline legacy-v1 --weights best_map_model.pth --out reports/clip.v1.json
uapd config
uapd run                         # V2 live app, run from the repo root (q quits, s restreams, t test frames)
uapd run path/to/clip.mp4        # the same on a recorded clip
python legacy/SOD_Main.py        # V1 live app, run from the repo root
```

`best_map_model.pth` is not in the repo; the download link is in `README.md`.

## Conventions

- Configuration comes from a `Config` object passed in. Never read settings at import time and never add module-level mutable settings. (V1's `from SOD_Constants import X` froze values at import, so live edits never reached half the system.)
- All pixel measurements are in full-resolution pixels of the cropped frame. If an image is downscaled for speed, scale measurements back before comparing or storing them.
- Every stored measurement must be the exact value that was used to make the decision.
- Record rejected candidates and the gate that rejected them; do not drop them silently. The one exception is by count: candidates whose brightest pixel is below `min_contrast` (about 90 specks of 1–2 px noise per live frame) are tallied per frame as `faint`, not recorded one by one.
- The live feed is 30 fps camera footage sent at 60 fps, so each camera frame arrives twice. The copies are not identical: a line along the top edge of the picture changes every frame. Do not rely on detecting repeated frames.
- Anything that produces output records the versions behind it: config fingerprint, model weights hash, code commit.
- Each milestone must be verifiable headlessly with `uapd eval`. New pipeline stages implement `uapd.pipeline.Pipeline` and register in `uapd/pipelines/__init__.py`.
- No self-training on anomaly positives. Pseudo-labels are allowed only for distilling the region model on abundant region classes.
- Keep tests free of the GPU, the weights file and the network. `tests/test_legacy_v1.py` shows how to drive real detection code with the model stubbed.

## Known V1 behaviour (do not port)

Fixed in `legacy/` by the July 2026 patch set, still present on `dev`:

- Candidates were measured on a 0.6× image but checked against full-scale thresholds, so the real minimum object size was about 8 px, not 5.
- A candidate needed at least 10 contour points, which rejected small and square-edged objects.
- The lens-flare guard was inert. In `legacy/` it now skips anomaly detection while `MAX_LENS_FLARES` or more flares are present.
- Saved `area` was the bounding-box area. It is now the contour area that gated the detection, with `bbox_area` beside it.

Still in `legacy/`. V2 code must not reproduce these:

- Detection still runs on a 0.6× image; sizes are scaled back before the gates, so small objects are measured coarsely.
- The live feed URL is hard-coded in `legacy/SOD_Main.py`; V2 keeps it in `Config.source.live_url`.
- The background ring is built with the morphology kernel (size 1), so it is empty: `bg_brightness` is always 0, `contrast` equals object brightness and `MAX_BG_BRIGHTNESS` never rejects anything.
- Area is `cv2.contourArea`, which is 16 for a 5×5 px object, so `MIN_CONTOUR_AREA = 25` rejects it. V2 counts pixels.
- The first RCNN run happens on frame 120, so the first two seconds of any run have no regions.
- `legacy/Context.md` states thresholds and cadences that no longer match the code. Trust `legacy/SOD_Constants.py`.
