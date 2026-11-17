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
  - `regions/` — region models; `rcnn.py` runs the V1 Faster R-CNN weights
  - `framestate.py` — whether a frame is searched (dark, no feed, no space)
  - `candidates.py` — full-resolution candidate generator; returns rejects with their gates
  - `pipelines/` — runnable pipelines; `core.py` is V2, `legacy_v1.py` wraps the V1 detector
  - `eval/headless.py` — the headless harness
  - `cli.py` — the `uapd` command
- `legacy/` — V1, unmodified. It is the baseline, not a place for fixes.
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
python legacy/SOD_Main.py        # V1 live app, run from the repo root
```

`best_map_model.pth` is not in the repo; the download link is in `README.md`.

## Conventions

- Configuration comes from a `Config` object passed in. Never read settings at import time and never add module-level mutable settings. (V1's `from SOD_Constants import X` froze values at import, so live edits never reached half the system.)
- All pixel measurements are in full-resolution pixels of the cropped frame. If an image is downscaled for speed, scale measurements back before comparing or storing them.
- Every stored measurement must be the exact value that was used to make the decision.
- Record rejected candidates and the gate that rejected them; do not drop them silently.
- Anything that produces output records the versions behind it: config fingerprint, model weights hash, code commit.
- Each milestone must be verifiable headlessly with `uapd eval`. New pipeline stages implement `uapd.pipeline.Pipeline` and register in `uapd/pipelines/__init__.py`.
- No self-training on anomaly positives. Pseudo-labels are allowed only for distilling the region model on abundant region classes.
- Keep tests free of the GPU, the weights file and the network. `tests/test_legacy_v1.py` shows how to drive real detection code with the model stubbed.

## Known V1 behaviour (do not port)

These are in `legacy/` and stay there so the baseline is honest. V2 code must not reproduce them:

- Candidates are measured on a 0.6× image but checked against full-scale thresholds, so the real minimum object size is about 8 px, not 5.
- A candidate needs at least 10 contour points, which rejects small and square-edged objects.
- The lens-flare guard is inert: `skip_save`, `consecutive_lf` and `lf_pause_until` are written and never read.
- Saved `area` is the bounding-box area, not the contour area that gated the detection.
- The live feed URL is hard-coded in `legacy/SOD_Main.py`; V2 keeps it in `Config.source.live_url`.
- The background ring is built with the morphology kernel (size 1), so it is empty: `bg_brightness` is always 0, `contrast` equals object brightness and `MAX_BG_BRIGHTNESS` never rejects anything.
- Area is `cv2.contourArea`, which is 16 for a 5×5 px object, so `MIN_CONTOUR_AREA = 25` rejects it. V2 counts pixels.
- The first RCNN run happens on frame 120, so the first two seconds of any run have no regions.
- `legacy/Context.md` states thresholds and cadences that no longer match the code. Trust `legacy/SOD_Constants.py`.
