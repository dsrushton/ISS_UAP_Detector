# ISS UAP Detector V2.0 — rewrite plan

Status: M0 built; M1 onward planned. Branch: `v2.0` (cut from `dev` at `3d46021`).
Source of the design: the "Self-improving UAP detection system architecture" discussion, mapped onto the V1 code as it stands on `dev`.

## 1. What V2 is

V1 is a detector with hand-tuned thresholds. V2 is a **data flywheel**: capture → review → retrain → better capture.

Three principles drive every decision below:

1. **We are training a rejector, not a finder.** Real anomalies are almost never seen, so the system learns what prosaic things look like (flare, ice, panel edge, debris, artifact) and surfaces the residue that matches nothing. The labelled data is a hard-negative stockpile.
2. **The intelligence is temporal.** Single-frame thresholds cannot tell compression noise from an object. Candidates become tracks, tracks become events, and events are what get classified, saved and labelled.
3. **Nothing is promoted without passing a gate.** Every new model or threshold profile must reject more labelled junk *without losing planted synthetic objects*, or it is rolled back automatically.

## 2. Pipeline

```
frame source (live / file / folder)
  → frame state      nofeed / total-darkness check, no model needed
  → region model     space / earth / iss / panel / sun / lf   (RCNN first, YOLO-seg later)
  → candidates       contour + brightness inside the space mask, permissive
  → tracker          IoU association across frames
  → events           tracks that persist N frames
  → event classifier verdict + score                          (threshold profile first, GBM later)
  → capture          clip + keyframes + JSON sidecar + SQLite row
  → review viewer    one verdict per event, FP taxonomy
  → retrain + gate   golden set + synthetic recall → promote or roll back
```

What changes from V1 at each stage:

| Stage | V1 today | V2 |
|---|---|---|
| Frame state | `td` and `nofeed` are RCNN classes | Cheap histogram check per frame |
| Regions | Faster R-CNN boxes every 120 frames, consumed stale, with a caching layer to cope | Same RCNN behind an interface at first; YOLO-seg masks every frame later, which removes the caching layer |
| Candidates | Gate cascade that drops rejects silently, capped at 2 per frame | Permissive generator that records every candidate with its full measurement vector and which gate it failed |
| Temporal | None | IoU tracker; persistence gating kills single-frame noise before anything is saved |
| Decision | AND-ed constants in `SOD_Constants.py` | Versioned profile, then a gradient-boosted classifier on track features |
| Storage | Numbered JPG/AVI files | Event folder + sidecar + SQLite index |
| Review | None | Built-in viewer, event-level verdicts |
| Learning | Manual console tweaks | Retrain after each review session, gated |

## 3. Package layout

```
uapd/
  config.py          typed config object, loaded from a versioned profile file; no import-time snapshots
  sources/           youtube, file, frame folder — one FrameSource interface
  framestate.py      nofeed / darkness
  regions/           base interface, rcnn.py (V1 weights), yolo_seg.py (later)
  candidates.py      contour/brightness generator, ported from V1 detect_anomalies
  tracking.py        IoU tracker, track features
  events.py          event lifecycle, pre/post buffer
  classify/          thresholds.py, gbm.py, registry.py (champion/challenger)
  store/             sidecar.py, db.py (SQLite), dataset.py (snapshots)
  synth.py           synthetic object injection
  eval/              headless.py, gate.py
  viewer/            FastAPI app + single-page UI
  train/             filter.py (GBM), regions.py (detector fine-tune / distillation)
  runtime/           live loop, display, recording, YouTube restream
  cli.py             uapd run | eval | review | train | promote
tests/
docs/
```

V1's `SOD_*.py` files stay untouched on `dev`. On `v2.0` they live in `legacy/`, where the headless harness runs them for comparison, and are deleted when V2 reaches parity. Since M1, `legacy/` carries the July 2026 patch set (scale fix, 3-point contour minimum, working flare guard, truthful `area`, cleanup), because it is the only live app until M2. The baseline in `uapd eval --pipeline legacy-v1` is therefore patched V1; unpatched V1 is on `dev`.

## 4. What is ported, rewritten, or dropped

**Ported (the designs are sound):**
- Async region inference with a size-1 queue
- Rolling pre-buffer and post-detection continuation for clips
- Threaded save queue
- Max-RGB brightness channel and the dark-region / bright-pixel mask logic
- Cheap-to-expensive ordering of checks
- YouTube restream and broadcast rotation (`SOD_Stream.py`, `SOD_YouTubeAPI.py`), moved under `runtime/` with minimal change

**Rewritten:**
- `SOD_Constants.py` singleton → config object passed explicitly. Fixes the V1 problem that console edits never reach modules that did `from SOD_Constants import X`.
- `SOD_Main.py` (1,166 lines) → small runtime loop over the pipeline stages
- `SOD_Console.py` (Tk) → parameter panel inside the viewer
- `SOD_Capture.py` / `SOD_Video.py` file numbering → event store

**V1 defects that must not be carried over:**
- Scale mismatch: candidates are measured on a 0.6× image but `analyze_object` applies full-scale thresholds, so the real minimum size is about 8 px, not 5. This discards the small, faint objects the project exists to find. V2 measures everything in full-resolution pixels.
- `len(contour) < 10` point-count gate, which also biases against small objects
- Empty background ring (found during M1): the ring is dilated with the size-1 morphology kernel, so `bg_brightness` is always 0, `contrast` is just object brightness and `MAX_BG_BRIGHTNESS` is inert
- `cv2.contourArea` as the area (found during M1): a 5×5 px object measures 16 and fails `MIN_CONTOUR_AREA = 25`. V2 counts the pixels the object covers
- Flare guard that does nothing: `skip_save`, `consecutive_lf` and `lf_pause_until` are written but never read
- Saved `area` is overwritten with bounding-box area, so metadata does not match what gated the detection
- `gpu_crop_frame` (slower than a slice), repeated re-cropping, per-frame `import`
- Dead code: `crop_and_prepare_for_rcnn`, `_combine_overlapping_boxes`, `force_rcnn_detection`, `pause_capture`, duplicate burst logic, commented-out logger calls, the second timing system in Main
- `Context.md`, which states thresholds and cadences that no longer match the code. Replaced by `CLAUDE.md` + this plan.

## 5. Data model

**Sidecar (one JSON per event, next to its clip):**
- event id, start/end time, source, frame range
- per-frame track: box, centroid, measurement vector (width, height, contour area, aspect ratio, object brightness, background brightness, contrast, distance to space-mask border, distance to nearest `lf`/`panel`/`iss` region)
- track features: duration, velocity, direction consistency, brightness profile, size change
- region context: classes, boxes/masks, scores at event start
- versions: region model, classifier, config profile, code commit
- classifier score and which gates passed or failed

Every value stored is exactly the value that was used to decide.

**SQLite tables:** `events`, `verdicts`, `models`, `profiles`, `eval_runs`, `golden_set`.

**Verdict taxonomy (starting set, editable):** `unknown-interesting`, `flare`, `ice`, `panel-edge`, `iss-structure`, `debris`, `earth-limb`, `star-or-planet`, `compression-artifact`, `hot-pixel`, `other-prosaic`.

**Labelling units:**
- Event verdict: one click, propagates to every frame in the track
- Whole-frame "nothing real here": free background data for detector training
- Boxes: only keyframe + interpolation, only when region-detector training needs them
- Masks: only for large regions, assisted (SAM2 inside CVAT or Label Studio), never for tiny targets

## 6. Evaluation gate

The gate scores a candidate model or profile on two things:

1. **Junk rejection** on a frozen golden set of labelled negatives, reported per taxonomy class
2. **Synthetic recall**: small moving dots with plausible size, brightness and trajectory planted into real archived footage. This is the tripwire against a rejector that grows greedy and starts eating the unknown space.

Promotion rule: the challenger must not lose synthetic recall beyond a set tolerance and must improve junk rejection. Otherwise the champion stays. Every run is recorded in `eval_runs`.

The golden set never enters training.

## 7. Milestones

Each milestone ends in something runnable and verifiable without a display.

**M0 — Skeleton and harness** (done)
- Package scaffold, `pyproject.toml`, config object, `CLAUDE.md`, tests wired up
- `uapd eval`: run a pipeline headlessly over a clip or frame folder and write detections + metrics to JSON
- Harness can run V1 from `legacy/` so later milestones have a baseline
- Remove committed `__pycache__`

**M1 — Core pipeline port** (built; check on real footage outstanding)
- Frame sources, frame state, RCNN region adapter, candidate generator with the scale fix
- Every candidate logged with its measurement vector and failed gate
- Check: on the same clips, V2 finds everything V1 finds plus additional small objects; differences are inspected by eye
- Status: `uapd eval --pipeline core` runs the pipeline; tests cover it on synthetic frames, including a side-by-side with V1. The check on real clips needs `best_map_model.pth` and archived footage.
- Deliberate differences from V1, to be judged in that check:
  - Contrast is now object minus a real background ring, so it reads up to about 10 lower than V1's (the ring lies in dark space, at or below `dark_region_threshold`). Objects with brightness 23–33 that V1 accepted can fail `min_contrast = 23`.
  - No limit of 2 accepted candidates per frame; up to `max_candidates` (200) are measured.
  - Three or more lens flares do not stop the search. The flare count is recorded per frame; candidates overlapping a flare box are still rejected. The rule is decided in M2.
  - Regions run in step with the frames (first frame, then every `cycle_frames`) rather than in a background thread. The async worker returns with the live runtime in M2.
  - `regions.max_boxes` is applied (best 10 regions); V1 never applied `MAX_RCNN_BOXES`.
  - Frame state still comes from the region model (`td`, `nofeed`); the histogram check waits for footage to tune it on.

**M2 — Tracker, events, store**
- IoU tracker, persistence gating, event lifecycle with pre/post buffer
- Sidecars and SQLite index
- Working flare handling, decided from data rather than the dead V1 heuristic
- Live runtime: display, recording, restream — V2 can replace V1 for daily running here

**M3 — Review viewer**
- FastAPI + single page: event list, clip playback, verdict buttons, keyboard shortcuts
- Queue sorted by uncertainty once a classifier exists; by time until then
- Parameter panel replacing the Tk console
- Importer that turns existing V1 `Detections/` captures into reviewable events (features limited to what can be recomputed)

**M4 — Synthetic injection and the gate**
- `synth.py`, built from V1's test-frame injection idea
- Golden-set tooling, `uapd eval --gate`, model/profile registry with promote and rollback

**M5 — Learned filter**
- Threshold-profile optimiser (Optuna) as the first automatic step
- Gradient-boosted classifier on track features, retrained after each review session, gated
- Auto-dismissal of very-high-confidence known junk, with a sampled spot-check stream so drift cannot hide

**M6 — Per-frame region model**
- Distil the RCNN into a YOLO-seg model from pseudo-labels on archived footage, corrected in the viewer
- Promote only after it beats the RCNN on the golden set
- Delete the stale-mask caching path

**Deferred until the GBM plateaus:** frozen-embedding memory bank for nearest-neighbour FP dismissal; open-set / novelty scoring (one-class models, reconstruction error) as an extra signal on the residue.

## 8. Decisions to confirm

Defaults are chosen so work can start; change any of them before M1.

| Decision | Default |
|---|---|
| Rewrite style | New `uapd/` package beside V1, V1 kept in `legacy/` as the baseline until parity |
| Python / tooling | Python 3.11, `uv`, `pyproject.toml`, `ruff`, `pytest` |
| Viewer | Custom FastAPI viewer for event verdicts; export to CVAT or Label Studio only for box and mask editing |
| First learned filter | LightGBM on track features |
| Dataset versioning | Dated snapshots tracked in SQLite; add DVC only if the dataset outgrows that |
| YOLO licence | Ultralytics is AGPL-3.0. Fine for a public repo under a compatible licence; needs a decision if the project should be licensed otherwise |
| Self-training | Never for anomaly positives. Allowed only for RCNN → YOLO distillation on abundant region classes |

## 9. Needed from outside the repo

- `best_map_model.pth` (linked from the README) for M1 onward
- Archived footage: clean passes, flare storms, ice events — the baseline clips for the harness and the raw material for the golden set
- Existing `Detections/` captures, to seed the hard-negative stockpile through the M3 importer
- The patched files, `CLAUDE.md` and `TASKS.md` from the earlier discussion, if you want them folded in; none of them are in the repo
- A GPU machine for anything that runs the region model; this plan was written without running V1
