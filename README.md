# ISS UAP Detector

## V2.0 (in progress, branch `v2.0`)

V2 rebuilds the detector as a self-improving system: detections become reviewable events, reviews become a labelled stockpile of prosaic look-alikes, and retrained filters are promoted only when they pass an evaluation gate. The plan is in [docs/V2_PLAN.md](docs/V2_PLAN.md).

What exists so far (milestone M0):

- `uapd/` package with a typed config object and a pipeline interface
- `uapd eval`, a headless harness that runs a pipeline over a video file or a folder of frames and writes detections and metrics to JSON
- An adapter that runs the unmodified V1 detector through the harness as the baseline

```bash
uv venv && uv pip install -e '.[torch,dev]'
pytest
uapd eval path/to/clip.mp4 --weights best_map_model.pth --out reports/clip.v1.json
```

## V1 (SpaceObjectDetector 1.1)

The V1 application now lives in `legacy/`. Run it from the repo root:

```bash
pip install -r legacy/requirements.txt
python legacy/SOD_Main.py
```

Download the RCNN model @ https://drive.google.com/file/d/13C2r1YmuHA-NZSbe8CKWmUgwFjo3502_/view?usp=sharing

 Vastly Improved ISS Object Detector: a RCNN and 1st Principles Analysis
![Screenshot 2025-01-10 150613](https://github.com/user-attachments/assets/fea0244c-b467-4ff7-92d5-e0dfecedeaf4)

![image](https://github.com/user-attachments/assets/dfa0580f-f82e-4f63-a39a-5a5f61240a40)

![image](https://github.com/user-attachments/assets/a5eda90b-5f44-4629-84fc-01938401d295)

![image](https://github.com/user-attachments/assets/d1aa0a8e-957f-4c41-9b97-9dcf7d707bb5)

![image](https://github.com/user-attachments/assets/e631a7e3-1069-45e1-ba6c-e4435543c81b)

