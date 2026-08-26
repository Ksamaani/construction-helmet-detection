# Construction Helmet Detection — PPE Compliance Monitor

An end-to-end computer vision system for **construction-site safety compliance**, built with
[Ultralytics](https://docs.ultralytics.com/) as the capstone project of
**Computer Vision for Developers with Ultralytics — SDAIA Academy**
(https://github.com/SDAIAAcademy).

> Status: work in progress — milestones are committed incrementally.

## The idea

Construction sites lose lives to preventable head injuries. This system watches site
camera feeds and:

1. **Detects** workers and helmets in real time (custom-trained YOLO11 detector).
2. **Segments** persons/helmets at pixel level and estimates worker **pose** (beyond-detection tasks).
3. **Tracks** every worker crossing a site entrance line and counts entries/exits (`model.track` + `ObjectCounter`).
4. Builds a **heatmap** of worker activity density across the shift (`ultralytics.solutions.Heatmap`).
5. Is **evaluated** on held-out data with mAP/precision/recall + confusion matrix, interpreted
   for the asymmetric cost of errors on a safety-critical system.
6. Deploys as **ONNX** + a minimal **Streamlit** app (upload image → detections).

## Repository layout

```
construction-helmet-detection/
├── README.md                  # this file
├── requirements.txt           # local CPU-stage dependencies
├── app.py                     # Streamlit demo app (deployment)
├── notebooks/
│   ├── 01_inference.ipynb     # detection + segmentation + pose on real site images
│   ├── 02_video_analytics.ipynb # tracking, entrance counting, heatmap over real video
│   └── 03_train_eval.ipynb    # custom training (Colab GPU) + evaluation
├── scripts/download_assets.py # reproducible fetch of sample images/video
├── scripts/export_onnx.py     # ONNX export + sanity inference
├── data/images|videos/        # sample assets (video fetched, not committed)
├── outputs/                   # annotated evidence frames / counts
├── logs/                      # captured run logs = execution evidence
├── models/                    # trained weights (gitignored; see README for how to obtain)
└── docs/                      # EVALUATION.md, TRAINING.md
```

## Quickstart (local, CPU-friendly)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
jupyter execute notebooks/01_inference.ipynb   # or open & run all
```

Full per-stage instructions, dataset download steps, and expected outputs are documented
below as each milestone lands.

## Attribution

Completed under the training program **"Computer Vision for Developers with Ultralytics" — SDAIA Academy**.
Program GitHub: https://github.com/SDAIAAcademy
