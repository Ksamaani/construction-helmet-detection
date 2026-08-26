"""Streamlit demo: upload a construction-site image -> PPE detections.

Deployment deliverable: serves the custom-trained helmet detector
(models/best.pt, or its ONNX export models/best.onnx) with a graceful
fallback to the COCO-pretrained yolo11n.pt when the custom weights are absent.

Run:  streamlit run app.py
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="PPE Compliance Monitor", page_icon="⛑", layout="wide")
st.title("Construction PPE Compliance Monitor")
st.caption(
    "Capstone — Computer Vision for Developers with Ultralytics · SDAIA Academy · "
    "https://github.com/SDAIAAcademy"
)


@st.cache_resource
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def available_models() -> list[tuple[str, str]]:
    options = []
    if (ROOT / "models" / "best.pt").exists():
        options.append(("models/best.pt", "Custom helmet model (PyTorch) — trained in 03_train_eval.ipynb"))
    if (ROOT / "models" / "best.onnx").exists():
        options.append(("models/best.onnx", "Custom helmet model (ONNX) — exported for deployment"))
    options.append(("yolo11n.pt", "COCO-pretrained yolo11n (fallback; no helmet classes)"))
    return options


models = available_models()
choice = st.sidebar.radio(
    "Model",
    [path for path, _ in models],
    format_func=lambda p: dict(models)[p],
)
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
custom = "best" in choice

if not custom:
    st.sidebar.warning(
        "COCO fallback has no helmet classes — it locates workers only. "
        "Add models/best.pt (from the Colab training) for compliance checks."
    )

model = load_model(choice)

upload = st.file_uploader("Upload a construction-site image", type=["jpg", "jpeg", "png"])
if upload is not None:
    bytes_img = np.frombuffer(upload.getvalue(), np.uint8)
    img = cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not decode that image.")
        st.stop()

    results = model.predict(source=img, conf=conf, verbose=False)[0]
    annotated = results.plot()

    left, right = st.columns(2)
    left.subheader("Input")
    left.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    right.subheader("Detections")
    right.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    names = results.names
    rows = []
    for box in results.boxes:
        rows.append(
            {
                "class": names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "x1": int(box.xyxy[0][0]),
                "y1": int(box.xyxy[0][1]),
                "x2": int(box.xyxy[0][2]),
                "y2": int(box.xyxy[0][3]),
            }
        )

    if rows:
        st.dataframe(rows, use_container_width=True)
        if custom:
            violations = [r for r in rows if r["class"].lower().startswith("no-")]
            if violations:
                st.error(
                    f"NON-COMPLIANCE: {len(violations)} violation(s) detected — "
                    f"{', '.join(sorted({r['class'] for r in violations}))}"
                )
            else:
                st.success("No PPE violations detected in this frame.")
    else:
        st.info("No objects above the confidence threshold.")
else:
    st.info("Upload an image to run the pipeline. Try one from data/images/.")
