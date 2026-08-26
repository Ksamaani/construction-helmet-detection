"""Export the trained helmet detector to ONNX and sanity-check it.

Deployment deliverable: model.export(format='onnx') + verification that the
exported model loads and predicts identically-shaped results through the same
Ultralytics API. ONNX Runtime (CPU) is used for inference of the export.

Usage:  python scripts/export_onnx.py [weights.pt]
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ultralytics import YOLO

weights = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "best.pt"
if not weights.exists():
    print(f"ERROR: {weights} not found. Train on Colab first and place best.pt in models/.")
    sys.exit(1)

sample = ROOT / "data" / "images" / "site_01.jpg"
log_lines = [f"=== ONNX EXPORT {time.strftime('%Y-%m-%d %H:%M:%S')} ===", f"weights: {weights}"]

print(f"[1] loading {weights}")
model = YOLO(str(weights))

print("[2] exporting to ONNX")
onnx_path = model.export(format="onnx", imgsz=640, opset=12, dynamic=False, simplify=True)
onnx_path = Path(onnx_path)
log_lines.append(f"exported: {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")

print("[3] sanity inference with the ONNX export (onnxruntime CPU)")
onnx_model = YOLO(str(onnx_path))
r_pt = model.predict(source=str(sample), conf=0.25, verbose=False)[0]
r_onnx = onnx_model.predict(source=str(sample), conf=0.25, verbose=False)[0]
print(f"    .pt : {len(r_pt.boxes)} detections")
print(f"    onnx: {len(r_onnx.boxes)} detections")
log_lines.append(f"detections .pt={len(r_pt.boxes)} onnx={len(r_onnx.boxes)} (sample: {sample.name})")
assert len(r_onnx.boxes) > 0, "ONNX model produced no detections - export broken"

summary = {
    "weights": str(weights),
    "onnx": str(onnx_path),
    "size_pt_mb": round(weights.stat().st_size / 1e6, 2),
    "size_onnx_mb": round(onnx_path.stat().st_size / 1e6, 2),
    "detections_pt": len(r_pt.boxes),
    "detections_onnx": len(r_onnx.boxes),
}
(ROOT / "models" / "export_summary.json").write_text(json.dumps(summary, indent=2))
log_lines.append(f"summary: {json.dumps(summary)}")
print(json.dumps(summary, indent=2))

log_path = ROOT / "logs" / "05_export_onnx.log"
log_path.parent.mkdir(exist_ok=True)
log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
print(f"log written: {log_path}")
