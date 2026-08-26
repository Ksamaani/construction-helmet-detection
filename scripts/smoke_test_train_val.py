"""Local CPU smoke test of the exact code paths used in the Colab notebook 03.

Validates: kagglehub download, data.yaml authoring, class distribution EDA,
model.train() (1 epoch, tiny imgsz), model.val(), per-class metrics extraction,
and a 2-point confidence sweep. NOT evidence of training - just API verification.
"""

import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import kagglehub
import yaml
from ultralytics import YOLO

t0 = time.time()
print("=== SMOKE TEST START ===")

print("\n[1] kagglehub download")
p = kagglehub.dataset_download("snehilsanyal/construction-site-safety-image-dataset-roboflow")
print("downloaded to:", p)
DATA_ROOT = Path(p) / "css-data"
print("exists:", DATA_ROOT.exists(), "| subdirs:", [d.name for d in DATA_ROOT.iterdir() if d.is_dir()])

print("\n[2] data.yaml authoring")
NAMES = {
    0: "Hardhat", 1: "Mask", 2: "NO-Hardhat", 3: "NO-Mask", 4: "NO-Safety Vest",
    5: "Person", 6: "Safety Cone", 7: "Safety Vest", 8: "machinery", 9: "vehicle",
}
yaml_path = Path("smoke_css_data.yaml").resolve()
yaml_path.write_text(yaml.safe_dump({
    "path": str(DATA_ROOT),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": NAMES,
}))
print(yaml_path.read_text())

print("\n[3] split sizes")
for split in ("train", "valid", "test"):
    n = len(list((DATA_ROOT / split / "images").glob("*")))
    print(f"  {split}: {n} images")

print("\n[4] label class distribution (train)")
from collections import Counter
cnt = Counter()
for lf in (DATA_ROOT / "train" / "labels").glob("*.txt"):
    for line in lf.read_text().splitlines():
        if line.strip():
            cnt[int(line.split()[0])] += 1
for cid in sorted(cnt):
    print(f"  {cid} {NAMES[cid]}: {cnt[cid]}")

print("\n[5] train 1 epoch @320 (CPU)")
model = YOLO("yolo11n.pt")
model.train(
    data=str(yaml_path), epochs=1, imgsz=320, batch=4, device="cpu",
    project="smoke_runs", name="smoke", plots=False, verbose=True,
)

print("\n[6] val with per-class metrics")
best = YOLO("smoke_runs/smoke/weights/best.pt")
metrics = best.val(data=str(yaml_path), split="val", plots=False, verbose=True)
print("mAP50:", round(float(metrics.box.map50), 4))
print("mAP50-95:", round(float(metrics.box.map), 4))
print("precision:", round(float(metrics.box.mp), 4))
print("recall:", round(float(metrics.box.mr), 4))
names = metrics.names
for i in metrics.box.ap_class_index:
    print(f"  {names[int(i)]}: P={metrics.box.p[list(metrics.box.ap_class_index).index(i)]:.3f} "
          f"R={metrics.box.r[list(metrics.box.ap_class_index).index(i)]:.3f} "
          f"AP50={metrics.box.ap50[list(metrics.box.ap_class_index).index(i)]:.3f}")
print("save_dir:", metrics.save_dir)

print("\n[7] confidence sweep (2 points)")
for conf in (0.10, 0.50):
    m = best.val(data=str(yaml_path), split="val", conf=conf, iou=0.6, plots=False, verbose=False)
    mp, mr = float(m.box.mp), float(m.box.mr)
    f1 = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
    print(f"  conf={conf}: P={mp:.3f} R={mr:.3f} F1={f1:.3f}")

print(f"\n=== SMOKE TEST OK in {time.time()-t0:.0f}s ===")
