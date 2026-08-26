"""One tracking pass over the site clip; evaluate candidate counting lines.

Metric = distinct track IDs that ever cross the line (matches ObjectCounter's
count-once-per-track semantics).
"""

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "videos" / "site_clip.mp4"

model = YOLO("yolo11n.pt")
tracks = defaultdict(list)

t0 = time.time()
results = model.track(source=str(VIDEO), stream=True, persist=True, classes=[0], conf=0.25, verbose=False)
for f, r in enumerate(results):
    if r.boxes.id is None:
        continue
    ids = r.boxes.id.int().tolist()
    xy = r.boxes.xywh.tolist()
    for tid, (cx, cy, w, h) in zip(ids, xy):
        tracks[tid].append((f, cx, cy))
print(f"tracked {len(tracks)} IDs in {time.time()-t0:.0f}s")

for frac in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65):
    line = 1280 * frac
    crossers = 0
    detail = []
    for tid, rows in tracks.items():
        rows = sorted(rows)
        side = None
        crossed = False
        for f, cx, cy in rows:
            s = 1 if cx > line else 0
            if side is not None and s != side:
                crossed = True
            side = s
        if crossed:
            crossers += 1
            detail.append(tid)
    print(f"line x={frac:.2f}W ({line:.0f}px): {crossers} distinct IDs cross -> {detail}")
