"""Build and execute notebooks/01_inference.ipynb (Deliverable 1).

Generates the notebook programmatically, executes it in-place with nbclient
so all cell outputs are captured as evidence, and writes a run log.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import nbformat
from nbclient import NotebookClient

NB_PATH = ROOT / "notebooks" / "01_inference.ipynb"
LOG_PATH = ROOT / "logs" / "01_inference_run.log"


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source)


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source)


cells = [
    md(
        "# 01 — Inference Pipeline: Detection, Segmentation & Pose\n"
        "\n"
        "**Capstone deliverable 1 (25 pts)** — *Core Vision Tasks & Inference*.\n"
        "\n"
        "This notebook runs the Ultralytics Python API (`model.predict`) over **real construction-site\n"
        "photographs** (Wikimedia Commons, CC-licensed — see `logs/01_assets_download.log`) and covers:\n"
        "\n"
        "1. **Object detection** with `yolo11n.pt` — locate every worker (COCO class `person`).\n"
        "2. **Instance segmentation** with `yolo11n-seg.pt` — pixel-accurate worker masks *(beyond detection #1)*.\n"
        "3. **Pose estimation** with `yolo11n-pose.pt` — 17-keypoint worker skeletons *(beyond detection #2)*.\n"
        "4. A **PPE gap analysis**: why helmet compliance needs the custom model trained in `03_train_eval.ipynb`.\n"
        "\n"
        "All annotated evidence images are written to `outputs/`."
    ),
    code(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "import cv2\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import ultralytics\n"
        "from ultralytics import YOLO\n"
        "\n"
        "ultralytics.checks()\n"
        "\n"
        "ROOT = Path.cwd()\n"
        "while not (ROOT / 'data').exists():\n"
        "    ROOT = ROOT.parent\n"
        "IMG_DIR = ROOT / 'data' / 'images'\n"
        "OUT_DIR = ROOT / 'outputs'\n"
        "OUT_DIR.mkdir(exist_ok=True)"
    ),
    md(
        "## 1 — The input data: real construction sites\n"
        "\n"
        "Eight real site photographs fetched by `scripts/download_assets.py`. They include workers\n"
        "**with** helmets (sites 01, 03, 05), workers in hi-vis without helmets (site 07), and\n"
        "challenging scenes (night break room, distant crane workers) to probe model robustness."
    ),
    code(
        "image_paths = sorted(IMG_DIR.glob('*.jpg'))\n"
        "print(f'Found {len(image_paths)} real construction-site images:')\n"
        "for p in image_paths:\n"
        "    print(f'  {p.name}  ({p.stat().st_size/1024:.0f} KB)')\n"
        "\n"
        "fig, axes = plt.subplots(2, 4, figsize=(20, 9))\n"
        "for ax, p in zip(axes.ravel(), image_paths):\n"
        "    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)\n"
        "    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\n"
        "    ax.set_title(p.name, fontsize=9)\n"
        "    ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 2 — Object detection (`yolo11n.pt`, COCO-pretrained)\n"
        "\n"
        "Detection is filtered to `classes=[0]` (person) because PPE compliance monitoring starts\n"
        "from knowing **where every worker is**. Confidence gate: `conf=0.25`."
    ),
    code(
        "det_model = YOLO('yolo11n.pt')\n"
        "\n"
        "rows = []\n"
        "results_by_image = {}\n"
        "for p in image_paths:\n"
        "    results = det_model.predict(source=str(p), classes=[0], conf=0.25, verbose=True)\n"
        "    r = results[0]\n"
        "    results_by_image[p.name] = r\n"
        "    confs = [float(c) for c in r.boxes.conf]\n"
        "    rows.append({\n"
        "        'image': p.name,\n"
        "        'persons_detected': len(r.boxes),\n"
        "        'mean_conf': round(float(np.mean(confs)), 3) if confs else None,\n"
        "        'max_conf': round(float(np.max(confs)), 3) if confs else None,\n"
        "    })\n"
        "\n"
        "det_summary = pd.DataFrame(rows)\n"
        "det_summary"
    ),
    code(
        "showcase = ['site_01.jpg', 'site_03.jpg', 'site_05.jpg']\n"
        "for name in showcase:\n"
        "    r = results_by_image[name]\n"
        "    out = OUT_DIR / f'01_detect_{name}'\n"
        "    r.save(filename=str(out))\n"
        "    print(f'saved {out.name}: {len(r.boxes)} person boxes')\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(22, 7))\n"
        "for ax, name in zip(axes, showcase):\n"
        "    img = cv2.imdecode(np.fromfile(str(OUT_DIR / f'01_detect_{name}'), dtype=np.uint8), cv2.IMREAD_COLOR)\n"
        "    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\n"
        "    ax.set_title(name)\n"
        "    ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 3 — Beyond detection #1: instance segmentation (`yolo11n-seg.pt`)\n"
        "\n"
        "Segmentation gives pixel-accurate worker silhouettes instead of rectangles. On a site this\n"
        "supports zone-intrusion analytics (is the worker's *body* inside the exclusion zone?) and\n"
        "cleaner occlusion handling between adjacent workers."
    ),
    code(
        "seg_model = YOLO('yolo11n-seg.pt')\n"
        "\n"
        "seg_names = ['site_03.jpg', 'site_01.jpg']\n"
        "for name in seg_names:\n"
        "    r = seg_model.predict(source=str(IMG_DIR / name), classes=[0], conf=0.25, verbose=True)[0]\n"
        "    n_masks = 0 if r.masks is None else len(r.masks)\n"
        "    poly_pts = 0 if r.masks is None else sum(len(xy) for xy in r.masks.xy)\n"
        "    shape = None if r.masks is None else tuple(r.masks.data.shape)\n"
        "    print(f'{name}: masks tensor {shape} -> {n_masks} person instances, '\n"
        "          f'{poly_pts} polygon vertices total')\n"
        "    r.save(filename=str(OUT_DIR / f'01_seg_{name}'))\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(18, 7))\n"
        "for ax, name in zip(axes, seg_names):\n"
        "    img = cv2.imdecode(np.fromfile(str(OUT_DIR / f'01_seg_{name}'), dtype=np.uint8), cv2.IMREAD_COLOR)\n"
        "    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\n"
        "    ax.set_title(f'segmented: {name}')\n"
        "    ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 4 — Beyond detection #2: pose estimation (`yolo11n-pose.pt`)\n"
        "\n"
        "Pose yields 17 body keypoints per worker. Safety-relevant uses: detecting bent/crouched\n"
        "postures in lifting operations, raised-arm danger gestures, or workers climbing where\n"
        "they should not."
    ),
    code(
        "pose_model = YOLO('yolo11n-pose.pt')\n"
        "\n"
        "pose_names = ['site_01.jpg', 'site_03.jpg', 'site_07.jpg']\n"
        "for name in pose_names:\n"
        "    r = pose_model.predict(source=str(IMG_DIR / name), conf=0.25, verbose=True)[0]\n"
        "    n = 0 if r.keypoints is None else len(r.keypoints)\n"
        "    shape = None if r.keypoints is None else tuple(r.keypoints.data.shape)\n"
        "    print(f'{name}: {n} worker skeletons, keypoints tensor {shape}')\n"
        "    r.save(filename=str(OUT_DIR / f'01_pose_{name}'))\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(22, 7))\n"
        "for ax, name in zip(axes, pose_names):\n"
        "    img = cv2.imdecode(np.fromfile(str(OUT_DIR / f'01_pose_{name}'), dtype=np.uint8), cv2.IMREAD_COLOR)\n"
        "    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\n"
        "    ax.set_title(f'pose: {name}')\n"
        "    ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 5 — PPE gap analysis: why a custom helmet model is required\n"
        "\n"
        "The COCO label space has no helmet class. A pretrained detector can find workers but can\n"
        "**never** decide whether a worker is compliant — the core capability of this project.\n"
        "That is exactly why deliverable 3 fine-tunes a custom helmet/head detector on a labeled\n"
        "PPE dataset."
    ),
    code(
        "coco_names = det_model.names\n"
        "helmet_like = [n for n in coco_names.values() if 'helmet' in n or 'hat' in n]\n"
        "print(f'COCO class count: {len(coco_names)}')\n"
        "print(f'Helmet-related classes in COCO: {helmet_like if helmet_like else \"NONE\"}')\n"
        "print()\n"
        "print('=> Pretrained models locate workers but cannot verify helmet compliance.')\n"
        "print('=> 03_train_eval.ipynb fine-tunes a custom helmet detector to close this gap.')\n"
        "det_summary"
    ),
    md(
        "## 6 — Evidence produced by this notebook"
    ),
    code(
        "print('Annotated evidence files:')\n"
        "for f in sorted(OUT_DIR.glob('01_*')):\n"
        "    print(f'  {f.name}  ({f.stat().st_size/1024:.0f} KB)')"
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}

NB_PATH.parent.mkdir(exist_ok=True)
start = time.time()
client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB_PATH.parent)}})
client.execute()
nbformat.write(nb, NB_PATH)
elapsed = time.time() - start

log = (
    "=== 01_inference.ipynb EXECUTION LOG ===\n"
    f"timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    f"notebook  : {NB_PATH}\n"
    f"status    : SUCCESS (all cells executed, outputs captured)\n"
    f"duration  : {elapsed:.1f}s\n"
    "models    : yolo11n.pt (detect), yolo11n-seg.pt (segment), yolo11n-pose.pt (pose)\n"
    "api calls : model.predict() x13 (8 detect + 2 seg + 3 pose)\n"
)
print(log)
LOG_PATH.write_text(log, encoding="utf-8")
