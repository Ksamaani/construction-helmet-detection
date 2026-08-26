# Training & Dataset Documentation

## Dataset

| | |
|---|---|
| **Name** | Construction Site Safety Image Dataset (Roboflow) |
| **Kaggle** | [`snehilsanyal/construction-site-safety-image-dataset-roboflow`](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) |
| **Origin** | Roboflow Universe — *Construction Site Safety* project |
| **License** | CC BY 4.0 |
| **Format** | YOLO (normalized `class cx cy w h` .txt labels) |
| **Images** | 2,641 — train 2,445 · valid 114 · test 82 |
| **Download** | Anonymous, no API key: `kagglehub.dataset_download("snehilsanyal/construction-site-safety-image-dataset-roboflow")` |

### Classes (10)

| id | name | id | name |
|---|---|---|---|
| 0 | `Hardhat` | 5 | `Person` |
| 1 | `Mask` | 6 | `Safety Cone` |
| 2 | `NO-Hardhat` | 7 | `Safety Vest` |
| 3 | `NO-Mask` | 8 | `machinery` |
| 4 | `NO-Safety Vest` | 9 | `vehicle` |

Compliance semantics come from the paired classes: a **violation is a positive detection**
of `NO-Hardhat` / `NO-Mask` / `NO-Safety Vest`, not the absence of a box — which is what
makes the FN-vs-FP analysis in the notebook meaningful.

Known caveats: valid/test splits are small (114/82) — good enough for model selection and
reporting, not for hyperparameter tuning; class counts are imbalanced (Hardhat/Person/Safety
Vest dominate; `machinery`/`vehicle`/mask classes are rare and will score lower).

## Training runs (Google Colab, T4 GPU)

Base weights: `yolo11n.pt` (COCO-pretrained, transfer learning). Seed 42 both runs.

| knob | Run A — baseline | Run B — tuned | rationale |
|---|---|---|---|
| epochs | 20 | 30 (patience 10) | headroom + early stop |
| imgsz | 640 | 640 | keep runtime comparable |
| batch | -1 (AutoBatch) | -1 (AutoBatch) | fill GPU memory |
| freeze | 0 | **10** | keep generic COCO backbone features, train PPE-specific head |
| weight_decay | 0.0005 (default) | **0.001** | stronger L2 against memorisation |
| hsv_v | 0.4 (default) | **0.5** | harsh site lighting / shadows |
| degrees | 0 | **10** | cameras never perfectly level |
| translate | 0.1 | **0.2** | workers appear off-centre |
| scale | 0.5 | **0.7** | large distance range from camera |
| fliplr | 0.5 | 0.5 | — |
| mosaic | 1.0 | 1.0 (close_mosaic 10) | disable late for stable finish |

## Results

> To be filled from the executed Colab notebook (`notebooks/03_train_eval.ipynb`) after the
> GPU run — see `metrics.json` and the run curves in the notebook output.

- Run A best mAP50-95: _pending_
- Run B best mAP50-95: _pending_
- Overfitting/underfitting observations: _pending_
- Chosen deployment conf threshold: _pending_

## Local verification

Before shipping the Colab notebook, every API call it makes (kagglehub download, data.yaml
authoring, `model.train`, `model.val` with per-class metrics, confidence sweep) was
smoke-tested locally on CPU with a 1-epoch/320px run — see `logs/03_smoke_test_local.log`.
