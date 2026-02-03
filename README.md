# Official Code for CUE-YOLO

**CUE-YOLO: An Efficient and Robust Detector for Metal Surface Defect Detection with Adaptive Attention and Task Alignment**

---

## 📌 Overview

This repository provides the **official PyTorch implementation** of **CUE-YOLO**, a task-driven architectural refinement of the YOLO framework for **metal surface defect detection**.

CUE-YOLO introduces targeted refinements across the detection pipeline, including **adaptive feature extraction**, **unified task-aligned prediction**, and **dynamic loss optimization**, which significantly improve sensitivity to subtle defects, reduce classification–localization misalignment, and enhance robustness under **severely imbalanced defect distributions**.

Extensive experiments on the **GC10-DET dataset** and cross-domain benchmarks demonstrate the effectiveness, efficiency, and generalization capability of the proposed method.

---

## 🚀 Contributions

The main contributions of this work are summarized as follows:

### 🔹 C3K2-ASA (Adaptive Sampling Attention)

To address the limitation of rigid convolutions in capturing faint and irregular defects, we propose **C3K2-ASA**, an adaptive sampling attention module that dynamically refines local receptive fields based on contextual cues.
This design enables more precise modeling of subtle defect patterns with **minimal computational overhead**.

---

### 🔹 UTAH (Unified Task-Aligned Head)

To alleviate spatial misalignment caused by decoupled detection heads, we design **UTAH**, a unified task-aligned detection head that harmonizes classification and localization through:

* Shared parameterization
* Cross-scale feature fusion
* Deformable refinement

This unified formulation effectively reduces task inconsistency and improves localization accuracy.

---

### 🔹 EMA-SlideLoss

To overcome the failure of static loss functions under severe class imbalance, we introduce **EMA-SlideLoss**, a dynamic loss adjustment strategy that:

* Incorporates **Exponential Moving Average (EMA)** updates
* Adaptively adjusts IoU thresholds during training

This mechanism provides stronger supervision for hard and minority defect samples.

---

## 🗂️ Repository Structure

The codebase is built upon the **Ultralytics YOLO framework** and organized as follows:

```
.
├── app_without_gt.py            # Visualization without ground truth
├── batch_detect.py              # Batch inference
├── batch_val.py                 # Batch validation
├── cfg                          # Model configuration files
│   ├── attention                # Adaptive attention modules (C3K2-ASA)
│   ├── head                     # Detection head definitions (UTAH)
│   ├── yolo                     # Base YOLO components
│   ├── yolo11-C3K2ASA.yaml
│   ├── yolo11-UTAH-C3K2ASA.yaml
│   └── yolo11-UTAH.yaml
├── detect.py                    # Detection entry
├── export.py                    # Model export (ONNX, etc.)
├── get_COCO_metrice.py          # COCO-style metrics
├── get_FPS.py                   # FPS benchmarking
├── get_model_erf.py             # Effective receptive field analysis
├── heatmap.py                   # Attention / Grad-CAM visualization
├── main_profile.py              # Profiling
├── merge_val_metrics.py         # Validation aggregation
├── plot_result.py               # Result visualization
├── summarize_val.py             # Validation summary
├── track.py                     # Object tracking
├── train.py                     # Training entry
├── transform_PGI.py             # Data preprocessing
├── ultralytics                  # Modified Ultralytics YOLO core
│   ├── models                   # Backbone / Neck / Head
│   ├── nn                       # Custom neural modules
│   └── utils                    # Utility functions
├── val.py                       # Validation entry
├── visualization_app_simple.py  # Lightweight visualization app
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-repo/CUE-YOLO.git
cd CUE-YOLO
pip install -r requirements.txt
```

**Requirements**

* Python ≥ 3.8
* PyTorch ≥ 1.12
* CUDA-enabled GPU (recommended)

---

## 🧪 Training

Train CUE-YOLO using the full configuration:

```bash
python train.py \
  --name CUE-YOLO \
  --model yolo11-UTAH-C3K2ASA \
  --loss EMASlideLoss \
  --dataset GC-DET-new
```

Available model variants:

* `yolo11-C3K2ASA.yaml`
* `yolo11-UTAH.yaml`
* `yolo11-UTAH-C3K2ASA.yaml` (Full model)

---

## 🔍 Evaluation

```bash
python val.py
```

Additional evaluation tools:

* `get_FPS.py` – inference speed
* `get_COCO_metrice.py` – COCO metrics
* `get_model_erf.py` – receptive field analysis

---



