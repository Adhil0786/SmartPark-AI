# 🚗 SmartPark AI: Autonomous Parking Space Vision & Occupancy Intelligence

SmartPark AI is an end-to-end computer vision and deep learning system engineered for academic research, real-time parking monitoring, and automated benchmark evaluation on the **PKLot** dataset.

The system extracts parking slot polygons from raw camera feeds and camera annotations, performs real-time vehicle detection using **Ultralytics YOLOv8**, identifies slot occupancy via spatial polygon-vehicle intersection, and computes evaluation metrics against ground-truth XML annotations with zero temporal leakage.

---

## 🌟 Key Features

- **PKLot Dataset Integration:** Automatically discovers and parses 12,400+ multi-weather images across **UFPR04** (28 bays), **UFPR05** (40 bays), and **PUCPR** (100 bays).
- **Exact Polygon Extraction:** Parses rotated bounding rectangles and contour coordinate points directly from PKLot XML annotations.
- **Multiple Intelligence Engines (No Fake Predictions):**
  1. **YOLOv8 + Spatial Overlap (Deep Learning):** Real-time vehicle detection (cars, buses, trucks, motorcycles) mapped to parking polygons using localized ROI spatial intersection analysis.
  2. **Ground Truth XML (Gold Standard):** Directly reads real camera annotations for 100% verified ground truth.
  3. **Scikit-Learn ML Classifier:** Feature extraction pipeline (HSV color histogram, Canny edge density, Laplacian variance) + trained linear classifier.
  4. **Classic Computer Vision Baseline:** Canny edge density and intensity variance heuristic.
- **Publication-Grade Visualizations:** Translucent polygon fills (green for vacant, red for occupied), crisp vector boundaries, slot ID tags, and HUD information banners.
- **Date-Grouped Benchmark Evaluation (Zero Data Leakage):**
  - Grouped train/test split strictly partitioned by capture date.
  - Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, and Balanced Accuracy.
  - Interactive $2 \times 2$ Confusion Matrix ($\text{TP}, \text{TN}, \text{FP}, \text{FN}$).
  - Per-slot accuracy breakdown across all individual bays.
- **Research Artifacts & Export Engine:**
  - One-click publication-ready academic **PDF Report** generation.
  - Per-slot accuracy CSV exports and detailed benchmark record logs.
  - Current scene CSV audits and Markdown summaries.
  - Annotated high-resolution image downloads.
- **Custom Image Upload:**
  - Support for uploading custom camera images with auto-projected custom bays or matching XML files.

---

## 📁 Project Structure

```
SmartPark-AI/
├── app.py                      # Interactive Streamlit dashboard
├── config.py                   # Global system configurations & hyperparameters
├── requirements.txt            # Python dependencies
├── packages.txt                # Linux system dependencies for Streamlit Cloud
├── README.md                   # Documentation & academic guide
│
├── computer_vision/            # Core Computer Vision algorithms
│   ├── __init__.py
│   ├── xml_parser.py           # PKLot XML parser (polygons, rotatedRects, ground truth)
│   ├── yolo_detector.py        # Pretrained YOLOv8 vehicle detector & COCO filtering
│   ├── occupancy_classifier.py # Spatial IoU/overlap, ML, and heuristic classifiers
│   ├── visualizer.py           # OpenCV translucent overlays, HUD banners, borders
│   ├── metrics.py              # Academic evaluation suite & confusion matrix
│   ├── vision_engine.py        # High-level vision orchestration engine
│   └── benchmark_evaluator.py  # Grouped date-split evaluation & PDF/CSV reporting
│
├── dataset/                    # Dataset management
│   ├── __init__.py
│   ├── data_loader.py          # PKLot lot/weather/date indexer & random sampler
│   ├── raw/PKLot/              # Full-resolution PKLot images and XML annotations
│   └── processed/              # Processed crops & cache
│
├── models/                     # Machine learning models & weights
│   ├── __init__.py
│   ├── yolov8n.pt              # YOLOv8 nano object detection weights
│   ├── patch_classifier.py     # Scikit-Learn feature extractor & classifier
│   └── patch_svm_model.joblib  # Trained patch classification model
│
├── reports/                    # Generated academic reports and CSV audits
│   ├── __init__.py
│   ├── benchmark/              # Publication PDF & CSV benchmark reports
│   └── report_generator.py     # CSV and Markdown report generation engine
│
└── utils/                      # Mathematical and system utilities
    ├── __init__.py
    ├── geometry.py             # Vectorized polygon area, ROI clipping, overlap ratio
    └── file_utils.py           # Safe image loading, Unicode file paths, XML pairing
```

---

## 🚀 Quickstart: How to Run Locally

### 1. Activate Environment

```bash
cd SmartPark-AI
source venv/bin/activate
```

*(If setting up on a new environment: `pip install -r requirements.txt`)*

### 2. Launch the Streamlit App

```bash
streamlit run app.py
```

Streamlit will print the local URL:
```
Local URL: http://localhost:8501
```
Open **`http://localhost:8501`** in your browser.

---

## 🧪 Automated Test Suites

SmartPark AI includes comprehensive test suites covering all modules:

1. **Date-Grouped Benchmark Evaluation Test:**
   ```bash
   python test_date_split_evaluation.py
   ```
   Evaluates YOLO on held-out capture dates (zero temporal leakage), computes per-slot accuracy across all bays, and exports publication PDF + CSV reports.

2. **Computer Vision & Detection Pipeline Test:**
   ```bash
   python test_cv_module.py
   ```
   Tests all 8 core CV requirements: XML parsing, polygon extraction, ground truth stats, YOLO detection, comparative audit, custom polygons, and perspective rectification.

3. **PKLot Dataset Loader Test:**
   ```bash
   python test_loader.py
   ```
   Tests XML parsing across 16 diverse scenes spanning all lots, weathers, and edge cases (missing attributes, uppercase tag anomalies).

---

## ☁️ Streamlit Community Cloud Deployment Guide

SmartPark AI is pre-configured for direct, one-click deployment on **[Streamlit Community Cloud](https://share.streamlit.io/)** without memory exhaustion or dependency conflicts.

### 📦 Repository Checklist for Deployment

The repository includes everything needed to run immediately:
1. **Entry Point:** `app.py` in the root folder.
2. **Lightweight Demo Sample:** `dataset/sample/PKLot` contains 18 working image/XML pairs across all 3 lots (UFPR04, UFPR05, PUCPR) and 3 weather types (5.2 MB total).
3. **Pretrained Models:** `models/yolov8n.pt` (6.2 MB) and `models/patch_svm_model.joblib` (2.1 KB) are checked into the repository.
4. **Python Dependencies:** `requirements.txt` specifies `opencv-python-headless`, `ultralytics`, `streamlit`, `altair`, `scikit-learn`, etc.
5. **System Dependencies:** `packages.txt` provides Debian libraries (`libgl1-mesa-glx`, `libglib2.0-0`) required by OpenCV in headless cloud environments.
6. **Zero Hardcoded Paths:** All path resolutions use dynamic relative locations via `Path(__file__).resolve().parent`.
7. **Exclusion of Large Dataset:** `.gitignore` automatically excludes the 6.6 GB `dataset/raw/` folder and `venv/` so repository size remains under ~15 MB total.

---

### 🚀 Step-by-Step Deployment Steps

1. **Push Repository to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "feat: complete SmartPark AI deployment build"
   git branch -M main
   git remote add origin https://github.com/<your-username>/SmartPark-AI.git
   git push -u origin main
   ```

2. **Connect to Streamlit Community Cloud**:
   - Log in at **[share.streamlit.io](https://share.streamlit.io/)** with your GitHub account.
   - Click **"Create app"** or **"Deploy an app"**.

3. **Configure App Settings**:
   - **Repository:** `<your-username>/SmartPark-AI`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** (Choose an available custom subdomain, e.g. `smartpark-ai.streamlit.app`)

4. **Deploy**:
   - Click **"Deploy!"**.
   - Streamlit Cloud will read `packages.txt`, install dependencies from `requirements.txt`, and boot the dashboard within ~2 minutes.

---

### 💡 Operating Modes on Streamlit Cloud

- **Demo Dataset Mode:** Automatically detects `dataset/sample/PKLot` when running in cloud mode. Users can test UFPR04, UFPR05, and PUCPR parking feeds, run YOLO vehicle inference, and generate real academic benchmarks.
- **Custom Image/XML Upload Mode:** Users can upload any parking image (JPG/PNG) and optional PKLot XML annotations. When no XML is uploaded, the system auto-projects a grid of customizable parking bays for live spatial tracking.
- **Full PKLot Dataset (Local Run):** When running locally with the full 12,400+ PKLot dataset in `dataset/raw/PKLot/PKLot`, the application automatically prioritizes the full archive.

---

## 🧠 Methodology & Mathematical Formulation

### 1. Spatial Overlap Ratio

Given a parking space polygon $\mathcal{P}$ and a set of detected vehicle bounding boxes $\mathcal{B} = \{B_1, B_2, \dots, B_k\}$, the spatial overlap ratio $\Omega(\mathcal{P})$ is defined as:

$$\Omega(\mathcal{P}) = \max_{B \in \mathcal{B}} \left( \frac{\text{Area}(\mathcal{P} \cap B)}{\text{Area}(\mathcal{P})} \right)$$

A parking slot is classified as **Occupied** if $\Omega(\mathcal{P}) \ge \tau_{\text{overlap}}$ (default $\tau = 0.20$), otherwise **Vacant**.

### 2. Evaluation Metrics

Comparing model predictions against ground truth labels:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}, \quad \text{F}_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{Balanced Accuracy} = \frac{\text{Recall} + \text{Specificity}}{2}$$

---

## 🔬 Dataset Overview (PKLot)

The system works directly with the **PKLot** benchmark:
- **UFPR04**: Federal University of Paraná (28 slots, elevated perspective)
- **UFPR05**: Federal University of Paraná (40 slots, top-right perspective)
- **PUCPR**: Pontifical Catholic University of Paraná (100 slots, high-altitude wide view)
- **Conditions**: Sunny, Cloudy, Rainy frames taken across different times of day.
