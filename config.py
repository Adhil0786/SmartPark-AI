"""
SmartPark AI - Configuration Module
Academic Computer Vision & Parking Occupancy Intelligence System
"""

import os
from pathlib import Path

# Set Ultralytics config dir to /tmp or models to prevent sandbox permission errors
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
os.environ["ULTRALYTICS_SETTINGS"] = "false"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Ensure Streamlit runs without interactive prompt
try:
    tmp_streamlit = Path("/tmp/.streamlit")
    tmp_streamlit.mkdir(parents=True, exist_ok=True)
    cred_file = tmp_streamlit / "credentials.toml"
    if not cred_file.exists():
        cred_file.write_text('[general]\nemail = ""\n', encoding="utf-8")
except Exception:
    pass

# Directory paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_PKLOT_DIR = DATASET_DIR / "raw" / "PKLot" / "PKLot"
SEGMENTED_PKLOT_DIR = DATASET_DIR / "raw" / "PKLot" / "PKLotSegmented"
SAMPLE_PKLOT_DIR = DATASET_DIR / "sample" / "PKLot"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
UTILS_DIR = BASE_DIR / "utils"
CV_DIR = BASE_DIR / "computer_vision"


def get_active_dataset_dir() -> Path:
    """
    Returns the primary dataset directory:
    1. Full RAW_PKLOT_DIR if present on disk.
    2. Fallback to lightweight SAMPLE_PKLOT_DIR (included in repo for Streamlit Cloud).
    3. Defaults to RAW_PKLOT_DIR.
    """
    if RAW_PKLOT_DIR.exists() and any(RAW_PKLOT_DIR.iterdir()):
        return RAW_PKLOT_DIR
    if SAMPLE_PKLOT_DIR.exists() and any(SAMPLE_PKLOT_DIR.iterdir()):
        return SAMPLE_PKLOT_DIR
    return RAW_PKLOT_DIR


ACTIVE_PKLOT_DIR = get_active_dataset_dir()

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
YOLO_WEIGHTS_PATH = MODELS_DIR / "yolov8n.pt"
PATCH_CLASSIFIER_PATH = MODELS_DIR / "patch_svm_model.joblib"

# Detection & Classification defaults
DEFAULT_YOLO_CONF = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_OVERLAP_THRESHOLD = 0.20  # Fraction of slot polygon occupied by vehicle box
ALPHA_OVERLAY = 0.35              # Polygon overlay transparency

# Vehicle classes in COCO (Ultralytics YOLO)
# 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Color configurations (BGR for OpenCV, RGB/Hex for Streamlit/UI)
COLORS = {
    "vacant_bgr": (46, 204, 113),      # Green #2ECC71
    "occupied_bgr": (60, 76, 231),     # Red #E74C3C
    "detected_box_bgr": (255, 165, 0), # Orange
    "highlight_bgr": (255, 255, 0),    # Cyan/Yellow
    "vacant_hex": "#2ECC71",
    "occupied_hex": "#E74C3C",
    "neutral_hex": "#3498DB",
    "bg_dark": "#0E1117"
}

# Known PKLot Lots
LOT_METADATA = {
    "UFPR04": {
        "name": "UFPR - Parking Lot 04",
        "spaces": 28,
        "camera_angle": "Elevated Oblique",
        "description": "Federal University of Paraná - Lot 04"
    },
    "UFPR05": {
        "name": "UFPR - Parking Lot 05",
        "spaces": 40,
        "camera_angle": "Top-Right Perspective",
        "description": "Federal University of Paraná - Lot 05"
    },
    "PUCPR": {
        "name": "PUCPR - Campus Curitiba",
        "spaces": 100,
        "camera_angle": "High Angle Wide View",
        "description": "Pontifical Catholic University of Paraná"
    }
}
