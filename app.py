"""
SmartPark AI - Modern Autonomous Parking Vision & Analytics Dashboard
Designed with a Dark Navy & White Professional Theme.
Integrates live PKLot dataset exploration, custom image uploads,
YOLOv8 vehicle detection, real ground-truth evaluation, and analytics reports.
"""

import io
import os
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# Custom modules
import config
from dataset.data_loader import PKLotDataset
from computer_vision.vision_engine import SmartParkVisionEngine
from computer_vision.xml_parser import PKLotXMLParser, ParkingSpace
from computer_vision.yolo_detector import YOLOVehicleDetector, DetectedVehicle
from computer_vision.occupancy_classifier import OccupancyClassifier, SlotClassificationResult
from computer_vision.visualizer import ParkingVisualizer
from computer_vision.metrics import EvaluationMetrics
from computer_vision.benchmark_evaluator import DateSplitEvaluator, BenchmarkResult
from reports.report_generator import ReportGenerator
from utils.file_utils import load_image
from components.polygon_editor import (
    render_polygon_editor,
    DEFAULT_2_ROW_TEMPLATE,
    normalized_to_pixel_spaces
)


# ==========================================================
# 1. PAGE SETUP & DARK NAVY / WHITE PROFESSIONAL THEME
# ==========================================================
st.set_page_config(
    page_title="SmartPark AI | Vision & Occupancy Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

DARK_NAVY_THEME = """
<style>
    /* Dark Navy & Crisp White Styling */
    :root {
        --navy-darkest: #070F2B;
        --navy-primary: #0B192C;
        --navy-secondary: #1E3E62;
        --navy-accent: #007BFF;
        --navy-light: #F0F4F8;
        --text-main: #0F172A;
        --text-muted: #64748B;
        --card-bg: #FFFFFF;
        --green-vacant: #10B981;
        --red-occupied: #EF4444;
        --amber-warning: #F59E0B;
    }

    /* Main Container Header */
    .dashboard-header {
        background: linear-gradient(135deg, #070F2B 0%, #0B192C 50%, #1E3E62 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(7, 15, 43, 0.2);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .dashboard-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .dashboard-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 6px;
    }
    .system-status-pill {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #10B981;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(11, 25, 44, 0.08);
        border-color: #CBD5E1;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
    }
    .c-total { color: #0B192C; }
    .c-vacant { color: #10B981; }
    .c-occupied { color: #EF4444; }
    .c-rate { color: #1E3E62; }
    .c-vehicles { color: #F59E0B; }

    /* Visual status pills in main view */
    .status-pill-vacant {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .status-pill-occupied {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .status-pill-vehicle {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Sidebar headers */
    .css-1d391kg {
        background-color: #0B192C;
    }
</style>
"""
st.markdown(DARK_NAVY_THEME, unsafe_allow_html=True)


# ==========================================================
# 2. CACHED ENGINE & DATASET ACCESS
# ==========================================================
@st.cache_resource(show_spinner="Initializing Computer Vision Engine & YOLOv8...")
def get_cv_engine(yolo_conf: float = config.DEFAULT_YOLO_CONF,
                  overlap_thresh: float = config.DEFAULT_OVERLAP_THRESHOLD) -> SmartParkVisionEngine:
    """Cache the high-level vision engine with YOLO."""
    return SmartParkVisionEngine(
        yolo_conf=yolo_conf,
        overlap_threshold=overlap_thresh
    )


@st.cache_resource(show_spinner="Indexing PKLot Dataset Structure...")
def get_dataset_loader() -> PKLotDataset:
    """Cache dataset explorer."""
    return PKLotDataset()


dataset_loader = get_dataset_loader()


@st.cache_data(show_spinner="Running reproducible date-grouped model evaluation...")
def run_cached_benchmark(target_lot: str,
                         max_frames: int = 15,
                         conf: float = config.DEFAULT_YOLO_CONF,
                         overlap: float = config.DEFAULT_OVERLAP_THRESHOLD):
    """Executes a reproducible train/test split grouped by capture date and evaluates YOLO."""
    evaluator = DateSplitEvaluator(dataset_loader)
    all_pairs = [p for p in dataset_loader.find_all_pairs() if p.has_xml and p.lot == target_lot]
    if not all_pairs:
        all_pairs = [p for p in dataset_loader.find_all_pairs() if p.has_xml]

    train_p, test_p, train_d, test_d = DateSplitEvaluator.split_by_capture_date(
        pairs=all_pairs, test_size=0.30, random_state=42
    )
    res = evaluator.run_evaluation(
        test_pairs=test_p,
        model_name="YOLOv8 + Spatial Overlap (Deep Learning)",
        max_frames=max_frames,
        yolo_conf=conf,
        overlap_threshold=overlap
    )
    res.train_dates = train_d
    pdf_p = DateSplitEvaluator.export_pdf_report(res)
    per_slot_csv, records_csv = DateSplitEvaluator.export_csv_reports(res)
    return res, str(pdf_p), str(per_slot_csv), str(records_csv), train_d, test_d


# ==========================================================
# 3. SIDEBAR NAVIGATION & CONTROLS
# ==========================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
        <span style="font-size: 2rem;">🚗</span>
        <div>
            <div style="font-weight: 800; font-size: 1.25rem; color: #FFFFFF; line-height: 1;">SmartPark AI</div>
            <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Vision Intelligence System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # SECTION 1: DATA SOURCE
    st.markdown("#### 📁 1. Data Source")
    source_choice = st.radio(
        "Choose image input mode:",
        ["PKLot Benchmark Dataset", "Upload Custom Image"],
        index=0,
        label_visibility="collapsed"
    )

    active_image_bgr: Optional[np.ndarray] = None
    active_parser: Optional[PKLotXMLParser] = None
    custom_spaces: List[ParkingSpace] = []
    lot_name = "UFPR05"
    weather_name = "Cloudy"
    date_name = "N/A"
    frame_name = "N/A"

    if source_choice == "PKLot Benchmark Dataset":
        lots = dataset_loader.get_lots()
        if lots:
            c_lot, c_rnd = st.columns([3, 1])
            with c_lot:
                selected_lot = st.selectbox("Parking Lot:", lots, index=lots.index("UFPR05") if "UFPR05" in lots else 0)
            with c_rnd:
                st.write("")
                st.write("")
                rnd_clicked = st.button("🎲", help="Sample random frame across dataset")

            lot_meta = dataset_loader.get_lot_info(selected_lot)
            st.caption(f"📍 {lot_meta.get('name')} • Capacity: **{lot_meta.get('spaces')}** bays")

            weathers = dataset_loader.get_weathers(selected_lot)
            selected_weather = st.selectbox("Weather Condition:", weathers, index=0)

            dates = dataset_loader.get_dates(selected_lot, selected_weather)
            selected_date = st.selectbox("Date Recorded:", dates, index=0)

            img_paths = dataset_loader.get_images(selected_lot, selected_weather, selected_date)
            if img_paths:
                img_names = [p.name for p in img_paths]

                if rnd_clicked and img_names:
                    import random
                    st.session_state["sel_frame_idx"] = random.randint(0, len(img_names) - 1)

                default_idx = st.session_state.get("sel_frame_idx", 0)
                if default_idx >= len(img_names):
                    default_idx = 0

                selected_frame = st.selectbox(
                    f"Frame Timestamp ({len(img_names)} captures):",
                    img_names,
                    index=default_idx
                )
                target_img_path = next((p for p in img_paths if p.name == selected_frame), img_paths[0])
                active_image_bgr, active_parser = dataset_loader.load_pair(target_img_path)
                lot_name = selected_lot
                weather_name = selected_weather
                date_name = selected_date
                frame_name = selected_frame
            else:
                st.warning("No images available in this subset.")
        else:
            st.warning("⚠️ No local PKLot dataset detected on server. Please switch to **'Upload Custom Image'** above to analyze parking camera feeds.")
    else:
        # CUSTOM UPLOAD MODE
        up_img = st.file_uploader("Upload Parking Camera Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        up_xml = st.file_uploader("Upload Matching XML (Optional)", type=["xml"])

        if up_img is not None:
            file_bytes = np.asarray(bytearray(up_img.read()), dtype=np.uint8)
            active_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            frame_name = up_img.name

            if up_xml is not None:
                xml_bytes = up_xml.read()
                try:
                    active_parser = PKLotXMLParser(xml_bytes)
                    lot_name = active_parser.parking_id
                except Exception as e:
                    st.error(f"Error parsing XML: {e}")
            else:
                lot_name = "Custom Upload (No XML)"
                weather_name = "Camera Feed"

                if "custom_workflow_stage" not in st.session_state:
                    st.session_state["custom_workflow_stage"] = "✏️ Step 1: Interactive Polygon Editor"
                if "custom_bays_normalized" not in st.session_state or not st.session_state["custom_bays_normalized"]:
                    st.session_state["custom_bays_normalized"] = [dict(b) for b in DEFAULT_2_ROW_TEMPLATE]

                st.markdown("##### 📐 Parking Bay Workflow")
                stage_choice = st.radio(
                    "Workflow Stage:",
                    [
                        "✏️ Step 1: Interactive Polygon Editor",
                        "🔍 Step 2: Occupancy Analysis"
                    ],
                    index=0 if "Editor" in st.session_state.get("custom_workflow_stage", "Editor") else 1
                )
                st.session_state["custom_workflow_stage"] = stage_choice
                st.caption(
                    f"📐 **{len(st.session_state['custom_bays_normalized'])} bays** currently configured. "
                    "In Step 1 you can drag corner handles to follow painted divider lines. "
                    "Step 2 performs YOLO vehicle detection and calculates occupancy."
                )

    st.markdown("---")

    # SECTION 2: DETECTION ENGINE
    st.markdown("#### 🧠 2. Detection Intelligence")
    engine_mode = st.selectbox(
        "Classification Methodology:",
        [
            "YOLOv8 + Spatial Overlap (Deep Learning)",
            "Ground Truth XML (Human Annotations)",
            "Scikit-Learn ML Patch Classifier",
            "Classic CV Edge Heuristic"
        ],
        index=0
    )

    # SECTION 3: VISUALIZATION MODE
    st.markdown("---")
    st.markdown("#### 🎨 3. Display & Overlays")
    view_mode = st.selectbox(
        "Rendering View Mode:",
        [
            "AI Prediction View (YOLO/ML)",
            "Ground Truth View (Real XML)",
            "Discrepancy Audit View (GT vs AI)"
        ],
        index=0
    )

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        show_ids = st.checkbox("Slot IDs (#)", value=True)
    with col_v2:
        show_boxes = st.checkbox("YOLO Boxes", value=True)

    show_hud = st.checkbox("Top HUD Banner", value=True)
    alpha_opacity = st.slider("Polygon Fill Opacity", min_value=0.10, max_value=0.70, value=0.35, step=0.05)

    # SECTION 4: HYPERPARAMETERS
    with st.expander("⚙️ Advanced Hyperparameters", expanded=False):
        yolo_conf = st.slider("YOLO Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
        overlap_thresh = st.slider("Slot Overlap IoU Threshold", min_value=0.05, max_value=0.80, value=0.20, step=0.05)
        edge_thresh = st.slider("Edge Heuristic Threshold", min_value=0.01, max_value=0.15, value=0.045, step=0.005)


# ==========================================================
# 4. TOP EXECUTIVE HEADER
# ==========================================================
st.markdown(f"""
<div class="dashboard-header">
    <div>
        <h1 class="dashboard-title">🚗 SmartPark AI • Autonomous Parking Intelligence</h1>
        <div class="dashboard-subtitle">
            Location: <b>{lot_name}</b> &nbsp;•&nbsp; Weather: <b>{weather_name}</b> &nbsp;•&nbsp; Frame: <code>{frame_name}</code> &nbsp;•&nbsp; Engine: <b>{engine_mode.split('(')[0].strip()}</b>
        </div>
    </div>
    <div style="text-align: right;">
        <span class="system-status-pill">● System Live & Active</span>
        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 5px;">Model: YOLOv8n + OpenCV</div>
    </div>
</div>
""", unsafe_allow_html=True)

if active_image_bgr is None:
    st.info("👈 Please select a scene from the PKLot dataset or upload an image to start monitoring.")
    st.stop()


# ==========================================================
# 5. CORE COMPUTER VISION PIPELINE EXECUTION
# ==========================================================
vision_engine = get_cv_engine(yolo_conf=yolo_conf, overlap_thresh=overlap_thresh)

# Resolve spaces
spaces: List[ParkingSpace] = []
if active_parser is not None:
    spaces = active_parser.get_spaces()
elif lot_name == "Custom Upload (No XML)":
    ih, iw = active_image_bgr.shape[:2]
    norm_bays = st.session_state.get("custom_bays_normalized", DEFAULT_2_ROW_TEMPLATE)
    pixel_bays = normalized_to_pixel_spaces(norm_bays, iw, ih)
    engine_inst = SmartParkVisionEngine()
    spaces = engine_inst.create_custom_spaces(pixel_bays)
elif custom_spaces:
    spaces = custom_spaces

# ==========================================================
# STEP 1: INTERACTIVE PARKING-SLOT POLYGON EDITOR VIEW
# ==========================================================
if lot_name == "Custom Upload (No XML)" and "Editor" in st.session_state.get("custom_workflow_stage", "Editor"):
    st.markdown("""
    <div style="background: #0B192C; border: 1px solid #1E3E62; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
        <h3 style="color: #FFFFFF; margin: 0 0 6px 0;">✏️ Step 1: Interactive Parking-Slot Polygon Editor</h3>
        <p style="color: #94A3B8; font-size: 0.88rem; margin: 0;">
            Displaying the uploaded parking image as the canvas background. <b>Each parking bay has its own quadrilateral polygon.</b>
            Follow the painted divider lines and perspective angles. You can drag corner handles to align boundaries, draw new bays, delete bays, or fine-tune coordinates.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Render interactive HTML5 Canvas polygon editor component
    editor_result = render_polygon_editor(
        image_bgr=active_image_bgr,
        initial_bays=st.session_state.get("custom_bays_normalized", DEFAULT_2_ROW_TEMPLATE),
        key="custom_polygon_editor_component"
    )

    if editor_result and isinstance(editor_result, dict):
        if editor_result.get("action") == "save" and "bays" in editor_result:
            st.session_state["custom_bays_normalized"] = editor_result["bays"]
            st.toast(f"💾 Saved layout with {len(editor_result['bays'])} parking bays!", icon="✅")
        elif editor_result.get("action") == "reset":
            st.session_state["custom_bays_normalized"] = [dict(b) for b in DEFAULT_2_ROW_TEMPLATE]
            st.toast("🔄 Reset to default 2-row perspective template!", icon="📐")

    # Workflow Action Buttons
    c_btn1, c_btn2, c_btn3 = st.columns([2, 1, 1])
    with c_btn1:
        if st.button("🚀 Save Layout & Run Occupancy Analysis (Step 2)", type="primary", use_container_width=True):
            st.session_state["custom_workflow_stage"] = "🔍 Step 2: Occupancy Analysis"
            st.rerun()
    with c_btn2:
        if st.button("📐 Reset to 2-Row Template", use_container_width=True):
            st.session_state["custom_bays_normalized"] = [dict(b) for b in DEFAULT_2_ROW_TEMPLATE]
            st.rerun()
    with c_btn3:
        if st.button("➕ Add Quadrilateral Bay", use_container_width=True):
            curr = st.session_state.get("custom_bays_normalized", [])
            new_id = f"BAY_{len(curr) + 1}"
            curr.append({
                "id": new_id,
                "points": [[0.35, 0.40], [0.50, 0.40], [0.49, 0.60], [0.34, 0.60]]
            })
            st.session_state["custom_bays_normalized"] = curr
            st.rerun()

    # Layout overview summary KPIs
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Mapped Bays", len(spaces))
    with m2:
        st.metric("Geometry", "4-Point Quadrilaterals")
    with m3:
        st.metric("Coordinate Storage", "Normalized [0.0 - 1.0]")
    with m4:
        st.metric("Occupancy Status", "Unanalyzed (Design Mode)")

    # Manual Coordinate Fine-Tuner & Bay Manager Expander
    with st.expander("🔧 Manual Coordinate Fine-Tuner & Bay List", expanded=False):
        st.caption("Inspect and fine-tune individual corner coordinates (normalized 0.000 to 1.000).")
        curr_bays = st.session_state.get("custom_bays_normalized", [])
        for idx, bay in enumerate(curr_bays):
            col_b_info, col_b_del = st.columns([4, 1])
            with col_b_info:
                st.markdown(f"**{bay['id']}** (Points: {len(bay['points'])})")
            with col_b_del:
                if st.button("🗑️ Delete", key=f"del_{bay['id']}_{idx}"):
                    curr_bays.pop(idx)
                    for j, b in enumerate(curr_bays):
                        b["id"] = f"BAY_{j + 1}"
                    st.session_state["custom_bays_normalized"] = curr_bays
                    st.rerun()

            c_p1, c_p2, c_p3, c_p4 = st.columns(4)
            labels = ["P1 (TL)", "P2 (TR)", "P3 (BR)", "P4 (BL)"]
            cols = [c_p1, c_p2, c_p3, c_p4]
            for p_i in range(min(4, len(bay["points"]))):
                with cols[p_i]:
                    nx = st.number_input(f"{bay['id']} {labels[p_i]} X", min_value=0.0, max_value=1.0,
                                         value=float(bay["points"][p_i][0]), step=0.01, format="%.3f",
                                         key=f"{bay['id']}_p{p_i}_x")
                    ny = st.number_input(f"{bay['id']} {labels[p_i]} Y", min_value=0.0, max_value=1.0,
                                         value=float(bay["points"][p_i][1]), step=0.01, format="%.3f",
                                         key=f"{bay['id']}_p{p_i}_y")
                    bay["points"][p_i] = [nx, ny]

    # Layout OpenCV Static Render Preview
    st.markdown("##### 👁️ Layout Preview (OpenCV Overlay)")
    visualizer_inst = ParkingVisualizer(alpha=alpha_opacity)
    layout_preview_bgr = visualizer_inst.draw_slot_layout(active_image_bgr, spaces, show_slot_ids=show_ids, show_hud=show_hud)
    layout_preview_rgb = cv2.cvtColor(layout_preview_bgr, cv2.COLOR_BGR2RGB)
    st.image(layout_preview_rgb, use_container_width=True, caption="OpenCV Layout Preview (Bays are unclassified until Step 2 is executed)")

    st.stop()  # Stop turn here so user can edit before analysis runs!

if lot_name == "Custom Upload (No XML)":
    c_nav1, c_nav2 = st.columns([3, 1])
    with c_nav1:
        st.info(f"🔍 Analyzing occupancy on **{len(spaces)} custom parking bays** using YOLO vehicle overlap.")
    with c_nav2:
        if st.button("✏️ Return to Polygon Editor", use_container_width=True):
            st.session_state["custom_workflow_stage"] = "✏️ Step 1: Interactive Polygon Editor"
            st.rerun()

# Run vehicle detection
detected_vehicles: List[DetectedVehicle] = vision_engine.detect_vehicles(active_image_bgr, conf=yolo_conf)
if getattr(vision_engine.yolo_detector, "load_error", None):
    st.caption(f"ℹ️ YOLO Detector Notice: {vision_engine.yolo_detector.load_error}")

# Run occupancy classification
classification_results: List[SlotClassificationResult] = []
if spaces:
    if "YOLOv8" in engine_mode:
        classification_results = vision_engine.predict_occupancy_with_yolo(
            spaces=spaces,
            detected_vehicles=detected_vehicles,
            overlap_threshold=overlap_thresh
        )
    elif "Ground Truth" in engine_mode:
        classification_results = vision_engine.get_ground_truth_results(spaces)
    elif "Scikit-Learn" in engine_mode:
        classification_results = OccupancyClassifier.classify_with_ml(active_image_bgr, spaces)
    elif "Edge Heuristic" in engine_mode:
        classification_results = OccupancyClassifier.classify_with_edge_heuristic(
            active_image_bgr, spaces, edge_density_threshold=edge_thresh
        )

# Compute live occupancy statistics
if classification_results:
    stats = vision_engine.calculate_statistics(classification_results)
elif spaces:
    stats = vision_engine.calculate_statistics(spaces)
else:
    stats = {"total": 0, "occupied": len(detected_vehicles), "vacant": 0, "unannotated": 0, "occupancy_rate": 0.0}

total_slots = stats["total"]
occupied_slots = stats["occupied"]
vacant_slots = stats["vacant"]
occupancy_rate = stats["occupancy_rate"]


# ==========================================================
# 6. OVERVIEW STATISTICS (5 KPI CARDS)
# ==========================================================
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value c-total">{total_slots}</div>
        <div class="metric-label">Total Parking Bays</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value c-vacant">{vacant_slots}</div>
        <div class="metric-label">Available Vacant</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value c-occupied">{occupied_slots}</div>
        <div class="metric-label">Occupied Bays</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value c-rate">{occupancy_rate:.1f}%</div>
        <div class="metric-label">Occupancy Rate</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value c-vehicles">{len(detected_vehicles)}</div>
        <div class="metric-label">YOLO Vehicles</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ==========================================================
# 7. RENDER ANNOTATED IMAGE
# ==========================================================
visualizer = ParkingVisualizer(
    alpha=alpha_opacity,
    vacant_color=config.COLORS["vacant_bgr"],
    occupied_color=config.COLORS["occupied_bgr"],
    box_color=config.COLORS["detected_box_bgr"]
)

if "Ground Truth" in view_mode and spaces:
    rendered_bgr = visualizer.draw_ground_truth(
        base_image=active_image_bgr,
        spaces=spaces,
        show_slot_ids=show_ids,
        show_hud=show_hud
    )
elif "Audit" in view_mode and spaces and classification_results:
    rendered_bgr = visualizer.draw_comparison_audit(
        base_image=active_image_bgr,
        spaces=spaces,
        classification_results=classification_results,
        detected_vehicles=detected_vehicles,
        show_hud=show_hud
    )
else:
    rendered_bgr = visualizer.draw_ai_predictions(
        base_image=active_image_bgr,
        spaces=spaces,
        classification_results=classification_results,
        detected_vehicles=detected_vehicles,
        show_slot_ids=show_ids,
        show_boxes=show_boxes,
        show_hud=show_hud
    )

rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)


# ==========================================================
# 8. MAIN DASHBOARD TABS
# ==========================================================
tab_map, tab_charts, tab_eval, tab_export = st.tabs([
    "🖥️ Spatial Vision Map",
    "📊 Occupancy & Vehicle Analytics",
    "🎯 Academic Accuracy Benchmark",
    "📥 Downloads & Research Reports"
])

# ----------------------------------------------------------
# TAB 1: SPATIAL VISION MAP
# ----------------------------------------------------------
with tab_map:
    col_view, col_legend = st.columns([3, 1])

    with col_view:
        st.image(rendered_rgb, use_container_width=True, caption=f"Analyzed Feed: {frame_name} ({view_mode})")

        # Action bar
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            success, enc_jpg = cv2.imencode(".jpg", rendered_bgr)
            if success:
                st.download_button(
                    label="💾 Download Annotated Vision Image",
                    data=enc_jpg.tobytes(),
                    file_name=f"smartpark_{lot_name}_{Path(frame_name).stem}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
        with col_act2:
            with st.expander("🔍 View Raw Original Image"):
                orig_rgb = cv2.cvtColor(active_image_bgr, cv2.COLOR_BGR2RGB)
                st.image(orig_rgb, use_container_width=True)

    with col_legend:
        st.markdown("#### 🎨 Visual Legend")
        st.markdown('<span class="status-pill-vacant">🟢 Green Polygon</span> &nbsp; **Vacant Slot**', unsafe_allow_html=True)
        st.write("")
        st.markdown('<span class="status-pill-occupied">🔴 Red Polygon</span> &nbsp; **Occupied Slot**', unsafe_allow_html=True)
        st.write("")
        st.markdown('<span class="status-pill-vehicle">🟠 Orange BBox</span> &nbsp; **YOLO Vehicle**', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ℹ️ Scene Details")
        st.markdown(f"- **Lot:** `{lot_name}`")
        st.markdown(f"- **Weather:** `{weather_name}`")
        st.markdown(f"- **View Mode:** `{view_mode}`")
        st.markdown(f"- **Total Bays:** `{total_slots}`")
        st.markdown(f"- **Available:** `{vacant_slots}`")
        st.markdown(f"- **Parked:** `{occupied_slots}`")
        st.markdown(f"- **Occupancy:** `{occupancy_rate:.1f}%`")

        if not spaces:
            st.warning("⚠️ No parking bay polygons defined for this image. Detections show free-roam YOLO vehicle tracking.")

# ----------------------------------------------------------
# TAB 2: OCCUPANCY & VEHICLE ANALYTICS CHARTS
# ----------------------------------------------------------
with tab_charts:
    st.subheader("📈 Real-Time Parking Analytics & Distributions")

    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown("##### 🍩 Occupancy Distribution")
        if total_slots > 0:
            donut_data = pd.DataFrame({
                "Status": ["Vacant (Available)", "Occupied (Parked)"],
                "Count": [vacant_slots, occupied_slots]
            })

            donut_chart = alt.Chart(donut_data).mark_arc(innerRadius=65).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(
                    field="Status",
                    type="nominal",
                    scale=alt.Scale(
                        domain=["Vacant (Available)", "Occupied (Parked)"],
                        range=["#10B981", "#EF4444"]
                    ),
                    legend=alt.Legend(title="Slot Status")
                ),
                tooltip=["Status", "Count"]
            ).properties(height=280)

            st.altair_chart(donut_chart, use_container_width=True)
        else:
            st.info("No parking slots to display distribution.")

    with c_chart2:
        st.markdown("##### 🚘 YOLO Detected Vehicles Breakdown")
        if detected_vehicles:
            v_data = pd.DataFrame([v.to_dict() for v in detected_vehicles])
            class_counts = v_data["class"].value_counts().reset_index()
            class_counts.columns = ["Vehicle Class", "Count"]

            bar_chart = alt.Chart(class_counts).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X("Vehicle Class:N", sort="-y"),
                y=alt.Y("Count:Q"),
                color=alt.Color("Vehicle Class:N", scale=alt.Scale(scheme="blues")),
                tooltip=["Vehicle Class", "Count"]
            ).properties(height=280)

            st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.info("No vehicles currently detected in this frame.")

    st.markdown("---")
    st.markdown("##### 📐 Parking Slot Scale & Area Distribution (Perspective Variation)")
    if spaces:
        slot_areas = [s.area for s in spaces if s.is_valid]
        if slot_areas:
            area_df = pd.DataFrame({"Slot Area (px²)": slot_areas})
            area_hist = alt.Chart(area_df).mark_bar(color="#1E3E62").encode(
                x=alt.X("Slot Area (px²):Q", bin=alt.Bin(maxbins=15), title="Slot Pixel Area (px²)"),
                y=alt.Y("count()", title="Number of Bays")
            ).properties(height=200)
            st.altair_chart(area_hist, use_container_width=True)
            st.caption(f"Min Bay Area: {min(slot_areas):.0f} px² • Max Bay Area: {max(slot_areas):.0f} px² • Perspective Ratio: {max(slot_areas)/min(slot_areas):.1f}x")

# ----------------------------------------------------------
# TAB 3: ACADEMIC ACCURACY BENCHMARK & DATE-GROUPED EVALUATION
# ----------------------------------------------------------
with tab_eval:
    eval_subtab1, eval_subtab2 = st.tabs([
        "🔬 Date-Grouped Benchmark (Zero Temporal Leakage)",
        "🎯 Current Frame Accuracy Audit"
    ])

    # ------------------------------------------------------
    # SUBTAB 1: REPRODUCIBLE DATE-GROUPED BENCHMARK
    # ------------------------------------------------------
    with eval_subtab1:
        st.markdown(f"#### 📅 Reproducible Model Evaluation Grouped by Capture Date ({lot_name})")
        st.caption("Frames are partitioned strictly by capture date so train and test sets have disjoint days, preventing temporal data leakage. All metrics compare actual YOLO predictions with real PKLot ground-truth XML.")

        c_bench_ctrl1, c_bench_ctrl2, c_bench_ctrl3 = st.columns([2, 2, 2])
        with c_bench_ctrl1:
            benchmark_frames = st.slider("Test Set Sample Size (Frames):", min_value=5, max_value=30, value=15, step=5)
        with c_bench_ctrl2:
            st.write("")
            st.write("")
            run_bench_btn = st.button("🚀 Run Date-Grouped Benchmark", type="primary", use_container_width=True)

        if run_bench_btn:
            run_cached_benchmark.clear()

        # Run benchmark
        try:
            bench_res, pdf_file_path, per_slot_csv_path, records_csv_path, train_dates_list, test_dates_list = run_cached_benchmark(
                target_lot=lot_name if lot_name in ["UFPR04", "UFPR05", "PUCPR"] else "UFPR05",
                max_frames=benchmark_frames,
                conf=yolo_conf,
                overlap=overlap_thresh
            )

            # Display date split summary pills
            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            c_s1.info(f"**Lot:** `{bench_res.lot}`")
            c_s2.info(f"**Train Dates:** `{len(train_dates_list)} days`")
            c_s3.info(f"**Held-Out Test Dates:** `{len(test_dates_list)} days`")
            c_s4.info(f"**Evaluated Slots:** `{bench_res.total_slots_evaluated:,} bays`")

            bm = bench_res.metrics

            # Top Macro Metric Cards
            b_col1, b_col2, b_col3, b_col4, b_col5, b_col6 = st.columns(6)
            b_col1.metric("Overall Accuracy", f"{bm.accuracy * 100.0:.2f}%")
            b_col2.metric("Precision", f"{bm.precision * 100.0:.2f}%")
            b_col3.metric("Recall (Sensitivity)", f"{bm.recall * 100.0:.2f}%")
            b_col4.metric("F1-Score", f"{bm.f1_score * 100.0:.2f}%")
            b_col5.metric("Specificity", f"{bm.specificity * 100.0:.2f}%")
            b_col6.metric("Bal. Accuracy", f"{bm.balanced_accuracy * 100.0:.2f}%")

            st.markdown("---")

            # Confusion Matrix & Error breakdown
            c_cm1, c_cm2 = st.columns([1, 1])
            with c_cm1:
                st.markdown("##### 🧮 Confusion Matrix (Test Dates)")
                st.dataframe(bm.confusion_matrix_df, use_container_width=True)
                st.caption(f"TP: {bm.tp:,} | TN: {bm.tn:,} | FP: {bm.fp:,} | FN: {bm.fn:,}")

            with c_cm2:
                st.markdown("##### 🔍 Error Distributions")
                err_summary_df = pd.DataFrame({
                    "Outcome": ["True Positives (TP)", "True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)"],
                    "Count": [bm.tp, bm.tn, bm.fp, bm.fn],
                    "Color": ["#EF4444", "#10B981", "#F59E0B", "#8B5CF6"]
                })
                cm_chart = alt.Chart(err_summary_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("Outcome:N", sort=None),
                    y=alt.Y("Count:Q"),
                    color=alt.Color("Color:N", scale=None),
                    tooltip=["Outcome", "Count"]
                ).properties(height=220)
                st.altair_chart(cm_chart, use_container_width=True)

            st.markdown("---")

            # PER-SLOT ACCURACY SECTION
            st.markdown(f"#### 📊 Per-Slot Accuracy Breakdown ({len(bench_res.per_slot_df)} Individual Parking Bays)")
            st.caption("Displays the exact classification accuracy for every individual slot across all held-out test dates.")

            # Altair bar chart for Per-Slot Accuracy
            slot_bar = alt.Chart(bench_res.per_slot_df).mark_bar().encode(
                x=alt.X("Slot ID:N", sort=None, title="Parking Bay ID"),
                y=alt.Y("Accuracy (%):Q", scale=alt.Scale(domain=[0, 105]), title="Accuracy (%)"),
                color=alt.condition(
                    alt.datum["Accuracy (%)"] >= 90.0,
                    alt.value("#10B981"), # Green
                    alt.condition(
                        alt.datum["Accuracy (%)"] >= 75.0,
                        alt.value("#F59E0B"), # Amber
                        alt.value("#EF4444")  # Red
                    )
                ),
                tooltip=["Slot ID", "Accuracy (%)", "Evaluations", "Correct", "TP", "TN", "FP", "FN"]
            ).properties(height=260)

            mean_rule = alt.Chart(pd.DataFrame({"Mean": [bm.accuracy * 100.0]})).mark_rule(
                color="#0B192C", strokeDash=[4, 4], size=1.5
            ).encode(y="Mean:Q")

            st.altair_chart(slot_bar + mean_rule, use_container_width=True)

            # Per-Slot Detailed Table
            with st.expander("📋 View Complete Per-Slot Accuracy Table", expanded=True):
                st.dataframe(bench_res.per_slot_df, use_container_width=True)

            # Benchmark Downloads Row
            st.markdown("##### 📥 Export Benchmark Artifacts")
            b_d1, b_d2, b_d3 = st.columns(3)
            with b_d1:
                if Path(pdf_file_path).exists():
                    with open(pdf_file_path, "rb") as f_pdf:
                        st.download_button(
                            label="📄 Download Academic PDF Report",
                            data=f_pdf.read(),
                            file_name=Path(pdf_file_path).name,
                            mime="application/pdf",
                            use_container_width=True
                        )
            with b_d2:
                if Path(per_slot_csv_path).exists():
                    with open(per_slot_csv_path, "rb") as f_slot_csv:
                        st.download_button(
                            label="📊 Download Per-Slot Accuracy (CSV)",
                            data=f_slot_csv.read(),
                            file_name=Path(per_slot_csv_path).name,
                            mime="text/csv",
                            use_container_width=True
                        )
            with b_d3:
                if Path(records_csv_path).exists():
                    with open(records_csv_path, "rb") as f_rec_csv:
                        st.download_button(
                            label="📋 Download Detailed Records (CSV)",
                            data=f_rec_csv.read(),
                            file_name=Path(records_csv_path).name,
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"Error running benchmark evaluation: {e}")

    # ------------------------------------------------------
    # SUBTAB 2: CURRENT FRAME ACCURACY AUDIT
    # ------------------------------------------------------
    with eval_subtab2:
        st.markdown(f"#### 🎯 Current Scene Accuracy Audit ({frame_name})")

        if not classification_results or not spaces:
            st.info("Accuracy evaluation requires parking bay polygons and ground truth annotations.")
        else:
            metrics = vision_engine.evaluate(classification_results)

            if metrics is None:
                st.warning("Ground-truth labels are not present in this custom upload to calculate benchmark accuracy.")
            else:
                # 6 metric cards
                e1, e2, e3, e4, e5, e6 = st.columns(6)
                e1.metric("Overall Accuracy", f"{metrics.accuracy * 100.0:.2f}%")
                e2.metric("Precision", f"{metrics.precision * 100.0:.2f}%")
                e3.metric("Recall (Sensitivity)", f"{metrics.recall * 100.0:.2f}%")
                e4.metric("F1-Score", f"{metrics.f1_score * 100.0:.2f}%")
                e5.metric("Specificity", f"{metrics.specificity * 100.0:.2f}%")
                e6.metric("Balanced Accuracy", f"{metrics.balanced_accuracy * 100.0:.2f}%")

                st.markdown("---")
                col_cm, col_err = st.columns([1, 1])

                with col_cm:
                    st.markdown("##### 🧮 Current Frame Confusion Matrix")
                    st.dataframe(metrics.confusion_matrix_df, use_container_width=True)

                    st.markdown(f"""
                    - **True Positives (TP):** `{metrics.tp}` &nbsp;*(Correctly identified occupied bays)*
                    - **True Negatives (TN):** `{metrics.tn}` &nbsp;*(Correctly identified empty bays)*
                    - **False Positives (FP):** `{metrics.fp}` &nbsp;*(Type I error: Empty bay flagged as occupied)*
                    - **False Negatives (FN):** `{metrics.fn}` &nbsp;*(Type II error: Parked vehicle missed)*
                    """)

                with col_err:
                    st.markdown("##### 🔍 Frame Error Analysis")
                    err_df = pd.DataFrame({
                        "Evaluation Outcome": ["True Positives (TP)", "True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)"],
                        "Count": [metrics.tp, metrics.tn, metrics.fp, metrics.fn],
                        "Color": ["#EF4444", "#10B981", "#F59E0B", "#8B5CF6"]
                    })
                    err_chart = alt.Chart(err_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                        x=alt.X("Evaluation Outcome:N", sort=None),
                        y=alt.Y("Count:Q"),
                        color=alt.Color("Color:N", scale=None),
                        tooltip=["Evaluation Outcome", "Count"]
                    ).properties(height=240)
                    st.altair_chart(err_chart, use_container_width=True)

                st.markdown("---")
                st.markdown("##### 📋 Current Frame Slot Audit")
                filter_mode = st.radio("Filter audit table:", ["All Slots", "Only Misclassifications (Errors)", "Only Correct Predictions"], horizontal=True)

                audit_df = metrics.detailed_df.copy()
                if filter_mode == "Only Misclassifications (Errors)":
                    audit_df = audit_df[audit_df["Evaluation"] == "Mismatch"]
                elif filter_mode == "Only Correct Predictions":
                    audit_df = audit_df[audit_df["Evaluation"] == "Correct"]

                st.dataframe(audit_df, use_container_width=True)

# ----------------------------------------------------------
# TAB 4: DOWNLOADS & RESEARCH REPORTS
# ----------------------------------------------------------
with tab_export:
    st.subheader("📑 Research Reports, CSV Audits & PDF Exports")

    st.markdown("##### 📦 1. Date-Grouped Benchmark Reports (Zero Data Leakage)")
    b_col_a, b_col_b, b_col_c = st.columns(3)
    bench_dir = Path("reports/benchmark")
    
    # Locate latest generated benchmark files
    pdf_files = sorted(list(bench_dir.glob("*.pdf")), key=lambda p: p.stat().st_mtime, reverse=True)
    slot_csv_files = sorted(list(bench_dir.glob("per_slot_accuracy_*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
    rec_csv_files = sorted(list(bench_dir.glob("benchmark_records_*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)

    with b_col_a:
        if pdf_files:
            latest_pdf = pdf_files[0]
            with open(latest_pdf, "rb") as f:
                st.download_button(
                    label="📄 Download Academic PDF Report",
                    data=f.read(),
                    file_name=latest_pdf.name,
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.button("📄 Academic PDF (Run benchmark first)", disabled=True, use_container_width=True)

    with b_col_b:
        if slot_csv_files:
            latest_slot_csv = slot_csv_files[0]
            with open(latest_slot_csv, "rb") as f:
                st.download_button(
                    label="📊 Download Per-Slot Accuracy (CSV)",
                    data=f.read(),
                    file_name=latest_slot_csv.name,
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.button("📊 Per-Slot CSV (Run benchmark first)", disabled=True, use_container_width=True)

    with b_col_c:
        if rec_csv_files:
            latest_rec_csv = rec_csv_files[0]
            with open(latest_rec_csv, "rb") as f:
                st.download_button(
                    label="📋 Download Detailed Records (CSV)",
                    data=f.read(),
                    file_name=latest_rec_csv.name,
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.button("📋 Detailed Records (Run benchmark first)", disabled=True, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 🖼️ 2. Current Scene Vision & Audit Exports")

    if classification_results and spaces:
        metrics = vision_engine.evaluate(classification_results)
        if metrics is not None:
            extra_meta = {
                "weather": weather_name,
                "date": date_name,
                "yolo_conf": yolo_conf,
                "overlap_threshold": overlap_thresh
            }

            rep_path, rep_md = ReportGenerator.generate_academic_report(
                lot_id=lot_name,
                image_name=frame_name,
                method_name=engine_mode,
                metrics=metrics,
                extra_metadata=extra_meta
            )

            csv_path, csv_str = ReportGenerator.export_csv(metrics)

            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                st.download_button(
                    label="📥 Download Frame Audit (CSV)",
                    data=csv_str,
                    file_name=f"smartpark_frame_audit_{lot_name}_{frame_name}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with d_col2:
                st.download_button(
                    label="📄 Download Academic Summary (Markdown)",
                    data=rep_md,
                    file_name=f"smartpark_report_{lot_name}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with d_col3:
                succ, enc_bgr = cv2.imencode(".jpg", rendered_bgr)
                if succ:
                    st.download_button(
                        label="🖼️ Download Annotated Image",
                        data=enc_bgr.tobytes(),
                        file_name=f"smartpark_vision_{Path(frame_name).stem}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )

            st.markdown("---")
            st.markdown("### 📄 Preview of Generated Academic Research Summary")
            st.markdown(rep_md)
        else:
            st.info("Ground truth labels not available to generate comparative report.")
    else:
        st.info("Load an image with annotations to generate academic evaluation reports.")
