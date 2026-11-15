"""
SmartPark AI - Interactive Parking Slot Polygon Editor Component
Enables precise 4-point quadrilateral polygon definition, interactive vertex dragging,
moving, resizing, deleting, template loading, and normalized coordinate storage.
"""

import base64
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2
import numpy as np
import streamlit.components.v1 as components

# Declare Streamlit custom component
_COMPONENT_PATH = os.path.dirname(os.path.abspath(__file__))
_polygon_editor_component = components.declare_component(
    "polygon_editor",
    path=_COMPONENT_PATH
)

# Standard 2-row perspective template (Upper row + Lower row)
DEFAULT_2_ROW_TEMPLATE = [
    # Upper row (distant)
    {
        "id": "BAY_1",
        "points": [[0.06, 0.30], [0.22, 0.29], [0.21, 0.44], [0.05, 0.45]]
    },
    {
        "id": "BAY_2",
        "points": [[0.24, 0.29], [0.40, 0.28], [0.39, 0.44], [0.23, 0.44]]
    },
    {
        "id": "BAY_3",
        "points": [[0.42, 0.28], [0.58, 0.28], [0.57, 0.43], [0.41, 0.44]]
    },
    {
        "id": "BAY_4",
        "points": [[0.60, 0.28], [0.76, 0.28], [0.75, 0.43], [0.59, 0.43]]
    },
    {
        "id": "BAY_5",
        "points": [[0.78, 0.28], [0.94, 0.28], [0.93, 0.43], [0.77, 0.43]]
    },
    # Lower row (near foreground)
    {
        "id": "BAY_6",
        "points": [[0.04, 0.54], [0.21, 0.53], [0.20, 0.78], [0.02, 0.79]]
    },
    {
        "id": "BAY_7",
        "points": [[0.23, 0.53], [0.40, 0.52], [0.39, 0.78], [0.22, 0.78]]
    },
    {
        "id": "BAY_8",
        "points": [[0.42, 0.52], [0.59, 0.52], [0.58, 0.77], [0.41, 0.78]]
    },
    {
        "id": "BAY_9",
        "points": [[0.61, 0.52], [0.78, 0.52], [0.77, 0.77], [0.60, 0.77]]
    },
    {
        "id": "BAY_10",
        "points": [[0.80, 0.52], [0.97, 0.52], [0.96, 0.77], [0.79, 0.77]]
    }
]


def render_polygon_editor(
    image_bgr: np.ndarray,
    initial_bays: Optional[List[Dict]] = None,
    key: Optional[str] = "smartpark_polygon_editor"
) -> Optional[Dict]:
    """
    Renders an interactive HTML5 canvas parking-slot polygon editor.
    
    Args:
        image_bgr: OpenCV BGR image array.
        initial_bays: Optional list of bay dictionaries with normalized points [[nx, ny], ...].
        key: Streamlit element key.
        
    Returns:
        Dict from component when user saves/resets, or None.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    # Encode image to JPEG base64 Data URL
    success, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        return None

    b64_str = base64.b64encode(buffer).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_str}"

    if initial_bays is None or len(initial_bays) == 0:
        initial_bays = DEFAULT_2_ROW_TEMPLATE

    return _polygon_editor_component(
        image_b64=data_url,
        initial_bays=initial_bays,
        key=key,
        default=None
    )


def normalized_to_pixel_spaces(
    normalized_bays: List[Dict],
    image_width: int,
    image_height: int
) -> List[Dict[str, Union[str, List[List[int]]]]]:
    """
    Converts normalized [0.0, 1.0] polygon coordinates to integer pixel coordinates.
    Scale-invariant: correctly positions polygons regardless of image resolution.
    """
    pixel_bays = []
    for bay in normalized_bays:
        bay_id = bay.get("id", f"BAY_{len(pixel_bays) + 1}")
        norm_pts = bay.get("points", [])
        pixel_pts = []
        for pt in norm_pts:
            px = max(0, min(image_width - 1, int(round(float(pt[0]) * image_width))))
            py = max(0, min(image_height - 1, int(round(float(pt[1]) * image_height))))
            pixel_pts.append([px, py])
        pixel_bays.append({
            "id": bay_id,
            "points": pixel_pts
        })
    return pixel_bays
