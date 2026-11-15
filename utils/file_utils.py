"""
SmartPark AI - File & Dataset Navigation Utilities
Robust handling for file discovery, image loading, and export operations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from config import RAW_PKLOT_DIR, REPORTS_DIR, get_active_dataset_dir


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Safely load an image from path into a BGR numpy array.
    Supports Windows/POSIX non-ASCII paths using imdecode.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        return None
    try:
        # np.fromfile handles non-ASCII paths properly
        data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return cv2.imread(str(image_path))


def save_image(output_path: Path, img: np.ndarray) -> bool:
    """Safely save a BGR numpy image to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ext = output_path.suffix.lower() or ".jpg"
        success, encoded = cv2.imencode(ext, img)
        if success:
            encoded.tofile(str(output_path))
            return True
        return False
    except Exception:
        return cv2.imwrite(str(output_path), img)


def get_current_pklot_dir() -> Path:
    """Returns the active PKLot directory (raw dataset or demo sample)."""
    return get_active_dataset_dir()


def discover_pklot_lots() -> List[str]:
    """List all available parking lots in active PKLot directory."""
    pklot_dir = get_current_pklot_dir()
    if not pklot_dir.exists():
        return []
    return sorted([
        d.name for d in pklot_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])


def discover_lot_weathers(lot: str) -> List[str]:
    """List weather conditions available for a given lot."""
    lot_dir = get_current_pklot_dir() / lot
    if not lot_dir.exists():
        return []
    return sorted([
        d.name for d in lot_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])


def discover_weather_dates(lot: str, weather: str) -> List[str]:
    """List date folders available for a given lot and weather."""
    weather_dir = get_current_pklot_dir() / lot / weather
    if not weather_dir.exists():
        return []
    return sorted([
        d.name for d in weather_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])


def list_lot_images(lot: str, weather: str, date: str) -> List[Path]:
    """List all JPG image files available for a given lot, weather, and date."""
    target_dir = get_current_pklot_dir() / lot / weather / date
    if not target_dir.exists():
        return []
    return sorted(list(target_dir.glob("*.jpg")))


def find_matching_xml(image_path: Path) -> Optional[Path]:
    """Find the corresponding XML file for a given image path."""
    image_path = Path(image_path)
    xml_path = image_path.with_suffix(".xml")
    if xml_path.is_file():
        return xml_path
    return None
