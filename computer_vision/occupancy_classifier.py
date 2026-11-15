"""
SmartPark AI - Parking Slot Occupancy Classifier
Provides multiple computer vision & AI approaches to determine slot occupancy:
1. Ground Truth (from PKLot XML annotations)
2. Spatial Overlap Analysis (IoU & polygon intersection with YOLO vehicle detections)
3. Classic Computer Vision Heuristic (Canny edge density & intensity variance)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from config import DEFAULT_OVERLAP_THRESHOLD
from computer_vision.xml_parser import ParkingSpace
from computer_vision.yolo_detector import DetectedVehicle
from utils.geometry import polygon_box_overlap, polygon_bounding_box
from models.patch_classifier import PatchClassifier


@dataclass
class SlotClassificationResult:
    """Stores the classification decision and metrics for a single parking slot."""
    slot_id: str
    occupied_pred: bool
    confidence: float
    method: str
    occupied_gt: Optional[bool] = None
    overlap_ratio: float = 0.0
    edge_density: float = 0.0
    matched_vehicle: Optional[DetectedVehicle] = None

    @property
    def is_correct(self) -> Optional[bool]:
        """Check if prediction matches ground truth if GT is available."""
        if self.occupied_gt is None:
            return None
        return self.occupied_pred == self.occupied_gt

    def to_dict(self) -> Dict:
        return {
            "slot_id": self.slot_id,
            "occupied_pred": self.occupied_pred,
            "status_pred": "Occupied" if self.occupied_pred else "Vacant",
            "occupied_gt": self.occupied_gt,
            "status_gt": ("Occupied" if self.occupied_gt else "Vacant") if self.occupied_gt is not None else "N/A",
            "is_correct": self.is_correct,
            "confidence": round(self.confidence, 3),
            "overlap_ratio": round(self.overlap_ratio, 3),
            "edge_density": round(self.edge_density, 4),
            "method": self.method
        }


class OccupancyClassifier:
    """
    Evaluates occupancy for parking spaces using various CV/AI methods.
    """

    @staticmethod
    def classify_by_ground_truth(spaces: List[ParkingSpace]) -> List[SlotClassificationResult]:
        """Extract occupancy directly from ground truth XML."""
        results = []
        for s in spaces:
            results.append(
                SlotClassificationResult(
                    slot_id=s.id,
                    occupied_pred=s.occupied_gt,
                    confidence=1.0,
                    method="Ground Truth XML",
                    occupied_gt=s.occupied_gt,
                    overlap_ratio=1.0 if s.occupied_gt else 0.0
                )
            )
        return results

    @staticmethod
    def classify_with_yolo(spaces: List[ParkingSpace],
                           detected_vehicles: List[DetectedVehicle],
                           overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD) -> List[SlotClassificationResult]:
        """
        Classifies slots using spatial overlap between slot polygons and YOLO vehicle detections.
        Computes the maximum overlap across all detected vehicles for each slot.
        """
        results = []
        for s in spaces:
            max_overlap = 0.0
            matched_v = None

            for v in detected_vehicles:
                overlap = polygon_box_overlap(s.points, v.bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    matched_v = v

            is_occupied = (max_overlap >= overlap_threshold)
            # Confidence is high if comfortably above or below threshold
            conf = min(1.0, max_overlap / max(1e-5, overlap_threshold)) if is_occupied else min(1.0, 1.0 - (max_overlap / max(1e-5, overlap_threshold)))

            results.append(
                SlotClassificationResult(
                    slot_id=s.id,
                    occupied_pred=is_occupied,
                    confidence=conf,
                    method="YOLO Overlap Analysis",
                    occupied_gt=s.occupied_gt,
                    overlap_ratio=max_overlap,
                    matched_vehicle=matched_v
                )
            )
        return results

    @staticmethod
    def classify_with_edge_heuristic(image: np.ndarray,
                                     spaces: List[ParkingSpace],
                                     edge_density_threshold: float = 0.045) -> List[SlotClassificationResult]:
        """
        Classic CV Baseline:
        Computes Canny edge density within each slot's polygon.
        Vacant asphalt/parking bays have low edge density; vehicles introduce contours.
        """
        # Convert image to grayscale and compute Canny edges
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        h, w = gray.shape[:2]

        results = []
        for s in spaces:
            pts = s.points
            min_x, min_y, max_x, max_y = polygon_bounding_box(pts)
            min_x, min_y = max(0, min_x), max(0, min_y)
            max_x, max_y = min(w, max_x), min(h, max_y)

            roi_w = max_x - min_x
            roi_h = max_y - min_y
            if roi_w <= 0 or roi_h <= 0:
                results.append(SlotClassificationResult(slot_id=s.id, occupied_pred=False, confidence=0.0, method="CV Edge Heuristic", occupied_gt=s.occupied_gt))
                continue

            # Local mask
            local_pts = pts.copy()
            local_pts[:, 0] -= min_x
            local_pts[:, 1] -= min_y

            local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.fillPoly(local_mask, [local_pts], 255)
            poly_pixels = np.sum(local_mask > 0)

            if poly_pixels == 0:
                density = 0.0
            else:
                roi_edges = edges[min_y:max_y, min_x:max_x]
                edge_pixels = np.sum((roi_edges > 0) & (local_mask > 0))
                density = float(edge_pixels) / float(poly_pixels)

            is_occupied = (density >= edge_density_threshold)
            conf = min(1.0, density / max(1e-5, edge_density_threshold)) if is_occupied else min(1.0, 1.0 - (density / max(1e-5, edge_density_threshold)))

            results.append(
                SlotClassificationResult(
                    slot_id=s.id,
                    occupied_pred=is_occupied,
                    confidence=conf,
                    method="CV Edge Heuristic",
                    occupied_gt=s.occupied_gt,
                    edge_density=density
                )
            )
        return results

    @staticmethod
    def classify_with_ml(image: np.ndarray,
                         spaces: List[ParkingSpace],
                         classifier: Optional[PatchClassifier] = None) -> List[SlotClassificationResult]:
        """
        Scikit-Learn ML Model Baseline:
        Crops each parking slot patch from the image and applies the trained
        feature-extraction pipeline and classifier (LogisticRegression / LinearSVC).
        """
        if classifier is None:
            classifier = PatchClassifier()

        h, w = image.shape[:2]
        results = []

        for s in spaces:
            pts = s.points
            min_x, min_y, max_x, max_y = polygon_bounding_box(pts)
            min_x, min_y = max(0, min_x), max(0, min_y)
            max_x, max_y = min(w, max_x), min(h, max_y)

            roi_w = max_x - min_x
            roi_h = max_y - min_y
            if roi_w < 5 or roi_h < 5:
                results.append(SlotClassificationResult(slot_id=s.id, occupied_pred=False, confidence=0.5, method="Scikit-Learn ML", occupied_gt=s.occupied_gt))
                continue

            patch = image[min_y:max_y, min_x:max_x]
            is_occupied, conf = classifier.predict_patch(patch)

            results.append(
                SlotClassificationResult(
                    slot_id=s.id,
                    occupied_pred=is_occupied,
                    confidence=conf,
                    method="Scikit-Learn ML",
                    occupied_gt=s.occupied_gt
                )
            )
        return results
