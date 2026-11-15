"""
SmartPark AI - Comprehensive Computer Vision Engine
Integrates OpenCV image processing, PKLot polygon extraction, ground-truth mapping,
scale-invariant & perspective-robust spatial intersection, YOLO vehicle detection,
custom polygon support for uploaded images, and comparative audit visualization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from config import (
    DEFAULT_YOLO_CONF,
    DEFAULT_OVERLAP_THRESHOLD,
    YOLO_WEIGHTS_PATH,
    REPORTS_DIR
)
from utils.file_utils import load_image, save_image, find_matching_xml
from utils.geometry import (
    polygon_area,
    polygon_centroid,
    polygon_bounding_box,
    polygon_box_overlap,
    perspective_aware_overlap,
    rectify_perspective
)
from computer_vision.xml_parser import PKLotXMLParser, ParkingSpace
from computer_vision.yolo_detector import YOLOVehicleDetector, DetectedVehicle
from computer_vision.occupancy_classifier import OccupancyClassifier, SlotClassificationResult
from computer_vision.visualizer import ParkingVisualizer
from computer_vision.metrics import EvaluationMetrics


class SmartParkVisionEngine:
    """
    High-level Computer Vision Engine for SmartPark AI.
    Executes parsing, spatial polygon processing, YOLO vehicle detection,
    custom slot configuration, and comparative audit rendering.
    """

    def __init__(self,
                 yolo_weights: Union[str, Path] = YOLO_WEIGHTS_PATH,
                 yolo_conf: float = DEFAULT_YOLO_CONF,
                 overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD):
        self.yolo_conf = yolo_conf
        self.overlap_threshold = overlap_threshold
        self.visualizer = ParkingVisualizer()
        self._yolo_detector: Optional[YOLOVehicleDetector] = None
        self._yolo_weights = yolo_weights

    @property
    def yolo_detector(self) -> YOLOVehicleDetector:
        """Lazy load YOLO vehicle detector on first access."""
        if self._yolo_detector is None:
            self._yolo_detector = YOLOVehicleDetector(
                weights_path=self._yolo_weights,
                confidence_threshold=self.yolo_conf
            )
        return self._yolo_detector

    # ==========================================================
    # 1. READ PARKING IMAGES AND MATCHING PKLOT XML ANNOTATIONS
    # ==========================================================
    def load_scene(self,
                   image_path: Union[str, Path],
                   xml_path: Optional[Union[str, Path]] = None) -> Tuple[Optional[np.ndarray], Optional[PKLotXMLParser]]:
        """
        Loads an image and its matching PKLot XML annotation file.
        Returns (image_bgr, xml_parser).
        """
        img_p = Path(image_path)
        img = load_image(img_p)
        if img is None:
            return None, None

        if xml_path is None:
            xml_p = find_matching_xml(img_p)
        else:
            xml_p = Path(xml_path)

        parser = None
        if xml_p is not None and xml_p.exists():
            try:
                parser = PKLotXMLParser(xml_p)
            except Exception:
                parser = None

        return img, parser

    # ==========================================================
    # 2. EXTRACT INDIVIDUAL PARKING-SLOT POLYGONS
    # ==========================================================
    def extract_slots_from_xml(self, xml_source: Union[str, Path, bytes]) -> List[ParkingSpace]:
        """
        Extracts all parking space polygons, centroids, areas, and labels from an XML annotation.
        """
        parser = PKLotXMLParser(xml_source)
        return parser.get_spaces()

    # ==========================================================
    # 3. IDENTIFY OCCUPIED/VACANT SLOTS USING ACTUAL ANNOTATIONS
    # ==========================================================
    def get_ground_truth_results(self, spaces: List[ParkingSpace]) -> List[SlotClassificationResult]:
        """
        Extracts real ground-truth occupancy from annotations without any synthetic predictions.
        """
        return OccupancyClassifier.classify_by_ground_truth(spaces)

    # ==========================================================
    # 4 & 5. CALCULATE OCCUPANCY STATISTICS
    # ==========================================================
    def calculate_statistics(self,
                             spaces_or_results: Union[List[ParkingSpace], List[SlotClassificationResult]]) -> Dict[str, Union[int, float]]:
        """
        Calculates total bays, occupied, vacant, unannotated count, and occupancy rate %.
        """
        if not spaces_or_results:
            return {"total": 0, "occupied": 0, "vacant": 0, "unannotated": 0, "occupancy_rate": 0.0}

        first_item = spaces_or_results[0]
        if isinstance(first_item, ParkingSpace):
            total = len(spaces_or_results)
            occupied = sum(1 for s in spaces_or_results if s.occupied_gt is True)
            vacant = sum(1 for s in spaces_or_results if s.occupied_gt is False)
            unannotated = sum(1 for s in spaces_or_results if s.occupied_gt is None)
        else:
            total = len(spaces_or_results)
            occupied = sum(1 for r in spaces_or_results if r.occupied_pred is True)
            vacant = sum(1 for r in spaces_or_results if r.occupied_pred is False)
            unannotated = sum(1 for r in spaces_or_results if r.occupied_gt is None and r.method == "Ground Truth XML")

        valid_labeled = occupied + vacant
        rate = (occupied / valid_labeled * 100.0) if valid_labeled > 0 else 0.0

        return {
            "total": total,
            "occupied": occupied,
            "vacant": vacant,
            "unannotated": unannotated,
            "occupancy_rate": round(rate, 2)
        }

    # ==========================================================
    # 6. OPTIONAL PRETRAINED YOLO VEHICLE DETECTION
    # ==========================================================
    def detect_vehicles(self,
                        image: np.ndarray,
                        conf: Optional[float] = None) -> List[DetectedVehicle]:
        """
        Runs pretrained YOLOv8 detection to find vehicles (cars, motorcycles, buses, trucks).
        """
        return self.yolo_detector.detect(image, conf=conf)

    def predict_occupancy_with_yolo(self,
                                   spaces: List[ParkingSpace],
                                   detected_vehicles: List[DetectedVehicle],
                                   overlap_threshold: Optional[float] = None) -> List[SlotClassificationResult]:
        """
        Predicts occupancy by calculating geometric overlap between slot polygons
        and detected vehicle bounding boxes.
        Handles perspective distortion and scale differences between bays.
        """
        thresh = overlap_threshold if overlap_threshold is not None else self.overlap_threshold
        results = []

        for s in spaces:
            if not s.is_valid:
                results.append(
                    SlotClassificationResult(
                        slot_id=s.id,
                        occupied_pred=False,
                        confidence=0.0,
                        method="YOLO Overlap Analysis",
                        occupied_gt=s.occupied_gt,
                        overlap_ratio=0.0
                    )
                )
                continue

            max_overlap = 0.0
            matched_v = None

            for v in detected_vehicles:
                # Use perspective-aware scale-invariant overlap calculation
                overlap = perspective_aware_overlap(s.points, v.bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    matched_v = v

            is_occupied = (max_overlap >= thresh)
            conf = min(1.0, max_overlap / max(1e-5, thresh)) if is_occupied else min(1.0, 1.0 - (max_overlap / max(1e-5, thresh)))

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

    # ==========================================================
    # 7. SUPPORT CUSTOM POLYGONS FOR UPLOADED IMAGES
    # ==========================================================
    def create_custom_spaces(self,
                             custom_slots: List[Dict[str, Union[str, List[List[int]]]]]) -> List[ParkingSpace]:
        """
        Constructs ParkingSpace objects from custom user-defined polygons.
        Format: [{'id': 'A1', 'points': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}, ...]
        Enables occupancy prediction on custom uploaded parking images without XML.
        """
        spaces = []
        for item in custom_slots:
            slot_id = str(item.get("id", len(spaces) + 1))
            pts_raw = item.get("points", [])
            poly_pts = np.array(pts_raw, dtype=np.int32)

            is_valid = len(poly_pts) >= 3
            area = polygon_area(poly_pts) if is_valid else 0.0
            center = polygon_centroid(poly_pts) if is_valid else (0, 0)
            bbox = polygon_bounding_box(poly_pts) if is_valid else (0, 0, 0, 0)

            spaces.append(
                ParkingSpace(
                    id=slot_id,
                    occupied_gt=None,  # Custom image has no ground-truth XML by default
                    points=poly_pts,
                    center=center,
                    area=area,
                    bbox=bbox,
                    status="Unannotated",
                    is_valid=is_valid
                )
            )
        return spaces

    # ==========================================================
    # 8. PERSPECTIVE DISTORTION HANDLING & SLOT RECTIFICATION
    # ==========================================================
    def rectify_slot(self,
                     image: np.ndarray,
                     space: ParkingSpace,
                     target_size: Tuple[int, int] = (64, 128)) -> np.ndarray:
        """
        Applies a perspective homography transform (cv2.warpPerspective)
        to remove camera perspective tilt and rectify an oblique slot into an overhead patch.
        """
        return rectify_perspective(image, space.points, target_size=target_size)

    # ==========================================================
    # 9. VISUALIZATION: CLEARLY DISTINGUISH GT vs AI PREDICTIONS
    # ==========================================================
    def render_ground_truth(self,
                            base_image: np.ndarray,
                            spaces: List[ParkingSpace],
                            show_slot_ids: bool = True,
                            show_hud: bool = True) -> np.ndarray:
        """
        Draws green vacant and red occupied polygons from actual annotations.
        Clearly labeled as [GROUND TRUTH].
        """
        return self.visualizer.draw_ground_truth(
            base_image=base_image,
            spaces=spaces,
            show_slot_ids=show_slot_ids,
            show_hud=show_hud
        )

    def render_ai_predictions(self,
                              base_image: np.ndarray,
                              spaces: List[ParkingSpace],
                              classification_results: List[SlotClassificationResult],
                              detected_vehicles: Optional[List[DetectedVehicle]] = None,
                              show_slot_ids: bool = True,
                              show_boxes: bool = True,
                              show_hud: bool = True) -> np.ndarray:
        """
        Draws AI Predictions (green predicted vacant, red predicted occupied)
        with optional YOLO vehicle detection bounding boxes (orange).
        Clearly labeled as [AI PREDICTION].
        """
        return self.visualizer.draw_ai_predictions(
            base_image=base_image,
            spaces=spaces,
            classification_results=classification_results,
            detected_vehicles=detected_vehicles,
            show_slot_ids=show_slot_ids,
            show_boxes=show_boxes,
            show_hud=show_hud
        )

    def render_comparison_audit(self,
                                base_image: np.ndarray,
                                spaces: List[ParkingSpace],
                                classification_results: List[SlotClassificationResult],
                                detected_vehicles: Optional[List[DetectedVehicle]] = None,
                                show_hud: bool = True) -> np.ndarray:
        """
        Draws an audit visualization comparing AI predictions against ground truth labels:
        - True Negatives: Green
        - True Positives: Red
        - False Positives (Errors): Yellow
        - False Negatives (Errors): Magenta
        Clearly labeled as [AUDIT COMPARISON].
        """
        return self.visualizer.draw_comparison_audit(
            base_image=base_image,
            spaces=spaces,
            classification_results=classification_results,
            detected_vehicles=detected_vehicles,
            show_hud=show_hud
        )

    def evaluate(self,
                 classification_results: List[SlotClassificationResult]) -> Optional[EvaluationMetrics]:
        """
        Computes academic evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
        comparing AI predictions against ground truth labels. No fake metrics.
        """
        return EvaluationMetrics.compute(classification_results)

    def save_image(self, output_path: Union[str, Path], image: np.ndarray) -> bool:
        """Saves annotated image to disk."""
        return save_image(Path(output_path), image)
