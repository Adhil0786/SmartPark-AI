"""
SmartPark AI - Computer Vision Visualizer
Provides clear, publication-grade OpenCV overlays:
- Pure Ground-Truth Annotation view
- Pure AI Prediction view (with YOLO vehicle bounding boxes)
- Comparative Verification view (highlighting True Positives, True Negatives, and Misclassifications)
- Support for custom user-defined polygons
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

from config import COLORS, ALPHA_OVERLAY
from computer_vision.xml_parser import ParkingSpace
from computer_vision.yolo_detector import DetectedVehicle
from computer_vision.occupancy_classifier import SlotClassificationResult


class ParkingVisualizer:
    """
    Renders visual overlays for parking slot polygons, YOLO detections,
    status badges, and evaluation indicators.
    Clearly distinguishes Ground Truth annotations from AI predictions.
    """

    def __init__(self,
                 alpha: float = ALPHA_OVERLAY,
                 vacant_color: Tuple[int, int, int] = COLORS["vacant_bgr"],
                 occupied_color: Tuple[int, int, int] = COLORS["occupied_bgr"],
                 box_color: Tuple[int, int, int] = COLORS["detected_box_bgr"]):
        self.alpha = alpha
        self.vacant_color = vacant_color      # Green
        self.occupied_color = occupied_color  # Red
        self.unannotated_color = (0, 200, 255) # Amber
        self.box_color = box_color            # Orange

        # Diagnostic colors for comparison mode
        self.tp_color = (60, 76, 231)         # Red (True Positive)
        self.tn_color = (46, 204, 113)        # Green (True Negative)
        self.fp_color = (0, 215, 255)         # Yellow/Gold (False Positive Error)
        self.fn_color = (255, 0, 255)         # Magenta/Purple (False Negative Error)
        self.layout_color = (255, 120, 0)     # Sky Blue (for Layout Design Mode)

    def draw_slot_layout(self,
                         base_image: np.ndarray,
                         spaces: List[ParkingSpace],
                         show_slot_ids: bool = True,
                         show_corners: bool = True,
                         show_hud: bool = True) -> np.ndarray:
        """
        Draws unclassified parking slot polygons for the interactive layout editor.
        Displays each bay's quadrilateral boundaries, corner points, and slot IDs.
        Does not mark bays as occupied or vacant.
        """
        output = base_image.copy()
        overlay = base_image.copy()

        # 1. Semi-transparent fill for mapped bays
        for s in spaces:
            if not s.is_valid or len(s.points) < 3:
                continue
            cv2.fillPoly(overlay, [s.points], self.layout_color)

        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)

        # 2. Draw crisp borders and corner points
        for s in spaces:
            if not s.is_valid or len(s.points) < 3:
                continue
            cv2.polylines(output, [s.points], isClosed=True, color=(255, 180, 50), thickness=2, lineType=cv2.LINE_AA)

            if show_corners:
                for pt in s.points:
                    cv2.circle(output, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(output, (int(pt[0]), int(pt[1])), 6, (0, 0, 0), 1, cv2.LINE_AA)

            if show_slot_ids:
                cx, cy = s.center
                label = f"#{s.id}"
                self._draw_text_badge(output, label, cx, cy)

        # 3. Top HUD Banner
        if show_hud:
            hud_text = f"[POLYGON LAYOUT EDITOR] Total Mapped Bays: {len(spaces)} | Mode: Unanalyzed (Ready for Detection)"
            self._draw_hud_banner(output, hud_text, bg_color=(15, 25, 45))

        return output

    def draw_ground_truth(self,
                          base_image: np.ndarray,
                          spaces: List[ParkingSpace],
                          show_slot_ids: bool = True,
                          show_hud: bool = True) -> np.ndarray:
        """
        Draws actual Ground-Truth annotations extracted from PKLot XML.
        Green = Vacant, Red = Occupied, Amber = Unannotated.
        Header explicitly declares '[GROUND TRUTH ANNOTATION]'.
        """
        output = base_image.copy()
        overlay = base_image.copy()

        # 1. Fill translucent polygons
        for s in spaces:
            if not s.is_valid or len(s.points) < 3:
                continue

            if s.occupied_gt is True:
                color = self.occupied_color
            elif s.occupied_gt is False:
                color = self.vacant_color
            else:
                color = self.unannotated_color

            cv2.fillPoly(overlay, [s.points], color)

        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)

        # 2. Draw crisp borders and slot ID labels
        for s in spaces:
            if not s.is_valid or len(s.points) < 3:
                continue

            if s.occupied_gt is True:
                border_color = self.occupied_color
            elif s.occupied_gt is False:
                border_color = self.vacant_color
            else:
                border_color = self.unannotated_color

            cv2.polylines(output, [s.points], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

            if show_slot_ids:
                cx, cy = s.center
                label = f"#{s.id}"
                self._draw_text_badge(output, label, cx, cy)

        # 3. Top HUD Header
        if show_hud:
            total = len(spaces)
            occ = sum(1 for s in spaces if s.occupied_gt is True)
            vac = sum(1 for s in spaces if s.occupied_gt is False)
            unann = sum(1 for s in spaces if s.occupied_gt is None)
            rate = (occ / (occ + vac) * 100.0) if (occ + vac) > 0 else 0.0

            hud_text = f"[GROUND TRUTH] Total: {total} | Vacant: {vac} | Occupied: {occ} | Rate: {rate:.1f}%"
            if unann > 0:
                hud_text += f" | Unannotated: {unann}"
            self._draw_hud_banner(output, hud_text, bg_color=(20, 30, 20))

        return output

    def draw_ai_predictions(self,
                            base_image: np.ndarray,
                            spaces: List[ParkingSpace],
                            classification_results: List[SlotClassificationResult],
                            detected_vehicles: Optional[List[DetectedVehicle]] = None,
                            show_slot_ids: bool = True,
                            show_boxes: bool = True,
                            show_hud: bool = True) -> np.ndarray:
        """
        Draws AI Predictions (e.g. YOLOv8 overlap analysis or ML patch classification).
        Clearly distinguished from Ground Truth annotations.
        """
        output = base_image.copy()
        overlay = base_image.copy()

        class_map: Dict[str, SlotClassificationResult] = {r.slot_id: r for r in classification_results}

        # 1. Fill translucent polygons based on AI prediction
        for s in spaces:
            if not s.is_valid or len(s.points) < 3:
                continue

            res = class_map.get(s.id)
            is_occ = res.occupied_pred if res else False
            color = self.occupied_color if is_occ else self.vacant_color
            cv2.fillPoly(overlay, [s.points], color)

        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)

        # 2. Draw borders and slot labels
        for s in spaces:
            if not s.is_valid or len(s.points) < 3:
                continue

            res = class_map.get(s.id)
            is_occ = res.occupied_pred if res else False
            border_color = self.occupied_color if is_occ else self.vacant_color

            cv2.polylines(output, [s.points], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

            if show_slot_ids:
                cx, cy = s.center
                conf_str = f"{res.confidence:.0%}" if res else ""
                label = f"#{s.id} ({conf_str})" if conf_str else f"#{s.id}"
                self._draw_text_badge(output, label, cx, cy)

        # 3. Draw YOLO Vehicle Bounding Boxes
        if show_boxes and detected_vehicles:
            for v in detected_vehicles:
                x1, y1, x2, y2 = v.int_bbox
                cv2.rectangle(output, (x1, y1), (x2, y2), self.box_color, 2, cv2.LINE_AA)
                box_label = f"YOLO: {v.class_name} {v.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(output, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), self.box_color, -1)
                cv2.putText(output, box_label, (x1 + 3, max(th, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # 4. Top HUD Header
        if show_hud:
            total = len(spaces)
            occ = sum(1 for r in classification_results if r.occupied_pred)
            vac = total - occ
            rate = (occ / total * 100.0) if total > 0 else 0.0
            num_v = len(detected_vehicles) if detected_vehicles else 0

            method_name = classification_results[0].method if classification_results else "AI Model"
            hud_text = f"[AI PREDICTION - {method_name}] Slots: {total} | Vacant: {vac} | Occupied: {occ} | Vehicles: {num_v} | Rate: {rate:.1f}%"
            self._draw_hud_banner(output, hud_text, bg_color=(35, 20, 20))

        return output

    def draw_comparison_audit(self,
                              base_image: np.ndarray,
                              spaces: List[ParkingSpace],
                              classification_results: List[SlotClassificationResult],
                              detected_vehicles: Optional[List[DetectedVehicle]] = None,
                              show_hud: bool = True) -> np.ndarray:
        """
        Draws an audit visualization comparing AI predictions against ground truth labels:
        - True Negative (Vacant -> Vacant): Green outline
        - True Positive (Occupied -> Occupied): Red outline
        - False Positive (Vacant -> Marked Occupied): Yellow outline (Type I error)
        - False Negative (Occupied -> Missed): Magenta outline (Type II error)
        """
        output = base_image.copy()
        overlay = base_image.copy()

        class_map: Dict[str, SlotClassificationResult] = {r.slot_id: r for r in classification_results}

        tp, tn, fp, fn = 0, 0, 0, 0

        # Draw diagnostic outlines
        for s in spaces:
            if not s.is_valid or len(s.points) < 3:
                continue

            res = class_map.get(s.id)
            if not res or s.occupied_gt is None:
                continue

            gt = s.occupied_gt
            pred = res.occupied_pred

            if gt is True and pred is True:
                color = self.tp_color
                tag = "TP"
                tp += 1
            elif gt is False and pred is False:
                color = self.tn_color
                tag = "TN"
                tn += 1
            elif gt is False and pred is True:
                color = self.fp_color
                tag = "FP!"
                fp += 1
            else:  # gt is True and pred is False
                color = self.fn_color
                tag = "FN!"
                fn += 1

            cv2.fillPoly(overlay, [s.points], color)

        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)

        for s in spaces:
            if not s.is_valid or len(s.points) < 3:
                continue

            res = class_map.get(s.id)
            if not res or s.occupied_gt is None:
                continue

            gt = s.occupied_gt
            pred = res.occupied_pred
            if gt is True and pred is True:
                color = self.tp_color
                tag = "TP"
            elif gt is False and pred is False:
                color = self.tn_color
                tag = "TN"
            elif gt is False and pred is True:
                color = self.fp_color
                tag = "FP!"
            else:
                color = self.fn_color
                tag = "FN!"

            cv2.polylines(output, [s.points], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            cx, cy = s.center
            self._draw_text_badge(output, f"#{s.id} {tag}", cx, cy)

        # Draw detected vehicles faintly
        if detected_vehicles:
            for v in detected_vehicles:
                x1, y1, x2, y2 = v.int_bbox
                cv2.rectangle(output, (x1, y1), (x2, y2), self.box_color, 1, cv2.LINE_AA)

        if show_hud:
            total = tp + tn + fp + fn
            acc = ((tp + tn) / total * 100.0) if total > 0 else 0.0
            hud_text = f"[AUDIT COMPARISON] Accuracy: {acc:.1f}% | TP: {tp} | TN: {tn} | FP(Errors): {fp} | FN(Errors): {fn}"
            self._draw_hud_banner(output, hud_text, bg_color=(25, 25, 35))

        return output

    # Backward compatibility
    def draw_parking_lot(self,
                         base_image: np.ndarray,
                         spaces: List[ParkingSpace],
                         classification_results: Optional[List[SlotClassificationResult]] = None,
                         detected_vehicles: Optional[List[DetectedVehicle]] = None,
                         show_slot_ids: bool = True,
                         show_boxes: bool = False,
                         show_hud: bool = True) -> np.ndarray:
        """Unified rendering method."""
        if classification_results is not None:
            return self.draw_ai_predictions(
                base_image=base_image,
                spaces=spaces,
                classification_results=classification_results,
                detected_vehicles=detected_vehicles,
                show_slot_ids=show_slot_ids,
                show_boxes=show_boxes,
                show_hud=show_hud
            )
        else:
            return self.draw_ground_truth(
                base_image=base_image,
                spaces=spaces,
                show_slot_ids=show_slot_ids,
                show_hud=show_hud
            )

    @staticmethod
    def _draw_text_badge(img: np.ndarray, text: str, cx: int, cy: int) -> None:
        """Helper to draw high-contrast legible text badge."""
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        bx1 = cx - tw // 2 - 2
        by1 = cy - th // 2 - 2
        bx2 = cx + tw // 2 + 2
        by2 = cy + th // 2 + 2

        h, w = img.shape[:2]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)

        cv2.rectangle(img, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.putText(img, text, (cx - tw // 2, cy + th // 2 - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_hud_banner(img: np.ndarray, text: str, bg_color: Tuple[int, int, int] = (15, 15, 20)) -> None:
        """Helper to draw semi-transparent top HUD banner."""
        h, w = img.shape[:2]
        header_bar = img.copy()
        cv2.rectangle(header_bar, (0, 0), (w, 36), bg_color, -1)
        cv2.addWeighted(header_bar, 0.85, img, 0.15, 0, img)
        cv2.putText(img, text, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
