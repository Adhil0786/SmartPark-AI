"""
SmartPark AI - Pretrained YOLO Vehicle Detector
Utilizes Ultralytics YOLOv8 for detecting vehicles (cars, motorcycles, buses, trucks)
in parking lot camera feeds with bounding boxes and confidence scores.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

from config import (
    YOLO_WEIGHTS_PATH,
    DEFAULT_YOLO_CONF,
    COCO_VEHICLE_CLASSES
)


@dataclass
class DetectedVehicle:
    """Represents a vehicle detected by the YOLO model."""
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

    @property
    def int_bbox(self) -> Tuple[int, int, int, int]:
        """Return integer [x1, y1, x2, y2] coordinates."""
        return (
            int(round(self.bbox[0])),
            int(round(self.bbox[1])),
            int(round(self.bbox[2])),
            int(round(self.bbox[3]))
        )

    @property
    def center(self) -> Tuple[int, int]:
        """Return center (x, y) coordinates of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)

    def to_dict(self) -> Dict:
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": [round(c, 1) for c in self.bbox],
            "center": self.center
        }


class YOLOVehicleDetector:
    """
    Wrapper for Ultralytics YOLOv8 object detection.
    Optimized for vehicle class filtering and inference.
    """

    def __init__(self,
                 weights_path: Union[str, Path] = YOLO_WEIGHTS_PATH,
                 confidence_threshold: float = DEFAULT_YOLO_CONF,
                 target_classes: Dict[int, str] = None):
        self.weights_path = Path(weights_path)
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or COCO_VEHICLE_CLASSES
        self.model = None
        self.load_error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model into memory."""
        self.load_error = None
        try:
            from ultralytics import YOLO
            if self.weights_path.exists():
                self.model = YOLO(str(self.weights_path))
            else:
                self.model = YOLO("yolov8n.pt")
        except Exception as e:
            self.load_error = str(e)
            self.model = None

    def detect(self,
               image: np.ndarray,
               conf: Optional[float] = None) -> List[DetectedVehicle]:
        """
        Run object detection on an input image (BGR or RGB numpy array).
        Filters predictions to vehicle classes (car, motorcycle, bus, truck).
        """
        if self.model is None:
            self._load_model()

        if self.model is None:
            return []

        try:
            conf_threshold = conf if conf is not None else self.confidence_threshold
            results = self.model(
                image,
                conf=conf_threshold,
                verbose=False,
                device="cpu"
            )
        except Exception as e:
            self.load_error = str(e)
            return []

        detections: List[DetectedVehicle] = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                # Filter only target vehicle categories
                if cls_id in self.target_classes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    class_name = self.target_classes[cls_id]
                    detections.append(
                        DetectedVehicle(
                            bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                            confidence=confidence,
                            class_id=cls_id,
                            class_name=class_name
                        )
                    )

        return detections
