"""
Computer Vision modules for SmartPark AI.
"""

from computer_vision.xml_parser import PKLotXMLParser, ParkingSpace
from computer_vision.yolo_detector import YOLOVehicleDetector, DetectedVehicle
from computer_vision.visualizer import ParkingVisualizer
from computer_vision.occupancy_classifier import OccupancyClassifier, SlotClassificationResult
from computer_vision.metrics import EvaluationMetrics
from computer_vision.vision_engine import SmartParkVisionEngine
from computer_vision.benchmark_evaluator import DateSplitEvaluator, BenchmarkResult

__all__ = [
    "PKLotXMLParser",
    "ParkingSpace",
    "YOLOVehicleDetector",
    "DetectedVehicle",
    "ParkingVisualizer",
    "OccupancyClassifier",
    "SlotClassificationResult",
    "EvaluationMetrics",
    "SmartParkVisionEngine",
    "DateSplitEvaluator",
    "BenchmarkResult"
]
