"""
SmartPark AI - Academic Evaluation Metrics
Calculates standard classification metrics, confusion matrices, and per-slot error breakdowns
comparing AI/CV predictions against PKLot ground-truth XML annotations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

from computer_vision.occupancy_classifier import SlotClassificationResult


@dataclass
class EvaluationMetrics:
    """Stores full statistical evaluation results."""
    total_slots: int
    tp: int  # True Positive (Occupied -> Occupied)
    tn: int  # True Negative (Vacant -> Vacant)
    fp: int  # False Positive (Vacant -> Occupied)
    fn: int  # False Negative (Occupied -> Vacant)
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    balanced_accuracy: float
    confusion_matrix_df: pd.DataFrame
    detailed_df: pd.DataFrame

    def to_dict(self) -> Dict:
        return {
            "Total Slots": self.total_slots,
            "Accuracy (%)": round(self.accuracy * 100.0, 2),
            "Precision (%)": round(self.precision * 100.0, 2),
            "Recall (%)": round(self.recall * 100.0, 2),
            "Specificity (%)": round(self.specificity * 100.0, 2),
            "F1-Score (%)": round(self.f1_score * 100.0, 2),
            "Balanced Accuracy (%)": round(self.balanced_accuracy * 100.0, 2),
            "True Positives (TP)": self.tp,
            "True Negatives (TN)": self.tn,
            "False Positives (FP)": self.fp,
            "False Negatives (FN)": self.fn,
        }

    @classmethod
    def compute(cls, results: List[SlotClassificationResult]) -> Optional["EvaluationMetrics"]:
        """
        Compute evaluation metrics comparing predicted occupancy with ground truth.
        Returns None if no ground truth is available.
        """
        valid_results = [r for r in results if r.occupied_gt is not None]
        if not valid_results:
            return None

        y_true = np.array([1 if r.occupied_gt else 0 for r in valid_results])
        y_pred = np.array([1 if r.occupied_pred else 0 for r in valid_results])

        # Confusion matrix: tn, fp, fn, tp
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        total = len(valid_results)
        acc = float(accuracy_score(y_true, y_pred))

        # Precision, Recall, F1
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        bal_acc = (r + spec) / 2.0

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Vacant (0)", "Actual Occupied (1)"],
            columns=["Predicted Vacant (0)", "Predicted Occupied (1)"]
        )

        rows = []
        for res in valid_results:
            gt_label = "Occupied" if res.occupied_gt else "Vacant"
            pred_label = "Occupied" if res.occupied_pred else "Vacant"
            status = "Correct" if res.is_correct else "Mismatch"

            rows.append({
                "Slot ID": f"#{res.slot_id}",
                "Ground Truth": gt_label,
                "Prediction": pred_label,
                "Evaluation": status,
                "Confidence": f"{res.confidence:.1%}",
                "Vehicle Overlap": f"{res.overlap_ratio:.1%}",
                "Edge Density": f"{res.edge_density:.4f}",
                "Method": res.method
            })

        detailed_df = pd.DataFrame(rows)

        return cls(
            total_slots=total,
            tp=int(tp),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            accuracy=acc,
            precision=float(p),
            recall=float(r),
            specificity=spec,
            f1_score=float(f1),
            balanced_accuracy=bal_acc,
            confusion_matrix_df=cm_df,
            detailed_df=detailed_df
        )
