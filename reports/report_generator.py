"""
SmartPark AI - Academic Evaluation Report Generator
Produces structured CSV audits, Markdown research summaries,
and performance reports for smart parking computer vision experiments.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

from config import REPORTS_DIR
from computer_vision.metrics import EvaluationMetrics


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Format pandas DataFrame as Markdown table without external tabulate dependency."""
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |"
    ]
    for _, row in df.iterrows():
        row_str = [str(val).replace("|", "\\|") for val in row.values]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)


class ReportGenerator:
    """
    Generates and saves research reports from parking occupancy evaluations.
    """

    @staticmethod
    def export_csv(metrics: EvaluationMetrics,
                   filename: Optional[str] = None) -> Tuple[Path, str]:
        """
        Export detailed per-slot evaluation results to a CSV file.
        Returns (saved_path, csv_string).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not filename:
            filename = f"smartpark_eval_{timestamp}.csv"

        out_path = REPORTS_DIR / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        csv_str = metrics.detailed_df.to_csv(index=False)
        out_path.write_text(csv_str, encoding="utf-8")
        return out_path, csv_str

    @staticmethod
    def generate_academic_report(lot_id: str,
                                 image_name: str,
                                 method_name: str,
                                 metrics: EvaluationMetrics,
                                 extra_metadata: Optional[Dict] = None) -> Tuple[Path, str]:
        """
        Build a formatted academic evaluation report in Markdown and save it to reports/.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{lot_id}_{file_ts}.md"
        out_path = REPORTS_DIR / filename

        meta = extra_metadata or {}
        weather = meta.get("weather", "N/A")
        date_val = meta.get("date", "N/A")
        conf_thresh = meta.get("yolo_conf", "N/A")
        overlap_thresh = meta.get("overlap_threshold", "N/A")

        # Markdown synthesis
        md_lines = [
            f"# SmartPark AI - Academic Evaluation Report",
            f"**Generated:** {timestamp}  ",
            f"**Parking Lot:** {lot_id}  ",
            f"**Image Source:** `{image_name}`  ",
            f"**Weather Condition:** {weather} | **Date:** {date_val}  ",
            f"**Evaluation Methodology:** {method_name}  ",
            f"**Parameters:** YOLO Conf={conf_thresh}, Overlap Threshold={overlap_thresh}  ",
            "",
            "---",
            "",
            "## 1. Executive Performance Summary",
            "",
            "| Metric | Score | Description |",
            "|---|---|---|",
            f"| **Total Evaluated Spaces** | **{metrics.total_slots}** | Total annotated parking bays |",
            f"| **Overall Accuracy** | **{metrics.accuracy * 100.0:.2f}%** | Overall correct classification rate |",
            f"| **Precision (Occupied)** | **{metrics.precision * 100.0:.2f}%** | True Positives / (TP + FP) |",
            f"| **Recall / Sensitivity** | **{metrics.recall * 100.0:.2f}%** | True Positives / (TP + FN) |",
            f"| **Specificity (Vacant)** | **{metrics.specificity * 100.0:.2f}%** | True Negatives / (TN + FP) |",
            f"| **F1-Score** | **{metrics.f1_score * 100.0:.2f}%** | Harmonic mean of Precision & Recall |",
            f"| **Balanced Accuracy** | **{metrics.balanced_accuracy * 100.0:.2f}%** | Macro average of sensitivity & specificity |",
            "",
            "---",
            "",
            "## 2. Confusion Matrix",
            "",
            "| Actual \\ Predicted | Predicted Vacant (0) | Predicted Occupied (1) |",
            "|---|---|---|",
            f"| **Actual Vacant (0)** | True Negatives (TN): **{metrics.tn}** | False Positives (FP): **{metrics.fp}** |",
            f"| **Actual Occupied (1)** | False Negatives (FN): **{metrics.fn}** | True Positives (TP): **{metrics.tp}** |",
            "",
            "---",
            "",
            "## 3. Computer Vision & Error Diagnostics",
            "",
            f"- **Correctly Classified Bays:** {metrics.tp + metrics.tn} / {metrics.total_slots} ({(metrics.tp + metrics.tn) / metrics.total_slots * 100.0:.1f}%)",
            f"- **Type I Errors (False Alarms / FP):** {metrics.fp} slots where shadows, neighboring vehicle overhang, or high edge texture triggered false occupancy.",
            f"- **Type II Errors (Missed Detections / FN):** {metrics.fn} slots where a parked vehicle had insufficient overlap or low confidence.",
            "",
            "---",
            "",
            "## 4. Per-Slot Detailed Audit Table",
            "",
            _df_to_markdown_table(metrics.detailed_df),
            "",
            "---",
            "",
            "*Generated automatically by SmartPark AI Computer Vision & Occupancy Intelligence System.*"
        ]

        report_content = "\n".join(md_lines)
        out_path.write_text(report_content, encoding="utf-8")
        return out_path, report_content
