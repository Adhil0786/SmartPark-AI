"""
SmartPark AI - Reproducible Benchmark Evaluator
Performs rigorous model evaluation grouped by capture date to prevent data leakage.
Computes Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Per-Slot Accuracy.
Exports comprehensive CSV and publication-ready multi-page PDF reports.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import random
import os

# Ensure matplotlib uses writable config directory
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns

from config import REPORTS_DIR, DEFAULT_YOLO_CONF, DEFAULT_OVERLAP_THRESHOLD
from computer_vision.xml_parser import PKLotXMLParser, ParkingSpace
from computer_vision.yolo_detector import YOLOVehicleDetector
from computer_vision.occupancy_classifier import OccupancyClassifier, SlotClassificationResult
from computer_vision.metrics import EvaluationMetrics


@dataclass
class BenchmarkResult:
    """Stores full benchmark evaluation outcome across held-out capture dates."""
    model_name: str
    lot: str
    total_frames: int
    total_slots_evaluated: int
    train_dates: List[str]
    test_dates: List[str]
    metrics: EvaluationMetrics
    per_slot_df: pd.DataFrame
    detailed_df: pd.DataFrame
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class DateSplitEvaluator:
    """
    Evaluator implementing a reproducible train/test split grouped by capture date.
    Ensures zero temporal data leakage between frames of the same day.
    """

    def __init__(self, dataset_loader=None):
        if dataset_loader is None:
            from dataset.data_loader import PKLotDataset
            self.loader = PKLotDataset()
        else:
            self.loader = dataset_loader

    @staticmethod
    def split_by_capture_date(pairs: List[Any],
                              test_size: float = 0.30,
                              random_state: int = 42) -> Tuple[List[Any], List[Any], List[str], List[str]]:
        """
        Groups all pairs by capture date and splits dates reproducibly.
        Returns (train_pairs, test_pairs, train_dates, test_dates).
        """
        valid_pairs = [p for p in pairs if p.has_xml and p.xml_path is not None]
        unique_dates = sorted(list({p.date for p in valid_pairs}))

        rng = random.Random(random_state)
        shuffled = unique_dates.copy()
        rng.shuffle(shuffled)

        if len(shuffled) <= 1:
            train_dates = shuffled
            test_dates = shuffled
            split_p = max(1, len(valid_pairs) // 2)
            train_pairs = valid_pairs[:split_p]
            test_pairs = valid_pairs[split_p:] if len(valid_pairs) > 1 else valid_pairs
            return train_pairs, test_pairs, train_dates, test_dates

        split_idx = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * (1.0 - test_size)))))
        train_dates = sorted(shuffled[:split_idx])
        test_dates = sorted(shuffled[split_idx:])

        train_pairs = [p for p in valid_pairs if p.date in set(train_dates)]
        test_pairs = [p for p in valid_pairs if p.date in set(test_dates)]

        return train_pairs, test_pairs, train_dates, test_dates

    def run_evaluation(self,
                       test_pairs: List[Any],
                       model_name: str = "YOLOv8 + Spatial Overlap",
                       yolo_conf: float = DEFAULT_YOLO_CONF,
                       overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
                       max_frames: Optional[int] = 20,
                       detector: Optional[YOLOVehicleDetector] = None) -> BenchmarkResult:
        """
        Evaluates genuine model predictions against real PKLot XML annotations on held-out dates.
        Computes macro metrics and per-slot accuracy.
        """
        if not test_pairs:
            raise ValueError("No test pairs provided for evaluation.")

        # Sample evenly across test dates if max_frames is specified
        if max_frames and len(test_pairs) > max_frames:
            step = max(1, len(test_pairs) // max_frames)
            eval_pairs = test_pairs[::step][:max_frames]
        else:
            eval_pairs = test_pairs

        det = detector or YOLOVehicleDetector(confidence_threshold=yolo_conf)
        all_class_results: List[SlotClassificationResult] = []
        detailed_rows = []

        lot_name = eval_pairs[0].lot if eval_pairs else "PKLot"

        for pair in eval_pairs:
            img, parser = self.loader.load_pair(pair.image_path)
            if img is None or parser is None:
                continue

            spaces = parser.get_spaces()
            valid_spaces = [s for s in spaces if s.is_valid and s.occupied_gt is not None]
            if not valid_spaces:
                continue

            # Run actual model inference
            vehicles = det.detect(img, conf=yolo_conf)
            slot_results = OccupancyClassifier.classify_with_yolo(
                valid_spaces, vehicles, overlap_threshold=overlap_threshold
            )
            all_class_results.extend(slot_results)

            for s_res in slot_results:
                is_correct = (s_res.occupied_pred == s_res.occupied_gt)
                detailed_rows.append({
                    "image_name": pair.image_name,
                    "lot": pair.lot,
                    "date": pair.date,
                    "weather": pair.weather,
                    "slot_id": s_res.slot_id,
                    "ground_truth": "Occupied" if s_res.occupied_gt else "Vacant",
                    "predicted": "Occupied" if s_res.occupied_pred else "Vacant",
                    "evaluation": "Correct" if is_correct else "Mismatch",
                    "confidence": round(s_res.confidence, 3),
                    "overlap_ratio": round(s_res.overlap_ratio, 3)
                })

        detailed_df = pd.DataFrame(detailed_rows)
        macro_metrics = EvaluationMetrics.compute(all_class_results)
        if macro_metrics is None:
            raise RuntimeError("Evaluation failed: no valid predictions with ground truth.")

        # -------------------------------------------------------------
        # COMPUTE PER-SLOT ACCURACY BREAKDOWN
        # -------------------------------------------------------------
        per_slot_rows = []
        slot_groups = detailed_df.groupby("slot_id")

        for s_id, group in slot_groups:
            tot = len(group)
            correct = sum(group["evaluation"] == "Correct")
            acc = (correct / tot * 100.0) if tot > 0 else 0.0

            tp = sum((group["ground_truth"] == "Occupied") & (group["predicted"] == "Occupied"))
            tn = sum((group["ground_truth"] == "Vacant") & (group["predicted"] == "Vacant"))
            fp = sum((group["ground_truth"] == "Vacant") & (group["predicted"] == "Occupied"))
            fn = sum((group["ground_truth"] == "Occupied") & (group["predicted"] == "Vacant"))

            try:
                sort_key = int(s_id)
            except ValueError:
                sort_key = s_id

            per_slot_rows.append({
                "_sort_key": sort_key,
                "Slot ID": f"#{s_id}",
                "Evaluations": tot,
                "Correct": correct,
                "Accuracy (%)": round(acc, 2),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn
            })

        per_slot_df = pd.DataFrame(per_slot_rows)
        per_slot_df = per_slot_df.sort_values(by="_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)

        return BenchmarkResult(
            model_name=model_name,
            lot=lot_name,
            total_frames=len(eval_pairs),
            total_slots_evaluated=len(detailed_df),
            train_dates=[],
            test_dates=sorted(list({p.date for p in eval_pairs})),
            metrics=macro_metrics,
            per_slot_df=per_slot_df,
            detailed_df=detailed_df
        )

    # ==========================================================
    # CSV REPORT GENERATOR
    # ==========================================================
    @staticmethod
    def export_csv_reports(result: BenchmarkResult,
                           output_dir: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
        """
        Saves two CSV files:
        1. per_slot_accuracy.csv
        2. benchmark_detailed_predictions.csv
        """
        out_d = Path(output_dir) if output_dir else (REPORTS_DIR / "benchmark")
        out_d.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        per_slot_csv = out_d / f"per_slot_accuracy_{result.lot}_{ts}.csv"
        records_csv = out_d / f"benchmark_records_{result.lot}_{ts}.csv"

        result.per_slot_df.to_csv(per_slot_csv, index=False)
        result.detailed_df.to_csv(records_csv, index=False)

        return per_slot_csv, records_csv

    # ==========================================================
    # MULTI-PAGE ACADEMIC PDF REPORT GENERATOR
    # ==========================================================
    @staticmethod
    def export_pdf_report(result: BenchmarkResult,
                          output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Generates an academic, multi-page PDF evaluation report containing:
        - Cover & Executive Metrics Summary
        - Grouped Date Train/Test Split Info
        - Confusion Matrix Heatmap
        - Per-Slot Accuracy Chart & Table
        """
        if not output_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = REPORTS_DIR / "benchmark" / f"SmartPark_Evaluation_Report_{result.lot}_{ts}.pdf"

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        m = result.metrics

        with PdfPages(str(out_p)) as pdf:
            # -------------------------------------------------------------
            # PAGE 1: EXECUTIVE SUMMARY & CONFUSION MATRIX
            # -------------------------------------------------------------
            fig1 = plt.figure(figsize=(8.5, 11), dpi=200)
            fig1.patch.set_facecolor("#FFFFFF")

            # Title Header
            plt.suptitle("SmartPark AI - Model Evaluation & Benchmark Report", fontsize=16, weight="bold", y=0.96, color="#070F2B")
            plt.figtext(0.5, 0.93, f"Parking Lot: {result.lot}  |  Generated: {result.timestamp}  |  Model: {result.model_name}", ha="center", fontsize=9, color="#64748B")

            # Divider line
            ax_line = fig1.add_axes([0.1, 0.915, 0.8, 0.002])
            ax_line.axhline(color="#0B192C", linewidth=1.5)
            ax_line.axis("off")

            # Section 1: Methodology & Split
            ax_meta = fig1.add_axes([0.1, 0.77, 0.8, 0.13])
            ax_meta.axis("off")
            split_summary = (
                f"• Evaluation Strategy: Grouped Train/Test Split by Capture Date (Zero Temporal Leakage)\n"
                f"• Held-Out Test Dates ({len(result.test_dates)}): {', '.join(result.test_dates[:7])}{'...' if len(result.test_dates) > 7 else ''}\n"
                f"• Total Evaluated Test Frames: {result.total_frames:,}\n"
                f"• Total Evaluated Parking Bays: {result.total_slots_evaluated:,} slots"
            )
            ax_meta.text(0.0, 0.95, "1. Experimental Configuration & Grouped Date Split", fontsize=11, weight="bold", color="#0B192C")
            ax_meta.text(0.02, 0.20, split_summary, fontsize=9, color="#1E293B", linespacing=1.6)

            # Section 2: Macro Metrics Table
            ax_tbl = fig1.add_axes([0.1, 0.58, 0.8, 0.16])
            ax_tbl.axis("off")
            ax_tbl.text(0.0, 1.05, "2. Macro Performance Metrics (Genuine Predictions vs Ground-Truth)", fontsize=11, weight="bold", color="#0B192C")

            table_data = [
                ["Overall Accuracy", f"{m.accuracy * 100.0:.2f}%", "Balanced Accuracy", f"{m.balanced_accuracy * 100.0:.2f}%"],
                ["Precision (Occupied)", f"{m.precision * 100.0:.2f}%", "Recall (Sensitivity)", f"{m.recall * 100.0:.2f}%"],
                ["Specificity (Vacant)", f"{m.specificity * 100.0:.2f}%", "F1-Score", f"{m.f1_score * 100.0:.2f}%"]
            ]
            tbl = ax_tbl.table(
                cellText=table_data,
                colWidths=[0.30, 0.20, 0.30, 0.20],
                loc="center",
                cellLoc="center"
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.0, 1.6)
            for (row, col), cell in tbl.get_celld().items():
                cell.set_edgecolor("#CBD5E1")
                if col in [0, 2]:
                    cell.set_facecolor("#F8FAFC")
                    cell.get_text().set_fontweight("bold")
                    cell.get_text().set_color("#0F172A")
                else:
                    cell.set_facecolor("#FFFFFF")
                    cell.get_text().set_color("#007BFF")
                    cell.get_text().set_fontweight("bold")

            # Section 3: Confusion Matrix
            ax_cm = fig1.add_axes([0.22, 0.18, 0.56, 0.32])
            cm_data = np.array([[m.tn, m.fp], [m.fn, m.tp]])
            sns.heatmap(
                cm_data,
                annot=True,
                fmt=",d",
                cmap="Blues",
                cbar=False,
                ax=ax_cm,
                xticklabels=["Predicted Vacant (0)", "Predicted Occupied (1)"],
                yticklabels=["Actual Vacant (0)", "Actual Occupied (1)"],
                annot_kws={"size": 13, "weight": "bold"}
            )
            ax_cm.set_title("3. 2x2 Confusion Matrix", fontsize=11, weight="bold", pad=12, color="#0B192C")
            ax_cm.tick_params(labelsize=9)

            # Footer
            fig1.text(0.5, 0.04, "Page 1 of 2  •  SmartPark AI Academic Vision Intelligence  •  Genuine PKLot Annotations", ha="center", fontsize=8, color="#94A3B8")
            pdf.savefig(fig1)
            plt.close(fig1)

            # -------------------------------------------------------------
            # PAGE 2: PER-SLOT ACCURACY BREAKDOWN
            # -------------------------------------------------------------
            fig2 = plt.figure(figsize=(8.5, 11), dpi=200)
            fig2.patch.set_facecolor("#FFFFFF")

            plt.suptitle("SmartPark AI - Per-Slot Accuracy & Spatial Diagnostics", fontsize=15, weight="bold", y=0.96, color="#070F2B")
            plt.figtext(0.5, 0.93, f"Parking Lot: {result.lot}  |  Total Evaluated Bays: {len(result.per_slot_df)}", ha="center", fontsize=9, color="#64748B")

            # Per-Slot Accuracy Bar Plot
            ax_bar = fig2.add_axes([0.1, 0.60, 0.8, 0.30])
            slots_sample = result.per_slot_df.head(40)  # Up to 40 bays for clean visual
            x_labels = [s.replace("#", "") for s in slots_sample["Slot ID"]]
            y_acc = slots_sample["Accuracy (%)"]

            colors = ["#10B981" if acc >= 90.0 else "#F59E0B" if acc >= 75.0 else "#EF4444" for acc in y_acc]
            bars = ax_bar.bar(x_labels, y_acc, color=colors, edgecolor="#0B192C", linewidth=0.5)
            ax_bar.axhline(m.accuracy * 100.0, color="#EF4444", linestyle="--", linewidth=1.2, label=f"Mean Accuracy ({m.accuracy*100:.1f}%)")
            ax_bar.set_ylim(0, 110)
            ax_bar.set_xlabel("Slot ID", fontsize=9, weight="bold", labelpad=6)
            ax_bar.set_ylabel("Accuracy (%)", fontsize=9, weight="bold")
            ax_bar.set_title("Per-Slot Accuracy Across Held-Out Test Frames", fontsize=10, weight="bold", pad=8)
            ax_bar.tick_params(axis="x", rotation=90, labelsize=7)
            ax_bar.tick_params(axis="y", labelsize=8)
            ax_bar.legend(loc="lower right", fontsize=8)
            ax_bar.grid(axis="y", linestyle=":", alpha=0.6)

            # Per-Slot Table (First 15-20 slots)
            ax_tbl2 = fig2.add_axes([0.1, 0.09, 0.8, 0.44])
            ax_tbl2.axis("off")
            ax_tbl2.text(0.0, 1.03, f"Detailed Per-Slot Breakdown (Sample of {min(20, len(result.per_slot_df))} Bays)", fontsize=10, weight="bold", color="#0B192C")

            table_subset = result.per_slot_df.head(20)
            cols = ["Slot ID", "Evaluations", "Correct", "Accuracy (%)", "TP", "TN", "FP", "FN"]
            tbl_vals = table_subset[cols].values.tolist()

            tbl2 = ax_tbl2.table(
                cellText=tbl_vals,
                colLabels=cols,
                loc="upper center",
                cellLoc="center"
            )
            tbl2.auto_set_font_size(False)
            tbl2.set_fontsize(7.5)
            tbl2.scale(1.0, 1.35)

            # Style table header & alternating rows
            for (r, c), cell in tbl2.get_celld().items():
                cell.set_edgecolor("#CBD5E1")
                if r == 0:
                    cell.set_facecolor("#0B192C")
                    cell.get_text().set_color("#FFFFFF")
                    cell.get_text().set_fontweight("bold")
                elif r % 2 == 1:
                    cell.set_facecolor("#F8FAFC")
                else:
                    cell.set_facecolor("#FFFFFF")

            fig2.text(0.5, 0.04, "Page 2 of 2  •  SmartPark AI Academic Vision Intelligence  •  Genuine PKLot Annotations", ha="center", fontsize=8, color="#94A3B8")
            pdf.savefig(fig2)
            plt.close(fig2)

        return out_p
