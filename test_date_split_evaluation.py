"""
SmartPark AI - Benchmark & Date Split Test Runner
Tests model evaluation against ground-truth PKLot annotations:
- Grouped Train/Test Split by Capture Date (disjoint dates)
- Actual model predictions (YOLOv8) compared against real XML ground-truth labels
- Computes Accuracy, Precision, Recall, F1, Specificity, Confusion Matrix, and Per-Slot Accuracy
- Generates CSV and PDF reports
"""

from pathlib import Path
from dataset.data_loader import PKLotDataset
from computer_vision.benchmark_evaluator import DateSplitEvaluator
from config import REPORTS_DIR


def main():
    print("=" * 80)
    print("🔬 SMARTPARK AI - DATE-GROUPED BENCHMARK EVALUATION TEST")
    print("=" * 80)

    loader = PKLotDataset()
    evaluator = DateSplitEvaluator(loader)

    # 1. Load dataset pairs for UFPR05
    target_lot = "UFPR05"
    print(f"\n[Step 1] Loading image/XML pairs for parking lot: {target_lot}...")
    all_pairs = [p for p in loader.find_all_pairs() if p.has_xml and p.lot == target_lot]
    print(f"  -> Total annotated pairs in {target_lot}: {len(all_pairs):,}")

    # 2. Perform Reproducible Train/Test Split Grouped by Capture Date
    print("\n[Step 2] Executing Grouped Train/Test Split by Capture Date (Zero Leakage)...")
    train_pairs, test_pairs, train_dates, test_dates = DateSplitEvaluator.split_by_capture_date(
        pairs=all_pairs,
        test_size=0.30,
        random_state=42
    )

    print(f"  -> Total Unique Capture Dates: {len(train_dates) + len(test_dates)}")
    print(f"  -> Training Dates ({len(train_dates)}): {train_dates[:5]}... ({len(train_pairs)} frames)")
    print(f"  -> Held-Out Test Dates ({len(test_dates)}): {test_dates[:5]}... ({len(test_pairs)} frames)")

    # 3. Run Benchmark on Held-Out Test Dates
    print(f"\n[Step 3] Running Real Model Evaluation on Held-Out Test Dates (Sample of 15 frames)...")
    benchmark_res = evaluator.run_evaluation(
        test_pairs=test_pairs,
        model_name="YOLOv8 + Spatial Overlap (Deep Learning)",
        max_frames=15,
        yolo_conf=0.25,
        overlap_threshold=0.20
    )

    m = benchmark_res.metrics
    print("\n[Step 4] Macro Evaluation Performance (Model vs Ground-Truth):")
    print(f"  • Total Evaluated Slots:   {benchmark_res.total_slots_evaluated:,} bays across {benchmark_res.total_frames} frames")
    print(f"  • Overall Accuracy:        {m.accuracy * 100.0:.2f}%")
    print(f"  • Precision (Occupied):    {m.precision * 100.0:.2f}%")
    print(f"  • Recall (Sensitivity):    {m.recall * 100.0:.2f}%")
    print(f"  • Specificity (Vacant):    {m.specificity * 100.0:.2f}%")
    print(f"  • F1-Score:                {m.f1_score * 100.0:.2f}%")
    print(f"  • Balanced Accuracy:       {m.balanced_accuracy * 100.0:.2f}%")

    print("\n  Confusion Matrix:")
    print(f"    TP: {m.tp:3d} (Correct Occupied)   |  FP: {m.fp:3d} (Type I Errors)")
    print(f"    FN: {m.fn:3d} (Type II Errors)     |  TN: {m.tn:3d} (Correct Vacant)")

    # 4. Display Per-Slot Accuracy
    print(f"\n[Step 5] Per-Slot Accuracy Breakdown ({len(benchmark_res.per_slot_df)} Individual Bays):")
    print(benchmark_res.per_slot_df.head(15).to_string(index=False))

    # 5. Export CSV & PDF Reports
    print("\n[Step 6] Generating CSV and PDF Reports...")
    out_dir = REPORTS_DIR / "benchmark"
    per_slot_csv, records_csv = DateSplitEvaluator.export_csv_reports(benchmark_res, out_dir)
    pdf_path = DateSplitEvaluator.export_pdf_report(benchmark_res)

    print(f"  -> Generated Per-Slot CSV:        {per_slot_csv.name}")
    print(f"  -> Generated Detailed Audit CSV:  {records_csv.name}")
    print(f"  -> Generated Publication PDF:     {pdf_path.name} ({pdf_path.stat().st_size:,} bytes)")

    print("\n" + "=" * 80)
    print("✅ BENCHMARK EVALUATION TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
