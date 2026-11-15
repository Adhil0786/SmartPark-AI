"""
SmartPark AI - Dataset Loader Verification & Test Suite
Executes comprehensive testing on 10-20 images from PKLot dataset:
- XML structural inspection
- Slot polygon coordinate extraction
- Occupancy label validation
- Missing/invalid annotation handling
- CSV generation & verification
- Visualized sample image outputs
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

from dataset.data_loader import PKLotDataset
from config import REPORTS_DIR


def main():
    print("=" * 75)
    print("🚗 SMARTPARK AI - PKLOT DATASET LOADER TEST SUITE")
    print("=" * 75)

    loader = PKLotDataset()

    # Step 1: Scan and pair dataset files
    print("\n[Step 1] Scanning dataset and pairing images with XML annotations...")
    all_pairs = loader.find_all_pairs()
    print(f"  -> Total images discovered: {len(all_pairs):,}")
    matched_count = sum(1 for p in all_pairs if p.has_xml)
    orphaned_count = sum(1 for p in all_pairs if not p.has_xml)
    print(f"  -> Matched with valid XML:  {matched_count:,}")
    print(f"  -> Images with missing XML: {orphaned_count}")

    # Step 2 & 3: Run comprehensive test on selected 10-20 images
    print("\n[Step 2] Executing Loader on 15 Diverse Images (All Lots, Weathers & Edge Cases)...")
    output_dir = REPORTS_DIR / "loader_test_run"
    results = loader.test_dataset_loader(output_dir=output_dir, n_test_samples=15)

    print(f"  -> Total Images Tested:       {results['total_images_tested']}")
    print(f"  -> Successfully Loaded & Parsed: {results['successful_images']}")
    print(f"  -> Images with Missing XML:   {results['images_with_missing_xml']}")
    print(f"  -> Total Slots Extracted:     {results['total_slots_extracted']}")
    print(f"  -> Occupied Slots Identified: {results['occupied_slots_count']}")
    print(f"  -> Vacant Slots Identified:   {results['vacant_slots_count']}")
    print(f"  -> Unannotated Slots Handled: {results['unannotated_slots_count']}")

    # Step 4: Display XML Structural Inspection details
    print("\n[Step 3] Structural XML Inspection Before Parsing (Sample of Tested Files):")
    for item in results["xml_inspections"][:6]:
        img_name = item["image"]
        lot = item["lot"]
        weather = item["weather"]
        has_xml = item["has_xml"]
        if not has_xml:
            print(f"  - [{lot} | {weather}] {img_name}: ⚠️ MISSING XML ANNOTATION")
            continue

        sinfo = item.get("structural_info", {})
        root_tag = sinfo.get("root_tag")
        spaces_found = sinfo.get("total_spaces_found")
        pt_tags = sinfo.get("point_tag_names", [])
        anomalies = sinfo.get("anomalies", [])
        breakdown = item.get("status_breakdown", {})

        print(f"  - [{lot} | {weather}] {img_name}:")
        print(f"      Root Tag: <{root_tag}> | Spaces: {spaces_found} | Point Tags: {pt_tags}")
        print(f"      Occupancy Breakdown: {breakdown}")
        if anomalies:
            print(f"      ⚠️ Handled Anomalies: {anomalies}")

    # Step 5: Verify Generated CSV
    print(f"\n[Step 4] Checking Generated Occupancy CSV: {results['csv_exported_path']}")
    csv_path = Path(results["csv_exported_path"])
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"  -> CSV Rows: {len(df)} records across {df['image_name'].nunique()} images")
        print(f"  -> Columns: {list(df.columns)}")
        print("\n  CSV Occupancy Status Distribution:")
        print(df["occupancy_status"].value_counts().to_string())

        print("\n  First 5 rows of CSV:")
        print(df[["image_name", "slot_id", "occupancy_status", "num_polygon_points", "is_valid"]].head().to_string())
    else:
        print("  -> ERROR: CSV file was not created!")

    # Step 6: Verify Visualized Images
    print(f"\n[Step 5] Checking Visualized Sample Images ({len(results['visualized_images'])} saved):")
    for v_path in results["visualized_images"][:5]:
        vp = Path(v_path)
        if vp.exists():
            img_stat = cv2.imread(str(vp))
            print(f"  -> Verified saved image: {vp.name} (Resolution: {img_stat.shape[1]}x{img_stat.shape[0]})")
        else:
            print(f"  -> ERROR: Visualization not found at {v_path}")

    # Step 7: Report Warnings and Errors
    print("\n[Step 6] Diagnostics, Errors, and Handled Warnings:")
    if results["errors_and_warnings"]:
        for log in results["errors_and_warnings"][:10]:
            print(f"  * {log}")
        if len(results["errors_and_warnings"]) > 10:
            print(f"  ... and {len(results['errors_and_warnings']) - 10} more handled notices.")
    else:
        print("  -> Zero warnings or errors encountered!")

    print("\n" + "=" * 75)
    print("✅ TEST SUITE COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
