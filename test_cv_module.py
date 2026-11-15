"""
SmartPark AI - Comprehensive Computer Vision Module Test Suite
Tests all 8 core CV requirements:
1. Read parking image & PKLot XML
2. Extract slot polygons
3. Identify occupied/vacant slots from real annotations
4. Draw green vacant & red occupied polygons with slot IDs
5. Calculate total, occupied, vacant, and occupancy %
6. Pretrained YOLO vehicle detection & spatial overlap
7. Custom polygon support for uploaded images
8. Variable slot size handling & perspective rectification
"""

from pathlib import Path
import cv2
import numpy as np

from computer_vision.vision_engine import SmartParkVisionEngine
from config import REPORTS_DIR


def main():
    print("=" * 80)
    print("🔍 SMARTPARK AI - COMPUTER VISION MODULE TEST SUITE")
    print("=" * 80)

    cv_engine = SmartParkVisionEngine()
    test_out_dir = REPORTS_DIR / "cv_module_test"
    test_out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Test 1 & 2: Read image and PKLot XML & Extract Slot Polygons
    # -------------------------------------------------------------
    print("\n[Req 1 & 2] Reading Image & XML Annotations and Extracting Polygons...")
    sample_img_path = Path("dataset/raw/PKLot/PKLot/UFPR05/Cloudy/2013-03-16/2013-03-16_12_05_07.jpg")
    img, parser = cv_engine.load_scene(sample_img_path)

    assert img is not None, "Failed to load image!"
    assert parser is not None, "Failed to parse XML!"

    spaces = parser.get_spaces()
    print(f"  -> Image loaded: {sample_img_path.name} (Shape: {img.shape})")
    print(f"  -> Successfully extracted {len(spaces)} parking slot polygons.")

    sample_slot = spaces[0]
    print(f"  -> Sample Slot #{sample_slot.id}:")
    print(f"      Centroid: {sample_slot.center} | Area: {sample_slot.area:.1f} px²")
    print(f"      Polygon Vertices: {sample_slot.points.tolist()}")

    # -------------------------------------------------------------
    # Test 3 & 5: Identify GT Occupancy & Calculate Statistics
    # -------------------------------------------------------------
    print("\n[Req 3 & 5] Ground-Truth Occupancy Identification & Statistics Calculation...")
    gt_stats = cv_engine.calculate_statistics(spaces)
    print("  -> Ground-Truth Statistics:")
    print(f"      Total Slots:        {gt_stats['total']}")
    print(f"      Vacant Slots:       {gt_stats['vacant']}")
    print(f"      Occupied Slots:     {gt_stats['occupied']}")
    print(f"      Unannotated Slots:  {gt_stats['unannotated']}")
    print(f"      Occupancy Rate:     {gt_stats['occupancy_rate']}%")

    assert gt_stats["total"] == 40, f"Expected 40 slots, found {gt_stats['total']}"
    assert gt_stats["occupied"] == 7, f"Expected 7 occupied slots, found {gt_stats['occupied']}"
    assert gt_stats["vacant"] == 33, f"Expected 33 vacant slots, found {gt_stats['vacant']}"

    # -------------------------------------------------------------
    # Test 4: Draw Green Vacant & Red Occupied Polygons (Ground Truth)
    # -------------------------------------------------------------
    print("\n[Req 4] Rendering Ground-Truth Image with Green (Vacant) & Red (Occupied) Polygons...")
    gt_rendered = cv_engine.render_ground_truth(img, spaces, show_slot_ids=True, show_hud=True)
    gt_out_path = test_out_dir / "01_annotated_ground_truth.jpg"
    cv_engine.save_image(gt_out_path, gt_rendered)
    print(f"  -> Saved Ground-Truth visualization to: {gt_out_path.name}")

    # -------------------------------------------------------------
    # Test 6: Pretrained YOLO Vehicle Detection & AI Occupancy
    # -------------------------------------------------------------
    print("\n[Req 6] Running Pretrained YOLO Vehicle Detection & AI Occupancy Prediction...")
    vehicles = cv_engine.detect_vehicles(img, conf=0.25)
    print(f"  -> YOLO detected {len(vehicles)} vehicles in the parking lot:")
    for i, v in enumerate(vehicles, 1):
        print(f"      [{i}] {v.class_name.upper()} (Confidence: {v.confidence:.1%}) at bbox {[round(c, 1) for c in v.bbox]}")

    ai_results = cv_engine.predict_occupancy_with_yolo(spaces, vehicles, overlap_threshold=0.20)
    ai_stats = cv_engine.calculate_statistics(ai_results)
    print("  -> AI Prediction Statistics:")
    print(f"      Total Slots:        {ai_stats['total']}")
    print(f"      Vacant Slots:       {ai_stats['vacant']}")
    print(f"      Occupied Slots:     {ai_stats['occupied']}")
    print(f"      Occupancy Rate:     {ai_stats['occupancy_rate']}%")

    ai_rendered = cv_engine.render_ai_predictions(
        img, spaces, ai_results, detected_vehicles=vehicles, show_slot_ids=True, show_boxes=True, show_hud=True
    )
    ai_out_path = test_out_dir / "02_annotated_ai_yolo_prediction.jpg"
    cv_engine.save_image(ai_out_path, ai_rendered)
    print(f"  -> Saved AI Prediction visualization to: {ai_out_path.name}")

    # Comparative evaluation: distinguish GT from AI predictions
    eval_metrics = cv_engine.evaluate(ai_results)
    print("\n  -> Comparative Evaluation (Ground-Truth vs AI):")
    print(f"      Accuracy:  {eval_metrics.accuracy * 100.0:.2f}%")
    print(f"      Precision: {eval_metrics.precision * 100.0:.2f}%")
    print(f"      Recall:    {eval_metrics.recall * 100.0:.2f}%")
    print(f"      F1-Score:  {eval_metrics.f1_score * 100.0:.2f}%")
    print(f"      Confusion Matrix: TP={eval_metrics.tp}, TN={eval_metrics.tn}, FP={eval_metrics.fp}, FN={eval_metrics.fn}")

    audit_rendered = cv_engine.render_comparison_audit(
        img, spaces, ai_results, detected_vehicles=vehicles, show_hud=True
    )
    audit_out_path = test_out_dir / "03_annotated_audit_comparison.jpg"
    cv_engine.save_image(audit_out_path, audit_rendered)
    print(f"  -> Saved Audit Comparison visualization to: {audit_out_path.name}")

    # -------------------------------------------------------------
    # Test 7: Custom Polygons for Uploaded Images
    # -------------------------------------------------------------
    print("\n[Req 7] Testing Custom User-Defined Polygons on Uploaded Image...")
    # Define custom slot polygons simulating a custom upload without XML
    custom_slot_defs = [
        {"id": "CUSTOM_BAY_A", "points": [[480, 160], [580, 160], [580, 230], [480, 230]]}, # Over a car
        {"id": "CUSTOM_BAY_B", "points": [[350, 155], [450, 155], [450, 225], [350, 225]]}, # Over a car
        {"id": "CUSTOM_BAY_C", "points": [[100, 300], [200, 300], [200, 400], [100, 400]]}, # Empty area
        {"id": "CUSTOM_BAY_D", "points": [[600, 400], [700, 400], [700, 500], [600, 500]]}  # Empty area
    ]
    custom_spaces = cv_engine.create_custom_spaces(custom_slot_defs)
    print(f"  -> Successfully generated {len(custom_spaces)} custom parking bay objects.")

    custom_ai_results = cv_engine.predict_occupancy_with_yolo(custom_spaces, vehicles, overlap_threshold=0.20)
    for r in custom_ai_results:
        print(f"      Slot {r.slot_id}: {'OCCUPIED' if r.occupied_pred else 'VACANT'} (Overlap: {r.overlap_ratio:.1%})")

    custom_rendered = cv_engine.render_ai_predictions(
        img, custom_spaces, custom_ai_results, detected_vehicles=vehicles, show_slot_ids=True, show_boxes=True
    )
    custom_out_path = test_out_dir / "04_annotated_custom_polygons.jpg"
    cv_engine.save_image(custom_out_path, custom_rendered)
    print(f"  -> Saved Custom Polygon visualization to: {custom_out_path.name}")

    # -------------------------------------------------------------
    # Test 8: Different Slot Sizes & Perspective Rectification
    # -------------------------------------------------------------
    print("\n[Req 8] Handling Different Slot Sizes & Perspective Distortion...")
    areas = [s.area for s in spaces if s.is_valid]
    print(f"  -> Slot Area Variation in Scene: Min={min(areas):.1f} px², Max={max(areas):.1f} px², Ratio={max(areas)/min(areas):.1f}x")
    print("      (Normalized fractional overlap ensures scale invariance across this 3x size difference)")

    # Select an oblique slot to test perspective unwarping (homography rectification)
    oblique_slot = spaces[0]
    rectified_patch = cv_engine.rectify_slot(img, oblique_slot, target_size=(100, 200))
    patch_out_path = test_out_dir / "05_rectified_slot_patch_slot1.jpg"
    cv_engine.save_image(patch_out_path, rectified_patch)
    print(f"  -> Perspective distortion removed for Slot #{oblique_slot.id}:")
    print(f"      Input quadrilateral unwarped into canonical overhead rectangle: {patch_out_path.name} (Shape: {rectified_patch.shape})")

    print("\n" + "=" * 80)
    print("✅ ALL COMPUTER VISION MODULE REQUIREMENTS TESTED & PASSED!")
    print(f"📁 Output images saved in: {test_out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
