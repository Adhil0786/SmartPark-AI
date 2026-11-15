"""
SmartPark AI - PKLot Dataset Explorer & Loader
Provides robust indexing, structural XML inspection, polygon & occupancy extraction,
missing annotation handling, CSV export, and sample visualization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import glob
import os
import random
import cv2
import numpy as np
import pandas as pd

from config import RAW_PKLOT_DIR, LOT_METADATA, REPORTS_DIR, get_active_dataset_dir
from utils.file_utils import (
    load_image,
    save_image,
    find_matching_xml,
    discover_pklot_lots,
    discover_lot_weathers,
    discover_weather_dates,
    list_lot_images
)
from computer_vision.xml_parser import PKLotXMLParser, ParkingSpace
from computer_vision.visualizer import ParkingVisualizer


@dataclass
class ImageXMLPair:
    """Represents an image and its matching XML annotation pair."""
    image_path: Path
    xml_path: Optional[Path]
    lot: str
    weather: str
    date: str
    image_name: str
    has_image: bool
    has_xml: bool


class PKLotDataset:
    """
    Complete dataset loader, inspector, and evaluator for PKLot dataset.
    Follows zero-assumption parsing, preserves original annotations,
    and handles missing or corrupted annotations gracefully.
    """

    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = Path(raw_dir) if raw_dir is not None else get_active_dataset_dir()
        self.visualizer = ParkingVisualizer()

    # ==========================================
    # 1. FIND PARKING IMAGES AND MATCHING XMLs
    # ==========================================
    def find_all_pairs(self, lot_filter: Optional[str] = None) -> List[ImageXMLPair]:
        """
        Scans raw PKLot directory, finds all parking images and matching XML files.
        Detects orphaned images (missing XML) or orphaned XML files (missing image).
        """
        search_dir = self.raw_dir if not lot_filter else (self.raw_dir / lot_filter)
        if not search_dir.exists():
            return []

        # Find all jpg and xml files
        jpg_paths = sorted(list(search_dir.glob("**/*.jpg")))
        xml_dict: Dict[str, Path] = {
            os.path.splitext(p)[0]: p for p in search_dir.glob("**/*.xml")
        }

        pairs: List[ImageXMLPair] = []
        for img_path in jpg_paths:
            base_key = os.path.splitext(img_path)[0]
            matched_xml = xml_dict.get(base_key)

            # Extract lot, weather, date from path hierarchy
            rel_parts = img_path.relative_to(self.raw_dir).parts
            lot = rel_parts[0] if len(rel_parts) > 0 else "Unknown"
            weather = rel_parts[1] if len(rel_parts) > 1 else "Unknown"
            date = rel_parts[2] if len(rel_parts) > 2 else "Unknown"

            pairs.append(
                ImageXMLPair(
                    image_path=img_path,
                    xml_path=matched_xml,
                    lot=lot,
                    weather=weather,
                    date=date,
                    image_name=img_path.name,
                    has_image=img_path.exists(),
                    has_xml=(matched_xml is not None and matched_xml.exists())
                )
            )

        return pairs

    # ==========================================
    # 2. INSPECT ACTUAL XML STRUCTURE
    # ==========================================
    def inspect_xml(self, xml_path: Union[str, Path]) -> Dict:
        """
        Inspects the actual XML structure before parsing.
        Reveals root tags, child tags, attributes, point naming, and anomalies.
        """
        return PKLotXMLParser.inspect_xml_structure(xml_path)

    # ==========================================
    # 3 & 4. EXTRACT SLOTS & HANDLE ANOMALIES
    # ==========================================
    def load_pair(self, image_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[PKLotXMLParser]]:
        """
        Loads an image and its matching PKLot XML annotation.
        Returns (image_bgr, xml_parser). Handles missing files cleanly.
        """
        img_p = Path(image_path)
        img = load_image(img_p)
        if img is None:
            return None, None

        xml_p = find_matching_xml(img_p)
        if xml_p is None or not xml_p.exists():
            return img, None

        try:
            parser = PKLotXMLParser(xml_p)
        except Exception:
            parser = None

        return img, parser

    # ==========================================
    # 5. CREATE CSV OF IMAGE, SLOT ID & OCCUPANCY
    # ==========================================
    def create_occupancy_dataframe(self, pairs: List[ImageXMLPair]) -> pd.DataFrame:
        """
        Extracts all slots from given image/xml pairs into a structured pandas DataFrame.
        Contains image metadata, slot ID, occupancy status, polygon coordinates, and validity.
        """
        rows = []
        for pair in pairs:
            if not pair.has_xml or pair.xml_path is None:
                # Record image with missing annotation
                rows.append({
                    "image_path": str(pair.image_path),
                    "image_name": pair.image_name,
                    "lot": pair.lot,
                    "weather": pair.weather,
                    "date": pair.date,
                    "slot_id": "N/A",
                    "occupancy_status": "Missing_Annotation",
                    "occupied_raw": "None",
                    "is_valid": False,
                    "num_polygon_points": 0,
                    "polygon_coords": "[]",
                    "error_reason": "No matching XML file found"
                })
                continue

            parser = PKLotXMLParser(pair.xml_path)
            spaces = parser.get_spaces()

            if not spaces:
                rows.append({
                    "image_path": str(pair.image_path),
                    "image_name": pair.image_name,
                    "lot": pair.lot,
                    "weather": pair.weather,
                    "date": pair.date,
                    "slot_id": "N/A",
                    "occupancy_status": "Empty_XML",
                    "occupied_raw": "None",
                    "is_valid": False,
                    "num_polygon_points": 0,
                    "polygon_coords": "[]",
                    "error_reason": "XML contains 0 space tags"
                })
                continue

            for s in spaces:
                rows.append({
                    "image_path": str(pair.image_path),
                    "image_name": pair.image_name,
                    "lot": pair.lot,
                    "weather": pair.weather,
                    "date": pair.date,
                    "slot_id": s.id,
                    "occupancy_status": s.status,  # "Occupied", "Vacant", "Unannotated", "Invalid"
                    "occupied_raw": str(s.occupied_gt) if s.occupied_gt is not None else "Missing",
                    "is_valid": s.is_valid,
                    "num_polygon_points": len(s.points) if s.points is not None else 0,
                    "polygon_coords": str(s.points.tolist()) if s.points is not None else "[]",
                    "error_reason": s.error_reason or "None"
                })

        return pd.DataFrame(rows)

    def export_occupancy_csv(self, pairs: List[ImageXMLPair], output_path: Union[str, Path]) -> Path:
        """
        Creates and saves a CSV containing image, slot ID and occupancy status.
        """
        df = self.create_occupancy_dataframe(pairs)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        return out_p

    # ==========================================
    # 6. VISUALIZE ANNOTATED SAMPLE IMAGES
    # ==========================================
    def visualize_sample(self,
                         image_path: Union[str, Path],
                         xml_path: Optional[Union[str, Path]] = None,
                         output_path: Optional[Union[str, Path]] = None,
                         show_hud: bool = True) -> Optional[np.ndarray]:
        """
        Visualizes parking-slot polygons directly from ground truth annotations.
        Green = Vacant, Red = Occupied, Amber = Unannotated/Missing attribute.
        Saves the output image if output_path is provided.
        """
        img_p = Path(image_path)
        img = load_image(img_p)
        if img is None:
            return None

        if xml_path is None:
            xml_p = find_matching_xml(img_p)
        else:
            xml_p = Path(xml_path)

        spaces: List[ParkingSpace] = []
        if xml_p is not None and xml_p.exists():
            parser = PKLotXMLParser(xml_p)
            spaces = parser.get_spaces()

        rendered = self.visualizer.draw_parking_lot(
            base_image=img,
            spaces=spaces,
            classification_results=None,
            detected_vehicles=None,
            show_slot_ids=True,
            show_boxes=False,
            show_hud=show_hud
        )

        if output_path:
            save_image(Path(output_path), rendered)

        return rendered

    # ==========================================
    # 7. COMPREHENSIVE TEST SUITE (10–20 IMAGES)
    # ==========================================
    def test_dataset_loader(self,
                            test_image_paths: Optional[List[Path]] = None,
                            output_dir: Optional[Path] = None,
                            n_test_samples: int = 15) -> Dict:
        """
        Executes an end-to-end test of the loader on 10-20 diverse images.
        Inspects XML, extracts slots, handles edge cases, creates CSV, and visualizes samples.
        """
        out_dir = Path(output_dir) if output_dir else (REPORTS_DIR / "loader_tests")
        vis_dir = out_dir / "visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Select diverse test images if not provided explicitly
        selected_pairs: List[ImageXMLPair] = []
        if test_image_paths:
            for p in test_image_paths:
                p = Path(p).resolve()
                xml_p = find_matching_xml(p)
                raw_res = self.raw_dir.resolve()
                rel_parts = p.relative_to(raw_res).parts if raw_res in p.parents else ("Custom", "Custom", "Custom")
                selected_pairs.append(
                    ImageXMLPair(
                        image_path=p,
                        xml_path=xml_p,
                        lot=rel_parts[0] if len(rel_parts) > 0 else "Custom",
                        weather=rel_parts[1] if len(rel_parts) > 1 else "Custom",
                        date=rel_parts[2] if len(rel_parts) > 2 else "Custom",
                        image_name=p.name,
                        has_image=p.exists(),
                        has_xml=(xml_p is not None and xml_p.exists())
                    )
                )
        else:
            all_pairs = self.find_all_pairs()
            # Pick a representative sample across lots and weathers, plus known edge cases
            lots = ["UFPR04", "UFPR05", "PUCPR"]
            weathers = ["Cloudy", "Sunny", "Rainy"]

            # Specifically include known edge cases
            known_edge_files = [
                # PUCPR rainy files with unannotated spaces
                "dataset/raw/PKLot/PKLot/PUCPR/Rainy/2012-10-11/2012-10-11_11_06_40.jpg",
                "dataset/raw/PKLot/PKLot/PUCPR/Rainy/2012-10-11/2012-10-11_07_56_34.jpg",
                # UFPR04 with <Point> capitalization
                "dataset/raw/PKLot/PKLot/UFPR04/Cloudy/2012-12-29/2012-12-29_14_30_37.jpg",
                # Orphan JPG without XML
                "dataset/raw/PKLot/PKLot/PUCPR/Sunny/2012-11-06/2012-11-06_18_48_46.jpg"
            ]

            edge_pairs = []
            for ef in known_edge_files:
                ep = Path(ef).resolve()
                if ep.exists():
                    exml = find_matching_xml(ep)
                    parts = ep.relative_to(self.raw_dir.resolve()).parts
                    edge_pairs.append(
                        ImageXMLPair(
                            image_path=ep,
                            xml_path=exml,
                            lot=parts[0] if len(parts) > 0 else "Unknown",
                            weather=parts[1] if len(parts) > 1 else "Unknown",
                            date=parts[2] if len(parts) > 2 else "Unknown",
                            image_name=ep.name,
                            has_image=True,
                            has_xml=(exml is not None and exml.exists())
                        )
                    )

            # Sample remaining from each lot/weather combination
            remaining_pairs = [p for p in all_pairs if str(p.image_path) not in known_edge_files]
            random.seed(42)
            stratified_pairs = []
            for lot in lots:
                for w in weathers:
                    matching = [p for p in remaining_pairs if p.lot == lot and p.weather == w]
                    if len(matching) >= 2:
                        stratified_pairs.extend(random.sample(matching, 2))
                    elif matching:
                        stratified_pairs.extend(matching)

            # Combine edge cases + stratified sample to reach ~16-18 images
            combined = edge_pairs + stratified_pairs
            selected_pairs = combined[:16]

        # Run tests on selected pairs
        results = {
            "total_images_tested": len(selected_pairs),
            "successful_images": 0,
            "images_with_missing_xml": 0,
            "total_slots_extracted": 0,
            "occupied_slots_count": 0,
            "vacant_slots_count": 0,
            "unannotated_slots_count": 0,
            "invalid_slots_count": 0,
            "xml_inspections": [],
            "visualized_images": [],
            "errors_and_warnings": []
        }

        # 1. Structural inspection & slot parsing per image
        for pair in selected_pairs:
            img_res = {
                "image": pair.image_name,
                "lot": pair.lot,
                "weather": pair.weather,
                "has_xml": pair.has_xml,
                "slots_count": 0,
                "status_breakdown": {}
            }

            if not pair.has_xml:
                results["images_with_missing_xml"] += 1
                results["errors_and_warnings"].append(
                    f"Warning: Image '{pair.image_name}' ({pair.lot}/{pair.weather}) has no matching XML."
                )
                results["xml_inspections"].append(img_res)
                continue

            # Inspect structure before parsing
            struct_info = self.inspect_xml(pair.xml_path)
            img_res["structural_info"] = struct_info

            # Parse XML
            parser = PKLotXMLParser(pair.xml_path)
            spaces = parser.get_spaces()
            summary = parser.get_summary()

            img_res["slots_count"] = summary["total_slots"]
            img_res["status_breakdown"] = {
                "Occupied": summary["occupied_slots"],
                "Vacant": summary["vacant_slots"],
                "Unannotated": summary["unannotated_slots"]
            }

            results["successful_images"] += 1
            results["total_slots_extracted"] += summary["total_slots"]
            results["occupied_slots_count"] += summary["occupied_slots"]
            results["vacant_slots_count"] += summary["vacant_slots"]
            results["unannotated_slots_count"] += summary["unannotated_slots"]

            for w in parser.warnings:
                results["errors_and_warnings"].append(f"[{pair.image_name}] Warning: {w}")
            for e in parser.errors:
                results["errors_and_warnings"].append(f"[{pair.image_name}] Error: {e}")

            # Visualize sample image
            vis_file = vis_dir / f"vis_{pair.lot}_{pair.weather}_{pair.image_name}"
            self.visualize_sample(pair.image_path, pair.xml_path, vis_file)
            results["visualized_images"].append(str(vis_file))

            results["xml_inspections"].append(img_res)

        # 2. Export Occupancy CSV
        csv_file = out_dir / "tested_images_occupancy.csv"
        self.export_occupancy_csv(selected_pairs, csv_file)
        results["csv_exported_path"] = str(csv_file)

        return results

    # Backward compatibility helpers for Streamlit UI
    def get_lots(self) -> List[str]:
        return discover_pklot_lots()

    def get_lot_info(self, lot: str) -> Dict:
        return LOT_METADATA.get(lot, {"name": lot, "spaces": 0, "camera_angle": "Unknown", "description": f"Parking lot {lot}"})

    def get_weathers(self, lot: str) -> List[str]:
        return discover_lot_weathers(lot)

    def get_dates(self, lot: str, weather: str) -> List[str]:
        return discover_weather_dates(lot, weather)

    def get_images(self, lot: str, weather: str, date: str) -> List[Path]:
        return list_lot_images(lot, weather, date)

    def sample_random(self, lot: Optional[str] = None, weather: Optional[str] = None):
        lots = [lot] if lot else self.get_lots()
        if not lots:
            return None
        chosen_lot = random.choice(lots)
        weathers = [weather] if weather else self.get_weathers(chosen_lot)
        if not weathers:
            return None
        chosen_weather = random.choice(weathers)
        dates = self.get_dates(chosen_lot, chosen_weather)
        if not dates:
            return None
        chosen_date = random.choice(dates)
        images = self.get_images(chosen_lot, chosen_weather, chosen_date)
        if not images:
            return None
        chosen_img_path = random.choice(images)
        img, parser = self.load_pair(chosen_img_path)
        return chosen_img_path, img, parser
