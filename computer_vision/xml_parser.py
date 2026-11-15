"""
SmartPark AI - PKLot XML Annotation Parser
Extracts parking slots, polygon coordinates, rotated bounding rectangles,
and real ground-truth occupancy labels from PKLot XML annotations.
Includes structural inspection, handles missing/invalid annotations, and supports tag variations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import numpy as np
import cv2

from utils.geometry import (
    polygon_area,
    polygon_centroid,
    polygon_bounding_box,
    rotated_rect_to_points
)


@dataclass
class ParkingSpace:
    """Represents a single annotated parking slot."""
    id: str
    occupied_gt: Optional[bool]  # True: Occupied, False: Vacant, None: Unannotated/Missing
    points: np.ndarray           # Shape (N, 2), dtype int32
    center: Tuple[int, int]
    area: float
    bbox: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    rotated_rect: Optional[Dict[str, float]] = None
    status: str = "Vacant"       # "Occupied", "Vacant", "Unannotated", "Invalid"
    is_valid: bool = True
    error_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "slot_id": self.id,
            "occupied_gt": self.occupied_gt,
            "status": self.status,
            "center_x": self.center[0],
            "center_y": self.center[1],
            "area_px": round(self.area, 2),
            "bbox": list(self.bbox),
            "polygon_points": self.points.tolist(),
            "is_valid": self.is_valid,
            "error_reason": self.error_reason or "None"
        }


class PKLotXMLParser:
    """
    Parser for PKLot dataset XML annotation files.
    Inspects XML structure before parsing and robustly handles anomalies.
    """

    def __init__(self, xml_source: Union[str, Path, bytes]):
        self.xml_source = xml_source
        self.parking_id: str = "Unknown"
        self.spaces: List[ParkingSpace] = []
        self.structural_info: Dict = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self._parse()

    @staticmethod
    def inspect_xml_structure(xml_source: Union[str, Path, bytes]) -> Dict:
        """
        Inspect actual XML structure before parsing.
        Returns tag names, attributes, point counts, and anomalies without modifying data.
        """
        info = {
            "root_tag": None,
            "root_attributes": {},
            "total_spaces_found": 0,
            "space_tags": set(),
            "space_attributes_found": set(),
            "has_rotated_rect": 0,
            "has_contour": 0,
            "point_tag_names": set(),
            "missing_occupied_count": 0,
            "missing_id_count": 0,
            "missing_contour_count": 0,
            "anomalies": []
        }

        try:
            if isinstance(xml_source, (str, Path)) and Path(xml_source).exists():
                tree = ET.parse(str(xml_source))
                root = tree.getroot()
            elif isinstance(xml_source, bytes):
                root = ET.fromstring(xml_source.decode("utf-8", errors="replace"))
            elif isinstance(xml_source, str):
                root = ET.fromstring(xml_source)
            else:
                info["anomalies"].append(f"Invalid XML source type: {type(xml_source)}")
                return info
        except Exception as e:
            info["anomalies"].append(f"XML Parse Failure: {e}")
            return info

        info["root_tag"] = root.tag
        info["root_attributes"] = root.attrib

        # Inspect children of root
        for child in root:
            info["space_tags"].add(child.tag)
            if child.tag.lower() == "space":
                info["total_spaces_found"] += 1
                for k in child.attrib:
                    info["space_attributes_found"].add(k)

                if "id" not in child.attrib:
                    info["missing_id_count"] += 1
                if "occupied" not in child.attrib:
                    info["missing_occupied_count"] += 1

                # Check sub-elements
                c_node = None
                r_node = None
                for sub in child:
                    if sub.tag.lower() == "contour":
                        c_node = sub
                        info["has_contour"] += 1
                    elif sub.tag.lower() == "rotatedrect":
                        r_node = sub
                        info["has_rotated_rect"] += 1

                if c_node is None:
                    info["missing_contour_count"] += 1
                else:
                    for pt in c_node:
                        info["point_tag_names"].add(pt.tag)

        # Convert sets to sorted lists for serialization
        info["space_tags"] = sorted(list(info["space_tags"]))
        info["space_attributes_found"] = sorted(list(info["space_attributes_found"]))
        info["point_tag_names"] = sorted(list(info["point_tag_names"]))

        if info["missing_occupied_count"] > 0:
            info["anomalies"].append(f"{info['missing_occupied_count']} space(s) missing 'occupied' attribute")
        if info["missing_id_count"] > 0:
            info["anomalies"].append(f"{info['missing_id_count']} space(s) missing 'id' attribute")
        if info["missing_contour_count"] > 0:
            info["anomalies"].append(f"{info['missing_contour_count']} space(s) missing contour element")
        if any(t == "Point" for t in info["point_tag_names"]):
            info["anomalies"].append("Detected capitalized '<Point>' tag casing in contour")

        return info

    def _parse(self) -> None:
        """Parse XML tree and construct ParkingSpace objects with full error handling."""
        # 1. Structural inspection first
        self.structural_info = self.inspect_xml_structure(self.xml_source)

        try:
            if isinstance(self.xml_source, (str, Path)) and Path(self.xml_source).exists():
                tree = ET.parse(str(self.xml_source))
                root = tree.getroot()
            elif isinstance(self.xml_source, bytes):
                root = ET.fromstring(self.xml_source.decode("utf-8", errors="replace"))
            elif isinstance(self.xml_source, str):
                root = ET.fromstring(self.xml_source)
            else:
                self.errors.append(f"Invalid XML source: {type(self.xml_source)}")
                return
        except Exception as e:
            self.errors.append(f"Failed to parse XML: {e}")
            return

        self.parking_id = root.attrib.get("id", "Unknown")

        # Find space elements (case-insensitive)
        space_nodes = [node for node in root if node.tag.lower() == "space"]
        if not space_nodes:
            self.warnings.append("No <space> elements found in XML.")
            return

        for idx, space_node in enumerate(space_nodes, start=1):
            slot_id = space_node.attrib.get("id")
            if not slot_id:
                slot_id = f"auto_{idx}"
                self.warnings.append(f"Space #{idx} has no 'id' attribute; assigned '{slot_id}'")

            occupied_raw = space_node.attrib.get("occupied")
            is_valid = True
            error_reason = None

            # Handle occupied attribute
            if occupied_raw is None:
                is_occupied = None
                status = "Unannotated"
                self.warnings.append(f"Slot #{slot_id}: Missing 'occupied' attribute (marked Unannotated)")
            elif str(occupied_raw).strip() == "1":
                is_occupied = True
                status = "Occupied"
            elif str(occupied_raw).strip() == "0":
                is_occupied = False
                status = "Vacant"
            else:
                is_occupied = None
                status = "Unannotated"
                self.warnings.append(f"Slot #{slot_id}: Unknown occupied value '{occupied_raw}'")

            # Extract rotatedRect if present
            rect_info = None
            rot_points = None
            rect_node = None
            for child in space_node:
                if child.tag.lower() == "rotatedrect":
                    rect_node = child
                    break

            if rect_node is not None:
                c_node = None
                s_node = None
                a_node = None
                for sub in rect_node:
                    stag = sub.tag.lower()
                    if stag == "center":
                        c_node = sub
                    elif stag == "size":
                        s_node = sub
                    elif stag == "angle":
                        a_node = sub

                if c_node is not None and s_node is not None and a_node is not None:
                    try:
                        cx = float(c_node.attrib.get("x", 0))
                        cy = float(c_node.attrib.get("y", 0))
                        w = float(s_node.attrib.get("w", 0))
                        h = float(s_node.attrib.get("h", 0))
                        angle = float(a_node.attrib.get("d", 0))
                        rect_info = {"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle}
                        rot_points = rotated_rect_to_points(cx, cy, w, h, angle)
                    except (ValueError, TypeError) as e:
                        self.warnings.append(f"Slot #{slot_id}: Malformed rotatedRect parameters: {e}")

            # Extract contour points (case-insensitive: handles <point> and <Point>)
            pts_list = []
            contour_node = None
            for child in space_node:
                if child.tag.lower() == "contour":
                    contour_node = child
                    break

            if contour_node is not None:
                for pt_node in contour_node:
                    if pt_node.tag.lower() == "point":
                        try:
                            px = int(round(float(pt_node.attrib.get("x", 0))))
                            py = int(round(float(pt_node.attrib.get("y", 0))))
                            pts_list.append([px, py])
                        except (ValueError, TypeError):
                            self.warnings.append(f"Slot #{slot_id}: Non-numeric point coordinates encountered")

            # Determine polygon points (prefer original contour, fallback to rotatedRect)
            if len(pts_list) >= 3:
                poly_pts = np.array(pts_list, dtype=np.int32)
            elif rot_points is not None:
                poly_pts = rot_points
                self.warnings.append(f"Slot #{slot_id}: Contour missing or < 3 points; used rotatedRect fallback")
            else:
                is_valid = False
                error_reason = "No valid contour or rotatedRect found"
                poly_pts = np.zeros((0, 2), dtype=np.int32)
                self.warnings.append(f"Slot #{slot_id}: {error_reason}")

            area = polygon_area(poly_pts) if is_valid else 0.0
            center = polygon_centroid(poly_pts) if is_valid else (0, 0)
            bbox = polygon_bounding_box(poly_pts) if is_valid else (0, 0, 0, 0)

            if is_valid and area <= 0:
                is_valid = False
                error_reason = "Degenerate polygon with zero area"

            self.spaces.append(
                ParkingSpace(
                    id=str(slot_id),
                    occupied_gt=is_occupied,
                    points=poly_pts,
                    center=center,
                    area=area,
                    bbox=bbox,
                    rotated_rect=rect_info,
                    status=status,
                    is_valid=is_valid,
                    error_reason=error_reason
                )
            )

    def get_spaces(self) -> List[ParkingSpace]:
        """Return all parsed parking spaces."""
        return self.spaces

    def get_valid_spaces(self) -> List[ParkingSpace]:
        """Return only spaces with valid polygons."""
        return [s for s in self.spaces if s.is_valid]

    def get_summary(self) -> Dict[str, Union[int, float, str]]:
        """Return occupancy metrics summary."""
        total = len(self.spaces)
        valid_count = sum(1 for s in self.spaces if s.is_valid)
        occupied = sum(1 for s in self.spaces if s.occupied_gt is True)
        vacant = sum(1 for s in self.spaces if s.occupied_gt is False)
        unannotated = sum(1 for s in self.spaces if s.occupied_gt is None)
        rate = (occupied / (occupied + vacant) * 100.0) if (occupied + vacant) > 0 else 0.0

        return {
            "parking_id": self.parking_id,
            "total_slots": total,
            "valid_slots": valid_count,
            "occupied_slots": occupied,
            "vacant_slots": vacant,
            "unannotated_slots": unannotated,
            "occupancy_rate": round(rate, 2),
            "warnings_count": len(self.warnings),
            "errors_count": len(self.errors)
        }

    @staticmethod
    def parse_file(path: Union[str, Path]) -> "PKLotXMLParser":
        """Convenience factory method to parse an XML file."""
        return PKLotXMLParser(path)
