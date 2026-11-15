"""
SmartPark AI - Geometry and Spatial Computations
Vectorized and OpenCV-accelerated polygon operations, perspective rectification,
and scale-invariant overlap metrics for parking slot analysis.
"""

from typing import List, Tuple, Union
import numpy as np
import cv2


def polygon_area(pts: np.ndarray) -> float:
    """Calculate the area of a 2D polygon."""
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(pts.astype(np.int32)))


def polygon_centroid(pts: np.ndarray) -> Tuple[int, int]:
    """Compute the integer (x, y) centroid of a 2D polygon."""
    M = cv2.moments(pts.astype(np.int32))
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    # Fallback to mean of coordinates
    mean_pt = np.mean(pts, axis=0)
    return int(round(mean_pt[0])), int(round(mean_pt[1]))


def polygon_bounding_box(pts: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (min_x, min_y, max_x, max_y) bounding box of a polygon."""
    min_x = int(np.min(pts[:, 0]))
    min_y = int(np.min(pts[:, 1]))
    max_x = int(np.max(pts[:, 0]))
    max_y = int(np.max(pts[:, 1]))
    return min_x, min_y, max_x, max_y


def boxes_intersect(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> bool:
    """Quick AABB intersection test between two [x1, y1, x2, y2] boxes."""
    return not (boxA[2] <= boxB[0] or boxA[0] >= boxB[2] or
                boxA[3] <= boxB[1] or boxA[1] >= boxB[3])


def box_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0
    return float(interArea / unionArea)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Sort 4 coordinates of a quadrilateral in clockwise order:
    [top-left, top-right, bottom-right, bottom-left].
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-left has smallest sum (x + y), bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest difference (y - x), bottom-left has largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def rectify_perspective(image: np.ndarray,
                        quad_pts: np.ndarray,
                        target_size: Tuple[int, int] = (64, 128)) -> np.ndarray:
    """
    Removes perspective distortion by mapping an oblique 4-point slot polygon
    to a canonical rectangular top-down patch using cv2.warpPerspective.
    target_size: (width, height) in pixels.
    """
    if len(quad_pts) != 4:
        # Fallback to bounding box crop if not quadrilateral
        min_x, min_y, max_x, max_y = polygon_bounding_box(quad_pts)
        h, w = image.shape[:2]
        crop = image[max(0, min_y):min(h, max_y), max(0, min_x):min(w, max_x)]
        if crop.size == 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        return cv2.resize(crop, target_size)

    rect = order_quad_points(quad_pts)
    dst_w, dst_h = target_size
    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
    return warped


def point_in_polygon(point: Tuple[float, float], poly_pts: np.ndarray) -> bool:
    """Check if a 2D point (x, y) is inside the polygon using cv2.pointPolygonTest."""
    if len(poly_pts) < 3:
        return False
    res = cv2.pointPolygonTest(poly_pts.astype(np.int32), (float(point[0]), float(point[1])), measureDist=False)
    return res >= 0


def polygon_box_overlap(poly_pts: np.ndarray,
                        box: Tuple[float, float, float, float],
                        img_shape: Tuple[int, int] = None) -> float:
    """
    Calculate the fractional overlap of a polygon covered by a bounding box:
    Overlap = (Area of Polygon ∩ Box) / Area of Polygon.
    Scale-invariant across variable slot sizes (large foreground bays vs small background bays).
    """
    if len(poly_pts) < 3:
        return 0.0

    poly_x1, poly_y1, poly_x2, poly_y2 = polygon_bounding_box(poly_pts)
    bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    # Fast AABB check: if bounding boxes don't intersect, overlap is 0
    if not boxes_intersect((poly_x1, poly_y1, poly_x2, poly_y2), (bx1, by1, bx2, by2)):
        return 0.0

    # Localized ROI around the polygon
    roi_w = max(1, poly_x2 - poly_x1 + 1)
    roi_h = max(1, poly_y2 - poly_y1 + 1)

    local_poly = poly_pts.copy()
    local_poly[:, 0] -= poly_x1
    local_poly[:, 1] -= poly_y1

    # Local mask for the polygon
    poly_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [local_poly.astype(np.int32)], 255)
    poly_area = float(np.sum(poly_mask > 0))

    if poly_area <= 0:
        return 0.0

    # Bounding box in local ROI coordinates
    local_bx1 = max(0, bx1 - poly_x1)
    local_by1 = max(0, by1 - poly_y1)
    local_bx2 = min(roi_w, bx2 - poly_x1)
    local_by2 = min(roi_h, by2 - poly_y1)

    if local_bx2 <= local_bx1 or local_by2 <= local_by1:
        return 0.0

    overlap_pixels = float(np.sum(poly_mask[local_by1:local_by2, local_bx1:local_bx2] > 0))
    return overlap_pixels / poly_area


def perspective_aware_overlap(poly_pts: np.ndarray,
                              box: Tuple[float, float, float, float]) -> float:
    """
    Computes a perspective-robust occupancy signal:
    Combines fractional area overlap with centroid containment.
    Accounts for camera tilt, vehicle overhang, and non-rectangular bays.
    """
    area_ratio = polygon_box_overlap(poly_pts, box)
    if area_ratio == 0:
        return 0.0

    # Check vehicle box center relative to polygon
    bx1, by1, bx2, by2 = box
    box_center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
    slot_center = polygon_centroid(poly_pts)

    center_in_poly = point_in_polygon(box_center, poly_pts)
    poly_center_in_box = (bx1 <= slot_center[0] <= bx2) and (by1 <= slot_center[1] <= by2)

    # Boost confidence when centers coincide, mitigating oblique overhang errors
    if center_in_poly or poly_center_in_box:
        return min(1.0, area_ratio * 1.25)
    return area_ratio


def rotated_rect_to_points(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    """Convert rotated rectangle parameters to 4 corner points [[x, y], ...]."""
    rect = ((float(cx), float(cy)), (float(w), float(h)), float(angle_deg))
    box = cv2.boxPoints(rect)
    return np.int32(box)
