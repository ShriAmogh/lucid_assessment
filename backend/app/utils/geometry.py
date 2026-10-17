"""
Geometry utilities for bounding box transformations, unclipping, and reading order sorting.
"""

from typing import List, Union
import numpy as np
import cv2
import pyclipper
from shapely.geometry import Polygon


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points in clockwise order starting from top-left:
    [top-left, top-right, bottom-right, bottom-left].
    """
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference of coordinates: top-right has smallest diff (x - y), bottom-left has largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def unclip_polygon(box: np.ndarray, unclip_ratio: float = 1.6) -> np.ndarray:
    """
    Expands a bounding polygon outwards using the Vatti clipping algorithm via pyclipper.
    D' = (Area * unclip_ratio) / Perimeter
    """
    poly = Polygon(box)
    area = poly.area
    perimeter = poly.length

    if perimeter <= 0 or area <= 0:
        return box

    distance = (area * unclip_ratio) / perimeter
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(box.astype(int).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = pco.Execute(distance)

    if not expanded or len(expanded) == 0:
        return box

    expanded_pts = np.array(expanded[0], dtype=np.float32)
    # Fit minimum area rectangle to expanded polygon
    rect = cv2.minAreaRect(expanded_pts)
    box_new = cv2.boxPoints(rect)
    return order_points_clockwise(box_new)


def sort_boxes_reading_order(boxes: List[np.ndarray], y_threshold: float = 12.0) -> List[np.ndarray]:
    """
    Sorts boxes in natural reading order:
    Top to bottom, and left to right within lines (using y_threshold for line grouping).
    """
    if not boxes:
        return []

    # Get center or top-left coordinates: (y_min, x_min)
    def box_key(b):
        pts = np.array(b)
        y_top = np.min(pts[:, 1])
        x_left = np.min(pts[:, 0])
        return y_top, x_left

    sorted_by_y = sorted(boxes, key=box_key)

    lines: List[List[np.ndarray]] = []
    current_line: List[np.ndarray] = []
    current_line_y = -1.0

    for box in sorted_by_y:
        pts = np.array(box)
        y_top = np.min(pts[:, 1])

        if current_line_y < 0 or abs(y_top - current_line_y) < y_threshold:
            current_line.append(box)
            if current_line_y < 0:
                current_line_y = y_top
        else:
            # Sort previous line by x_left
            current_line.sort(key=lambda b: np.min(np.array(b)[:, 0]))
            lines.append(current_line)
            current_line = [box]
            current_line_y = y_top

    if current_line:
        current_line.sort(key=lambda b: np.min(np.array(b)[:, 0]))
        lines.append(current_line)

    result = []
    for line in lines:
        result.extend(line)

    return result
