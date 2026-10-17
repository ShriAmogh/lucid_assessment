"""
Unit tests for geometry utilities.
Tests clockwise ordering, polygon unclipping, and reading order sorting.
"""

import numpy as np
import pytest

from backend.app.utils.geometry import (
    order_points_clockwise,
    unclip_polygon,
    sort_boxes_reading_order,
)


def test_order_points_clockwise():
    """Unordered 4 corners should be ordered as TL, TR, BR, BL."""
    pts = np.array([[100, 20], [20, 20], [20, 60], [100, 60]], dtype=np.float32)
    ordered = order_points_clockwise(pts)
    assert ordered.shape == (4, 2)
    # TL should be approx (20, 20)
    assert np.allclose(ordered[0], [20, 20], atol=1e-2)
    # TR should be approx (100, 20)
    assert np.allclose(ordered[1], [100, 20], atol=1e-2)
    # BR should be approx (100, 60)
    assert np.allclose(ordered[2], [100, 60], atol=1e-2)
    # BL should be approx (20, 60)
    assert np.allclose(ordered[3], [20, 60], atol=1e-2)


def test_unclip_polygon():
    """Unclipping should expand the polygon outwards."""
    box = np.array([[20, 20], [100, 20], [100, 60], [20, 60]], dtype=np.float32)
    expanded = unclip_polygon(box, unclip_ratio=1.5)
    assert expanded is not None
    assert len(expanded) == 4
    # Expanded box should have larger width and height
    w_orig = 80
    h_orig = 40
    w_new = np.linalg.norm(expanded[0] - expanded[1])
    h_new = np.linalg.norm(expanded[0] - expanded[3])
    assert w_new > w_orig
    assert h_new > h_orig


def test_sort_boxes_reading_order():
    """Boxes should be sorted top-to-bottom, left-to-right."""
    # Box 1: bottom-left
    b1 = np.array([[10, 200], [50, 200], [50, 220], [10, 220]], dtype=np.float32)
    # Box 2: top-left
    b2 = np.array([[10, 20], [50, 20], [50, 40], [10, 40]], dtype=np.float32)
    # Box 3: top-right
    b3 = np.array([[100, 20], [150, 20], [150, 40], [100, 40]], dtype=np.float32)

    boxes = [b1, b3, b2]
    sorted_boxes = sort_boxes_reading_order(boxes)

    # First should be b2 (top-left), second b3 (top-right), third b1 (bottom-left)
    assert np.allclose(sorted_boxes[0], b2)
    assert np.allclose(sorted_boxes[1], b3)
    assert np.allclose(sorted_boxes[2], b1)
