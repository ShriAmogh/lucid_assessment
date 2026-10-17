"""
Unit tests for DBNet TextDetector.
Tests model initialization, forward pass, and bounding box output formatting.
"""

from pathlib import Path
import pytest
import numpy as np
import cv2

from backend.app.config import settings
from backend.app.pipeline.preprocessor import ImagePreprocessor
from backend.app.pipeline.detector import TextDetector, DetectionBox


@pytest.fixture
def detector():
    model_path = settings.models.det_model_path
    assert model_path.exists(), f"Detection model must exist at {model_path}"
    return TextDetector(
        model_path=str(model_path),
        num_threads=settings.models.intra_op_num_threads,
        thresh=settings.models.det_db_thresh,
        box_thresh=settings.models.det_db_box_thresh,
        unclip_ratio=settings.models.det_db_unclip_ratio,
    )


@pytest.fixture
def preprocessor():
    return ImagePreprocessor()


def test_detector_inference_on_synthetic_text(detector, preprocessor):
    """Detector should find text bounding boxes on an image with rendered text."""
    # Create white canvas and render black text
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "GERMAN IDENTITY CARD", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "MUSTERMANN ERIKA", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    boxes = detector.detect(img, preprocessor)
    assert isinstance(boxes, list)
    assert len(boxes) >= 2, f"Expected at least 2 text regions, got {len(boxes)}"
    
    for box in boxes:
        assert isinstance(box, DetectionBox)
        assert len(box.points) == 4
        assert 0.0 <= box.score <= 1.0
        # Ensure points are within image dimensions
        pts = np.array(box.points)
        assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= 600)
        assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] <= 400)
