"""
Integration test for OCREngine.
Tests end-to-end processing and latency profiling on actual sample images.
"""

from pathlib import Path
import pytest

from backend.app.config import settings, IMAGES_DIR
from backend.app.pipeline.ocr_engine import OCREngine
from backend.app.models.schemas import ExtractionResponse


@pytest.fixture(scope="module")
def ocr_engine():
    return OCREngine()


def test_engine_german_image(ocr_engine):
    """End-to-end test on german_id_1.png."""
    img_path = IMAGES_DIR / "german_id_1.png"
    assert img_path.exists()

    with open(img_path, "rb") as f:
        image_bytes = f.read()

    response = ocr_engine.process_image(image_bytes, language_hint="german")
    assert isinstance(response, ExtractionResponse)
    assert response.success is True
    assert len(response.raw_detections) > 5
    assert response.latency.total_ms > 0
    assert response.fields.full_name.value is not None
    assert "PAPADOPOULOS" in response.fields.full_name.value


def test_engine_greek_image(ocr_engine):
    """End-to-end test on greek_id_1.png."""
    img_path = IMAGES_DIR / "greek_id_1.png"
    assert img_path.exists()

    with open(img_path, "rb") as f:
        image_bytes = f.read()

    response = ocr_engine.process_image(image_bytes, language_hint="greek")
    assert isinstance(response, ExtractionResponse)
    assert response.success is True
    assert len(response.raw_detections) > 5
    assert "ΠΑΠΑΔΟΠΟΥΛΟΣ" in response.fields.full_name.value


def test_engine_romanian_image(ocr_engine):
    """End-to-end test on roman_id_1.png."""
    img_path = IMAGES_DIR / "roman_id_1.png"
    assert img_path.exists()

    with open(img_path, "rb") as f:
        image_bytes = f.read()

    response = ocr_engine.process_image(image_bytes, language_hint="romanian")
    assert isinstance(response, ExtractionResponse)
    assert response.success is True
    assert len(response.raw_detections) > 5
    assert "PAPADOPOULOS" in response.fields.full_name.value
    assert response.fields.expiry_date.value == "2030-03-28"
