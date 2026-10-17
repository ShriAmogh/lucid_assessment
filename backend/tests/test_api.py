"""
Integration tests for FastAPI endpoints.
Tests /api/health, /api/samples, /api/samples/{name}, and /api/extract.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import IMAGES_DIR

client = TestClient(app)


def test_health_endpoint():
    """GET /api/health should return status ok and model information."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data
    assert data["models_loaded"]["detection"] is True
    assert data["models_loaded"]["latin_rec"] is True
    assert data["models_loaded"]["greek_rec"] is True


def test_samples_endpoint():
    """GET /api/samples should return list of available test images."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    assert len(samples) >= 4
    filenames = [s["filename"] for s in samples]
    assert "german_id_1.png" in filenames
    assert "greek_id_1.png" in filenames
    assert "roman_id_1.png" in filenames


def test_sample_image_fetch():
    """GET /api/samples/{filename} should return image binary bytes."""
    response = client.get("/api/samples/german_id_1.png")
    assert response.status_code == 200
    assert response.headers["content-type"] in ["image/png", "image/jpeg"]
    assert len(response.content) > 1000


def test_extract_endpoint_with_sample():
    """POST /api/extract with an uploaded image should return structured extraction JSON."""
    img_path = IMAGES_DIR / "german_id_1.png"
    assert img_path.exists()

    with open(img_path, "rb") as f:
        response = client.post(
            "/api/extract",
            files={"file": ("german_id_1.png", f, "image/png")},
            data={"language_hint": "german"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "PAPADOPOULOS" in data["fields"]["full_name"]["value"]
    assert data["latency"]["total_ms"] > 0
    assert len(data["raw_detections"]) > 5
