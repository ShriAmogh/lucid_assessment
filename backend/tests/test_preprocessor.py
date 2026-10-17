"""
Unit tests for ImagePreprocessor.
Following Test-First development strategy.
"""

import io
import pytest
import numpy as np
from PIL import Image

from backend.app.pipeline.preprocessor import ImagePreprocessor


@pytest.fixture
def preprocessor():
    return ImagePreprocessor(det_limit_side_len=960)


@pytest.fixture
def sample_rgba_bytes():
    """Generates an RGBA test image in PNG format with semi-transparent pixels."""
    im = Image.new("RGBA", (400, 300), (255, 0, 0, 128))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_rgb_image():
    """Generates an RGB numpy array (H=300, W=400, C=3)."""
    return np.zeros((300, 400, 3), dtype=np.uint8)


def test_load_from_bytes_rgba(preprocessor, sample_rgba_bytes):
    """RGBA images must be blended onto white background and returned as 3-channel RGB/BGR."""
    img = preprocessor.load_from_bytes(sample_rgba_bytes)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[0] == 300
    assert img.shape[1] == 400


def test_preprocess_for_detection_shapes(preprocessor, sample_rgb_image):
    """Detection tensor must have dimensions [1, 3, H, W] where H and W are multiples of 32."""
    tensor, ratio_h, ratio_w = preprocessor.preprocess_for_detection(sample_rgb_image)
    assert tensor.ndim == 4
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3
    
    H, W = tensor.shape[2], tensor.shape[3]
    assert H % 32 == 0, f"Height {H} must be divisible by 32"
    assert W % 32 == 0, f"Width {W} must be divisible by 32"
    assert tensor.dtype == np.float32
    assert ratio_h > 0 and ratio_w > 0


def test_crop_box_perspective(preprocessor):
    """Oriented 4-point bounding box must be extracted as a rectangular crop."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Define a 40x80 box inside
    box = np.array([[20, 20], [100, 20], [100, 60], [20, 60]], dtype=np.float32)
    crop = preprocessor.crop_box(img, box)
    assert crop.ndim == 3
    assert crop.shape[2] == 3
    assert crop.shape[0] == pytest.approx(40, abs=2)
    assert crop.shape[1] == pytest.approx(80, abs=2)


def test_preprocess_for_recognition(preprocessor):
    """Recognition tensor must have fixed height 48 and proportional width."""
    crop = np.zeros((40, 120, 3), dtype=np.uint8)
    tensor = preprocessor.preprocess_for_recognition(crop, target_height=48)
    assert tensor.ndim == 4
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3
    assert tensor.shape[2] == 48
    assert tensor.shape[3] > 0
    assert tensor.dtype == np.float32
