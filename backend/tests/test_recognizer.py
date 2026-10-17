"""
Unit tests for SVTR TextRecognizer and CTC decoding.
Tests character dictionary loading, CTC greedy decoding, and batch inference.
"""

from pathlib import Path
import pytest
import numpy as np
import cv2

from backend.app.config import settings
from backend.app.pipeline.preprocessor import ImagePreprocessor
from backend.app.pipeline.recognizer import TextRecognizer, RecognizedText


@pytest.fixture
def latin_recognizer():
    return TextRecognizer(
        model_path=str(settings.models.rec_latin_model_path),
        dict_path=str(settings.models.dict_latin_path),
        num_threads=settings.models.intra_op_num_threads,
    )


@pytest.fixture
def greek_recognizer():
    return TextRecognizer(
        model_path=str(settings.models.rec_greek_model_path),
        dict_path=str(settings.models.dict_greek_path),
        num_threads=settings.models.intra_op_num_threads,
    )


@pytest.fixture
def preprocessor():
    return ImagePreprocessor()


def test_dictionary_loading(latin_recognizer, greek_recognizer):
    """Character dictionary must be loaded with blank token and characters."""
    assert len(latin_recognizer.characters) > 100
    assert len(greek_recognizer.characters) > 100
    assert latin_recognizer.characters[0] == ""  # blank token at index 0


def test_ctc_greedy_decoding(latin_recognizer):
    """CTC greedy decoder should collapse duplicates and omit blank tokens."""
    # Synthetic logits: batch_size=1, time_steps=6, num_classes=504
    logits = np.zeros((1, 6, 504), dtype=np.float32)

    # Let index 1 be 'a', index 2 be 'b', index 0 be blank
    # Sequence: 1, 1, 0, 2, 2, 0 -> should decode to "ab"
    logits[0, 0, 1] = 10.0
    logits[0, 1, 1] = 10.0
    logits[0, 2, 0] = 10.0
    logits[0, 3, 2] = 10.0
    logits[0, 4, 2] = 10.0
    logits[0, 5, 0] = 10.0

    texts, confs = latin_recognizer.decode_ctc(logits)
    assert len(texts) == 1
    char_1 = latin_recognizer.characters[1]
    char_2 = latin_recognizer.characters[2]
    assert texts[0] == f"{char_1}{char_2}"
    assert confs[0] > 0.9


def test_latin_recognition_inference(latin_recognizer, preprocessor):
    """Recognize text on a synthetic crop with text 'TEST123'."""
    crop = np.ones((48, 200, 3), dtype=np.uint8) * 255
    cv2.putText(crop, "TEST123", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    results = latin_recognizer.recognize_crops([crop], preprocessor=preprocessor)
    assert len(results) == 1
    assert isinstance(results[0], RecognizedText)
    assert "TEST" in results[0].text or "123" in results[0].text
    assert results[0].confidence > 0.5
