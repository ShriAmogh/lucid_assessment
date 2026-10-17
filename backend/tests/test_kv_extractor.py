"""
Unit tests for Key-Value Extractor and MRZ parser.
Tests semantic field parsing across German, Greek, Romanian, and English ID cards.
"""

import pytest
from backend.app.pipeline.recognizer import RecognizedText
from backend.app.pipeline.kv_extractor import KeyValueExtractor
from backend.app.models.schemas import ExtractedFields


@pytest.fixture
def extractor():
    return KeyValueExtractor()


def test_extract_german_fields(extractor):
    """Should correctly extract full name, DOB, nationality from German ID text tokens."""
    texts = [
        RecognizedText(text="BUNDESREPUBLIK DEUTSCHLAND", confidence=0.99),
        RecognizedText(text="PERSONALAUSWEIS", confidence=0.99),
        RecognizedText(text="FAMILIENNAME", confidence=0.99),
        RecognizedText(text="PAPADOPOULOS", confidence=0.99),
        RecognizedText(text="VORNAME(N)", confidence=0.99),
        RecognizedText(text="IOHANNES", confidence=0.99),
        RecognizedText(text="STAATSANGEHÖRIGKEIT", confidence=0.99),
        RecognizedText(text="DEUTSCH", confidence=0.99),
        RecognizedText(text="GEBURTSDATUM", confidence=0.99),
        RecognizedText(text="29.03.1990", confidence=0.99),
    ]

    result = extractor.extract(texts, default_doc_type=None)
    assert isinstance(result, ExtractedFields)
    assert result.document_type_detected == "German ID (Personalausweis)"
    assert "PAPADOPOULOS" in result.full_name.value
    assert "IOHANNES" in result.full_name.value
    assert result.date_of_birth.value == "1990-03-29"
    assert result.nationality.value == "DEUTSCH"


def test_extract_greek_fields(extractor):
    """Should correctly extract full name, DOB from Greek ID text tokens."""
    texts = [
        RecognizedText(text="ΕΛΛΗΝΙΚΗ ΔΗΜΟΚΡΑΤΙΑ", confidence=0.98),
        RecognizedText(text="ΕΠΩΝΥΜΟ", confidence=0.99),
        RecognizedText(text="ΠΑΠΑΔΟΠΟΥΛΟΣ", confidence=0.99),
        RecognizedText(text="ONOMA", confidence=0.99),
        RecognizedText(text="ΙΩΑΝΝΗΣ", confidence=0.98),
        RecognizedText(text="ΗΜΕΡ. ΓΕΝΝΗΣΗΣ", confidence=0.99),
        RecognizedText(text="29.03.1990", confidence=0.99),
    ]

    result = extractor.extract(texts, default_doc_type="Greek ID")
    assert isinstance(result, ExtractedFields)
    assert "ΠΑΠΑΔΟΠΟΥΛΟΣ" in result.full_name.value
    assert "ΙΩΑΝΝΗΣ" in result.full_name.value
    assert result.date_of_birth.value == "1990-03-29"


def test_extract_romanian_fields(extractor):
    """Should correctly extract full name, DOB, and Expiry from Romanian ID text tokens."""
    texts = [
        RecognizedText(text="REPUBLICA ROMÂNIA", confidence=0.98),
        RecognizedText(text="CARTE DE IDENTITATE", confidence=0.98),
        RecognizedText(text="NUME", confidence=0.99),
        RecognizedText(text="PAPADOPOULOS", confidence=0.99),
        RecognizedText(text="PRENUME", confidence=0.99),
        RecognizedText(text="IOHANNES", confidence=0.99),
        RecognizedText(text="CETÄTENIE", confidence=0.99),
        RecognizedText(text="ROMÂNÃ", confidence=0.99),
        RecognizedText(text="DATA NAŞTERII", confidence=0.98),
        RecognizedText(text="29.03.1990", confidence=0.99),
        RecognizedText(text="VALABIL PÁN LA", confidence=0.98),
        RecognizedText(text="29.03.2030", confidence=0.99),
    ]

    result = extractor.extract(texts, default_doc_type="Romanian ID")
    assert isinstance(result, ExtractedFields)
    assert "PAPADOPOULOS" in result.full_name.value
    assert "IOHANNES" in result.full_name.value
    assert result.date_of_birth.value == "1990-03-29"
    assert result.expiry_date.value == "2030-03-29"
    assert result.nationality.value == "ROMÂNÃ"


def test_extract_inline_delimiter_format(extractor):
    """Should correctly extract fields when anchor and value are on same line with colons."""
    texts = [
        RecognizedText(text="UNITED KINGDOM IDENTITY DOCUMENT", confidence=0.99),
        RecognizedText(text="SURNAME: Papadopoulos", confidence=0.99),
        RecognizedText(text="FORENAME: Johannes", confidence=0.99),
        RecognizedText(text="DATE OF BIRTH: 25.03.1980", confidence=0.99),
    ]

    result = extractor.extract(texts, default_doc_type="English ID")
    assert isinstance(result, ExtractedFields)
    assert "Papadopoulos" in result.full_name.value
    assert "Johannes" in result.full_name.value
    assert result.date_of_birth.value == "1980-03-25"
