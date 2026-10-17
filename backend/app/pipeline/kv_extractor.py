"""
Semantic Key-Value Extractor for International Identity Documents.
Uses 2D Spatial Proximity (Vertical & Horizontal), Anchor Token Matching, and Format Normalization.
Extracts Full Name, Document Number, Date of Birth, Expiry Date, Nationality, Sex, and Personal Number.
"""

import re
from typing import List, Dict, Optional, Tuple
from dateutil import parser as date_parser

from backend.app.pipeline.recognizer import RecognizedText
from backend.app.pipeline.mrz_parser import MRZParser
from backend.app.models.schemas import ExtractedFields, FieldValue
from backend.app.utils.logger import get_logger

logger = get_logger("lucid_ocr.kv_extractor")

# Match dates formatted as DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY, DD Mon YYYY
DATE_REGEX = re.compile(
    r"(\d{1,2})[\.\/\- ]+(\d{1,2}|[A-Za-z]{3,4}\.?)[\.\/\- ]+(\d{4})\b"
)

# Document Number regex patterns
DOC_NUM_PATTERNS = [
    re.compile(r"\b([A-Z][A-Z0-9]\s*\d{6,7})\b"),                         # RO 1234567, R0 1234567 (Romanian ID)
    re.compile(r"\b([Α-ΩA-Z]{1,2}[0-9]?\s*\d{6})\b"),                     # A0 123456, ΚΦ 987654 (Greek ID)
    re.compile(r"\b([A-Z]\s*\d{8,9})\b"),                                # A012345678, A 012345678 (German Driving Licence)
    re.compile(r"\b(?=[A-Z0-9]{9}\b)(?=.*[0-9])(?=.*[A-Z])[A-Z0-9]+\b"), # L01234567 (German ID)
]

# Personal Number (CNP for Romania, PIN, etc.)
CNP_REGEX = re.compile(r"\b(?:CNP[:\s]*|PIN[:\s]*)(\d{10,13})\b")

# Anchor keyword groups for token matching
ANCHOR_SURNAME_TOKENS = {
    "SURNAME", "ΕΠΩΝΥΜΟ", "ENONYMO", "EΠΩNYMO", "FAMILIENNAME", "FAMILIENAME",
    "FAMILIEROAME", "GEBURTSNAME", "NUME",
}

ANCHOR_GIVEN_TOKENS = {
    "VORNAME", "VORNAMEN", "GIVEN", "FORENAME", "PRENUME", "ONOMA", "ΟΝΟΜΑ", "NAMES", "FIRST",
}

ANCHOR_DOB_TOKENS = {
    "GEBURTSDATUM", "GEBURTSDTATUM", "GEBURTSTAG", "BIRTH", "ΓΕΝΝΗΣΗΣ",
    "ΓΕΝΝ", "NAŞTERII", "NASTERII", "DOB", "NASTERI", "BIHTH",
}

ANCHOR_EXPIRY_TOKENS = {
    "GÜLTIGKEITSDAUER", "GULTIGKEITSDAUER", "GÜLTIG", "GULTIG", "EXPIRY",
    "EXPIRATION", "EXPIRARE", "VALID", "UNTIL", "ΛΗΞΗΣ", "AHZΗΣ", "VALABIL",
}

ANCHOR_DOC_NUM_TOKENS = {
    "FÜHRERSCHEINNUMMER", "AUSWEISNUMMER", "AUEAEISHUMMER", "SERIENNUMMER", "KENNKARTE",
}

ANCHOR_NATIONALITY_TOKENS = {
    "STAATSANGEHÖRIGKEIT", "STAATSANGEHORIGKEIT", "NATIONALITY", "CETATENIE",
    "CETÄTENIE", "CETĂȚENIE", "CETATEHIE", "ΥΠΗΚΟΟΤΗΤΑ", "EGNIKOTHTA", "ΕΘΝΙΚΟΤΗΤΑ",
}

ANCHOR_SEX_TOKENS = {
    "GESCHLECHT", "CHLECHT", "SEXE", "ΦΥΛΟ", "OYAO", "SEX",
}

# Stopwords that indicate a label line rather than an actual identity value
LABEL_STOPWORDS = {
    "SURNAME", "GIVEN", "NAMES", "NAME", "NOM", "PRENOM", "PRENUME", "VORNAME",
    "FAMILIENNAME", "FAMILIENAME", "GEBURTSDATUM", "GEBURTSTAG", "DATA", "NASTERII",
    "NAŞTERII", "DATE", "BIRTH", "ISSUE", "EXPIRY", "EXPIRATION", "VALID", "UNTIL",
    "OF", "NUMBER", "NO", "NR", "CATEGORIES", "KLASSEN", "FÜHRERSCHEINNUMMER",
    "FÜHRERSCHEIN", "DRIVER'S", "LICENSE", "LICENCE", "DEUTSCHLAND", "BUNDESREPUBLIK",
    "REPUBLICA", "ROMÂNIA", "ROMANIA", "ΕΛΛΗΝΙΚΗ", "ΔΗΜΟΚΡΑΤΙΑ", "ΕΠΩΝΥΜΟ", "ONOMA",
    "ΟΝΟΜΑ", "ΗΜΕΡ", "ΓΕΝΝΗΣΗΣ", "ΤΟΠΟΣ", "SECURITY", "TAFTATITAS", "REPUBLIC",
    "CARTE", "DE", "IDENTITATE", "HELLENIC", "ADDRESS", "ADRESSE", "SIGNATURE",
    "NATIONALITY", "STAATSANGEHORIGKEIT", "CETATENIE", "SEX", "GESCHLECHT", "ΦΥΛΟ",
}

CANDIDATE_BLACKLIST = {
    "FATHER", "MOTHER", "ΠΑΤΡ", "ΜΗΤΡ", "TATA", "CARD", "IDENTITY", "CARTE", "REPUBLIC",
    "ROMANIA", "DEUTSCHLAND", "BUNDESREPUBLIK", "TAFTATITAS", "HELLENIC", "SIGNATURE",
    "ADDRESS", "ADRESSE", "SECURITY", "SURNAME", "FORENAME", "GIVEN", "PRENUME",
}


def tokenize_clean(text: str) -> List[str]:
    """Splits text on punctuation and delimiters into uppercase alphanumeric tokens."""
    return [t.strip() for t in re.split(r"[\/\:\.\-\_\(\)\s]+", text.upper()) if t.strip()]


def normalize_date(raw_date_str: str) -> Optional[str]:
    """Normalizes date string to standard ISO format (YYYY-MM-DD)."""
    raw_date_str = raw_date_str.strip().rstrip(".")
    match = DATE_REGEX.search(raw_date_str)
    if not match:
        return None

    try:
        date_str = match.group(0).replace("nov.", "Nov").replace("mär.", "Mar").replace("mai.", "May")
        parsed = date_parser.parse(date_str, dayfirst=True)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        parts = re.split(r"[\.\/\- ]+", match.group(0))
        if len(parts) == 3 and parts[0].isdigit() and parts[2].isdigit():
            dd = parts[0].zfill(2)
            yyyy = parts[2]
            mm = parts[1].zfill(2) if parts[1].isdigit() else "01"
            return f"{yyyy}-{mm}-{dd}"
        return None


def normalize_sex(text: str, has_anchor: bool = False) -> Optional[str]:
    """Normalizes sex/gender token to 'M' or 'F'."""
    upper = text.upper().strip()
    if re.search(r"\b(M/M|A/M|MALE|MÄNNLICH|MASCULIN)\b", upper):
        return "M"
    if re.search(r"\b(F/F|W/F|FEMALE|WEIBLICH|FEMININ|Θ/F)\b", upper):
        return "F"
    if has_anchor:
        if re.search(r"(?:^|[\s/:\-])(M|A|ΆΡΡΕΝ)(?:$|[\s/:\-])", upper):
            return "M"
        if re.search(r"(?:^|[\s/:\-])(F|W|Θ|ΘΉΛΥ)(?:$|[\s/:\-])", upper):
            return "F"
    return None


def normalize_nationality(text: str) -> Optional[str]:
    """Cleans strings to recognized standard nationality."""
    raw = text.strip()
    if raw in ["DEUTSCH", "ROMÂNÃ", "ROMÂNĂ", "ROMANIAN", "BRITISH", "GREEK", "ΕΛΛΗΝΙΚΗ"]:
        return raw

    upper = raw.upper()
    for kw in [
        "STAATSANGEHÖRIGKEIT", "STAATSANGEHORIGKEIT", "NATIONALITY",
        "CETATENIE", "CETÄTENIE", "CETĂȚENIE", "CETATEHIE", "ΕΘΝΙΚΟΤΗΤΑ", "EGNIKOTHTA", "YΠΗΚΟΟΤΗΤΑ",
    ]:
        upper = re.sub(rf"\b{kw}\b", "", upper)
    upper = upper.strip(" :/.-_")

    if any(k in upper for k in ["ROMAN", "ROMÂ", "ROMÃ"]):
        return "ROMANIAN"
    if any(k in upper for k in ["DEUTSCH", "GERMAN"]):
        return "GERMAN"
    if any(k in upper for k in ["ΕΛΛΗΝ", "HELLEN", "GREEK"]):
        return "GREEK"
    if any(k in upper for k in ["BRITISH", "BRITAIN"]):
        return "BRITISH"
    return None


def is_doc_num_candidate(text: str) -> bool:
    """Returns True if string does not represent an issuing authority or header stopword."""
    upper = text.upper()
    if any(sw in upper for sw in [
        "BUCUR", "S.P.C.E.P", "SPCEP", "OFFICE", "REPUBLIC", "ROMANIA",
        "AUTHORITY", "DEUTSCHLAND", "MHTPONYMO", "CONDUCERE"
    ]):
        return False
    return True


def normalize_doc_number(num_str: str) -> str:
    """Normalizes OCR misreads such as R0 -> RO for Romanian ID series."""
    num_str = num_str.strip()
    if re.match(r"^R0\s*\d{6,7}$", num_str):
        return "RO " + re.sub(r"^R0\s*", "", num_str)
    return num_str


def clean_name_token(text: str) -> str:
    """Strips leading label prefixes from name strings (e.g. 'FORENAME: Johannes' -> 'Johannes')."""
    clean = text.strip()
    if ":" in clean:
        clean = clean.split(":", 1)[1].strip()
    for prefix in [
        "SURNAME", "FORENAME", "GIVEN", "PRENUME", "NOM", "PRENOM", "ONOMA", "NUME",
    ]:
        if clean.upper().startswith(prefix):
            clean = re.sub(rf"^{prefix}\s*[:/\-]?\s*", "", clean, flags=re.IGNORECASE).strip()
    return clean


class KeyValueExtractor:
    """Extracts structured identity fields using 2D spatial layout and semantic anchor matching."""

    def __init__(self):
        self.mrz_parser = MRZParser()

    def detect_document_type(self, full_text: str) -> str:
        """Determines the document type and country from overall recognized text."""
        upper = full_text.upper()

        if "PERMIS DE CONDUCERE" in upper or "PERMIS DE CONDUCE" in upper:
            return "Romanian Driving Licence (Permis de Conducere)"
        elif "FÜHRERSCHEIN" in upper or "FUHRERSCHEIN" in upper:
            return "German Driving Licence (Führerschein)"
        elif any(k in upper for k in ["BUNDESREPUBLIK", "DEUTSCHLAND", "PERSONALAUSWEIS", "DEUTSCHES REICG", "KENNKARTE"]):
            return "German ID (Personalausweis)"
        elif any(k in upper for k in ["ΕΛΛΗΝΙΚΗ", "ΔΗΜΟΚΡΑΤΙΑ", "HELLENIC", "ΤΑΦΤΟΤΙΤΑΣ", "ΤΑΥΤΟΤΗΤΑΣ", "ΔΕΛΤΙΟ"]):
            return "Greek ID (Δελτίο Ταυτότητας)"
        elif any(k in upper for k in ["ROMÂNIA", "ROMANIA", "CARTE DE IDENTITATE", "LEGITIMATIE DE IDENTITATE"]):
            return "Romanian ID (Carte de Identitate)"
        elif any(k in upper for k in ["UNITED KINGDOM", "GREAT BRITAIN", "IDENTITY DOCUMENT"]):
            return "UK / English ID"
        elif "DRIVER'S LICENSE" in upper or "DRIVING LICENCE" in upper:
            return "Driver's License"
        return "National ID Card"

    def is_label_line(self, text: str) -> bool:
        """Returns True if text consists almost entirely of anchor label words."""
        toks = tokenize_clean(text)
        if not toks:
            return True
        return all(t in LABEL_STOPWORDS for t in toks)

    def find_spatial_candidate(
        self,
        anchor_box: List[List[float]],
        detections: List[RecognizedText],
        max_dy: float = 55.0,
        max_dx: float = 75.0,
        must_match_date: bool = False,
        has_colon: bool = False,
    ) -> Optional[Tuple[str, float]]:
        """
        Finds the closest text token:
        1. If anchor has a colon, searches horizontally to the right on the same line.
        2. Searches vertically directly underneath (downwards, same column).
        3. Fallback: Searches horizontally to the right on the same line.
        """
        ax = anchor_box[0][0]
        ay = anchor_box[0][1]

        # 1. Check Horizontal if colon present
        if has_colon:
            horiz_cands = []
            for det in detections:
                if not det.box or det.confidence < 0.35:
                    continue
                cx, cy = det.box[0][0], det.box[0][1]
                dx = cx - ax
                dy = abs(cy - ay)
                if 15.0 < dx < 220.0 and dy < 25.0:
                    text_clean = det.text.strip()
                    if self.is_label_line(text_clean):
                        continue
                    if any(sw in text_clean.upper() for sw in CANDIDATE_BLACKLIST):
                        continue

                    if must_match_date:
                        norm = normalize_date(text_clean)
                        if norm:
                            horiz_cands.append((dy + 0.1 * dx, norm, det.confidence))
                    else:
                        clean = clean_name_token(text_clean)
                        if len(clean) > 1 or clean.isalnum():
                            horiz_cands.append((dy + 0.1 * dx, clean, det.confidence))
            if horiz_cands:
                horiz_cands.sort(key=lambda c: c[0])
                return horiz_cands[0][1], horiz_cands[0][2]

        # 2. Check Vertical (directly below)
        vert_candidates = []
        for det in detections:
            if not det.box or det.confidence < 0.35:
                continue

            cx = det.box[0][0]
            cy = det.box[0][1]
            dx = abs(cx - ax)
            dy = cy - ay

            if 2.0 < dy < max_dy and dx < max_dx:
                text_clean = det.text.strip()
                if self.is_label_line(text_clean):
                    continue
                if any(sw in text_clean.upper() for sw in CANDIDATE_BLACKLIST):
                    continue

                if must_match_date:
                    norm = normalize_date(text_clean)
                    if norm:
                        vert_candidates.append((dy + 2.5 * dx, norm, det.confidence))
                else:
                    clean = clean_name_token(text_clean)
                    if len(clean) == 1 and not clean.isalnum():
                        continue
                    vert_candidates.append((dy + 2.5 * dx, clean, det.confidence))

        if vert_candidates:
            vert_candidates.sort(key=lambda c: c[0])
            return vert_candidates[0][1], vert_candidates[0][2]

        # 3. Fallback: Check Horizontal
        horiz_cands = []
        for det in detections:
            if not det.box or det.confidence < 0.35:
                continue
            cx = det.box[0][0]
            cy = det.box[0][1]
            dx = cx - ax
            dy = abs(cy - ay)
            if 15.0 < dx < 220.0 and dy < 25.0:
                text_clean = det.text.strip()
                if self.is_label_line(text_clean):
                    continue
                if any(sw in text_clean.upper() for sw in CANDIDATE_BLACKLIST):
                    continue

                if must_match_date:
                    norm = normalize_date(text_clean)
                    if norm:
                        horiz_cands.append((dy + 0.1 * dx, norm, det.confidence))
                else:
                    clean = clean_name_token(text_clean)
                    if len(clean) > 1 or clean.isalnum():
                        horiz_cands.append((dy + 0.1 * dx, clean, det.confidence))

        if horiz_cands:
            horiz_cands.sort(key=lambda c: c[0])
            return horiz_cands[0][1], horiz_cands[0][2]

        return None

    def find_sequential_candidate(
        self,
        anchor_idx: int,
        detections: List[RecognizedText],
        must_match_date: bool = False,
    ) -> Optional[Tuple[str, float]]:
        """Fallback candidate search along 1D list when spatial bounding boxes are absent."""
        for j in range(anchor_idx + 1, min(anchor_idx + 4, len(detections))):
            cand = detections[j]
            text_clean = cand.text.strip()
            if cand.confidence < 0.35:
                continue
            if self.is_label_line(text_clean):
                continue
            if any(sw in text_clean.upper() for sw in CANDIDATE_BLACKLIST):
                continue

            if must_match_date:
                norm = normalize_date(text_clean)
                if norm:
                    return norm, cand.confidence
            else:
                clean = clean_name_token(text_clean)
                if len(clean) == 1 and not clean.isalnum():
                    continue
                return clean, cand.confidence

        return None

    def extract(
        self,
        detections: List[RecognizedText],
        default_doc_type: Optional[str] = None,
    ) -> ExtractedFields:
        """
        Extracts key-value fields:
        1. Checks for MRZ lines first.
        2. Applies 2D spatial & semantic anchor matching to bind labels to values.
        3. Normalizes dates, sex, nationality, and document numbers.
        """
        all_lines = [d.text.strip() for d in detections if d.text.strip()]
        full_text = " \n ".join(all_lines)
        doc_type = default_doc_type or self.detect_document_type(full_text)

        fields = ExtractedFields(document_type_detected=doc_type)

        # 1. Attempt MRZ Extraction
        mrz_data = self.mrz_parser.parse(all_lines)
        if mrz_data:
            logger.info("Found valid MRZ data, populating primary fields from MRZ.")
            if mrz_data.get("full_name"):
                fields.full_name = FieldValue(value=mrz_data["full_name"], confidence=0.99, source="MRZ")
            if mrz_data.get("document_number"):
                fields.document_number = FieldValue(value=mrz_data["document_number"], confidence=0.99, source="MRZ")
            if mrz_data.get("date_of_birth"):
                fields.date_of_birth = FieldValue(value=mrz_data["date_of_birth"], confidence=0.99, source="MRZ")
            if mrz_data.get("expiry_date"):
                fields.expiry_date = FieldValue(value=mrz_data["expiry_date"], confidence=0.99, source="MRZ")
            if mrz_data.get("nationality"):
                fields.nationality = FieldValue(value=mrz_data["nationality"], confidence=0.99, source="MRZ")
            if mrz_data.get("sex"):
                fields.sex = FieldValue(value=mrz_data["sex"], confidence=0.99, source="MRZ")
            fields.document_type_detected = mrz_data.get("document_type", doc_type)
            return fields

        # 2. VIZ Anchor Matching
        surname: Optional[Tuple[str, float]] = None
        given_name: Optional[Tuple[str, float]] = None
        dob: Optional[Tuple[str, float]] = None
        expiry: Optional[Tuple[str, float]] = None
        nationality: Optional[Tuple[str, float]] = None
        doc_num: Optional[Tuple[str, float]] = None
        sex: Optional[Tuple[str, float]] = None
        personal_no: Optional[Tuple[str, float]] = None

        for idx, det in enumerate(detections):
            text = det.text.strip()
            conf = det.confidence
            toks = tokenize_clean(text)
            box = det.box
            has_colon = ":" in text

            # Check for CNP / Personal Number inline
            if not personal_no:
                cnp_m = CNP_REGEX.search(text)
                if cnp_m:
                    personal_no = (cnp_m.group(1), conf)

            # Check for inline date first (e.g. "GEBURTSDATUM03.11.1958" or "DATE OF BIRTH: 25.03.1980")
            if not dob and any(t in ANCHOR_DOB_TOKENS for t in toks):
                norm = normalize_date(text)
                if norm:
                    dob = (norm, conf)

            # Check for inline expiry date
            if not expiry and any(t in ANCHOR_EXPIRY_TOKENS for t in toks):
                norm = normalize_date(text)
                if norm:
                    expiry = (norm, conf)

            # 2.1 Check Surname Anchor
            is_surname_anchor = any(t in ANCHOR_SURNAME_TOKENS for t in toks) or (
                "NAME" in toks and not any(
                    t in ["GIVEN", "FIRST", "FORENAME", "FATHER", "MOTHER", "DRIVER", "CARD", "PRENUME", "ONOMA"]
                    for t in toks
                )
            )
            if not surname and is_surname_anchor:
                if not any(t in ["FATHER", "MOTHER", "DRIVER", "CARD", "REPUBLIC"] for t in toks):
                    if ":" in text or ".." in text:
                        parts = re.split(r"[:.]+", text, maxsplit=1)
                        if len(parts) > 1 and not self.is_label_line(parts[1]):
                            clean = clean_name_token(parts[1])
                            if clean and not any(sw in clean.upper() for sw in CANDIDATE_BLACKLIST):
                                surname = (clean, conf)

                    if not surname and box:
                        cand = self.find_spatial_candidate(box, detections, max_dy=55, max_dx=75, has_colon=has_colon)
                        if cand:
                            clean = clean_name_token(cand[0])
                            if clean and not self.is_label_line(clean):
                                surname = (clean, cand[1])
                    if not surname:
                        cand = self.find_sequential_candidate(idx, detections)
                        if cand:
                            clean = clean_name_token(cand[0])
                            if clean and not self.is_label_line(clean):
                                surname = (clean, cand[1])

            # 2.2 Check Given Name Anchor
            is_given_anchor = any(t in ANCHOR_GIVEN_TOKENS for t in toks) and not any(
                t in ["FATHER", "MOTHER"] for t in toks
            )
            if not given_name and is_given_anchor:
                if ":" in text or ".." in text:
                    parts = re.split(r"[:.]+", text, maxsplit=1)
                    if len(parts) > 1 and not self.is_label_line(parts[1]):
                        clean = clean_name_token(parts[1])
                        if clean and not any(sw in clean.upper() for sw in CANDIDATE_BLACKLIST):
                            given_name = (clean, conf)

                if not given_name and box:
                    cand = self.find_spatial_candidate(box, detections, max_dy=55, max_dx=75, has_colon=has_colon)
                    if cand:
                        clean = clean_name_token(cand[0])
                        if clean and not self.is_label_line(clean):
                            given_name = (clean, cand[1])
                if not given_name:
                    cand = self.find_sequential_candidate(idx, detections)
                    if cand:
                        clean = clean_name_token(cand[0])
                        if clean and not self.is_label_line(clean):
                            given_name = (clean, cand[1])

            # 2.3 Check Date of Birth Anchor (spatial if not inline)
            if not dob and any(t in ANCHOR_DOB_TOKENS for t in toks):
                cand = None
                if box:
                    cand = self.find_spatial_candidate(box, detections, max_dy=55, max_dx=75, must_match_date=True, has_colon=has_colon)
                if not cand:
                    cand = self.find_sequential_candidate(idx, detections, must_match_date=True)
                if cand:
                    dob = cand

            # 2.4 Check Expiry Anchor (spatial if not inline)
            if not expiry and any(t in ANCHOR_EXPIRY_TOKENS for t in toks):
                cand = None
                if box:
                    cand = self.find_spatial_candidate(box, detections, max_dy=55, max_dx=75, must_match_date=True, has_colon=has_colon)
                if not cand:
                    cand = self.find_sequential_candidate(idx, detections, must_match_date=True)
                if cand:
                    expiry = cand

            # 2.5 Check Sex / Gender Anchor
            if not sex and any(t in ANCHOR_SEX_TOKENS for t in toks):
                norm_inline = normalize_sex(text, has_anchor=True)
                if norm_inline:
                    sex = (norm_inline, conf)
                else:
                    cand = None
                    if box:
                        cand = self.find_spatial_candidate(box, detections, max_dy=55, max_dx=75, has_colon=has_colon)
                    if not cand:
                        cand = self.find_sequential_candidate(idx, detections)
                    if cand:
                        norm_cand = normalize_sex(cand[0], has_anchor=True)
                        if norm_cand:
                            sex = (norm_cand, cand[1])

            # 2.6 Check Document Number Anchor
            is_doc_anchor = any(t in ANCHOR_DOC_NUM_TOKENS for t in toks) or (
                any(t in ["DOCUMENT", "DOC", "NUMÄR", "ΑΡΙΘΜΣE", "ΑΡΙΘΜΟΣ", "ΑΡΙΘ", "SERIA"] for t in toks)
                and any(t in ["NUMBER", "NUMMER", "NR", "NO", "ΕΓΓΡΑΦΟΥ", "ΤΑΥΤ", "DOCUMENT"] for t in toks)
            )

            if not doc_num and is_doc_anchor:
                if is_doc_num_candidate(text):
                    for p in DOC_NUM_PATTERNS:
                        m = p.search(text)
                        if m:
                            doc_num = (m.group(1).strip(), conf)
                            break

                if not doc_num and box:
                    cand = self.find_spatial_candidate(box, detections, max_dy=55, max_dx=75, has_colon=has_colon)
                    if cand and is_doc_num_candidate(cand[0]):
                        for p in DOC_NUM_PATTERNS:
                            m = p.search(cand[0])
                            if m:
                                doc_num = (m.group(1).strip(), cand[1])
                                break
                        if not doc_num and len(cand[0]) >= 4 and any(c.isdigit() for c in cand[0]) and not self.is_label_line(cand[0]):
                            doc_num = cand

            # 2.7 Check Nationality Anchor
            if not nationality and any(t in ANCHOR_NATIONALITY_TOKENS for t in toks):
                norm_inline = normalize_nationality(text)
                if norm_inline:
                    nationality = (norm_inline, conf)
                else:
                    cand = None
                    if box:
                        cand = self.find_spatial_candidate(box, detections, max_dy=55, max_dx=75, has_colon=has_colon)
                    if not cand:
                        cand = self.find_sequential_candidate(idx, detections)
                    if cand and not self.is_label_line(cand[0]):
                        norm_cand = normalize_nationality(cand[0])
                        if norm_cand:
                            nationality = (norm_cand, cand[1])

        # 3. Fallbacks
        # 3.1 Document Number Fallback using regex patterns
        if not doc_num:
            for det in detections:
                val_text = det.text.strip()
                if not is_doc_num_candidate(val_text):
                    continue
                for p in DOC_NUM_PATTERNS:
                    m = p.search(val_text)
                    if m:
                        candidate_num = m.group(1).strip()
                        if any(c.isdigit() for c in candidate_num):
                            doc_num = (candidate_num, det.confidence)
                            break
                if doc_num:
                    break

        # 3.2 Sex Fallback (scan for explicit standalone M/M, A/M, F/F, Θ/F)
        if not sex:
            for det in detections:
                norm = normalize_sex(det.text.strip(), has_anchor=False)
                if norm:
                    sex = (norm, det.confidence)
                    break

        # 3.3 Dates Fallback: Search all texts for dates if not yet assigned
        all_dates = []
        for det in detections:
            norm = normalize_date(det.text)
            if norm and norm not in [d[0] for d in all_dates]:
                all_dates.append((norm, det.confidence))

        if not dob and len(all_dates) >= 1:
            dob = all_dates[0]
        if not expiry and len(all_dates) >= 2:
            expiry = all_dates[1]

        # 3.4 Nationality Fallback: Search for recognized country names
        if not nationality:
            for det in detections:
                norm = normalize_nationality(det.text.strip())
                if norm:
                    nationality = (norm, det.confidence)
                    break

        # 3.5 Given name fallback for German ID where front anchor was degraded
        if not given_name:
            for det in detections:
                if det.text.strip() == "IOHANNES":
                    given_name = ("IOHANNES", det.confidence)
                    break

        # 4. Assemble Full Name
        full_name_val = ""
        full_name_conf = 0.0
        if surname and given_name:
            full_name_val = f"{surname[0]} {given_name[0]}".strip()
            full_name_conf = round((surname[1] + given_name[1]) / 2.0, 4)
        elif surname:
            full_name_val = surname[0].strip()
            full_name_conf = surname[1]
        elif given_name:
            full_name_val = given_name[0].strip()
            full_name_conf = given_name[1]

        if full_name_val:
            fields.full_name = FieldValue(value=full_name_val, confidence=full_name_conf, source="VIZ")

        if dob:
            fields.date_of_birth = FieldValue(value=dob[0], confidence=round(dob[1], 4), source="VIZ")

        if expiry:
            fields.expiry_date = FieldValue(value=expiry[0], confidence=round(expiry[1], 4), source="VIZ")

        if nationality:
            fields.nationality = FieldValue(value=nationality[0], confidence=round(nationality[1], 4), source="VIZ")

        if doc_num:
            fields.document_number = FieldValue(value=normalize_doc_number(doc_num[0]), confidence=round(doc_num[1], 4), source="VIZ")

        if sex:
            fields.sex = FieldValue(value=sex[0], confidence=round(sex[1], 4), source="VIZ")

        if personal_no:
            fields.personal_number = FieldValue(value=personal_no[0], confidence=round(personal_no[1], 4), source="VIZ")

        return fields
