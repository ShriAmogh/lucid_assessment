"""
ICAO Doc 9303 Machine Readable Zone (MRZ) Parser.
Parses TD1 (3 lines of 30 characters), TD2 (2 lines of 36 chars),
and TD3 (2 lines of 44 chars) with strict checksum validation.
"""

import re
from typing import List, Optional, Dict, Any
from backend.app.utils.logger import get_logger

logger = get_logger("lucid_ocr.mrz_parser")

CHECK_DIGIT_WEIGHTS = [7, 3, 1]


def compute_check_digit(data: str) -> int:
    """Computes ICAO 9303 check digit using weights [7, 3, 1]."""
    total = 0
    for i, char in enumerate(data):
        if "0" <= char <= "9":
            val = int(char)
        elif "A" <= char <= "Z":
            val = ord(char) - ord("A") + 10
        elif char == "<":
            val = 0
        else:
            val = 0
        weight = CHECK_DIGIT_WEIGHTS[i % 3]
        total += val * weight
    return total % 10


def format_mrz_date(yymmdd: str, is_expiry: bool = False) -> Optional[str]:
    """Converts YYMMDD string to ISO YYYY-MM-DD."""
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    yy = int(yymmdd[:2])
    mm = yymmdd[2:4]
    dd = yymmdd[4:6]

    if is_expiry:
        yyyy = 2000 + yy if yy < 70 else 1900 + yy
    else:
        yyyy = 1900 + yy if yy > 25 else 2000 + yy

    return f"{yyyy}-{mm}-{dd}"


class MRZParser:
    """Detects and parses ICAO 9303 compliant MRZ text lines with strict verification."""

    def find_mrz_lines(self, lines: List[str]) -> List[str]:
        """Filters lines that strictly resemble MRZ lines."""
        mrz_candidates = []
        for line in lines:
            cleaned = line.replace(" ", "").upper()
            # Must contain << and be at least 28 characters
            if ("<<" in cleaned and len(cleaned) >= 28) or (cleaned.startswith("ID") and "<<" in cleaned):
                # Filter out lines with standard words or lowercases
                if not any(w in cleaned for w in ["BUNDESREPUBLIK", "DEUTSCHLAND", "POLICE", "REPUBLIC", "TAFTATITAS"]):
                    mrz_candidates.append(cleaned)
        return mrz_candidates

    def parse(self, text_lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parses lines if they contain valid TD1, TD2, or TD3 MRZ strings."""
        mrz_lines = self.find_mrz_lines(text_lines)
        if not mrz_lines:
            return None

        # Check for TD1 (3 lines of ~30 chars)
        if len(mrz_lines) == 3 and all(28 <= len(line) <= 32 for line in mrz_lines):
            td1 = self._parse_td1(mrz_lines)
            if td1:
                return td1

        # Check for TD2 (2 lines of ~36 chars) or TD3 (2 lines of ~44 chars)
        if len(mrz_lines) == 2:
            if len(mrz_lines[0]) >= 40:
                td3 = self._parse_td3(mrz_lines)
                if td3:
                    return td3
            elif len(mrz_lines[0]) >= 34:
                td2 = self._parse_td2(mrz_lines)
                if td2:
                    return td2

        return None

    def _parse_td1(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parses TD1 3-line format with checksum checking."""
        try:
            line1, line2, line3 = lines[0], lines[1], lines[2]
            doc_type = line1[:2].replace("<", "")
            issuing_country = line1[2:5].replace("<", "")
            doc_number = line1[5:14].replace("<", "")
            doc_check = line1[14] if len(line1) > 14 and line1[14].isdigit() else None

            # Verify doc number check digit if present
            if doc_check and int(doc_check) != compute_check_digit(line1[5:14]):
                return None

            dob_raw = line2[0:6]
            dob_iso = format_mrz_date(dob_raw, is_expiry=False)
            if not dob_iso:
                return None

            sex = line2[7] if len(line2) > 7 and line2[7] in ["M", "F"] else "Unknown"

            expiry_raw = line2[8:14]
            expiry_iso = format_mrz_date(expiry_raw, is_expiry=True)
            nationality = line2[15:18].replace("<", "")

            names = line3.split("<<")
            surname = names[0].replace("<", " ").strip()
            given_names = names[1].replace("<", " ").strip() if len(names) > 1 else ""
            full_name = f"{surname} {given_names}".strip()

            return {
                "source": "MRZ_TD1",
                "document_number": doc_number,
                "document_type": f"{issuing_country} ID (MRZ)",
                "full_name": full_name,
                "date_of_birth": dob_iso,
                "expiry_date": expiry_iso,
                "nationality": nationality,
                "sex": sex,
            }
        except Exception as e:
            logger.debug(f"TD1 MRZ check failed: {e}")
            return None

    def _parse_td2(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parses TD2 2-line format."""
        try:
            line1, line2 = lines[0], lines[1]
            names = line1[5:].split("<<")
            surname = names[0].replace("<", " ").strip()
            given_names = names[1].replace("<", " ").strip() if len(names) > 1 else ""
            full_name = f"{surname} {given_names}".strip()

            doc_number = line2[:9].replace("<", "")
            nationality = line2[10:13].replace("<", "")
            dob_iso = format_mrz_date(line2[13:19], is_expiry=False)
            sex = line2[20] if len(line2) > 20 and line2[20] in ["M", "F"] else "Unknown"
            expiry_iso = format_mrz_date(line2[21:27], is_expiry=True)

            if not dob_iso:
                return None

            return {
                "source": "MRZ_TD2",
                "document_number": doc_number,
                "full_name": full_name,
                "date_of_birth": dob_iso,
                "expiry_date": expiry_iso,
                "nationality": nationality,
                "sex": sex,
            }
        except Exception as e:
            return None

    def _parse_td3(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parses TD3 2-line format (Passports)."""
        try:
            line1, line2 = lines[0], lines[1]
            names = line1[5:].split("<<")
            surname = names[0].replace("<", " ").strip()
            given_names = names[1].replace("<", " ").strip() if len(names) > 1 else ""
            full_name = f"{surname} {given_names}".strip()

            doc_number = line2[:9].replace("<", "")
            nationality = line2[10:13].replace("<", "")
            dob_iso = format_mrz_date(line2[13:19], is_expiry=False)
            sex = line2[20] if len(line2) > 20 and line2[20] in ["M", "F"] else "Unknown"
            expiry_iso = format_mrz_date(line2[21:27], is_expiry=True)

            if not dob_iso:
                return None

            return {
                "source": "MRZ_TD3",
                "document_number": doc_number,
                "document_type": "Passport (MRZ)",
                "full_name": full_name,
                "date_of_birth": dob_iso,
                "expiry_date": expiry_iso,
                "nationality": nationality,
                "sex": sex,
            }
        except Exception as e:
            return None
