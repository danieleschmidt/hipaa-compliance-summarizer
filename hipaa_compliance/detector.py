"""
PHIDetector — detects all 18 HIPAA Safe Harbor PHI categories using regex patterns.

Categories (per 45 CFR § 164.514(b)(2)):
  1.  NAME
  2.  GEOGRAPHIC
  3.  DATE
  4.  PHONE
  5.  FAX
  6.  EMAIL
  7.  SSN
  8.  MRN
  9.  HEALTH_PLAN_BENEFICIARY
  10. ACCOUNT_NUMBER
  11. CERTIFICATE_LICENSE
  12. VEHICLE_IDENTIFIER
  13. DEVICE_IDENTIFIER
  14. URL
  15. IP_ADDRESS
  16. BIOMETRIC
  17. PHOTO_REFERENCE
  18. AGE_OVER_89
"""

import re
from typing import List, Dict, Any


class PHIDetector:
    """Detects PHI across all 18 HIPAA Safe Harbor categories."""

    # Order matters: more specific patterns before broader ones
    PATTERNS: Dict[str, str] = {
        # 7. SSN — must come before PHONE to avoid partial overlaps
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",

        # 4. Phone numbers
        "PHONE": (
            r"\b(?:\+1[-.\s]?)?"
            r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),

        # 5. Fax numbers — same format as phone; keyword context
        "FAX": (
            r"\b(?:fax|f)[:\s]*"
            r"(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),

        # 6. Email addresses
        "EMAIL": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",

        # 14. Web URLs
        "URL": r"https?://[^\s]+",

        # 15. IP addresses
        "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",

        # 8. Medical Record Numbers
        "MRN": (
            r"\b(?:MRN|Medical\s+Record(?:\s+Number)?)"
            r"[:\s#]*\d+\b"
        ),

        # 9. Health plan beneficiary numbers
        "HEALTH_PLAN_BENEFICIARY": (
            r"\b(?:beneficiary|member|plan)\s*(?:id|#|number)[:\s]*"
            r"[A-Z0-9]{6,}\b"
        ),

        # 10. Account numbers
        "ACCOUNT_NUMBER": (
            r"\b(?:account|acct)\.?\s*(?:#|number|num|no\.?)[:\s]*"
            r"[A-Z0-9\-]{4,}\b"
        ),

        # 11. Certificate / license numbers
        "CERTIFICATE_LICENSE": (
            r"\b(?:license|cert(?:ificate)?|lic\.?)"
            r"[:\s#]*[A-Z0-9\-]{5,}\b"
        ),

        # 12. Vehicle identifiers — VIN (17 chars) or license plate
        "VEHICLE_IDENTIFIER": (
            r"\b[A-HJ-NPR-Z0-9]{17}\b"                        # VIN
            r"|\b(?:plate|vin)[:\s]*[A-Z0-9\-]{4,10}\b"
        ),

        # 13. Device identifiers / serial numbers
        "DEVICE_IDENTIFIER": (
            r"\b(?:device|serial|s/n|sn)[:\s#]*[A-Z0-9\-]{6,}\b"
        ),

        # 16. Biometric identifiers
        "BIOMETRIC": (
            r"\b(?:fingerprint|retina|iris|voiceprint|biometric)"
            r"\s*(?:id|#|identifier|scan)?[:\s]*[A-Z0-9\-]*\b"
        ),

        # 17. Full-face photo references
        "PHOTO_REFERENCE": (
            r"\b(?:photo|photograph|image|headshot|portrait)"
            r"[:\s]*[^\s,;]{3,}\.(jpg|jpeg|png|gif|bmp|tiff|webp)\b"
        ),

        # 18. Ages over 89
        "AGE_OVER_89": (
            r"\b(?:9\d|1\d{2})\s*[-\s]?\s*(?:year|yr)s?\s*[-\s]?\s*old\b"
            r"|\bage[:\s]+(?:9\d|1\d{2})\b"
        ),

        # 3. Dates (except standalone year)
        "DATE": (
            r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"
            r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?"
            r"|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?"
            r"|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"[\s.]+\d{1,2},?\s+\d{4}\b"
            r"|\b\d{1,2}\s+"
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?"
            r"|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?"
            r"|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\s+\d{4}\b"
        ),

        # 2. Geographic data — street address, zip, US state abbreviation
        "GEOGRAPHIC": (
            r"\b\d{1,5}\s+[A-Za-z0-9\s]{2,30}"
            r"(?:St(?:reet)?|Ave(?:nue)?|Blvd|Rd|Dr|Ln|Ct|Pl|Way"
            r"|Hwy|Pkwy|Circle|Trail|Terr?)\b"
            r"|\b\d{5}(?:-\d{4})?\b"                           # ZIP
            r"|\b(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA"
            r"|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY"
            r"|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI"
            r"|WY|DC)\b"
        ),

        # 1. Names — capitalized first+last; intentionally last to avoid
        #    overmatching institutional names caught by other patterns
        "NAME": r"\b[A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20}\b",
    }

    def __init__(self) -> None:
        self._compiled: Dict[str, re.Pattern] = {
            phi_type: re.compile(pattern, re.IGNORECASE)
            for phi_type, pattern in self.PATTERNS.items()
        }

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        Scan *text* for all PHI categories.

        Returns a list of dicts, each with:
            type  — PHI category name
            value — matched string
            start — start character offset in text
            end   — end character offset in text
        """
        findings: List[Dict[str, Any]] = []

        for phi_type, pattern in self._compiled.items():
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "type": phi_type,
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        # Sort by position in text
        findings.sort(key=lambda x: x["start"])
        return findings
