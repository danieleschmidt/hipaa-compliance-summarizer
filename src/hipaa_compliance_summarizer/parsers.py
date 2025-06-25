from __future__ import annotations

from pathlib import Path


def _load_text(path_or_text: str) -> str:
    """Return the text content from a file path or raw string."""
    if not path_or_text:
        return ""
    path = Path(path_or_text)
    if path.is_file():
        return path.read_text()
    return path_or_text


def parse_medical_record(data: str) -> str:
    """Extract text from a medical record file or string."""
    return _load_text(data).strip()


def parse_clinical_note(data: str) -> str:
    """Extract text from a clinical note."""
    return _load_text(data).strip()


def parse_insurance_form(data: str) -> str:
    """Extract text from an insurance form."""
    return _load_text(data).strip()
