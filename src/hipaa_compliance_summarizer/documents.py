from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DocumentType(str, Enum):
    """Supported healthcare document categories."""

    MEDICAL_RECORD = "medical_record"
    CLINICAL_NOTE = "clinical_note"
    INSURANCE_FORM = "insurance_form"
    UNKNOWN = "unknown"


@dataclass
class Document:
    """Represents a healthcare-related document."""

    path: str
    type: DocumentType


def detect_document_type(path_or_name: str) -> DocumentType:
    """Infer a :class:`DocumentType` from a file path or name."""
    name = Path(path_or_name).stem.lower()
    if any(key in name for key in {"medical", "record"}):
        return DocumentType.MEDICAL_RECORD
    if any(key in name for key in {"clinical", "note"}):
        return DocumentType.CLINICAL_NOTE
    if any(key in name for key in {"insurance", "claim", "form"}):
        return DocumentType.INSURANCE_FORM
    return DocumentType.UNKNOWN
