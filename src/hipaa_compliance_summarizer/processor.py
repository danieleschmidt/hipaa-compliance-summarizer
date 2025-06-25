from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from .documents import Document, DocumentType
from .parsers import (
    parse_clinical_note,
    parse_insurance_form,
    parse_medical_record,
)

from .phi import PHIRedactor, RedactionResult
import textwrap


class ComplianceLevel(str, Enum):
    """Supported compliance strictness levels."""

    STRICT = "strict"
    STANDARD = "standard"
    MINIMAL = "minimal"


@dataclass
class ProcessingResult:
    """Outcome of HIPAA-compliant document processing."""

    summary: str
    compliance_score: float
    phi_detected_count: int
    redacted: RedactionResult


class HIPAAProcessor:
    """Process medical documents with PHI redaction and summarization."""

    def __init__(
        self,
        compliance_level: ComplianceLevel = ComplianceLevel.STANDARD,
        *,
        redactor: Optional[PHIRedactor] = None,
    ) -> None:
        self.compliance_level = ComplianceLevel(compliance_level)
        self.redactor = redactor or PHIRedactor()

    def _load_text(self, path_or_text: str) -> str:
        path = Path(path_or_text)
        if path.exists():
            return path.read_text()
        return path_or_text

    def _summarize(self, text: str) -> str:
        """Naive summarization by shortening the text."""
        width = 400 if self.compliance_level == ComplianceLevel.STRICT else 600
        return textwrap.shorten(text, width=width, placeholder="...")

    def _score(self, redacted: RedactionResult) -> float:
        base = 1.0
        penalty = min(len(redacted.entities) * 0.01, 0.2)
        if self.compliance_level == ComplianceLevel.STRICT:
            penalty *= 1.5
        return max(0.0, base - penalty)

    def process_document(self, document: Union[str, Document]) -> ProcessingResult:
        """Process a document path, raw text, or :class:`Document` instance."""
        if isinstance(document, Document):
            if document.type == DocumentType.MEDICAL_RECORD:
                text = parse_medical_record(document.path)
            elif document.type == DocumentType.CLINICAL_NOTE:
                text = parse_clinical_note(document.path)
            elif document.type == DocumentType.INSURANCE_FORM:
                text = parse_insurance_form(document.path)
            else:
                text = self._load_text(document.path)
        else:
            text = self._load_text(document)
        redacted = self.redactor.redact(text)
        summary = self._summarize(redacted.text)
        score = self._score(redacted)
        return ProcessingResult(
            summary=summary,
            compliance_score=score,
            phi_detected_count=len(redacted.entities),
            redacted=redacted,
        )
