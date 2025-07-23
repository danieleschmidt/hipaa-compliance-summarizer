from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import os
import time
from pathlib import Path
from typing import Optional, Union

from .constants import SECURITY_LIMITS
from .documents import Document, DocumentType, DocumentError, validate_document
from .parsers import (
    parse_clinical_note,
    parse_insurance_form,
    parse_medical_record,
    ParsingError,
)

from .phi import PHIRedactor, RedactionResult
from .security import validate_file_for_processing, SecurityError
from .config import CONFIG
import textwrap

logger = logging.getLogger(__name__)


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
        # Only treat as path if it looks like one and is reasonable length
        if (len(path_or_text) < 4096 and 
            not '\n' in path_or_text and 
            not path_or_text.strip().startswith(' ') and
            (os.path.sep in path_or_text or '.' in path_or_text)):
            
            path = Path(path_or_text)
            try:
                if path.exists():
                    try:
                        # Apply security validation for file paths
                        validated_path = validate_file_for_processing(str(path))
                        try:
                            return validated_path.read_text(encoding='utf-8', errors='strict')
                        except UnicodeDecodeError as e:
                            logger.error("File encoding error for %s: %s", path, e)
                            raise ValueError(f"File contains invalid UTF-8 encoding: {e}")
                        except IOError as e:
                            logger.error("File read error for %s: %s", path, e)
                            raise IOError(f"Cannot read file: {e}")
                    except SecurityError as e:
                        logger.error("Security validation failed for file %s: %s", path, e)
                        raise SecurityError(f"File security validation failed: {e}")
                else:
                    # File doesn't exist, treat as text content
                    logger.debug("Path %s does not exist. Treating as text content.", path_or_text)
            except OSError as e:
                # If path checking fails (e.g., filename too long, permission denied), treat as text
                logger.debug("Path validation failed for %s: %s. Treating as text content.", path_or_text, e)
            except PermissionError as e:
                # Permission denied accessing file system
                logger.warning("Permission denied for path %s: %s. Treating as text content.", path_or_text, e)
        
        return path_or_text

    def _summarize(self, text: str) -> str:
        """Naive summarization by shortening the text."""
        width = 400 if self.compliance_level == ComplianceLevel.STRICT else 600
        return textwrap.shorten(text, width=width, placeholder="...")

    def _score(self, redacted: RedactionResult) -> float:
        scoring = CONFIG.get("scoring", {})
        per_entity = scoring.get("penalty_per_entity", 0.01)
        cap = scoring.get("penalty_cap", 0.2)
        strict_mul = scoring.get("strict_multiplier", 1.5)

        base = 1.0
        penalty = min(len(redacted.entities) * per_entity, cap)
        if self.compliance_level == ComplianceLevel.STRICT:
            penalty *= strict_mul
        return max(0.0, base - penalty)

    def _validate_input_text(self, text: str) -> str:
        """Validate and sanitize input text for security."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Check for excessive length to prevent DoS
        max_length = SECURITY_LIMITS.MAX_DOCUMENT_SIZE
        if len(text) > max_length:
            raise ValueError(f"Text too large: {len(text)} characters (max {max_length})")
        
        # Check for null bytes that could indicate binary content
        if '\x00' in text:
            raise ValueError("Text contains null bytes - possibly binary content")
        
        # Log suspicious patterns but don't reject (medical text might contain these)
        suspicious_patterns = ['<script', 'javascript:', 'data:text/html']
        for pattern in suspicious_patterns:
            if pattern.lower() in text.lower():
                logger.warning("Suspicious pattern detected in text: %s", pattern)
        
        return text

    def process_document(self, document: Union[str, Document]) -> ProcessingResult:
        """Process a document path, raw text, or :class:`Document` instance."""
        logger.info("Processing document %s", getattr(document, 'path', document))
        start = time.perf_counter()
        
        try:
            if isinstance(document, Document):
                # Validate the Document object first
                try:
                    validate_document(document)
                except DocumentError as e:
                    logger.error("Invalid document object: %s", e)
                    raise RuntimeError(f"Document validation failed: {e}")
                
                # For Document objects, validate the file path first
                try:
                    validated_path = validate_file_for_processing(document.path)
                    logger.info("File validation successful for %s", validated_path)
                except SecurityError as e:
                    logger.error("Security validation failed for document %s: %s", document.path, e)
                    raise SecurityError(f"Document security validation failed: {e}")
                
                # Parse document based on type with error handling
                try:
                    if document.type == DocumentType.MEDICAL_RECORD:
                        text = parse_medical_record(str(validated_path))
                    elif document.type == DocumentType.CLINICAL_NOTE:
                        text = parse_clinical_note(str(validated_path))
                    elif document.type == DocumentType.INSURANCE_FORM:
                        text = parse_insurance_form(str(validated_path))
                    else:
                        # For UNKNOWN type, use generic text loading
                        text = self._load_text(str(validated_path))
                except ParsingError as e:
                    logger.error("Failed to parse document %s: %s", document.path, e)
                    raise RuntimeError(f"Document parsing failed: {e}")
            else:
                text = self._load_text(document)
            
            # Validate input text for security
            text = self._validate_input_text(text)
            
            redacted = self.redactor.redact(text)
            summary = self._summarize(redacted.text)
            score = self._score(redacted)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Processed document in %.2f ms with score %.2f and %d entities",
                elapsed_ms,
                score,
                len(redacted.entities),
            )
            return ProcessingResult(
                summary=summary,
                compliance_score=score,
                phi_detected_count=len(redacted.entities),
                redacted=redacted,
            )
        except SecurityError:
            # Re-raise security errors as-is
            raise
        except Exception as e:
            logger.error("Error processing document: %s", e)
            raise RuntimeError(f"Document processing failed: {e}")
