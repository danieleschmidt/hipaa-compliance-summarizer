from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class DocumentError(Exception):
    """Base exception for document-related errors."""
    pass


class DocumentTypeError(DocumentError):
    """Raised when document type detection fails."""
    pass


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

    def __post_init__(self):
        """Validate document after initialization."""
        if not isinstance(self.path, str):
            raise DocumentError(f"Document path must be a string, got {type(self.path).__name__}")
        
        if not self.path.strip():
            raise DocumentError("Document path cannot be empty")
        
        if not isinstance(self.type, DocumentType):
            raise DocumentError(f"Document type must be a DocumentType, got {type(self.type).__name__}")


def detect_document_type(path_or_name: Union[str, Path]) -> DocumentType:
    """Infer a :class:`DocumentType` from a file path or name with robust error handling.
    
    Args:
        path_or_name: File path, filename, or Path object
        
    Returns:
        Detected DocumentType
        
    Raises:
        DocumentTypeError: If detection fails due to invalid input
        TypeError: If input type is invalid
    """
    try:
        # Handle None input
        if path_or_name is None:
            logger.warning("Received None input for document type detection")
            raise DocumentTypeError("Cannot detect document type from None input")
        
        # Convert to string if it's a Path object
        if hasattr(path_or_name, '__fspath__'):
            path_str = str(path_or_name)
        elif isinstance(path_or_name, str):
            path_str = path_or_name
        else:
            logger.error("Invalid input type for document detection: %s", type(path_or_name).__name__)
            raise TypeError(f"Input must be a string or Path object, got {type(path_or_name).__name__}")
        
        # Handle empty string
        if not path_str.strip():
            logger.debug("Empty path provided, returning UNKNOWN")
            return DocumentType.UNKNOWN
        
        # Handle very long paths (potential attack or corruption)
        if len(path_str) > 10000:
            logger.warning("Extremely long path provided (%d chars), truncating for analysis", len(path_str))
            path_str = path_str[:1000]  # Truncate for analysis
        
        try:
            # Extract filename stem safely
            path_obj = Path(path_str)
            name = path_obj.stem.lower()
        except (ValueError, OSError) as e:
            logger.warning("Failed to parse path '%s': %s, using full string", path_str[:100], e)
            # Fall back to using the full string for analysis
            name = path_str.lower()
        
        # Remove any null bytes or control characters that might cause issues
        name = ''.join(char for char in name if ord(char) >= 32 or char in '\t\n\r')
        
        # Document type detection logic with logging
        if any(key in name for key in {"medical", "record"}):
            logger.debug("Detected MEDICAL_RECORD type for: %s", path_str[:100])
            return DocumentType.MEDICAL_RECORD
        
        if any(key in name for key in {"clinical", "note"}):
            logger.debug("Detected CLINICAL_NOTE type for: %s", path_str[:100])
            return DocumentType.CLINICAL_NOTE
        
        if any(key in name for key in {"insurance", "claim", "form"}):
            logger.debug("Detected INSURANCE_FORM type for: %s", path_str[:100])
            return DocumentType.INSURANCE_FORM
        
        logger.debug("No specific type detected, returning UNKNOWN for: %s", path_str[:100])
        return DocumentType.UNKNOWN
        
    except Exception as e:
        logger.error("Unexpected error in document type detection: %s", e)
        raise DocumentTypeError(f"Document type detection failed: {e}") from e


def validate_document(document: Document) -> bool:
    """Validate a Document object for basic consistency.
    
    Args:
        document: Document to validate
        
    Returns:
        True if document is valid
        
    Raises:
        DocumentError: If document is invalid
    """
    if not isinstance(document, Document):
        raise DocumentError(f"Expected Document object, got {type(document).__name__}")
    
    if not document.path or not document.path.strip():
        raise DocumentError("Document path cannot be empty")
    
    if not isinstance(document.type, DocumentType):
        raise DocumentError(f"Invalid document type: {document.type}")
    
    return True
