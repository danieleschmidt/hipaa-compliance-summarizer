from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)


class DocumentError(Exception):
    """Base exception for document-related errors.
    
    Attributes:
        document_type: Type of document that caused the error
        validation_errors: List of validation errors if applicable
        original_error: Original exception that triggered this error
    """
    
    def __init__(self, message: str, document_type: Optional[str] = None,
                 validation_errors: Optional[list] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.document_type = document_type
        self.validation_errors = validation_errors or []
        self.original_error = original_error
    
    def get_context(self) -> dict:
        """Get error context information for logging and debugging."""
        context = {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "document_type": self.document_type,
            "validation_errors": self.validation_errors,
        }
        if self.original_error:
            context["original_error"] = {
                "type": self.original_error.__class__.__name__,
                "message": str(self.original_error)
            }
        return context
    
    def to_dict(self) -> dict:
        """Convert error to dictionary for serialization."""
        return self.get_context()
    
    def get_user_message(self) -> str:
        """Get user-friendly error message without technical details."""
        base_message = str(self)
        if self.document_type:
            return f"Error processing {self.document_type} document: {base_message}"
        return f"Document processing error: {base_message}"


class DocumentTypeError(DocumentError):
    """Raised when document type detection fails.
    
    Attributes:
        type_candidates: List of possible document types that were considered
        confidence_scores: Dictionary of type -> confidence mappings if available
    """
    
    def __init__(self, message: str, type_candidates: Optional[list] = None,
                 confidence_scores: Optional[dict] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.type_candidates = type_candidates or []
        self.confidence_scores = confidence_scores or {}


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
