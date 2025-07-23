"""Tests for proper implementation of exception classes."""

import pytest
from pathlib import Path

from hipaa_compliance_summarizer.security import SecurityError
from hipaa_compliance_summarizer.parsers import ParsingError, FileReadError, EncodingError
from hipaa_compliance_summarizer.documents import DocumentError, DocumentTypeError


class TestSecurityError:
    """Test SecurityError exception implementation."""
    
    def test_security_error_basic_functionality(self):
        """Test basic SecurityError functionality."""
        error = SecurityError("Test security violation")
        assert str(error) == "Test security violation"
        assert isinstance(error, Exception)
    
    def test_security_error_with_file_path(self):
        """Test SecurityError with file path context."""
        error = SecurityError("Invalid file", file_path="/dangerous/path")
        assert str(error) == "Invalid file"
        assert error.file_path == "/dangerous/path"
        assert "file_path" in error.get_context()
    
    def test_security_error_with_violation_type(self):
        """Test SecurityError with violation type."""
        error = SecurityError("Path traversal detected", violation_type="path_traversal")
        assert error.violation_type == "path_traversal"
        assert "violation_type" in error.get_context()
    
    def test_security_error_context_dict(self):
        """Test SecurityError context dictionary."""
        error = SecurityError("Test", file_path="/test", violation_type="test_violation")
        context = error.get_context()
        assert context["file_path"] == "/test"
        assert context["violation_type"] == "test_violation"


class TestParsingError:
    """Test ParsingError exception implementation."""
    
    def test_parsing_error_basic_functionality(self):
        """Test basic ParsingError functionality."""
        error = ParsingError("Parsing failed")
        assert str(error) == "Parsing failed"
        assert isinstance(error, Exception)
    
    def test_parsing_error_with_file_path(self):
        """Test ParsingError with file path."""
        error = ParsingError("Cannot parse", file_path="document.pdf")
        assert error.file_path == "document.pdf"
    
    def test_parsing_error_with_parser_type(self):
        """Test ParsingError with parser type."""
        error = ParsingError("Parse failed", parser_type="pdf_parser")
        assert error.parser_type == "pdf_parser"


class TestFileReadError:
    """Test FileReadError exception implementation."""
    
    def test_file_read_error_inherits_from_parsing_error(self):
        """Test FileReadError inheritance."""
        error = FileReadError("Cannot read file")
        assert isinstance(error, ParsingError)
        assert isinstance(error, Exception)
    
    def test_file_read_error_with_permissions(self):
        """Test FileReadError with permission context."""
        error = FileReadError("Access denied", file_path="/restricted", permission_error=True)
        assert error.permission_error is True
        assert error.file_path == "/restricted"


class TestEncodingError:
    """Test EncodingError exception implementation."""
    
    def test_encoding_error_inherits_from_parsing_error(self):
        """Test EncodingError inheritance."""
        error = EncodingError("Encoding failed")
        assert isinstance(error, ParsingError)
    
    def test_encoding_error_with_encoding_info(self):
        """Test EncodingError with encoding details."""
        error = EncodingError("Bad encoding", attempted_encodings=["utf-8", "latin-1"])
        assert error.attempted_encodings == ["utf-8", "latin-1"]


class TestDocumentError:
    """Test DocumentError exception implementation."""
    
    def test_document_error_basic_functionality(self):
        """Test basic DocumentError functionality."""
        error = DocumentError("Document validation failed")
        assert str(error) == "Document validation failed"
        assert isinstance(error, Exception)
    
    def test_document_error_with_document_type(self):
        """Test DocumentError with document type."""
        error = DocumentError("Invalid document", document_type="clinical_note")
        assert error.document_type == "clinical_note"
    
    def test_document_error_with_validation_details(self):
        """Test DocumentError with validation context."""
        error = DocumentError("Validation failed", validation_errors=["Missing field", "Invalid format"])
        assert error.validation_errors == ["Missing field", "Invalid format"]


class TestDocumentTypeError:
    """Test DocumentTypeError exception implementation."""
    
    def test_document_type_error_inherits_from_document_error(self):
        """Test DocumentTypeError inheritance."""
        error = DocumentTypeError("Cannot detect type")
        assert isinstance(error, DocumentError)
        assert isinstance(error, Exception)
    
    def test_document_type_error_with_candidates(self):
        """Test DocumentTypeError with type candidates."""
        error = DocumentTypeError("Ambiguous type", type_candidates=["medical_record", "clinical_note"])
        assert error.type_candidates == ["medical_record", "clinical_note"]