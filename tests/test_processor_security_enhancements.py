"""Tests for enhanced security validation in the processor module."""

import tempfile
import os
from pathlib import Path
import pytest

from hipaa_compliance_summarizer.processor import HIPAAProcessor, ComplianceLevel
from hipaa_compliance_summarizer.documents import Document, DocumentType
from hipaa_compliance_summarizer.security import SecurityError


class TestProcessorSecurityEnhancements:
    """Test security enhancements in the main processor."""

    def test_processor_validates_file_paths(self):
        """Test that processor validates file paths for security."""
        processor = HIPAAProcessor()
        
        # Test with potentially dangerous path
        dangerous_paths = [
            "../../../etc/passwd",
            "~/sensitive_file.txt",
            "/proc/version",
            "file_with_null\x00byte.txt"
        ]
        
        for dangerous_path in dangerous_paths:
            doc = Document(dangerous_path, DocumentType.CLINICAL_NOTE)
            with pytest.raises(SecurityError):
                processor.process_document(doc)

    def test_processor_validates_file_content(self):
        """Test that processor validates file content for security."""
        processor = HIPAAProcessor()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write executable signature
            f.write('#!/bin/bash\necho "malicious"')
            temp_path = f.name
        
        try:
            doc = Document(temp_path, DocumentType.CLINICAL_NOTE)
            with pytest.raises(SecurityError):
                processor.process_document(doc)
        finally:
            os.unlink(temp_path)

    def test_processor_validates_text_input(self):
        """Test that processor validates direct text input."""
        processor = HIPAAProcessor()
        
        # Test with null bytes - should raise RuntimeError wrapping ValueError
        with pytest.raises(RuntimeError, match="null bytes"):
            processor.process_document("Text with null\x00byte")
        
        # Test with excessive length - should raise RuntimeError wrapping ValueError
        large_text = "A" * (51 * 1024 * 1024)  # 51MB
        with pytest.raises(RuntimeError, match="Text too large"):
            processor.process_document(large_text)
        
        # Test with non-string input
        with pytest.raises(ValueError, match="Input must be a string"):
            processor._validate_input_text(123)

    def test_processor_warns_about_suspicious_content(self, caplog):
        """Test that processor logs warnings for suspicious content."""
        processor = HIPAAProcessor()
        
        suspicious_texts = [
            "Patient data <script>alert('xss')</script>",
            "Medical record javascript:malicious()",
            "Report contains data:text/html,<h1>test</h1>"
        ]
        
        for text in suspicious_texts:
            # Should not raise error but should log warning
            result = processor.process_document(text)
            assert result is not None
            assert "Suspicious pattern detected" in caplog.text

    def test_processor_handles_valid_medical_text(self):
        """Test that processor handles valid medical text correctly."""
        processor = HIPAAProcessor()
        
        valid_medical_text = """
        Patient: John Doe
        SSN: 123-45-6789
        Medical History: Patient presents with chest pain.
        Diagnosis: Acute myocardial infarction.
        Treatment: Administered nitroglycerin.
        """
        
        result = processor.process_document(valid_medical_text)
        assert result is not None
        assert result.compliance_score >= 0.0
        assert result.phi_detected_count > 0  # Should detect SSN

    def test_processor_with_valid_file(self):
        """Test processor with a valid, secure file."""
        processor = HIPAAProcessor()
        
        # Create a valid medical document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Patient John Smith, DOB: 01/15/1980, condition stable.")
            temp_path = f.name
        
        try:
            doc = Document(temp_path, DocumentType.CLINICAL_NOTE)
            result = processor.process_document(doc)
            assert result is not None
            assert result.compliance_score >= 0.0
        finally:
            os.unlink(temp_path)

    def test_processor_rejects_oversized_files(self):
        """Test that processor rejects files that are too large."""
        processor = HIPAAProcessor()
        
        # Create a large file (over 100MB limit)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write enough data to exceed the limit
            chunk = "A" * 1024  # 1KB chunk
            for _ in range(105 * 1024):  # 105MB
                f.write(chunk)
            temp_path = f.name
        
        try:
            doc = Document(temp_path, DocumentType.CLINICAL_NOTE)
            with pytest.raises(SecurityError, match="File too large"):
                processor.process_document(doc)
        finally:
            os.unlink(temp_path)

    def test_processor_rejects_invalid_extensions(self):
        """Test that processor rejects files with invalid extensions."""
        processor = HIPAAProcessor()
        
        # Create file with invalid extension
        with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
            f.write(b"Some content")
            temp_path = f.name
        
        try:
            doc = Document(temp_path, DocumentType.CLINICAL_NOTE)
            with pytest.raises(SecurityError, match="not allowed"):
                processor.process_document(doc)
        finally:
            os.unlink(temp_path)

    def test_processor_error_handling(self):
        """Test that processor handles various error conditions gracefully."""
        processor = HIPAAProcessor()
        
        # Test with non-existent file
        doc = Document("/nonexistent/file.txt", DocumentType.CLINICAL_NOTE)
        with pytest.raises(SecurityError, match="does not exist"):
            processor.process_document(doc)

    def test_processor_compliance_levels_with_security(self):
        """Test that different compliance levels work with security validation."""
        for compliance_level in [ComplianceLevel.STRICT, ComplianceLevel.STANDARD, ComplianceLevel.MINIMAL]:
            processor = HIPAAProcessor(compliance_level=compliance_level)
            
            valid_text = "Patient: Jane Doe, SSN: 987-65-4321"
            result = processor.process_document(valid_text)
            
            assert result is not None
            assert result.compliance_score >= 0.0
            
            # Strict compliance should have shorter summaries
            if compliance_level == ComplianceLevel.STRICT:
                assert len(result.summary) <= 400

    def test_processor_direct_path_processing(self):
        """Test processor with direct file path (string) input."""
        processor = HIPAAProcessor()
        
        # Create a valid file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Medical report for patient with SSN 111-22-3333")
            temp_path = f.name
        
        try:
            # Process using direct path string
            result = processor.process_document(temp_path)
            assert result is not None
            assert result.phi_detected_count > 0
        finally:
            os.unlink(temp_path)

    def test_processor_edge_case_text_validation(self):
        """Test edge cases in text validation."""
        processor = HIPAAProcessor()
        
        # Empty string should be valid, but needs space to not be confused with path
        result = processor.process_document(" ")  # Single space
        assert result is not None
        
        # Very long but valid text should work
        long_valid_text = "Patient record. " * 1000  # ~16KB
        result = processor.process_document(long_valid_text)
        assert result is not None
        
        # Text with unicode characters should work
        unicode_text = "Patient: José María, condition: stable 🏥"
        result = processor.process_document(unicode_text)
        assert result is not None