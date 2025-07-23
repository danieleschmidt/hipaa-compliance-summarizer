"""Test security enhancements and input validation."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from hipaa_compliance_summarizer.security import (
    SecurityError,
    validate_file_path,
    validate_file_size,
    validate_file_extension,
    validate_directory_path,
    sanitize_filename,
    validate_content_type,
    validate_file_for_processing,
    get_security_recommendations,
)
from hipaa_compliance_summarizer import BatchProcessor


class TestSecurityValidation:
    """Test security validation functions."""

    def test_validate_file_path_success(self, tmp_path):
        """Test successful file path validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        result = validate_file_path(str(test_file))
        assert result == test_file.resolve()

    def test_validate_file_path_rejects_traversal(self):
        """Test that path traversal attempts are rejected."""
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "~/secret_file",
            "/etc/shadow",
            "/proc/version",
        ]
        
        for path in dangerous_paths:
            with pytest.raises(SecurityError, match="dangerous path pattern"):
                validate_file_path(path)

    def test_validate_file_path_rejects_empty(self):
        """Test that empty or invalid paths are rejected."""
        with pytest.raises(SecurityError, match="non-empty string"):
            validate_file_path("")
        
        with pytest.raises(SecurityError, match="non-empty string"):
            validate_file_path(None)

    def test_validate_file_path_rejects_long_path(self):
        """Test that overly long paths are rejected."""
        long_path = "a" * 5000
        with pytest.raises(SecurityError, match="path too long"):
            validate_file_path(long_path)

    def test_validate_file_size_success(self, tmp_path):
        """Test successful file size validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Valid content")
        
        # Should not raise an exception
        validate_file_size(test_file)

    def test_validate_file_size_rejects_large_file(self, tmp_path):
        """Test that large files are rejected."""
        test_file = tmp_path / "large.txt"
        
        # Mock a large file size
        from hipaa_compliance_summarizer.constants import SECURITY_LIMITS
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = SECURITY_LIMITS.MAX_FILE_SIZE_LARGE
            
            with pytest.raises(SecurityError, match="File too large"):
                validate_file_size(test_file)

    def test_validate_file_extension_success(self, tmp_path):
        """Test successful file extension validation."""
        allowed_files = [
            "document.txt",
            "report.pdf", 
            "data.csv",
            "config.json",
        ]
        
        for filename in allowed_files:
            test_file = tmp_path / filename
            validate_file_extension(test_file)  # Should not raise

    def test_validate_file_extension_rejects_disallowed(self, tmp_path):
        """Test that disallowed extensions are rejected."""
        dangerous_files = [
            "script.exe",
            "malware.bat",
            "trojan.scr",
            "virus.com",
        ]
        
        for filename in dangerous_files:
            test_file = tmp_path / filename
            with pytest.raises(SecurityError, match="not allowed"):
                validate_file_extension(test_file)

    def test_validate_directory_path_success(self, tmp_path):
        """Test successful directory validation."""
        result = validate_directory_path(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_validate_directory_path_rejects_nonexistent(self):
        """Test that non-existent directories are rejected."""
        with pytest.raises(SecurityError, match="does not exist"):
            validate_directory_path("/nonexistent/directory")

    def test_validate_directory_path_rejects_file(self, tmp_path):
        """Test that files are rejected when directory expected."""
        test_file = tmp_path / "not_a_dir.txt"
        test_file.write_text("content")
        
        with pytest.raises(SecurityError, match="not a directory"):
            validate_directory_path(str(test_file))

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        dangerous_name = 'file<>:"/\\|?*\x00name.txt'
        result = sanitize_filename(dangerous_name)
        # Should replace dangerous characters with underscores
        assert '<' not in result
        assert '>' not in result
        assert '"' not in result
        assert '|' not in result
        assert '?' not in result
        assert '*' not in result
        assert result.startswith("file")
        assert result.endswith("name.txt")

    def test_sanitize_filename_empty(self):
        """Test sanitization of empty or problematic filenames."""
        assert sanitize_filename("") == "unknown_file"
        assert sanitize_filename("...") == "unknown_file"
        assert sanitize_filename("   ") == "unknown_file"

    def test_sanitize_filename_long(self):
        """Test sanitization of overly long filenames."""
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 255
        assert result.endswith(".txt")

    def test_validate_content_type_text_file(self, tmp_path):
        """Test content type validation for text files."""
        test_file = tmp_path / "safe.txt"
        test_file.write_text("This is safe content.")
        
        assert validate_content_type(test_file) is True

    def test_validate_content_type_rejects_scripts(self, tmp_path):
        """Test that script content is detected and rejected."""
        dangerous_contents = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "<body onload='malicious()'>",
        ]
        
        for content in dangerous_contents:
            test_file = tmp_path / "dangerous.html"
            test_file.write_text(content)
            
            assert validate_content_type(test_file) is False

    def test_validate_content_type_rejects_executables(self, tmp_path):
        """Test that executable file signatures are detected."""
        # PE executable signature
        test_file = tmp_path / "executable.txt"
        test_file.write_bytes(b'MZ\x90\x00' + b'\x00' * 100)
        
        assert validate_content_type(test_file) is False

    def test_validate_file_for_processing_comprehensive(self, tmp_path):
        """Test comprehensive file validation for processing."""
        # Create a valid file
        test_file = tmp_path / "valid_document.txt"
        test_file.write_text("This is valid HIPAA document content.")
        
        result = validate_file_for_processing(str(test_file))
        assert result == test_file.resolve()

    def test_validate_file_for_processing_rejects_invalid(self, tmp_path):
        """Test that invalid files are rejected by comprehensive validation."""
        # Create a file with dangerous content
        test_file = tmp_path / "dangerous.txt"
        test_file.write_text("<script>alert('malicious')</script>")
        
        with pytest.raises(SecurityError, match="potentially dangerous"):
            validate_file_for_processing(str(test_file))

    def test_get_security_recommendations(self):
        """Test that security recommendations are provided."""
        recommendations = get_security_recommendations()
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


class TestBatchProcessorSecurity:
    """Test security integration in BatchProcessor."""

    def test_batch_processor_validates_input_directory(self):
        """Test that BatchProcessor validates input directories."""
        processor = BatchProcessor()
        
        with pytest.raises(ValueError, match="Security validation failed"):
            processor.process_directory("../../../etc")

    def test_batch_processor_validates_output_directory(self, tmp_path):
        """Test that BatchProcessor validates output directories."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create a test file
        test_file = input_dir / "test.txt"
        test_file.write_text("content")
        
        # Try to use a dangerous output path
        with pytest.raises(ValueError, match="Security validation failed"):
            processor.process_directory(str(input_dir), output_dir="../../../tmp/dangerous")

    def test_batch_processor_sanitizes_output_filenames(self, tmp_path):
        """Test that output filenames are sanitized."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        
        # Create a file with dangerous characters in name
        dangerous_file = input_dir / 'dangerous<>"|?*.txt'
        dangerous_file.write_text("Safe content")
        
        # Mock the processor to avoid actual processing
        with patch.object(processor.processor, 'process_document') as mock_process:
            from hipaa_compliance_summarizer.processor import ProcessingResult
            from hipaa_compliance_summarizer.phi import RedactionResult
            
            mock_process.return_value = ProcessingResult(
                summary="Processed content",
                compliance_score=1.0,
                phi_detected_count=0,
                redacted=RedactionResult(text="Processed content", entities=[])
            )
            
            results = processor.process_directory(
                str(input_dir), 
                output_dir=str(output_dir)
            )
            
            # Check that output file was created with sanitized name
            output_files = list(output_dir.glob("*"))
            assert len(output_files) == 1
            
            # Filename should be sanitized
            output_filename = output_files[0].name
            assert '<' not in output_filename
            assert '>' not in output_filename
            assert '|' not in output_filename

    def test_batch_processor_handles_security_errors(self, tmp_path):
        """Test that security errors are properly handled."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create a file that will fail security validation
        dangerous_file = input_dir / "malware.exe"
        dangerous_file.write_bytes(b'MZ\x90\x00' + b'\x00' * 100)  # PE signature
        
        results = processor.process_directory(str(input_dir))
        
        # Should have one error result
        assert len(results) == 1
        assert hasattr(results[0], 'error')
        assert results[0].error_type == "SecurityError"
        assert "Security validation failed" in results[0].error

    def test_batch_processor_prevents_file_overwrite(self, tmp_path):
        """Test that output files don't overwrite existing files."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create input file
        test_file = input_dir / "document.txt"
        test_file.write_text("Content")
        
        # Create existing output file
        existing_output = output_dir / "document.txt"
        existing_output.write_text("Existing content")
        
        # Mock the processor
        with patch.object(processor.processor, 'process_document') as mock_process:
            from hipaa_compliance_summarizer.processor import ProcessingResult
            from hipaa_compliance_summarizer.phi import RedactionResult
            
            mock_process.return_value = ProcessingResult(
                summary="New content",
                compliance_score=1.0,
                phi_detected_count=0,
                redacted=RedactionResult(text="New content", entities=[])
            )
            
            processor.process_directory(
                str(input_dir), 
                output_dir=str(output_dir)
            )
            
            # Should have created a new file with suffix
            output_files = list(output_dir.glob("document*.txt"))
            assert len(output_files) == 2  # Original + new numbered file
            
            # Original file should be unchanged
            assert existing_output.read_text() == "Existing content"