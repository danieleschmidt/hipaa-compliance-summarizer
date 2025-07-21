"""Comprehensive unit tests for the security module - covering edge cases and error conditions."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import stat

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
    MAX_FILE_SIZE,
    MAX_PATH_LENGTH,
    ALLOWED_EXTENSIONS,
    BLOCKED_PATTERNS,
)


class TestSecurityValidationEdgeCases:
    """Test edge cases and error conditions in security validation."""

    def test_validate_file_path_with_invalid_path_object(self):
        """Test validation with path that raises OSError during resolution."""
        # Test with a path that contains invalid characters that cause Path.resolve() to fail
        with patch('pathlib.Path.resolve', side_effect=OSError("Invalid path")):
            with pytest.raises(SecurityError, match="Invalid file path"):
                validate_file_path("some_path")

    def test_validate_file_path_with_value_error(self):
        """Test validation with path that raises ValueError during resolution."""
        with patch('pathlib.Path.resolve', side_effect=ValueError("Path error")):
            with pytest.raises(SecurityError, match="Invalid file path"):
                validate_file_path("some_path")

    def test_validate_file_path_with_dotdot_in_resolved_path(self):
        """Test validation that detects .. in resolved path."""
        # Create a mock Path object that returns a path containing '..'
        with patch('pathlib.Path.resolve') as mock_resolve:
            mock_path = Mock()
            mock_path.__str__ = Mock(return_value="/some/path/../other")
            mock_resolve.return_value = mock_path
            
            with pytest.raises(SecurityError, match="Path traversal detected in resolved path"):
                validate_file_path("some_path")

    def test_validate_file_size_with_empty_file_warning(self, tmp_path, caplog):
        """Test validation logs warning for empty files."""
        empty_file = tmp_path / "empty.txt"
        empty_file.touch()  # Creates empty file
        
        validate_file_size(empty_file)  # Should not raise but should warn
        assert "Empty file detected" in caplog.text

    def test_validate_file_size_with_os_error(self, tmp_path):
        """Test validation handles OSError when accessing file stats."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        # Mock stat to raise OSError
        with patch.object(Path, 'stat', side_effect=OSError("Permission denied")):
            with pytest.raises(SecurityError, match="Cannot access file"):
                validate_file_size(test_file)

    def test_validate_directory_path_with_no_read_permission(self, tmp_path):
        """Test validation rejects directory without read permission."""
        test_dir = tmp_path / "no_read"
        test_dir.mkdir()
        
        # Mock os.access to return False for read permission
        with patch('os.access', return_value=False):
            with pytest.raises(SecurityError, match="not readable"):
                validate_directory_path(str(test_dir))

    def test_validate_content_type_with_unicode_decode_error(self, tmp_path):
        """Test content validation handles binary files that can't be decoded."""
        binary_file = tmp_path / "binary.txt"
        # Write binary data that can't be decoded as UTF-8
        binary_file.write_bytes(b'\xff\xfe\x00\x00invalid_utf8')
        
        # Should return True since it's not an executable and can't be decoded as text
        result = validate_content_type(binary_file)
        assert result is True

    def test_validate_content_type_with_os_error(self, tmp_path, caplog):
        """Test content validation handles OSError when reading file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        # Mock open to raise OSError
        with patch('builtins.open', side_effect=OSError("Permission denied")):
            result = validate_content_type(test_file)
            assert result is False
            assert "Cannot read file for content validation" in caplog.text

    def test_validate_file_for_processing_nonexistent_file(self):
        """Test comprehensive validation rejects non-existent files."""
        with pytest.raises(SecurityError, match="does not exist"):
            validate_file_for_processing("/nonexistent/file.txt")

    def test_validate_file_for_processing_not_regular_file(self, tmp_path):
        """Test comprehensive validation rejects non-regular files."""
        # Create a directory instead of a file
        test_dir = tmp_path / "notafile"
        test_dir.mkdir()
        
        with pytest.raises(SecurityError, match="not a regular file"):
            validate_file_for_processing(str(test_dir))

    def test_validate_file_for_processing_no_read_permission(self, tmp_path):
        """Test comprehensive validation rejects files without read permission."""
        test_file = tmp_path / "no_read.txt"
        test_file.write_text("content")
        
        # Mock os.access to return False for read permission
        with patch('os.access', return_value=False):
            with pytest.raises(SecurityError, match="not readable"):
                validate_file_for_processing(str(test_file))


class TestSecurityConstantsAndEdgeCases:
    """Test security constants and boundary conditions."""

    def test_security_constants_values(self):
        """Test that security constants have expected values."""
        assert MAX_FILE_SIZE == 100 * 1024 * 1024  # 100MB
        assert MAX_PATH_LENGTH == 4096
        assert len(ALLOWED_EXTENSIONS) >= 8  # Should have common extensions
        assert len(BLOCKED_PATTERNS) >= 6   # Should have common dangerous patterns
        
        # Test specific blocked patterns
        assert r'\.\./' in BLOCKED_PATTERNS
        assert r'/etc/' in BLOCKED_PATTERNS
        assert r'/proc/' in BLOCKED_PATTERNS

    def test_security_recommendations_completeness(self):
        """Test that security recommendations cover key areas."""
        recommendations = get_security_recommendations()
        
        # Convert to lowercase for easier checking
        rec_text = " ".join(recommendations).lower()
        
        # Should cover these key security areas
        assert "privilege" in rec_text or "permission" in rec_text
        assert "encrypt" in rec_text
        assert "https" in rec_text
        assert "audit" in rec_text or "log" in rec_text
        assert "update" in rec_text or "patch" in rec_text

    def test_sanitize_filename_with_reserved_names(self):
        """Test filename sanitization with Windows reserved names."""
        reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
        
        for name in reserved_names:
            # These should be sanitized but not cause errors
            result = sanitize_filename(name + ".txt")
            assert result  # Should not be empty
            assert len(result) > 0

    def test_sanitize_filename_unicode_handling(self):
        """Test filename sanitization with Unicode characters."""
        unicode_filename = "文档_test_файл.txt"
        result = sanitize_filename(unicode_filename)
        
        # Should preserve Unicode characters that are safe
        assert "test" in result
        assert result.endswith(".txt")

    def test_validate_file_extension_case_insensitive(self, tmp_path):
        """Test that file extension validation is case insensitive."""
        # Test uppercase extensions
        test_files = [
            "document.TXT",
            "report.PDF", 
            "data.CSV",
            "config.JSON",
        ]
        
        for filename in test_files:
            test_file = tmp_path / filename
            validate_file_extension(test_file)  # Should not raise

    def test_blocked_patterns_comprehensive(self):
        """Test that blocked patterns catch various attack vectors."""
        dangerous_paths = [
            "../../../etc/passwd",           # Unix path traversal
            "..\\..\\..\\windows\\system32", # Windows path traversal  
            "~/../../etc/shadow",            # Home dir + traversal
            "/etc/passwd",                   # Direct system file access
            "/proc/self/environ",            # Process information
            "/sys/class/net",                # System information
        ]
        
        for path in dangerous_paths:
            with pytest.raises(SecurityError, match="dangerous path pattern"):
                validate_file_path(path)


class TestSecurityErrorConditions:
    """Test various error conditions and exception handling."""

    def test_security_error_inheritance(self):
        """Test that SecurityError is properly inherited from Exception."""
        error = SecurityError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_validate_file_path_with_none_input(self):
        """Test path validation with None input."""
        with pytest.raises(SecurityError, match="non-empty string"):
            validate_file_path(None)

    def test_validate_file_path_with_non_string_input(self):
        """Test path validation with non-string input."""
        with pytest.raises(SecurityError, match="non-empty string"):
            validate_file_path(123)
        
        with pytest.raises(SecurityError, match="non-empty string"):
            validate_file_path([])

    def test_validate_file_path_boundary_length(self):
        """Test path validation at boundary length."""
        # Test path at exactly max length
        max_length_path = "a" * MAX_PATH_LENGTH
        # This should work (assuming it doesn't hit blocked patterns)
        try:
            validate_file_path(max_length_path)
        except SecurityError as e:
            # Could fail due to path resolution, but not due to length
            assert "too long" not in str(e)
        
        # Test path over max length
        over_length_path = "a" * (MAX_PATH_LENGTH + 1)
        with pytest.raises(SecurityError, match="too long"):
            validate_file_path(over_length_path)

    def test_validate_file_size_boundary_conditions(self, tmp_path):
        """Test file size validation at boundaries."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        # Mock file size at exactly max size
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = MAX_FILE_SIZE
            validate_file_size(test_file)  # Should not raise
        
        # Mock file size over max size
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = MAX_FILE_SIZE + 1
            with pytest.raises(SecurityError, match="too large"):
                validate_file_size(test_file)


class TestAllSecurityFunctions:
    """Test all security functions are properly exported."""

    def test_all_exports_completeness(self):
        """Test that __all__ contains all expected security functions."""
        from hipaa_compliance_summarizer.security import __all__
        
        expected_exports = [
            "SecurityError",
            "validate_file_path",
            "validate_file_size", 
            "validate_file_extension",
            "validate_directory_path",
            "sanitize_filename",
            "validate_content_type",
            "validate_file_for_processing",
            "get_security_recommendations",
        ]
        
        for export in expected_exports:
            assert export in __all__, f"{export} should be in __all__"

    def test_content_type_validation_with_different_file_types(self, tmp_path):
        """Test content type validation with various file types."""
        # Test ELF executable
        elf_file = tmp_path / "elf_test.txt"
        elf_file.write_bytes(b'\x7fELF' + b'\x00' * 100)
        assert validate_content_type(elf_file) is False
        
        # Test Mach-O universal binary
        macho_file = tmp_path / "macho_test.txt"
        macho_file.write_bytes(b'\xca\xfe\xba\xbe' + b'\x00' * 100)
        assert validate_content_type(macho_file) is False
        
        # Test shell script variations
        shell_scripts = [
            b'#!/bin/sh\necho test',
            b'#!/usr/bin/env python\nprint("test")',
            b'#!/usr/bin/bash\necho test'
        ]
        
        for script_content in shell_scripts:
            script_file = tmp_path / f"script_{len(script_content)}.txt"
            script_file.write_bytes(script_content)
            assert validate_content_type(script_file) is False

    def test_dangerous_script_patterns_in_html_xml(self, tmp_path):
        """Test detection of dangerous patterns in HTML/XML files."""
        # These patterns should be detected as dangerous based on the actual implementation
        dangerous_patterns = [
            '<script type="text/javascript">alert("xss")</script>',
            'javascript:void(document.cookie="stolen")',
            'vbscript:msgbox("dangerous")',
            '<body onload="maliciousFunction()">content</body>',  # onload= is detected
        ]
        
        for i, pattern in enumerate(dangerous_patterns):
            for file_ext in ['.html', '.xml']:
                test_file = tmp_path / f"dangerous_{i}{file_ext}"
                test_file.write_text(pattern)
                assert validate_content_type(test_file) is False, f"Pattern {i} should be detected as dangerous"
        
        # Test a pattern that should NOT be detected (for negative testing)
        safe_pattern = '<img src="x" alt="safe image">'
        safe_file = tmp_path / "safe.html"
        safe_file.write_text(safe_pattern)
        assert validate_content_type(safe_file) is True, "Safe pattern should not be flagged"


if __name__ == "__main__":
    pytest.main([__file__])