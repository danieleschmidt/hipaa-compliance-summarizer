"""Comprehensive tests for error handling edge cases across all modules."""

import tempfile
import os
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from hipaa_compliance_summarizer.parsers import parse_medical_record, parse_clinical_note, parse_insurance_form, ParsingError
from hipaa_compliance_summarizer.documents import Document, DocumentType, detect_document_type, DocumentTypeError
from hipaa_compliance_summarizer.processor import HIPAAProcessor
from hipaa_compliance_summarizer.batch import BatchProcessor
from hipaa_compliance_summarizer.constants import TEST_CONSTANTS


class TestParserErrorHandling:
    """Test error handling in parser functions."""

    def test_parse_medical_record_with_nonexistent_file(self):
        """Test parsing non-existent file path."""
        with pytest.raises(ParsingError, match="File not found"):
            parse_medical_record("/nonexistent/path/to/file.txt")

    def test_parse_medical_record_with_permission_error(self):
        """Test parsing file with permission issues."""
        # Create a real file first, then mock the read operation
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            # Mock the file reading to simulate permission error
            with patch('pathlib.Path.read_text', side_effect=PermissionError("Permission denied")):
                with pytest.raises(ParsingError, match="Cannot read file"):
                    parse_medical_record(temp_path)
        finally:
            os.unlink(temp_path)

    def test_parse_medical_record_with_binary_file(self):
        """Test parsing binary file that can't be decoded."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write binary data that will cause encoding issues
            f.write(b'\xff\xfe\x00\x00invalid\x00utf8\x00data')
            temp_path = f.name
        
        try:
            # Should handle encoding errors gracefully
            result = parse_medical_record(temp_path)
            assert isinstance(result, str)  # Should return some string, even if garbled
        finally:
            os.unlink(temp_path)

    def test_parse_medical_record_with_empty_file(self):
        """Test parsing empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_path = f.name
        
        try:
            result = parse_medical_record(temp_path)
            assert result == ""
        finally:
            os.unlink(temp_path)

    def test_parse_medical_record_with_large_file(self):
        """Test parsing extremely large file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            # Create a large file
            large_content = "A" * TEST_CONSTANTS.TEST_MODERATE_SIZE  # 10MB
            f.write(large_content)
            temp_path = f.name
        
        try:
            result = parse_medical_record(temp_path)
            assert len(result) == TEST_CONSTANTS.TEST_MODERATE_SIZE
        finally:
            os.unlink(temp_path)

    def test_parse_with_directory_instead_of_file(self):
        """Test parsing when given a directory path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ParsingError, match="Path is a directory"):
                parse_medical_record(temp_dir)

    def test_parse_with_device_file(self):
        """Test parsing when given a device file path."""
        # Skip on non-Unix systems
        device_file = "/dev/null"
        if os.path.exists(device_file):
            result = parse_medical_record(device_file)
            assert result == ""  # Should handle device files gracefully

    def test_parse_with_malformed_utf8(self):
        """Test parsing file with malformed UTF-8 sequences."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            # Write invalid UTF-8 sequence
            f.write(b'Valid text \xff\xfe invalid sequence \xc0\x80 more text')
            temp_path = f.name
        
        try:
            result = parse_medical_record(temp_path)
            assert isinstance(result, str)
            assert "Valid text" in result
        finally:
            os.unlink(temp_path)

    def test_parse_clinical_note_error_scenarios(self):
        """Test clinical note parsing with various error scenarios."""
        # Test with None input
        with pytest.raises(ParsingError, match="Input must be a string"):
            parse_clinical_note(None)
        
        # Test with integer input
        with pytest.raises(ParsingError, match="Input must be a string"):
            parse_clinical_note(123)

    def test_parse_insurance_form_network_path(self):
        """Test parsing with network/UNC path."""
        # This should handle network paths gracefully
        if os.name == 'nt':  # Windows
            with pytest.raises(OSError):
                parse_insurance_form("\\\\nonexistent\\share\\file.txt")


class TestDocumentTypeDetectionErrorHandling:
    """Test error handling in document type detection."""

    def test_detect_document_type_with_none(self):
        """Test document type detection with None input."""
        with pytest.raises(DocumentTypeError, match="Cannot detect document type from None"):
            detect_document_type(None)

    def test_detect_document_type_with_integer(self):
        """Test document type detection with integer input."""
        with pytest.raises(DocumentTypeError, match="Input must be a string or Path object"):
            detect_document_type(123)

    def test_detect_document_type_with_empty_string(self):
        """Test document type detection with empty string."""
        result = detect_document_type("")
        assert result == DocumentType.UNKNOWN

    def test_detect_document_type_with_very_long_path(self):
        """Test document type detection with extremely long path."""
        long_path = "a" * 10000 + "/file.txt"
        result = detect_document_type(long_path)
        assert result == DocumentType.UNKNOWN

    def test_detect_document_type_with_special_characters(self):
        """Test document type detection with special characters."""
        special_path = "file_with_\x00_null_byte.txt"
        result = detect_document_type(special_path)
        assert result == DocumentType.UNKNOWN

    def test_detect_document_type_with_unicode(self):
        """Test document type detection with Unicode characters."""
        unicode_path = "файл_медицинской_записи.txt"
        result = detect_document_type(unicode_path)
        assert result == DocumentType.UNKNOWN  # Since keywords are in English


class TestProcessorErrorHandling:
    """Test error handling in processor components."""

    def test_processor_with_corrupted_config(self):
        """Test processor behavior with corrupted configuration."""
        processor = HIPAAProcessor()
        
        # Test with text that would cause processing errors
        problematic_text = "\x00" * 1000  # Null bytes
        with pytest.raises(RuntimeError, match="null bytes"):
            processor.process_document(problematic_text)

    def test_processor_with_memory_exhaustion_simulation(self):
        """Test processor behavior under memory pressure."""
        processor = HIPAAProcessor()
        
        # Simulate memory exhaustion scenario
        very_large_text = "Medical record " * 1000000  # Very large input
        
        # Should handle gracefully or raise appropriate error
        try:
            result = processor.process_document(very_large_text)
            assert result is not None
        except (MemoryError, RuntimeError) as e:
            # Either should be acceptable
            assert "memory" in str(e).lower() or "too large" in str(e).lower()

    def test_processor_with_malformed_document_object(self):
        """Test processor with malformed Document objects."""
        from hipaa_compliance_summarizer.documents import DocumentError
        
        # Test that Document validation catches empty path
        with pytest.raises(DocumentError, match="Document path cannot be empty"):
            # This should fail at Document creation time
            Document("", DocumentType.MEDICAL_RECORD)


class TestBatchProcessorErrorHandling:
    """Test error handling in batch processing."""

    def test_batch_processor_with_mixed_file_errors(self):
        """Test batch processing with mix of valid and problematic files."""
        processor = BatchProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mix of files
            files_to_create = [
                ("valid.txt", "Patient: John Doe, SSN: 123-45-6789"),
                ("empty.txt", ""),
                ("binary.dat", b"\xff\xfe\x00\x00"),  # Binary content
                ("large.txt", "Data " * 100000),  # Large file
            ]
            
            for filename, content in files_to_create:
                file_path = os.path.join(temp_dir, filename)
                if isinstance(content, bytes):
                    with open(file_path, 'wb') as f:
                        f.write(content)
                else:
                    with open(file_path, 'w') as f:
                        f.write(content)
            
            # Process directory - should handle errors gracefully
            results = processor.process_directory(temp_dir)
            
            # Should have results for all files, some successful, some errors
            assert len(results) == 4
            
            # Check that we have both success and error results
            successful = [r for r in results if hasattr(r, 'compliance_score')]
            errors = [r for r in results if hasattr(r, 'error')]
            
            assert len(successful) >= 1  # At least valid.txt should succeed
            assert len(errors) >= 1  # At least binary.dat should fail

    def test_batch_processor_with_permission_errors(self):
        """Test batch processing with permission-denied scenarios."""
        processor = BatchProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file
            file_path = os.path.join(temp_dir, "test.txt")
            with open(file_path, 'w') as f:
                f.write("Test data")
            
            # Mock the processor to simulate permission error
            with patch.object(processor.processor, 'process_document', 
                             side_effect=PermissionError("Permission denied")):
                results = processor.process_directory(temp_dir)
                
                # Should handle permission errors gracefully
                assert len(results) == 1
                assert hasattr(results[0], 'error')
                assert 'permission' in results[0].error.lower() or 'error' in results[0].error.lower()

    def test_batch_processor_with_network_interruption_simulation(self):
        """Test batch processing with simulated network/IO interruptions."""
        processor = BatchProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            file_path = os.path.join(temp_dir, "test.txt")
            with open(file_path, 'w') as f:
                f.write("Test content")
            
            # Mock the processor to simulate network interruption
            with patch.object(processor.processor, 'process_document', 
                             side_effect=OSError("Network unreachable")):
                results = processor.process_directory(temp_dir)
                
                # Should handle IO errors gracefully
                assert len(results) == 1
                assert hasattr(results[0], 'error')

    def test_batch_processor_with_invalid_worker_count(self):
        """Test batch processor with invalid worker configurations."""
        processor = BatchProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            file_path = os.path.join(temp_dir, "test.txt")
            with open(file_path, 'w') as f:
                f.write("Test content")
            
            # Test with invalid max_workers
            with pytest.raises(ValueError, match="max_workers must be positive"):
                processor.process_directory(temp_dir, max_workers=0)
            
            with pytest.raises(ValueError, match="max_workers must be positive"):
                processor.process_directory(temp_dir, max_workers=-1)


class TestErrorRecoveryMechanisms:
    """Test error recovery and fallback mechanisms."""

    def test_graceful_degradation_with_partial_failures(self):
        """Test that system continues operating with partial failures."""
        processor = HIPAAProcessor()
        
        # Test that one failure doesn't affect subsequent operations
        try:
            processor.process_document("\x00invalid")
        except RuntimeError as e:
            # Expected to fail - log for testing verification
            assert "processing failed" in str(e).lower()
        
        # Next operation should work fine
        result = processor.process_document("Valid medical text")
        assert result is not None
        assert result.compliance_score >= 0

    def test_cache_recovery_after_errors(self):
        """Test that caches can recover after errors."""
        processor = BatchProcessor()
        
        # Force an error that might corrupt cache state
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a problematic file
            file_path = os.path.join(temp_dir, "problem.txt")
            with open(file_path, 'wb') as f:
                f.write(b'\xff\xfe\x00\x00')  # Binary content
            
            # Process (should handle gracefully)
            results = processor.process_directory(temp_dir)
            
            # Clear and verify cache still works
            processor.clear_cache()
            cache_info = processor.get_cache_performance()
            
            # Cache should be operational
            assert cache_info is not None
            assert 'phi_detection' in cache_info

    def test_resource_cleanup_after_errors(self):
        """Test that resources are properly cleaned up after errors."""
        processor = BatchProcessor()
        
        # Get initial cache state
        initial_cache = processor.get_file_cache_info()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files that will cause various errors
            for i in range(5):
                file_path = os.path.join(temp_dir, f"test_{i}.txt")
                with open(file_path, 'w') as f:
                    f.write(f"Test content {i}")
            
            # Process directory
            results = processor.process_directory(temp_dir)
            
            # Cache should be managed properly
            final_cache = processor.get_file_cache_info()
            assert final_cache['file_cache_size'] <= final_cache['file_cache_max_size']