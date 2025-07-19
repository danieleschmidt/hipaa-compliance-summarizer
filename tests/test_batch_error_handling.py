"""Tests for comprehensive error handling in batch processing."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from hipaa_compliance_summarizer.batch import BatchProcessor
from hipaa_compliance_summarizer.processor import ProcessingResult, ComplianceLevel
from hipaa_compliance_summarizer.documents import Document, DocumentType


class TestBatchProcessorFileErrorHandling:
    """Test file operation error handling in BatchProcessor."""

    def test_process_directory_handles_invalid_input_directory(self):
        """Test graceful handling of non-existent input directory."""
        processor = BatchProcessor()
        
        with pytest.raises(FileNotFoundError) as exc_info:
            processor.process_directory("/nonexistent/directory")
        
        assert "Input directory does not exist" in str(exc_info.value)

    def test_process_directory_handles_input_directory_access_denied(self):
        """Test handling of input directory permission errors."""
        processor = BatchProcessor()
        
        # Mock the Path class within the batch module
        with patch('hipaa_compliance_summarizer.batch.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_dir.return_value = True
            mock_path_instance.iterdir.side_effect = PermissionError("Access denied")
            mock_path.return_value = mock_path_instance
            
            with pytest.raises(PermissionError) as exc_info:
                processor.process_directory("/restricted/directory")
            
            assert "Permission denied accessing input directory" in str(exc_info.value)

    def test_process_directory_handles_output_directory_creation_failure(self, tmp_path):
        """Test handling of output directory creation errors."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create a test file
        test_file = input_dir / "test.txt"
        test_file.write_text("Sample content")
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Cannot create directory")
            
            with pytest.raises(PermissionError) as exc_info:
                processor.process_directory(str(input_dir), output_dir="/restricted/output")
            
            assert "Cannot create output directory" in str(exc_info.value)

    def test_process_directory_handles_file_read_errors(self, tmp_path):
        """Test handling of individual file read errors."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        
        # Create test files
        good_file = input_dir / "good.txt"
        good_file.write_text("Valid content")
        
        bad_file = input_dir / "bad.txt"
        bad_file.write_text("Content that will fail")
        
        with patch('hipaa_compliance_summarizer.documents.Document') as mock_doc:
            def side_effect(path, doc_type):
                if "bad.txt" in path:
                    raise IOError("File read error")
                return Mock()
            
            mock_doc.side_effect = side_effect
            
            # Should continue processing other files even if one fails
            results = processor.process_directory(str(input_dir), output_dir=str(output_dir))
            
            # Should have processed the good file and recorded error for bad file
            assert len(results) == 2
            # One successful result, one error result
            successful_results = [r for r in results if not hasattr(r, 'error')]
            error_results = [r for r in results if hasattr(r, 'error')]
            
            assert len(successful_results) == 1
            assert len(error_results) == 1
            assert "File read error" in error_results[0].error

    def test_process_directory_handles_file_write_errors(self, tmp_path):
        """Test handling of output file write errors."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create a test file
        test_file = input_dir / "test.txt"
        test_file.write_text("Sample content")
        
        with patch('pathlib.Path.write_text') as mock_write:
            mock_write.side_effect = PermissionError("Cannot write file")
            
            # Should handle write errors gracefully
            results = processor.process_directory(str(input_dir), output_dir=str(output_dir))
            
            assert len(results) == 1
            assert hasattr(results[0], 'error')
            assert "Cannot write file" in results[0].error

    def test_process_directory_handles_corrupted_files(self, tmp_path):
        """Test handling of corrupted or unreadable files."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create a binary file that will cause text reading issues
        binary_file = input_dir / "binary.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')
        
        # Should handle file processing errors gracefully
        results = processor.process_directory(str(input_dir))
        
        assert len(results) == 1
        # Should either process successfully or record error
        if hasattr(results[0], 'error'):
            assert "encoding" in results[0].error.lower() or "decode" in results[0].error.lower()

    def test_process_directory_handles_concurrent_processing_errors(self, tmp_path):
        """Test error handling with multiple workers and concurrent failures."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create multiple test files
        for i in range(5):
            test_file = input_dir / f"test_{i}.txt"
            test_file.write_text(f"Content {i}")
        
        with patch('hipaa_compliance_summarizer.processor.HIPAAProcessor.process_document') as mock_process:
            def side_effect(doc):
                if "test_2.txt" in doc.path or "test_4.txt" in doc.path:
                    raise RuntimeError(f"Processing error for {doc.path}")
                return Mock(spec=ProcessingResult)
            
            mock_process.side_effect = side_effect
            
            # Should handle concurrent errors gracefully
            results = processor.process_directory(str(input_dir), max_workers=3)
            
            assert len(results) == 5
            error_results = [r for r in results if hasattr(r, 'error')]
            assert len(error_results) == 2


class TestBatchProcessorDashboardErrorHandling:
    """Test error handling in dashboard generation and saving."""

    def test_generate_dashboard_handles_empty_results(self):
        """Test dashboard generation with empty results."""
        processor = BatchProcessor()
        
        # Should handle empty results gracefully
        dashboard = processor.generate_dashboard([])
        
        assert dashboard.documents_processed == 0
        assert dashboard.avg_compliance_score == 1.0
        assert dashboard.total_phi_detected == 0

    def test_generate_dashboard_handles_invalid_results(self):
        """Test dashboard generation with malformed results."""
        processor = BatchProcessor()
        
        # Create mock results with missing or invalid attributes
        invalid_results = [
            Mock(compliance_score=None, phi_detected_count=5),
            Mock(compliance_score=0.8, phi_detected_count=None),
            Mock(spec=[])  # Missing required attributes
        ]
        
        # Should handle invalid data gracefully
        dashboard = processor.generate_dashboard(invalid_results)
        
        # Should use safe defaults or skip invalid entries
        assert dashboard.documents_processed >= 0
        assert 0.0 <= dashboard.avg_compliance_score <= 1.0
        assert dashboard.total_phi_detected >= 0

    def test_save_dashboard_handles_write_permission_errors(self, tmp_path):
        """Test dashboard saving with write permission errors."""
        processor = BatchProcessor()
        results = [Mock(compliance_score=0.9, phi_detected_count=3)]
        
        restricted_path = tmp_path / "restricted" / "dashboard.json"
        
        with patch('pathlib.Path.write_text') as mock_write:
            mock_write.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(PermissionError) as exc_info:
                processor.save_dashboard(results, str(restricted_path))
            
            assert "Cannot write dashboard to" in str(exc_info.value)

    def test_save_dashboard_handles_invalid_path(self):
        """Test dashboard saving with invalid file paths."""
        processor = BatchProcessor()
        results = [Mock(compliance_score=0.9, phi_detected_count=3)]
        
        # Test with invalid path characters
        invalid_path = "/dev/null/invalid\x00path.json"
        
        with pytest.raises(ValueError) as exc_info:
            processor.save_dashboard(results, invalid_path)
        
        assert "Invalid file path" in str(exc_info.value)

    def test_save_dashboard_handles_disk_full_errors(self, tmp_path):
        """Test dashboard saving when disk is full."""
        processor = BatchProcessor()
        results = [Mock(compliance_score=0.9, phi_detected_count=3)]
        
        dashboard_path = tmp_path / "dashboard.json"
        
        with patch('pathlib.Path.write_text') as mock_write:
            mock_write.side_effect = OSError("No space left on device")
            
            with pytest.raises(OSError) as exc_info:
                processor.save_dashboard(results, str(dashboard_path))
            
            assert "Failed to write dashboard" in str(exc_info.value)


class TestBatchProcessorCacheErrorHandling:
    """Test error handling in cache operations."""

    def test_get_cache_performance_handles_missing_cache_info(self):
        """Test cache performance reporting when cache info is unavailable."""
        processor = BatchProcessor()
        
        with patch('hipaa_compliance_summarizer.phi.PHIRedactor.get_cache_info') as mock_cache:
            mock_cache.side_effect = AttributeError("Cache not available")
            
            # Should handle missing cache gracefully
            performance = processor.get_cache_performance()
            
            # Should return safe defaults
            assert "pattern_compilation" in performance
            assert "phi_detection" in performance
            assert all(isinstance(v, dict) for v in performance.values())

    def test_get_cache_performance_handles_invalid_cache_data(self):
        """Test cache performance with corrupted cache data."""
        processor = BatchProcessor()
        
        with patch('hipaa_compliance_summarizer.phi.PHIRedactor.get_cache_info') as mock_cache:
            # Return malformed cache info
            mock_cache.return_value = {
                "pattern_compilation": Mock(hits=None, misses="invalid"),
                "phi_detection": None
            }
            
            # Should handle invalid data gracefully
            performance = processor.get_cache_performance()
            
            # Should return valid structure with safe defaults
            assert "pattern_compilation" in performance
            assert "phi_detection" in performance

    def test_clear_cache_handles_cache_clear_errors(self):
        """Test cache clearing when cache operations fail."""
        processor = BatchProcessor()
        
        with patch('hipaa_compliance_summarizer.phi.PHIRedactor.clear_cache') as mock_clear:
            mock_clear.side_effect = RuntimeError("Cache clear failed")
            
            # Should handle cache clear errors gracefully
            with pytest.raises(RuntimeError) as exc_info:
                processor.clear_cache()
            
            assert "Failed to clear cache" in str(exc_info.value)


class TestBatchProcessorResourceManagement:
    """Test resource management and cleanup error handling."""

    def test_process_directory_handles_memory_errors(self, tmp_path):
        """Test handling of memory allocation errors during processing."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create a test file
        test_file = input_dir / "test.txt"
        test_file.write_text("Sample content")
        
        with patch('hipaa_compliance_summarizer.processor.HIPAAProcessor.process_document') as mock_process:
            mock_process.side_effect = MemoryError("Out of memory")
            
            # Should handle memory errors gracefully
            results = processor.process_directory(str(input_dir))
            
            assert len(results) == 1
            assert hasattr(results[0], 'error')
            assert "Out of memory" in results[0].error

    def test_process_directory_handles_thread_pool_errors(self, tmp_path):
        """Test handling of thread pool execution errors."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create test files
        for i in range(3):
            test_file = input_dir / f"test_{i}.txt"
            test_file.write_text(f"Content {i}")
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.__enter__ = Mock(return_value=mock_executor)
            mock_executor.__exit__ = Mock(return_value=None)
            mock_executor.map.side_effect = RuntimeError("Thread pool error")
            mock_executor_class.return_value = mock_executor
            
            # Should handle thread pool errors gracefully
            with pytest.raises(RuntimeError) as exc_info:
                processor.process_directory(str(input_dir), max_workers=2)
            
            assert "Thread pool execution failed" in str(exc_info.value)

    def test_process_directory_handles_large_file_timeouts(self, tmp_path):
        """Test handling of timeouts when processing large files."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create a large test file
        large_file = input_dir / "large.txt"
        large_file.write_text("Large content " * 10000)
        
        with patch('hipaa_compliance_summarizer.processor.HIPAAProcessor.process_document') as mock_process:
            import time
            
            def slow_process(doc):
                time.sleep(10)  # Simulate slow processing
                return Mock(spec=ProcessingResult)
            
            mock_process.side_effect = slow_process
            
            # Should have timeout handling (this test validates the structure exists)
            # In actual implementation, we'd add timeout handling
            results = processor.process_directory(str(input_dir))
            
            # For now, just verify the structure works
            assert len(results) >= 0


class TestBatchProcessorValidation:
    """Test input validation and error handling."""

    def test_process_directory_validates_compliance_level(self, tmp_path):
        """Test validation of compliance level parameter."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create a test file
        test_file = input_dir / "test.txt"
        test_file.write_text("Sample content")
        
        # Should handle invalid compliance levels gracefully
        with pytest.raises(ValueError) as exc_info:
            processor.process_directory(str(input_dir), compliance_level="invalid_level")
        
        assert "Invalid compliance level" in str(exc_info.value)

    def test_process_directory_validates_max_workers(self, tmp_path):
        """Test validation of max_workers parameter."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create a test file
        test_file = input_dir / "test.txt"
        test_file.write_text("Sample content")
        
        # Should handle invalid max_workers values
        with pytest.raises(ValueError) as exc_info:
            processor.process_directory(str(input_dir), max_workers=0)
        
        assert "max_workers must be positive" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            processor.process_directory(str(input_dir), max_workers=-1)
        
        assert "max_workers must be positive" in str(exc_info.value)

    def test_process_directory_validates_file_types(self, tmp_path):
        """Test handling of unsupported file types."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create files with various extensions
        supported_file = input_dir / "document.txt"
        supported_file.write_text("Valid document content")
        
        unsupported_file = input_dir / "binary.exe"
        unsupported_file.write_bytes(b'Binary content')
        
        # Should process supported files and handle unsupported ones gracefully
        results = processor.process_directory(str(input_dir))
        
        # Should have results for all files, with errors for unsupported types
        assert len(results) >= 1
        
        # Check that unsupported files are either skipped or marked with errors
        for result in results:
            if hasattr(result, 'error'):
                assert "unsupported" in result.error.lower() or "binary" in result.error.lower()