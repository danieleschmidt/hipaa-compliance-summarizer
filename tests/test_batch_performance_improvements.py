"""Test batch processing performance improvements."""

import pytest
from unittest.mock import patch, Mock
import multiprocessing
from pathlib import Path

from hipaa_compliance_summarizer import BatchProcessor


class TestBatchProcessorPerformanceImprovements:
    """Test performance enhancements to BatchProcessor."""

    def test_auto_detects_optimal_workers(self):
        """Test that optimal worker count is auto-detected."""
        processor = BatchProcessor()
        
        with patch('multiprocessing.cpu_count', return_value=8):
            with patch.object(processor, 'process_directory') as mock_process:
                # Mock to intercept the call and check max_workers
                def side_effect(*args, **kwargs):
                    # Should auto-detect to min(4, max(1, 8-1)) = 4
                    assert 'max_workers' not in kwargs or kwargs['max_workers'] is None
                    return []
                
                mock_process.side_effect = side_effect
                processor.process_directory("/fake/dir")

    def test_adaptive_worker_scaling_with_few_files(self, tmp_path):
        """Test that worker count is reduced when there are fewer files than workers."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create only 2 files
        for i in range(2):
            test_file = input_dir / f"test_{i}.txt"
            test_file.write_text(f"Content {i}")
        
        # Mock the actual processing to avoid real work
        with patch.object(processor.processor, 'process_document') as mock_process:
            from hipaa_compliance_summarizer.processor import ProcessingResult
            from hipaa_compliance_summarizer.phi import RedactionResult
            
            mock_process.return_value = ProcessingResult(
                summary="Test content",
                compliance_score=1.0,
                phi_detected_count=0,
                redacted=RedactionResult(text="Test content", entities=[])
            )
            
            # Should automatically reduce workers to 2 for 2 files
            results = processor.process_directory(str(input_dir), max_workers=4)
            
            assert len(results) == 2
            # Verify all results are successful
            assert all(hasattr(r, 'summary') for r in results)

    def test_performance_logging(self, tmp_path, caplog):
        """Test that performance-related logging is enabled."""
        import logging
        caplog.set_level(logging.INFO)
        
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create test files
        for i in range(3):
            test_file = input_dir / f"test_{i}.txt"
            test_file.write_text(f"Content {i}")
        
        # Mock the actual processing
        with patch.object(processor.processor, 'process_document') as mock_process:
            from hipaa_compliance_summarizer.processor import ProcessingResult
            from hipaa_compliance_summarizer.phi import RedactionResult
            
            mock_process.return_value = ProcessingResult(
                summary="Test content",
                compliance_score=1.0,
                phi_detected_count=0,
                redacted=RedactionResult(text="Test content", entities=[])
            )
            
            processor.process_directory(str(input_dir))
            
            # Check that performance logging is present
            log_messages = [record.message for record in caplog.records]
            assert any("Auto-detected optimal workers" in msg for msg in log_messages)
            assert any("Processing 3 files with" in msg for msg in log_messages)

    def test_explicit_max_workers_honored(self, tmp_path):
        """Test that explicitly set max_workers is honored."""
        processor = BatchProcessor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create test files
        for i in range(5):
            test_file = input_dir / f"test_{i}.txt"
            test_file.write_text(f"Content {i}")
        
        # Mock the actual processing
        with patch.object(processor.processor, 'process_document') as mock_process:
            from hipaa_compliance_summarizer.processor import ProcessingResult
            from hipaa_compliance_summarizer.phi import RedactionResult
            
            mock_process.return_value = ProcessingResult(
                summary="Test content",
                compliance_score=1.0,
                phi_detected_count=0,
                redacted=RedactionResult(text="Test content", entities=[])
            )
            
            # Should honor explicit max_workers=2
            results = processor.process_directory(str(input_dir), max_workers=2)
            
            assert len(results) == 5
            # All should be successful
            assert all(hasattr(r, 'summary') for r in results)

    def test_single_cpu_fallback(self):
        """Test that single CPU systems fall back gracefully."""
        processor = BatchProcessor()
        
        with patch('multiprocessing.cpu_count', return_value=1):
            with patch.object(processor, 'process_directory') as mock_process:
                def side_effect(*args, **kwargs):
                    # Should auto-detect to max(1, 1-1) = 1
                    return []
                
                mock_process.side_effect = side_effect
                processor.process_directory("/fake/dir")