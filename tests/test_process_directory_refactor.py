"""Test suite for process_directory refactoring."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from hipaa_compliance_summarizer.batch import BatchProcessor
from hipaa_compliance_summarizer.processor import ComplianceLevel


def test_validate_and_setup_workers():
    """Test worker validation and setup functionality."""
    processor = BatchProcessor()
    
    # Test auto-detection
    workers = processor._validate_and_setup_workers(None)
    assert workers > 0
    
    # Test explicit value
    workers = processor._validate_and_setup_workers(4)
    assert workers == 4
    
    # Test invalid value
    with pytest.raises(ValueError, match="max_workers must be positive"):
        processor._validate_and_setup_workers(0)


def test_validate_compliance_level():
    """Test compliance level validation."""
    processor = BatchProcessor()
    
    # Test valid compliance level
    processor._validate_compliance_level("standard")
    assert processor.processor.compliance_level == ComplianceLevel.STANDARD
    
    # Test None (should not change current level)
    processor._validate_compliance_level(None)
    
    # Test invalid compliance level
    with pytest.raises(ValueError, match="Invalid compliance level"):
        processor._validate_compliance_level("invalid")


def test_validate_input_directory(tmp_path):
    """Test input directory validation."""
    processor = BatchProcessor()
    
    # Test valid directory
    result = processor._validate_input_directory(str(tmp_path))
    assert result == tmp_path
    
    # Test non-existent directory
    with pytest.raises(FileNotFoundError):
        processor._validate_input_directory(str(tmp_path / "nonexistent"))
    
    # Test file instead of directory
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    with pytest.raises(ValueError, match="not a directory"):
        processor._validate_input_directory(str(test_file))


def test_setup_output_directory(tmp_path):
    """Test output directory setup."""
    processor = BatchProcessor()
    
    # Test None output directory
    result = processor._setup_output_directory(None)
    assert result is None
    
    # Test valid output directory
    output_dir = tmp_path / "output"
    result = processor._setup_output_directory(str(output_dir))
    assert result == output_dir
    assert output_dir.exists()
    
    # Test existing directory
    result = processor._setup_output_directory(str(output_dir))
    assert result == output_dir


def test_collect_files_to_process(tmp_path):
    """Test file collection functionality."""
    processor = BatchProcessor()
    
    # Create test files
    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")
    (tmp_path / "subdir").mkdir()
    
    files = processor._collect_files_to_process(tmp_path)
    assert len(files) == 2
    assert all(f.is_file() for f in files)
    assert all(f.name.endswith('.txt') for f in files)


def test_optimize_processing_setup(tmp_path):
    """Test processing optimization."""
    processor = BatchProcessor()
    
    # Create files of different sizes
    (tmp_path / "small.txt").write_text("small")
    (tmp_path / "large.txt").write_text("large content " * 100)
    
    files = [tmp_path / "large.txt", tmp_path / "small.txt"]
    optimized_files, workers = processor._optimize_processing_setup(files, 4)
    
    # Should be sorted by size (small first)
    assert optimized_files[0].name == "small.txt"
    assert optimized_files[1].name == "large.txt"
    
    # Workers should be adjusted for small file count
    assert workers <= len(files)


def test_execute_file_processing(tmp_path):
    """Test file processing execution."""
    processor = BatchProcessor()
    
    # Create test files
    (tmp_path / "test1.txt").write_text("Patient data")
    (tmp_path / "test2.txt").write_text("Medical record")
    
    files = [tmp_path / "test1.txt", tmp_path / "test2.txt"]
    results = processor._execute_file_processing(
        files, None, False, False, 1
    )
    
    assert len(results) == 2


def test_process_directory_integration(tmp_path):
    """Test full process_directory integration after refactoring."""
    # Create test files
    (tmp_path / "patient1.txt").write_text("Patient: John Doe, SSN: 123-45-6789")
    (tmp_path / "patient2.txt").write_text("Medical record for Jane Smith")
    
    processor = BatchProcessor()
    results = processor.process_directory(
        str(tmp_path),
        output_dir=None,
        compliance_level="standard",
        generate_summaries=False,
        show_progress=False,
        max_workers=2
    )
    
    assert len(results) == 2
    # Results should still work exactly as before
    dashboard = processor.generate_dashboard(results)
    assert dashboard.documents_processed == 2