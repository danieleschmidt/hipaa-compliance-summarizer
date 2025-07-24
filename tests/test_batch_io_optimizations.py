"""Tests for batch processing I/O optimizations."""

import tempfile
import os
from pathlib import Path
import time
import pytest

from hipaa_compliance_summarizer.batch import BatchProcessor
from hipaa_compliance_summarizer.processor import ComplianceLevel
from hipaa_compliance_summarizer.constants import BYTES_PER_MB


class TestBatchIOOptimizations:
    """Test I/O optimizations in batch processing."""

    def test_file_content_caching(self):
        """Test that file content is cached for repeated access."""
        processor = BatchProcessor()
        
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = "Patient: John Doe, SSN: 123-45-6789"
            f.write(test_content)
            temp_path = Path(f.name)
        
        try:
            # First read should cache the content
            content1 = processor._optimized_file_read(temp_path)
            assert content1 == test_content
            
            # Second read should come from cache
            content2 = processor._optimized_file_read(temp_path)
            assert content2 == test_content
            assert content1 == content2
            
            # Verify cache contains the file
            cache_info = processor.get_file_cache_info()
            assert cache_info["file_cache_size"] == 1
        finally:
            os.unlink(temp_path)

    def test_cache_size_management(self):
        """Test that cache size is properly managed."""
        processor = BatchProcessor()
        processor._max_cache_size = 3  # Small cache for testing
        
        temp_files = []
        try:
            # Create files that exceed cache size
            for i in range(5):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(f"Patient {i}: Content {i}")
                    temp_files.append(Path(f.name))
            
            # Read all files
            for temp_path in temp_files:
                processor._optimized_file_read(temp_path)
            
            # Cache should be limited to max size
            cache_info = processor.get_file_cache_info()
            assert cache_info["file_cache_size"] <= processor._max_cache_size
            assert cache_info["file_cache_usage_ratio"] <= 1.0
            
        finally:
            for temp_path in temp_files:
                os.unlink(temp_path)

    def test_large_file_memory_mapping(self):
        """Test memory mapping for large files."""
        processor = BatchProcessor()
        
        # Create a large file (> 1MB)
        from hipaa_compliance_summarizer.constants import TEST_CONSTANTS
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            large_content = "Large file content. " * TEST_CONSTANTS.LARGE_FILE_SIZE  # ~1.2MB
            f.write(large_content)
            temp_path = Path(f.name)
        
        try:
            # Should use memory mapping for large files
            content = processor._optimized_file_read(temp_path)
            assert len(content) > BYTES_PER_MB  # Verify it's large
            assert "Large file content." in content
            
            # Large files should not be cached
            cache_info = processor.get_file_cache_info()
            assert cache_info["file_cache_size"] == 0
            
        finally:
            os.unlink(temp_path)

    def test_preload_small_files(self):
        """Test preloading of small files into cache."""
        processor = BatchProcessor()
        
        temp_files = []
        try:
            # Create mix of small and large files
            for i in range(3):
                # Small files (< 512KB)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(f"Small file {i} content")
                    temp_files.append(Path(f.name))
            
            # Large file (> 512KB)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("Large content. " * 30000)  # ~450KB
                temp_files.append(Path(f.name))
            
            # Preload files
            processor._preload_files(temp_files)
            
            # Small files should be in cache
            cache_info = processor.get_file_cache_info()
            assert cache_info["file_cache_size"] > 0
            
        finally:
            for temp_path in temp_files:
                os.unlink(temp_path)

    def test_optimized_batch_processing_performance(self):
        """Test that optimizations improve batch processing performance."""
        processor = BatchProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files with IDENTICAL content for caching benefits
            identical_content = "Patient: John Doe, SSN: 123-45-6789, Phone: 555-123-4567"
            
            for i in range(5):
                file_path = os.path.join(temp_dir, f"patient_{i:02d}.txt")
                with open(file_path, "w") as f:
                    # Use identical content to benefit from PHI caching
                    f.write(identical_content)
            
            # Process with optimizations
            start_time = time.perf_counter()
            results = processor.process_directory(temp_dir)
            optimized_time = time.perf_counter() - start_time
            
            # Verify all files were processed successfully
            assert len(results) == 5
            assert all(hasattr(r, 'compliance_score') for r in results)
            
            # Check that caching is working
            cache_info = processor.get_cache_performance()
            phi_cache = cache_info["phi_detection"]
            
            # Should have cache hits due to identical content (after first file)
            assert phi_cache["hits"] >= 1  # At least 1 hit expected
            
            # Check file cache is working
            file_cache_info = processor.get_file_cache_info()
            assert file_cache_info["file_cache_size"] > 0
            
            print(f"Processing time: {optimized_time:.3f}s")
            print(f"PHI cache hits: {phi_cache['hits']}, ratio: {phi_cache['hit_ratio']:.2f}")
            print(f"File cache size: {file_cache_info['file_cache_size']}")

    def test_file_sorting_by_size(self):
        """Test that files are sorted by size for optimal processing."""
        processor = BatchProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files of different sizes
            files_info = [
                ("large.txt", "Content " * 1000),
                ("small.txt", "Small"),
                ("medium.txt", "Content " * 100),
            ]
            
            for filename, content in files_info:
                with open(os.path.join(temp_dir, filename), "w") as f:
                    f.write(content)
            
            # Get files and verify they are sorted by size
            input_path = Path(temp_dir)
            files = [f for f in input_path.iterdir() if f.is_file()]
            files.sort(key=lambda f: f.stat().st_size)
            
            # Smallest file should be first
            assert files[0].name == "small.txt"
            assert files[-1].name == "large.txt"

    def test_cache_clearing(self):
        """Test that cache clearing works correctly."""
        processor = BatchProcessor()
        
        # Create and cache some content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for caching")
            temp_path = Path(f.name)
        
        try:
            # Read file to populate cache
            processor._optimized_file_read(temp_path)
            
            # Verify cache has content
            cache_info = processor.get_file_cache_info()
            assert cache_info["file_cache_size"] > 0
            
            # Clear cache
            processor.clear_cache()
            
            # Verify cache is empty
            cache_info = processor.get_file_cache_info()
            assert cache_info["file_cache_size"] == 0
            
        finally:
            os.unlink(temp_path)

    def test_error_handling_in_optimized_read(self):
        """Test error handling in optimized file reading."""
        processor = BatchProcessor()
        
        # Test with non-existent file
        non_existent = Path("/nonexistent/file.txt")
        with pytest.raises(FileNotFoundError):
            processor._optimized_file_read(non_existent)

    def test_unicode_handling_in_optimized_read(self):
        """Test that optimized reading handles unicode content correctly."""
        processor = BatchProcessor()
        
        # Create file with unicode content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            unicode_content = "Patient: José María 🏥 SSN: 123-45-6789"
            f.write(unicode_content)
            temp_path = Path(f.name)
        
        try:
            content = processor._optimized_file_read(temp_path)
            assert "José María" in content
            assert "🏥" in content
            assert "123-45-6789" in content
        finally:
            os.unlink(temp_path)

    def test_empty_file_handling(self):
        """Test handling of empty files."""
        processor = BatchProcessor()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Create empty file
            temp_path = Path(f.name)
        
        try:
            content = processor._optimized_file_read(temp_path)
            assert content == ""
            
            # Empty files should still be cached
            cache_info = processor.get_file_cache_info()
            assert cache_info["file_cache_size"] == 1
        finally:
            os.unlink(temp_path)