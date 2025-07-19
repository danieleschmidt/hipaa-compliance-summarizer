import tempfile
import os
from hipaa_compliance_summarizer.batch import BatchProcessor
from hipaa_compliance_summarizer.phi import PHIRedactor


def test_batch_processor_cache_performance():
    """Test that batch processor can report cache performance metrics."""
    
    # Clear caches to start fresh
    PHIRedactor.clear_cache()
    
    processor = BatchProcessor()
    
    # Get initial performance (after processor creation)
    initial_performance = processor.get_cache_performance()
    initial_pattern_total = initial_performance["pattern_compilation"]["hits"] + initial_performance["pattern_compilation"]["misses"]
    initial_phi_total = initial_performance["phi_detection"]["hits"] + initial_performance["phi_detection"]["misses"]
    
    # Create test files with duplicate content
    with tempfile.TemporaryDirectory() as temp_dir:
        content = "Patient John Doe, SSN: 123-45-6789, Phone: 555-123-4567"
        
        # Create multiple files with identical content
        for i in range(3):
            file_path = os.path.join(temp_dir, f"patient_{i}.txt")
            with open(file_path, "w") as f:
                f.write(content)
        
        # Process directory
        results = processor.process_directory(temp_dir)
        
        # Check cache performance after processing
        final_performance = processor.get_cache_performance()
        
        # Should have processed files
        assert len(results) == 3
        
        # Should have increased cache activity
        final_pattern_total = final_performance["pattern_compilation"]["hits"] + final_performance["pattern_compilation"]["misses"]
        final_phi_total = final_performance["phi_detection"]["hits"] + final_performance["phi_detection"]["misses"]
        
        # Should have more activity after processing
        assert final_pattern_total >= initial_pattern_total
        assert final_phi_total >= initial_phi_total
        
        # Should have PHI detection activity (duplicate content should show cache hits)
        assert final_phi_total > initial_phi_total
        
        # Hit ratios should be calculable
        assert 0 <= final_performance["pattern_compilation"]["hit_ratio"] <= 1
        assert 0 <= final_performance["phi_detection"]["hit_ratio"] <= 1


def test_batch_processor_cache_clearing():
    """Test that batch processor can clear caches."""
    
    # Process some data to populate caches
    processor = BatchProcessor()
    redactor = PHIRedactor()
    
    # Do some detection to populate cache
    redactor.detect("Patient SSN: 123-45-6789")
    redactor.detect("Patient Phone: 555-123-4567")
    
    # Should have cache entries
    performance_before = processor.get_cache_performance()
    total_entries_before = (
        performance_before["pattern_compilation"]["current_size"] +
        performance_before["phi_detection"]["current_size"]
    )
    assert total_entries_before > 0
    
    # Clear cache
    processor.clear_cache()
    
    # Should have no cache entries
    performance_after = processor.get_cache_performance()
    assert performance_after["pattern_compilation"]["current_size"] == 0
    assert performance_after["phi_detection"]["current_size"] == 0
    assert performance_after["pattern_compilation"]["hits"] == 0
    assert performance_after["pattern_compilation"]["misses"] == 0
    assert performance_after["phi_detection"]["hits"] == 0
    assert performance_after["phi_detection"]["misses"] == 0


def test_cache_performance_with_many_unique_texts():
    """Test cache behavior with many unique text inputs."""
    
    PHIRedactor.clear_cache()
    processor = BatchProcessor()
    
    # Create many files with unique SSNs
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(10):
            ssn = f"{123+i:03d}-{45+i:02d}-{6789+i:04d}"
            content = f"Patient SSN: {ssn}"
            
            file_path = os.path.join(temp_dir, f"patient_{i}.txt")
            with open(file_path, "w") as f:
                f.write(content)
        
        # Process directory
        results = processor.process_directory(temp_dir)
        
        # Check results
        assert len(results) == 10
        
        # Each file should have detected one SSN
        for result in results:
            assert result.phi_detected_count == 1
        
        # Check cache performance
        performance = processor.get_cache_performance()
        
        # Should have pattern cache hits (patterns reused)
        assert performance["pattern_compilation"]["hits"] > 0
        
        # PHI detection should mostly be misses (unique content)
        phi_misses = performance["phi_detection"]["misses"]
        assert phi_misses > 0  # Each unique text is a cache miss


def test_cache_performance_with_duplicate_content():
    """Test cache behavior with duplicate content."""
    
    PHIRedactor.clear_cache()
    processor = BatchProcessor()
    
    # Create many files with identical content
    with tempfile.TemporaryDirectory() as temp_dir:
        content = "Patient SSN: 123-45-6789, Phone: 555-123-4567"
        
        for i in range(10):
            file_path = os.path.join(temp_dir, f"duplicate_{i}.txt")
            with open(file_path, "w") as f:
                f.write(content)
        
        # Process directory
        results = processor.process_directory(temp_dir)
        
        # Check results
        assert len(results) == 10
        
        # All files should have identical results
        base_result = results[0]
        for result in results[1:]:
            assert result.phi_detected_count == base_result.phi_detected_count
            assert abs(result.compliance_score - base_result.compliance_score) < 0.01
        
        # Check cache performance
        performance = processor.get_cache_performance()
        
        # Should have high hit ratio for PHI detection (duplicate content)
        phi_hit_ratio = performance["phi_detection"]["hit_ratio"]
        if performance["phi_detection"]["hits"] + performance["phi_detection"]["misses"] > 1:
            assert phi_hit_ratio > 0.5  # Should have significant cache hits


def test_concurrent_batch_processing_cache_sharing():
    """Test that concurrent batch processing shares caches effectively."""
    import concurrent.futures
    
    PHIRedactor.clear_cache()
    
    def process_batch(batch_id):
        processor = BatchProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use same content across all batches to test cache sharing
            content = f"Batch {batch_id}: Patient SSN: 123-45-6789"
            
            file_path = os.path.join(temp_dir, f"batch_{batch_id}.txt")
            with open(file_path, "w") as f:
                f.write(content)
            
            results = processor.process_directory(temp_dir)
            return len(results), processor.get_cache_performance()
    
    # Process multiple batches concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_batch, i) for i in range(5)]
        batch_results = [future.result() for future in futures]
    
    # All batches should have processed files
    for count, performance in batch_results:
        assert count == 1  # One file per batch
    
    # Check final cache state
    final_processor = BatchProcessor()
    final_performance = final_processor.get_cache_performance()
    
    # Should have cache activity from all batches
    assert final_performance["pattern_compilation"]["hits"] + final_performance["pattern_compilation"]["misses"] > 0
    assert final_performance["phi_detection"]["hits"] + final_performance["phi_detection"]["misses"] > 0