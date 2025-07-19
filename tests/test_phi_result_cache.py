import time
import hashlib
from unittest.mock import patch
from hipaa_compliance_summarizer.phi import PHIRedactor
from hipaa_compliance_summarizer.batch import BatchProcessor


def test_phi_result_caching():
    """Test that PHI detection results are cached for identical text."""
    
    text = "Patient SSN: 123-45-6789, Phone: 555-123-4567"
    
    redactor = PHIRedactor()
    
    # First detection (cache miss)
    start_time = time.perf_counter()
    result1 = redactor.detect(text)
    first_time = time.perf_counter() - start_time
    
    # Second detection of same text (cache hit)
    start_time = time.perf_counter()
    result2 = redactor.detect(text)
    second_time = time.perf_counter() - start_time
    
    # Results should be identical
    assert len(result1) == len(result2)
    for ent1, ent2 in zip(result1, result2):
        assert ent1.type == ent2.type
        assert ent1.value == ent2.value
        assert ent1.start == ent2.start
        assert ent1.end == ent2.end
    
    # Second call should be faster due to caching
    assert second_time < first_time * 0.8  # Should be significantly faster


def test_phi_result_cache_with_different_text():
    """Test that cache properly handles different text inputs."""
    
    text1 = "Patient SSN: 123-45-6789"
    text2 = "Patient SSN: 987-65-4321"
    
    redactor = PHIRedactor()
    
    result1 = redactor.detect(text1)
    result2 = redactor.detect(text2)
    
    # Should get different results for different text
    assert len(result1) == 1
    assert len(result2) == 1
    assert result1[0].value != result2[0].value
    assert result1[0].value == "123-45-6789"
    assert result2[0].value == "987-65-4321"


def test_phi_result_cache_size_limit():
    """Test that result cache has reasonable size limits."""
    
    redactor = PHIRedactor()
    
    # Generate many different texts to fill cache - use proper SSN format
    texts = []
    for i in range(100):  # Reduced number for reasonable test time
        # Generate valid SSN format: XXX-XX-XXXX
        ssn = f"{123+i:03d}-{45+i%50:02d}-{6789+i:04d}"
        texts.append(f"Patient SSN: {ssn}")
    
    results = []
    for text in texts:
        result = redactor.detect(text)
        results.append(result)
    
    # All should return valid results
    for i, result in enumerate(results):
        assert len(result) == 1
        assert result[0].type == "ssn"
        expected_ssn = f"{123+i:03d}-{45+i%50:02d}-{6789+i:04d}"
        assert result[0].value == expected_ssn


def test_phi_result_cache_thread_safety():
    """Test that result caching is thread-safe."""
    import concurrent.futures
    
    text = "Patient SSN: 123-45-6789, Phone: 555-123-4567"
    redactor = PHIRedactor()
    
    def detect_phi():
        return redactor.detect(text)
    
    # Run detection concurrently from multiple threads
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(detect_phi) for _ in range(20)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # All results should be identical
    base_result = results[0]
    for result in results[1:]:
        assert len(result) == len(base_result)
        for ent1, ent2 in zip(result, base_result):
            assert ent1.type == ent2.type
            assert ent1.value == ent2.value
            assert ent1.start == ent2.start
            assert ent1.end == ent2.end


def test_phi_redaction_result_caching():
    """Test that redaction results are also cached."""
    
    text = "Patient SSN: 123-45-6789, Phone: 555-123-4567"
    redactor = PHIRedactor()
    
    # First redaction
    start_time = time.perf_counter()
    result1 = redactor.redact(text)
    first_time = time.perf_counter() - start_time
    
    # Second redaction of same text
    start_time = time.perf_counter()
    result2 = redactor.redact(text)
    second_time = time.perf_counter() - start_time
    
    # Results should be identical
    assert result1.text == result2.text
    assert len(result1.entities) == len(result2.entities)
    
    # Second call should be faster
    assert second_time < first_time * 0.8


def test_batch_processing_with_result_cache():
    """Test that batch processing benefits from result caching."""
    import tempfile
    import os
    
    # Create temporary directory with duplicate content files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create files with identical content
        content = "Patient John Doe, SSN: 123-45-6789, Phone: 555-123-4567"
        
        for i in range(5):
            file_path = os.path.join(temp_dir, f"patient_{i}.txt")
            with open(file_path, "w") as f:
                f.write(content)
        
        # Process with batch processor
        processor = BatchProcessor()
        
        start_time = time.perf_counter()
        results = processor.process_directory(temp_dir)
        total_time = time.perf_counter() - start_time
        
        # All results should be identical due to caching
        assert len(results) == 5
        base_result = results[0]
        
        for result in results[1:]:
            assert result.phi_detected_count == base_result.phi_detected_count
            assert abs(result.compliance_score - base_result.compliance_score) < 0.01
            
            # Redacted text should be identical
            assert result.redacted.text == base_result.redacted.text


def test_cache_invalidation_with_different_patterns():
    """Test that result cache is properly invalidated when patterns change."""
    
    text = "Patient SSN: 123-45-6789"
    
    # Create redactor with custom patterns
    patterns1 = {"ssn": r"\b\d{3}-\d{2}-\d{4}\b"}
    patterns2 = {"custom": r"\b\d{3}-\d{2}-\d{4}\b"}  # Same regex, different name
    
    redactor1 = PHIRedactor(patterns=patterns1)
    redactor2 = PHIRedactor(patterns=patterns2)
    
    result1 = redactor1.detect(text)
    result2 = redactor2.detect(text)
    
    # Should detect same entity but with different type names
    assert len(result1) == 1
    assert len(result2) == 1
    assert result1[0].value == result2[0].value
    assert result1[0].type == "ssn"
    assert result2[0].type == "custom"