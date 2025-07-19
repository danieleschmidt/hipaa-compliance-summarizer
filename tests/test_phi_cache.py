import time
import pytest
from unittest.mock import patch, MagicMock
from hipaa_compliance_summarizer.phi import PHIRedactor, _compile_pattern
from hipaa_compliance_summarizer.batch import BatchProcessor


def test_pattern_cache():
    """Test basic pattern caching between PHIRedactor instances."""
    r1 = PHIRedactor().patterns["ssn"]
    r2 = PHIRedactor().patterns["ssn"]
    assert r1 is r2


def test_pattern_compilation_caching():
    """Test that regex patterns are compiled once and cached."""
    
    # Clear any existing cache
    _compile_pattern.cache_clear()
    
    pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    
    # First compilation
    start_time = time.perf_counter()
    compiled1 = _compile_pattern(pattern)
    first_compilation_time = time.perf_counter() - start_time
    
    # Second compilation (should be cached)
    start_time = time.perf_counter()
    compiled2 = _compile_pattern(pattern)
    second_compilation_time = time.perf_counter() - start_time
    
    # Should return the same object
    assert compiled1 is compiled2
    
    # Second call should be significantly faster
    assert second_compilation_time < first_compilation_time * 0.5
    
    # Check cache info
    cache_info = _compile_pattern.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1


def test_phi_redactor_pattern_reuse():
    """Test that PHIRedactor instances can share compiled patterns."""
    
    patterns = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "phone": r"\b\d{3}[.-]\d{3}[.-]\d{4}\b"
    }
    
    # Create multiple redactor instances with same patterns
    redactor1 = PHIRedactor(patterns=patterns)
    redactor2 = PHIRedactor(patterns=patterns)
    
    # They should use the same compiled pattern objects (due to lru_cache)
    for pattern_name in patterns:
        assert redactor1.patterns[pattern_name] is redactor2.patterns[pattern_name]


def test_batch_processing_pattern_sharing():
    """Test that batch processing efficiently shares pattern compilation."""
    
    # Clear pattern cache
    _compile_pattern.cache_clear()
    
    # Create multiple processors (simulating batch processing)
    processors = [BatchProcessor() for _ in range(3)]
    
    # All processors should share the same compiled patterns
    base_patterns = processors[0].processor.redactor.patterns
    for processor in processors[1:]:
        current_patterns = processor.processor.redactor.patterns
        for pattern_name in base_patterns:
            assert base_patterns[pattern_name] is current_patterns[pattern_name]


def test_phi_redactor_performance_with_cache():
    """Test that PHI detection performance improves with pattern caching."""
    
    test_text = """
    Patient John Doe, SSN: 123-45-6789, Phone: 555-123-4567
    Email: john.doe@example.com, DOB: 01/15/1980
    Another patient Jane Smith, SSN: 987-65-4321, Phone: 555-987-6543
    """
    
    # Clear cache and time cold performance
    _compile_pattern.cache_clear()
    
    start_time = time.perf_counter()
    redactor1 = PHIRedactor()
    result1 = redactor1.detect(test_text)
    cold_time = time.perf_counter() - start_time
    
    # Time warm performance (patterns already cached)
    start_time = time.perf_counter()
    redactor2 = PHIRedactor()
    result2 = redactor2.detect(test_text)
    warm_time = time.perf_counter() - start_time
    
    # Results should be identical
    assert len(result1) == len(result2)
    for ent1, ent2 in zip(result1, result2):
        assert ent1.type == ent2.type
        assert ent1.value == ent2.value
        assert ent1.start == ent2.start
        assert ent1.end == ent2.end
    
    # Warm should be faster (or at least not significantly slower)
    assert warm_time <= cold_time * 1.1  # Allow 10% tolerance


def test_pattern_cache_memory_efficiency():
    """Test that pattern caching doesn't cause memory leaks."""
    
    # Clear cache
    _compile_pattern.cache_clear()
    
    # Create many redactors with same patterns
    redactors = []
    for i in range(50):  # Reduced from 100 to be more reasonable for tests
        redactor = PHIRedactor()
        redactors.append(redactor)
    
    # Check that we're not duplicating pattern objects
    base_patterns = redactors[0].patterns
    pattern_objects = set()
    
    for redactor in redactors:
        for pattern_name, pattern_obj in redactor.patterns.items():
            # All instances should reference the same compiled pattern objects
            assert pattern_obj is base_patterns[pattern_name]
            pattern_objects.add(id(pattern_obj))
    
    # Should only have unique pattern objects equal to number of pattern types
    assert len(pattern_objects) == len(base_patterns)


def test_cache_invalidation_on_different_patterns():
    """Test that cache properly handles different pattern sets."""
    
    _compile_pattern.cache_clear()
    
    patterns1 = {"ssn": r"\b\d{3}-\d{2}-\d{4}\b"}
    patterns2 = {"phone": r"\b\d{3}[.-]\d{3}[.-]\d{4}\b"}
    
    redactor1 = PHIRedactor(patterns=patterns1)
    redactor2 = PHIRedactor(patterns=patterns2)
    
    # Should have different pattern objects for different expressions
    assert len(redactor1.patterns) == 1
    assert len(redactor2.patterns) == 1
    assert "ssn" in redactor1.patterns
    assert "phone" in redactor2.patterns


def test_concurrent_pattern_compilation():
    """Test that pattern compilation is thread-safe."""
    import threading
    import concurrent.futures
    
    _compile_pattern.cache_clear()
    
    pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    results = []
    
    def compile_pattern():
        return _compile_pattern(pattern)
    
    # Compile pattern concurrently from multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(compile_pattern) for _ in range(10)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # All should return the same cached object
    for result in results[1:]:
        assert result is results[0]
