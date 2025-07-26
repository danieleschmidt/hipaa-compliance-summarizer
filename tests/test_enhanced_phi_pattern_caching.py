"""
Tests for enhanced PHI pattern caching optimizations.

This module tests the additional caching layers added to the PHI pattern system:
- Compiled pattern caching with hash-based invalidation
- Category-based pattern caching
- Validation result caching
- Statistics caching with TTL
"""

import time
import pytest
from unittest.mock import patch
from hipaa_compliance_summarizer.phi_patterns import PHIPatternManager, PHIPatternConfig, pattern_manager
from hipaa_compliance_summarizer.phi import PHIRedactor


class TestEnhancedPatternCaching:
    """Test enhanced caching functionality in PHI pattern system."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Clear all caches before each test
        pattern_manager.clear_all_caches()
        PHIRedactor.clear_cache()
    
    def test_compiled_pattern_caching(self):
        """Test that compiled patterns are cached with hash-based invalidation."""
        manager = PHIPatternManager()
        manager.load_default_patterns()
        
        # First call should populate cache
        patterns1 = manager.get_compiled_patterns()
        cache_info1 = manager.get_cache_info()
        
        # Second call should hit cache
        patterns2 = manager.get_compiled_patterns()
        cache_info2 = manager.get_cache_info()
        
        # Should return same patterns
        assert patterns1 == patterns2
        
        # Cache hits should increase
        assert cache_info2["compiled_patterns"]["hits"] > cache_info1["compiled_patterns"]["hits"]
    
    def test_category_pattern_caching(self):
        """Test that category-specific patterns are cached."""
        manager = PHIPatternManager()
        manager.load_default_patterns()
        
        # Get patterns for a specific category
        core_patterns1 = manager.get_patterns_by_category("core")
        cache_info1 = manager.get_cache_info()
        
        # Second call should hit cache
        core_patterns2 = manager.get_patterns_by_category("core")
        cache_info2 = manager.get_cache_info()
        
        # Should return same patterns
        assert len(core_patterns1) == len(core_patterns2)
        for name in core_patterns1:
            assert name in core_patterns2
            assert core_patterns1[name].pattern == core_patterns2[name].pattern
        
        # Cache hits should increase
        assert cache_info2["category_patterns"]["hits"] > cache_info1["category_patterns"]["hits"]
    
    def test_validation_caching(self):
        """Test that pattern validation results are cached."""
        manager = PHIPatternManager()
        manager.load_default_patterns()
        
        # First validation should populate cache
        errors1 = manager.validate_all_patterns()
        cache_info1 = manager.get_cache_info()
        
        # Second validation should use cache
        errors2 = manager.validate_all_patterns()
        cache_info2 = manager.get_cache_info()
        
        # Results should be identical
        assert errors1 == errors2
        
        # Validation cache should have entries
        assert cache_info2["validation_cache_size"] > 0
    
    def test_statistics_caching_with_ttl(self):
        """Test that statistics are cached with TTL."""
        manager = PHIPatternManager()
        manager.load_default_patterns()
        
        # First call should calculate fresh stats
        stats1 = manager.get_pattern_statistics()
        cache_info1 = manager.get_cache_info()
        
        # Second call within TTL should use cache
        stats2 = manager.get_pattern_statistics()
        cache_info2 = manager.get_cache_info()
        
        # Should return same stats
        assert stats1 == stats2
        
        # Cache should be valid
        assert cache_info2["stats_cache_valid"] is True
    
    def test_cache_invalidation_on_pattern_changes(self):
        """Test that caches are properly invalidated when patterns change."""
        manager = PHIPatternManager()
        manager.load_default_patterns()
        
        # Populate caches
        patterns = manager.get_compiled_patterns()
        stats = manager.get_pattern_statistics()
        validation = manager.validate_all_patterns()
        
        # Get initial cache info
        cache_info_before = manager.get_cache_info()
        
        # Disable a pattern (should invalidate caches)
        manager.disable_pattern("ssn")
        
        # Get cache info after change
        cache_info_after = manager.get_cache_info()
        
        # Validation cache should be cleared
        assert cache_info_after["validation_cache_size"] == 0
        
        # Stats cache should be invalidated
        assert cache_info_after["stats_cache_valid"] is False
    
    def test_pattern_manager_integration_with_phi_redactor(self):
        """Test that PHIRedactor integrates with pattern manager caches."""
        # Create redactor which should use cached patterns
        redactor = PHIRedactor()
        
        # Get cache info including pattern manager
        cache_info = redactor.get_cache_info()
        
        # Should include pattern manager cache information
        assert "pattern_manager" in cache_info
        assert "compiled_patterns" in cache_info["pattern_manager"]
        assert "category_patterns" in cache_info["pattern_manager"]
        assert "validation_cache_size" in cache_info["pattern_manager"]
        assert "stats_cache_valid" in cache_info["pattern_manager"]
    
    def test_concurrent_cache_access(self):
        """Test that caches are thread-safe during concurrent access."""
        import threading
        import concurrent.futures
        
        manager = PHIPatternManager()
        manager.load_default_patterns()
        
        def access_caches():
            """Access various cached methods."""
            patterns = manager.get_compiled_patterns()
            stats = manager.get_pattern_statistics()
            validation = manager.validate_all_patterns()
            core_patterns = manager.get_patterns_by_category("core")
            return len(patterns), stats["total_patterns"], len(validation), len(core_patterns)
        
        # Access caches concurrently from multiple threads
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(access_caches) for _ in range(10)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        # All results should be identical (indicating consistent caching)
        for result in results[1:]:
            assert result == results[0]
    
    def test_memory_efficiency_with_caching(self):
        """Test that caching doesn't cause memory leaks."""
        manager = PHIPatternManager()
        manager.load_default_patterns()
        
        # Access caches multiple times
        for _ in range(50):
            patterns = manager.get_compiled_patterns()
            stats = manager.get_pattern_statistics()
            validation = manager.validate_all_patterns()
            core_patterns = manager.get_patterns_by_category("core")
        
        cache_info = manager.get_cache_info()
        
        # Caches should have reasonable sizes
        assert cache_info["compiled_patterns"]["currsize"] <= 32  # maxsize=32
        assert cache_info["category_patterns"]["currsize"] <= 16  # maxsize=16
        assert cache_info["validation_cache_size"] <= 10  # Should not grow indefinitely
    
    def test_cache_performance_improvement(self):
        """Test that caching improves performance."""
        manager = PHIPatternManager()
        manager.load_default_patterns()
        
        # Time cold access (cache miss)
        manager.clear_all_caches()
        start_time = time.perf_counter()
        patterns1 = manager.get_compiled_patterns()
        stats1 = manager.get_pattern_statistics()
        cold_time = time.perf_counter() - start_time
        
        # Time warm access (cache hit)
        start_time = time.perf_counter()
        patterns2 = manager.get_compiled_patterns()
        stats2 = manager.get_pattern_statistics()
        warm_time = time.perf_counter() - start_time
        
        # Results should be identical
        assert patterns1 == patterns2
        assert stats1 == stats2
        
        # Warm access should be faster (or at least not significantly slower)
        assert warm_time <= cold_time * 1.1  # Allow 10% tolerance
    
    def test_manual_cache_clearing(self):
        """Test manual cache clearing functionality."""
        manager = PHIPatternManager()
        manager.load_default_patterns()
        
        # Populate caches
        patterns = manager.get_compiled_patterns()
        stats = manager.get_pattern_statistics()
        validation = manager.validate_all_patterns()
        core_patterns = manager.get_patterns_by_category("core")
        
        # Verify caches are populated
        cache_info_before = manager.get_cache_info()
        assert cache_info_before["compiled_patterns"]["currsize"] > 0
        assert cache_info_before["validation_cache_size"] > 0
        assert cache_info_before["stats_cache_valid"] is True
        
        # Clear all caches
        manager.clear_all_caches()
        
        # Verify caches are cleared
        cache_info_after = manager.get_cache_info()
        assert cache_info_after["compiled_patterns"]["currsize"] == 0
        assert cache_info_after["category_patterns"]["currsize"] == 0
        assert cache_info_after["validation_cache_size"] == 0
        assert cache_info_after["stats_cache_valid"] is False


def test_phi_redactor_clear_cache_integration():
    """Test that PHIRedactor.clear_cache() clears pattern manager caches too."""
    # Populate both PHI redactor and pattern manager caches
    redactor = PHIRedactor()
    test_text = "Patient SSN: 123-45-6789"
    entities = redactor.detect(test_text)
    
    # Get initial cache info
    cache_info_before = PHIRedactor.get_cache_info()
    
    # Should have some cache activity
    assert cache_info_before["phi_detection"].currsize > 0
    
    # Clear all caches
    PHIRedactor.clear_cache()
    
    # Verify all caches are cleared
    cache_info_after = PHIRedactor.get_cache_info()
    assert cache_info_after["phi_detection"].currsize == 0
    assert cache_info_after["pattern_compilation"].currsize == 0
    assert cache_info_after["pattern_manager"]["compiled_patterns"]["currsize"] == 0