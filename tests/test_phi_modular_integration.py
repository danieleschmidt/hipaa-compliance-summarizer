"""
Integration tests for the modular PHI pattern system with PHIRedactor.
"""

import pytest
from hipaa_compliance_summarizer.phi import PHIRedactor, Entity
from hipaa_compliance_summarizer.phi_patterns import PHIPatternConfig, pattern_manager


class TestPHIRedactorModularIntegration:
    """Test integration between PHIRedactor and the modular pattern system."""
    
    def setup_method(self):
        """Set up for each test."""
        # Reset global pattern manager
        pattern_manager.categories.clear()
        pattern_manager._default_patterns_loaded = False
    
    def test_redactor_uses_default_patterns(self):
        """Test that redactor automatically loads and uses default patterns."""
        redactor = PHIRedactor()
        
        # Should have loaded default patterns
        assert len(redactor.patterns) > 0
        patterns = redactor.list_patterns()
        assert "ssn" in patterns
        assert "email" in patterns
        assert "phone" in patterns
    
    def test_redactor_backward_compatibility(self):
        """Test that legacy pattern parameter still works."""
        legacy_patterns = {
            "test_ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "test_phone": r"\b\d{3}-\d{3}-\d{4}\b"
        }
        
        redactor = PHIRedactor(patterns=legacy_patterns)
        
        # Should use legacy patterns instead of defaults
        assert len(redactor.patterns) == 2
        assert "test_ssn" in redactor.patterns
        assert "test_phone" in redactor.patterns
    
    def test_redactor_detects_phi_with_modular_patterns(self):
        """Test PHI detection using modular pattern system."""
        redactor = PHIRedactor()
        
        test_text = "Patient SSN: 123-45-6789, email: john@example.com, phone: 555-123-4567"
        entities = redactor.detect(test_text)
        
        # Should detect multiple PHI entities
        assert len(entities) >= 3
        
        entity_types = {entity.type for entity in entities}
        assert "ssn" in entity_types
        assert "email" in entity_types
        assert "phone" in entity_types
    
    def test_redactor_redacts_phi_with_modular_patterns(self):
        """Test PHI redaction using modular pattern system."""
        redactor = PHIRedactor(mask="[REDACTED]")
        
        test_text = "Patient SSN: 123-45-6789, email: john@example.com"
        result = redactor.redact(test_text)
        
        # Should redact detected PHI
        assert "123-45-6789" not in result.text
        assert "john@example.com" not in result.text
        assert "[REDACTED]" in result.text
        assert len(result.entities) >= 2
    
    def test_add_custom_pattern_to_redactor(self):
        """Test adding custom patterns through the redactor interface."""
        redactor = PHIRedactor()
        
        # Add a custom pattern
        redactor.add_custom_pattern(
            name="employee_id",
            pattern=r"\bEMP-\d{4}\b",
            description="Employee ID pattern",
            category="custom"
        )
        
        # Verify pattern was added
        patterns = redactor.list_patterns()
        assert "employee_id" in patterns
        assert patterns["employee_id"]["category"] == "custom"
        
        # Test detection with custom pattern
        test_text = "Employee ID: EMP-1234"
        entities = redactor.detect(test_text)
        
        employee_entities = [e for e in entities if e.type == "employee_id"]
        assert len(employee_entities) == 1
        assert employee_entities[0].value == "EMP-1234"
    
    def test_disable_enable_pattern_through_redactor(self):
        """Test disabling and enabling patterns through redactor interface."""
        redactor = PHIRedactor()
        
        # Initially should detect SSN
        test_text = "SSN: 123-45-6789"
        entities_before = redactor.detect(test_text)
        ssn_entities_before = [e for e in entities_before if e.type == "ssn"]
        assert len(ssn_entities_before) == 1
        
        # Disable SSN pattern
        result = redactor.disable_pattern("ssn")
        assert result is True
        
        # Should no longer detect SSN
        entities_after_disable = redactor.detect(test_text)
        ssn_entities_after_disable = [e for e in entities_after_disable if e.type == "ssn"]
        assert len(ssn_entities_after_disable) == 0
        
        # Re-enable SSN pattern
        result = redactor.enable_pattern("ssn")
        assert result is True
        
        # Should detect SSN again
        entities_after_enable = redactor.detect(test_text)
        ssn_entities_after_enable = [e for e in entities_after_enable if e.type == "ssn"]
        assert len(ssn_entities_after_enable) == 1
    
    def test_pattern_statistics_through_redactor(self):
        """Test getting pattern statistics through redactor interface."""
        redactor = PHIRedactor()
        
        stats = redactor.get_pattern_statistics()
        
        assert "total_categories" in stats
        assert "total_patterns" in stats
        assert "enabled_patterns" in stats
        assert "disabled_patterns" in stats
        
        # Should have at least the default patterns
        assert stats["total_patterns"] > 0
        assert stats["enabled_patterns"] > 0
    
    def test_pattern_caching_with_modular_system(self):
        """Test that pattern caching works correctly with the modular system."""
        redactor = PHIRedactor()
        
        test_text = "SSN: 123-45-6789"
        
        # Clear caches to start fresh
        PHIRedactor.clear_cache()
        
        # First detection - should populate cache
        entities1 = redactor.detect(test_text)
        cache_info_after_first = PHIRedactor.get_cache_info()
        
        # Second detection - should use cache
        entities2 = redactor.detect(test_text)
        cache_info_after_second = PHIRedactor.get_cache_info()
        
        # Results should be identical
        assert len(entities1) == len(entities2)
        assert entities1[0].type == entities2[0].type
        assert entities1[0].value == entities2[0].value
        
        # Cache should have been used
        assert cache_info_after_second["phi_detection"].hits > cache_info_after_first["phi_detection"].hits
    
    def test_pattern_refresh_clears_cache(self):
        """Test that pattern changes properly clear the detection cache."""
        redactor = PHIRedactor()
        
        test_text = "SSN: 123-45-6789"
        
        # Populate cache
        redactor.detect(test_text)
        cache_info_before = PHIRedactor.get_cache_info()
        
        # Add a custom pattern (should trigger cache clear)
        redactor.add_custom_pattern(
            name="test_pattern",
            pattern=r"\bTEST-\d+\b",
            description="Test pattern"
        )
        
        cache_info_after = PHIRedactor.get_cache_info()
        
        # Detection cache should have been cleared (but pattern compilation cache preserved)
        assert cache_info_after["phi_detection"].hits == 0
        assert cache_info_after["phi_detection"].misses == 0
    
    def test_medical_patterns_detection(self):
        """Test detection of medical-specific patterns."""
        redactor = PHIRedactor()
        
        test_text = """
        Patient medical record: MRN: ABC123456789
        DEA Number: AB1234567
        Insurance ID: POLICY123456789
        """
        
        entities = redactor.detect(test_text)
        
        # Should detect medical patterns
        entity_types = {entity.type for entity in entities}
        assert "mrn" in entity_types or any("mrn" in t.lower() for t in entity_types)
        assert "dea" in entity_types or any("dea" in t.lower() for t in entity_types)
        assert "insurance" in entity_types or any("insurance" in t.lower() for t in entity_types)
    
    def test_pattern_confidence_thresholds(self):
        """Test that pattern confidence thresholds are preserved in the system."""
        redactor = PHIRedactor()
        
        patterns = redactor.list_patterns()
        
        # All patterns should have confidence thresholds
        for pattern_name, pattern_info in patterns.items():
            assert "confidence_threshold" in pattern_info
            threshold = pattern_info["confidence_threshold"]
            assert 0.0 <= threshold <= 1.0
    
    def test_error_handling_for_invalid_pattern_operations(self):
        """Test error handling for invalid pattern operations."""
        redactor = PHIRedactor()
        
        # Test disabling nonexistent pattern
        result = redactor.disable_pattern("nonexistent_pattern")
        assert result is False
        
        # Test enabling nonexistent pattern
        result = redactor.enable_pattern("nonexistent_pattern")
        assert result is False
        
        # Test adding pattern with invalid regex
        with pytest.raises(ValueError):
            redactor.add_custom_pattern(
                name="invalid_pattern",
                pattern="[invalid regex",  # Missing closing bracket
                description="Invalid pattern"
            )
    
    def test_pattern_categories_in_redactor(self):
        """Test that pattern categories are properly managed in the redactor."""
        redactor = PHIRedactor()
        
        patterns = redactor.list_patterns()
        
        # Should have patterns from multiple categories
        categories = {pattern_info["category"] for pattern_info in patterns.values()}
        assert "core" in categories
        assert "medical" in categories
        
        # Add custom pattern and verify category
        redactor.add_custom_pattern(
            name="test_custom",
            pattern=r"\bCUST-\d+\b",
            description="Test custom pattern",
            category="test_category"
        )
        
        updated_patterns = redactor.list_patterns()
        assert "test_custom" in updated_patterns
        assert updated_patterns["test_custom"]["category"] == "test_category"


class TestPHIRedactorStreamingWithModularPatterns:
    """Test streaming redaction with modular patterns."""
    
    def test_file_redaction_with_modular_patterns(self, tmp_path):
        """Test file-based redaction using modular pattern system."""
        redactor = PHIRedactor()
        
        # Create test file with PHI
        test_file = tmp_path / "test_document.txt"
        test_content = """
        Patient Information:
        SSN: 123-45-6789
        Email: patient@hospital.com
        Phone: 555-123-4567
        MRN: ABC123456789
        """
        test_file.write_text(test_content)
        
        # Redact file
        result = redactor.redact_file(str(test_file))
        
        # Should redact all PHI
        assert "123-45-6789" not in result.text
        assert "patient@hospital.com" not in result.text
        assert "555-123-4567" not in result.text
        assert "[REDACTED]" in result.text
        assert len(result.entities) >= 3  # At least SSN, email, phone