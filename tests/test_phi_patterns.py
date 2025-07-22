"""
Tests for the modular PHI pattern configuration system.
"""

import pytest
import re
from pathlib import Path
import tempfile
import yaml

from hipaa_compliance_summarizer.phi_patterns import (
    PHIPatternConfig, 
    PHIPatternCategory, 
    PHIPatternManager,
    pattern_manager
)


class TestPHIPatternConfig:
    """Test the PHIPatternConfig dataclass."""
    
    def test_valid_pattern_creation(self):
        """Test creating a valid PHI pattern configuration."""
        pattern = PHIPatternConfig(
            name="test_ssn",
            pattern=r"\b\d{3}-\d{2}-\d{4}\b",
            description="Social Security Number",
            confidence_threshold=0.95
        )
        
        assert pattern.name == "test_ssn"
        assert pattern.pattern == r"\b\d{3}-\d{2}-\d{4}\b"
        assert pattern.description == "Social Security Number"
        assert pattern.confidence_threshold == 0.95
        assert pattern.enabled is True
        assert pattern.compiled_pattern is not None
        assert isinstance(pattern.compiled_pattern, re.Pattern)
    
    def test_invalid_pattern_regex(self):
        """Test that invalid regex patterns raise ValueError."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            PHIPatternConfig(
                name="invalid",
                pattern="[invalid regex",  # Missing closing bracket
                description="Invalid pattern"
            )
    
    def test_empty_name_validation(self):
        """Test that empty pattern names are rejected."""
        with pytest.raises(ValueError, match="Pattern name cannot be empty"):
            PHIPatternConfig(
                name="",
                pattern=r"\d+",
                description="Empty name test"
            )
    
    def test_empty_pattern_validation(self):
        """Test that empty pattern strings are rejected."""
        with pytest.raises(ValueError, match="Pattern string cannot be empty"):
            PHIPatternConfig(
                name="empty_pattern",
                pattern="",
                description="Empty pattern test"
            )
    
    def test_invalid_confidence_threshold(self):
        """Test that invalid confidence thresholds are rejected."""
        with pytest.raises(ValueError, match="Confidence threshold must be between"):
            PHIPatternConfig(
                name="invalid_confidence",
                pattern=r"\d+",
                confidence_threshold=1.5  # Invalid - greater than 1.0
            )
        
        with pytest.raises(ValueError, match="Confidence threshold must be between"):
            PHIPatternConfig(
                name="invalid_confidence2",
                pattern=r"\d+",
                confidence_threshold=-0.1  # Invalid - less than 0.0
            )


class TestPHIPatternCategory:
    """Test the PHIPatternCategory class."""
    
    def test_category_creation(self):
        """Test creating a pattern category."""
        category = PHIPatternCategory("medical", "Medical identifiers")
        
        assert category.name == "medical"
        assert category.description == "Medical identifiers"
        assert len(category.patterns) == 0
    
    def test_add_pattern_to_category(self):
        """Test adding patterns to a category."""
        category = PHIPatternCategory("test_category", "Test patterns")
        pattern = PHIPatternConfig(
            name="test_pattern",
            pattern=r"\d+",
            description="Test pattern"
        )
        
        category.add_pattern(pattern)
        
        assert len(category.patterns) == 1
        assert "test_pattern" in category.patterns
        assert category.patterns["test_pattern"].category == "test_category"
    
    def test_get_enabled_patterns(self):
        """Test filtering enabled patterns in a category."""
        category = PHIPatternCategory("test", "Test category")
        
        enabled_pattern = PHIPatternConfig("enabled", r"\d+", enabled=True)
        disabled_pattern = PHIPatternConfig("disabled", r"\w+", enabled=False)
        
        category.add_pattern(enabled_pattern)
        category.add_pattern(disabled_pattern)
        
        enabled = category.get_enabled_patterns()
        
        assert len(enabled) == 1
        assert "enabled" in enabled
        assert "disabled" not in enabled


class TestPHIPatternManager:
    """Test the PHIPatternManager class."""
    
    def setup_method(self):
        """Set up a fresh pattern manager for each test."""
        self.manager = PHIPatternManager()
    
    def test_load_default_patterns(self):
        """Test loading default PHI patterns."""
        self.manager.load_default_patterns()
        
        assert len(self.manager.categories) >= 2  # At least core and medical
        assert "core" in self.manager.categories
        assert "medical" in self.manager.categories
        
        core_patterns = self.manager.get_patterns_by_category("core")
        assert "ssn" in core_patterns
        assert "email" in core_patterns
        assert "phone" in core_patterns
        assert "date" in core_patterns
        
        medical_patterns = self.manager.get_patterns_by_category("medical")
        assert "mrn" in medical_patterns
        assert "dea" in medical_patterns
        assert "insurance_id" in medical_patterns
    
    def test_load_patterns_from_config(self):
        """Test loading patterns from configuration dictionary."""
        config = {
            "patterns": {
                "custom_id": r"\bCUST-\d{6}\b",
                "test_email": r"\w+@test\.com"
            }
        }
        
        self.manager.load_patterns_from_config(config)
        
        all_patterns = self.manager.get_all_patterns()
        assert "custom_id" in all_patterns
        assert "test_email" in all_patterns
    
    def test_load_patterns_from_file(self):
        """Test loading patterns from YAML file."""
        config_data = {
            "patterns": {
                "file_pattern": r"\bFILE-\d{4}\b",
                "another_pattern": r"\w{3,8}"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            
            self.manager.load_patterns_from_file(f.name)
            
            all_patterns = self.manager.get_all_patterns()
            assert "file_pattern" in all_patterns
            assert "another_pattern" in all_patterns
            
        Path(f.name).unlink()  # Clean up
    
    def test_load_patterns_from_nonexistent_file(self):
        """Test loading patterns from a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.manager.load_patterns_from_file("/nonexistent/file.yml")
    
    def test_add_custom_pattern(self):
        """Test adding custom patterns."""
        pattern = PHIPatternConfig(
            name="custom_test",
            pattern=r"\bTEST-\d{3}\b",
            description="Custom test pattern"
        )
        
        self.manager.add_custom_pattern(pattern, "custom")
        
        assert "custom" in self.manager.categories
        custom_patterns = self.manager.get_patterns_by_category("custom")
        assert "custom_test" in custom_patterns
    
    def test_disable_enable_pattern(self):
        """Test disabling and enabling patterns."""
        self.manager.load_default_patterns()
        
        # Test disabling a pattern
        result = self.manager.disable_pattern("ssn")
        assert result is True
        
        enabled_patterns = self.manager.get_all_patterns()
        assert "ssn" not in enabled_patterns
        
        # Test enabling the pattern back
        result = self.manager.enable_pattern("ssn")
        assert result is True
        
        enabled_patterns = self.manager.get_all_patterns()
        assert "ssn" in enabled_patterns
        
        # Test disabling nonexistent pattern
        result = self.manager.disable_pattern("nonexistent")
        assert result is False
    
    def test_get_compiled_patterns(self):
        """Test getting compiled regex patterns."""
        self.manager.load_default_patterns()
        
        compiled = self.manager.get_compiled_patterns()
        
        assert len(compiled) > 0
        for name, pattern in compiled.items():
            assert isinstance(pattern, re.Pattern)
    
    def test_validate_all_patterns(self):
        """Test validation of all loaded patterns."""
        self.manager.load_default_patterns()
        
        errors = self.manager.validate_all_patterns()
        
        # Default patterns should have no validation errors
        assert len(errors) == 0
        
        # Add an invalid pattern and test
        invalid_pattern = PHIPatternConfig.__new__(PHIPatternConfig)
        invalid_pattern.name = "invalid"
        invalid_pattern.pattern = "[invalid"
        invalid_pattern.enabled = True
        
        self.manager.categories["test"] = PHIPatternCategory("test")
        self.manager.categories["test"].patterns["invalid"] = invalid_pattern
        
        errors = self.manager.validate_all_patterns()
        assert len(errors) > 0
    
    def test_get_pattern_statistics(self):
        """Test getting pattern statistics."""
        self.manager.load_default_patterns()
        
        stats = self.manager.get_pattern_statistics()
        
        assert "total_categories" in stats
        assert "total_patterns" in stats
        assert "enabled_patterns" in stats
        assert "disabled_patterns" in stats
        
        assert stats["total_categories"] >= 2
        assert stats["total_patterns"] > 0
        assert stats["enabled_patterns"] > 0
        assert stats["disabled_patterns"] >= 0
    
    def test_determine_category(self):
        """Test automatic category determination for patterns."""
        # Test medical category detection
        assert self.manager._determine_category("mrn_number") == "medical"
        assert self.manager._determine_category("patient_id") == "medical"
        assert self.manager._determine_category("dea_number") == "medical"
        
        # Test core category detection
        assert self.manager._determine_category("ssn_pattern") == "core"
        assert self.manager._determine_category("phone_number") == "core"
        assert self.manager._determine_category("email_addr") == "core"
        
        # Test custom category for unknown patterns
        assert self.manager._determine_category("unknown_pattern") == "custom"


class TestGlobalPatternManager:
    """Test the global pattern manager instance."""
    
    def test_global_pattern_manager_exists(self):
        """Test that the global pattern manager is available."""
        assert pattern_manager is not None
        assert isinstance(pattern_manager, PHIPatternManager)
    
    def test_global_pattern_manager_defaults(self):
        """Test that the global pattern manager can load defaults."""
        # Reset the global manager
        pattern_manager.categories.clear()
        pattern_manager._default_patterns_loaded = False
        
        pattern_manager.load_default_patterns()
        
        assert pattern_manager._default_patterns_loaded is True
        assert len(pattern_manager.categories) >= 2


class TestPHIPatternIntegration:
    """Integration tests for the PHI pattern system."""
    
    def test_pattern_matching_functionality(self):
        """Test that patterns actually work for detecting PHI."""
        manager = PHIPatternManager()
        manager.load_default_patterns()
        
        compiled_patterns = manager.get_compiled_patterns()
        
        # Test SSN pattern
        ssn_pattern = compiled_patterns["ssn"]
        test_text = "Patient SSN: 123-45-6789"
        matches = ssn_pattern.findall(test_text)
        assert len(matches) == 1
        assert "123-45-6789" in matches[0]
        
        # Test email pattern
        email_pattern = compiled_patterns["email"]
        test_text = "Contact: patient@hospital.com"
        matches = email_pattern.findall(test_text)
        assert len(matches) == 1
        assert "patient@hospital.com" in matches[0]
    
    def test_custom_pattern_functionality(self):
        """Test that custom patterns work correctly."""
        manager = PHIPatternManager()
        
        custom_pattern = PHIPatternConfig(
            name="hospital_id",
            pattern=r"\bHOSP-\d{4}\b",
            description="Hospital ID pattern"
        )
        
        manager.add_custom_pattern(custom_pattern, "custom")
        
        compiled_patterns = manager.get_compiled_patterns()
        hospital_pattern = compiled_patterns["hospital_id"]
        
        test_text = "Hospital ID: HOSP-1234"
        matches = hospital_pattern.findall(test_text)
        assert len(matches) == 1
        assert "HOSP-1234" in matches[0]