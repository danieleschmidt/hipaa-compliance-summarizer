"""
Sample test demonstrating the testing infrastructure for HIPAA Compliance Summarizer.

This test serves as an example of how to use the testing utilities and fixtures
provided by the testing infrastructure.
"""

import pytest
import time
from tests.utils import (
    SyntheticDataGenerator,
    TestSecurityUtils,
    MockHIPAAConfig,
    requires_phi_detection,
    hipaa_compliance_test,
    performance_test
)


class TestInfrastructureSample:
    """Sample tests demonstrating testing infrastructure capabilities."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation utilities."""
        # Generate synthetic clinical note
        note = SyntheticDataGenerator.generate_clinical_note()
        assert "Chief Complaint" in note
        assert "John Doe" in note  # Synthetic name
        
        # Generate note without PHI
        clean_note = SyntheticDataGenerator.generate_clinical_note(include_phi=False)
        assert "Chief Complaint" in clean_note
        assert "John Doe" not in clean_note
    
    def test_security_utilities(self):
        """Test security testing utilities."""
        # Test PHI exposure detection
        phi_text = "Patient John Doe, SSN: 999-99-9999"
        with pytest.raises(AssertionError):
            TestSecurityUtils.assert_no_phi_exposure(phi_text)
        
        # Test clean text passes
        clean_text = "Patient presents with chest pain"
        TestSecurityUtils.assert_no_phi_exposure(clean_text)
    
    def test_mock_config(self):
        """Test mock HIPAA configuration."""
        config = MockHIPAAConfig.get_test_config()
        assert config["compliance"]["level"] == "strict"
        assert config["security"]["encryption_key_rotation"] == 90
    
    def test_file_manager_fixture(self, file_manager):
        """Test file manager fixture for temporary files."""
        # Create a temporary file
        test_content = "This is test content"
        temp_file = file_manager.create_temp_file(test_content)
        
        # Verify file exists and contains content
        with open(temp_file, 'r') as f:
            assert f.read() == test_content
        
        # File will be automatically cleaned up by fixture
    
    def test_performance_tracking(self, performance_tracker):
        """Test performance tracking utilities."""
        # Simulate some operation
        start_time = time.time()
        time.sleep(0.1)  # Simulate work
        duration = time.time() - start_time
        
        # Record performance
        performance_tracker.record_time("test_operation", duration)
        
        # Check metrics
        avg_time = performance_tracker.get_average_time("test_operation")
        assert avg_time > 0.05  # Should be at least 50ms
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test with unit test marker."""
        assert True
    
    @pytest.mark.fast
    def test_fast_marker(self):
        """Test with fast marker."""
        assert True
    
    @pytest.mark.phi_test
    @requires_phi_detection
    def test_phi_detection_decorator(self):
        """Test with PHI detection decorator."""
        # This would test PHI detection functionality
        synthetic_text = "Patient John Doe has MRN123456"
        # Simulate PHI detection
        assert "John Doe" in synthetic_text
    
    @pytest.mark.hipaa
    @hipaa_compliance_test
    def test_hipaa_compliance_decorator(self):
        """Test with HIPAA compliance decorator."""
        # This would test HIPAA compliance features
        config = MockHIPAAConfig.get_test_config()
        assert config["compliance"]["audit_logging"] is True
    
    @pytest.mark.performance
    @performance_test(max_time=5.0)
    def test_performance_decorator(self):
        """Test with performance decorator."""
        # This test must complete within 5 seconds
        time.sleep(0.1)  # Simulate some work
        assert True
    
    def test_synthetic_fixtures(self, synthetic_clinical_note, synthetic_lab_report):
        """Test synthetic data fixtures."""
        # Test clinical note fixture
        assert "Chief Complaint" in synthetic_clinical_note
        assert "John Doe" in synthetic_clinical_note
        
        # Test lab report fixture
        assert "LABORATORY REPORT" in synthetic_lab_report
        assert "Troponin" in synthetic_lab_report
    
    def test_config_fixture(self, test_hipaa_config):
        """Test HIPAA configuration fixture."""
        assert test_hipaa_config["compliance"]["level"] == "strict"
        assert test_hipaa_config["redaction"]["method"] == "synthetic_replacement"
    
    @pytest.mark.slow
    def test_slow_operation(self):
        """Test marked as slow for demonstration."""
        time.sleep(0.2)  # Simulate slow operation
        assert True
    
    @pytest.mark.security
    def test_security_marker(self):
        """Test with security marker."""
        # This would test security features
        assert True
    
    @pytest.mark.compliance
    def test_compliance_marker(self):
        """Test with compliance marker."""
        # This would test compliance features
        assert True
    
    def test_no_phi_fixture(self, no_phi_clinical_note):
        """Test clinical note without PHI."""
        # Verify no synthetic PHI patterns
        TestSecurityUtils.assert_no_phi_exposure(no_phi_clinical_note)
        
        # Should still contain clinical content
        assert "Chief Complaint" in no_phi_clinical_note
    
    def test_secure_temp_file(self, secure_temp_file):
        """Test secure temporary file fixture."""
        import os
        import stat
        
        # Verify file exists
        assert os.path.exists(secure_temp_file)
        
        # Verify restrictive permissions
        file_mode = os.stat(secure_temp_file).st_mode
        permissions = stat.filemode(file_mode)
        assert "rw-------" in permissions  # Only owner can read/write


class TestHealthcareSpecificMarkers:
    """Test healthcare-specific marker functionality."""
    
    @pytest.mark.clinical_note
    def test_clinical_note_processing(self):
        """Test clinical note processing."""
        note = SyntheticDataGenerator.generate_clinical_note()
        assert "History of Present Illness" in note
    
    @pytest.mark.lab_report
    def test_lab_report_processing(self):
        """Test lab report processing."""
        report = SyntheticDataGenerator.generate_lab_report()
        assert "Test Results" in report
    
    @pytest.mark.synthetic_data
    def test_synthetic_data_only(self):
        """Test using only synthetic data."""
        # This test should only use synthetic data
        note = SyntheticDataGenerator.generate_clinical_note()
        # Verify it contains known synthetic markers
        assert any(name in note for name in SyntheticDataGenerator.SYNTHETIC_NAMES)
    
    @pytest.mark.no_phi
    def test_no_phi_requirement(self):
        """Test that must not contain PHI."""
        clean_text = "Patient presents with symptoms"
        TestSecurityUtils.assert_no_phi_exposure(clean_text)


# Example parameterized test
@pytest.mark.parametrize("compliance_level", ["strict", "standard", "minimal"])
def test_compliance_levels(compliance_level):
    """Test different compliance levels."""
    config = MockHIPAAConfig.get_test_config()
    config["compliance"]["level"] = compliance_level
    assert config["compliance"]["level"] in ["strict", "standard", "minimal"]


# Example test with multiple markers
@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.synthetic_data
@pytest.mark.no_phi
def test_multiple_markers():
    """Test with multiple markers."""
    # Fast unit test using synthetic data with no PHI
    clean_note = SyntheticDataGenerator.generate_clinical_note(include_phi=False)
    TestSecurityUtils.assert_no_phi_exposure(clean_note)
    assert len(clean_note) > 0