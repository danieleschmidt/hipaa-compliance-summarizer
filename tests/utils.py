"""
Test utilities for HIPAA Compliance Summarizer test suite.

This module provides common testing utilities, fixtures, and helper functions
for healthcare-specific testing scenarios.
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock, patch

import pytest


class SyntheticDataGenerator:
    """Generate synthetic healthcare data for testing purposes."""
    
    SYNTHETIC_NAMES = [
        "John Doe", "Jane Smith", "Robert Johnson", "Mary Williams",
        "Michael Brown", "Patricia Davis", "William Miller", "Jennifer Wilson"
    ]
    
    SYNTHETIC_ADDRESSES = [
        "123 Main St, Anytown, ST 12345",
        "456 Oak Ave, Somewhere, ST 67890",
        "789 Pine Rd, Nowhere, ST 11111"
    ]
    
    SYNTHETIC_PHONES = [
        "(555) 123-4567", "(555) 987-6543", "(555) 555-0123"
    ]
    
    SYNTHETIC_SSNS = [
        "999-99-9999", "888-88-8888", "777-77-7777"
    ]
    
    SYNTHETIC_MRNS = [
        "MRN123456", "MRN789012", "MRN345678"
    ]
    
    @classmethod
    def generate_clinical_note(cls, include_phi: bool = True) -> str:
        """Generate a synthetic clinical note."""
        base_note = """
        CLINICAL NOTE
        
        Chief Complaint: Patient presents with chest pain and shortness of breath.
        
        History of Present Illness:
        The patient is a 45-year-old presenting with acute onset chest pain
        that began approximately 2 hours ago. Pain is described as crushing,
        substernal, and radiating to the left arm.
        
        Past Medical History:
        - Hypertension
        - Diabetes mellitus type 2
        - Hyperlipidemia
        
        Assessment and Plan:
        1. Acute coronary syndrome - rule out MI
           - Serial cardiac enzymes
           - EKG monitoring
           - Aspirin, nitroglycerin as needed
        
        2. Continue current medications
        
        Follow-up as needed.
        """
        
        if include_phi:
            phi_note = f"""
            Patient Name: {cls.SYNTHETIC_NAMES[0]}
            MRN: {cls.SYNTHETIC_MRNS[0]}
            DOB: 01/01/1978
            Address: {cls.SYNTHETIC_ADDRESSES[0]}
            Phone: {cls.SYNTHETIC_PHONES[0]}
            SSN: {cls.SYNTHETIC_SSNS[0]}
            
            """ + base_note
            return phi_note
        
        return base_note
    
    @classmethod
    def generate_lab_report(cls, include_phi: bool = True) -> str:
        """Generate a synthetic lab report."""
        base_report = """
        LABORATORY REPORT
        
        Test Results:
        - Troponin I: 0.04 ng/mL (Normal: <0.04)
        - CK-MB: 3.2 ng/mL (Normal: 0.0-6.3)
        - Total Cholesterol: 245 mg/dL (High)
        - LDL: 165 mg/dL (High)
        - HDL: 35 mg/dL (Low)
        - Glucose: 156 mg/dL (High)
        
        Interpretation: Elevated glucose and lipid levels.
        """
        
        if include_phi:
            phi_report = f"""
            Patient: {cls.SYNTHETIC_NAMES[1]}
            MRN: {cls.SYNTHETIC_MRNS[1]}
            Collection Date: 2024-01-15
            Report Date: 2024-01-15
            
            """ + base_report
            return phi_report
        
        return base_report


class TestSecurityUtils:
    """Security testing utilities."""
    
    @staticmethod
    def assert_no_phi_exposure(text: str) -> None:
        """Assert that text does not contain PHI patterns."""
        # Check for synthetic PHI markers
        synthetic_patterns = [
            r"\d{3}-\d{2}-\d{4}",  # SSN pattern
            r"MRN\d+",  # MRN pattern
            r"\(\d{3}\) \d{3}-\d{4}",  # Phone pattern
        ]
        
        for pattern in synthetic_patterns:
            import re
            if re.search(pattern, text):
                pytest.fail(f"Potential PHI pattern found: {pattern}")
    
    @staticmethod
    def assert_encrypted_content(content: bytes) -> None:
        """Assert that content appears to be encrypted."""
        # Simple heuristic: encrypted content should not contain common text patterns
        if b"patient" in content.lower() or b"name" in content.lower():
            pytest.fail("Content appears to be unencrypted")
    
    @staticmethod
    def create_secure_temp_file() -> str:
        """Create a secure temporary file for testing."""
        fd, path = tempfile.mkstemp(prefix="hipaa_test_", suffix=".tmp")
        os.close(fd)
        os.chmod(path, 0o600)  # Restrictive permissions
        return path


class MockHIPAAConfig:
    """Mock HIPAA configuration for testing."""
    
    @staticmethod
    def get_test_config() -> Dict[str, Any]:
        """Get a test HIPAA configuration."""
        return {
            "compliance": {
                "level": "strict",
                "audit_logging": True,
                "encryption_at_rest": True,
                "phi_detection_threshold": 0.95
            },
            "redaction": {
                "method": "synthetic_replacement",
                "preserve_clinical_context": True,
                "maintain_document_structure": True
            },
            "security": {
                "encryption_key_rotation": 90,
                "access_logging": True,
                "data_retention_policy": 2555,
                "secure_deletion": True
            }
        }


class TestFileManager:
    """Manage test files and cleanup."""
    
    def __init__(self):
        self.temp_files: List[str] = []
        self.temp_dirs: List[str] = []
    
    def create_temp_file(self, content: str, suffix: str = ".txt") -> str:
        """Create a temporary file with content."""
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="hipaa_test_")
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        self.temp_files.append(path)
        return path
    
    def create_temp_dir(self) -> str:
        """Create a temporary directory."""
        path = tempfile.mkdtemp(prefix="hipaa_test_")
        self.temp_dirs.append(path)
        return path
    
    def cleanup(self) -> None:
        """Clean up all temporary files and directories."""
        import shutil
        
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except OSError:
                pass
        
        for dir_path in self.temp_dirs:
            try:
                shutil.rmtree(dir_path)
            except OSError:
                pass
        
        self.temp_files.clear()
        self.temp_dirs.clear()


class PerformanceTracker:
    """Track performance metrics during testing."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record_time(self, operation: str, duration: float) -> None:
        """Record execution time for an operation."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
    
    def get_average_time(self, operation: str) -> float:
        """Get average execution time for an operation."""
        times = self.metrics.get(operation, [])
        return sum(times) / len(times) if times else 0.0
    
    def assert_performance_threshold(self, operation: str, max_time: float) -> None:
        """Assert that operation meets performance threshold."""
        avg_time = self.get_average_time(operation)
        if avg_time > max_time:
            pytest.fail(f"Performance threshold exceeded: {operation} "
                       f"averaged {avg_time}s, expected <{max_time}s")


# Common test fixtures
@pytest.fixture
def synthetic_clinical_note():
    """Provide a synthetic clinical note for testing."""
    return SyntheticDataGenerator.generate_clinical_note()


@pytest.fixture
def synthetic_lab_report():
    """Provide a synthetic lab report for testing."""
    return SyntheticDataGenerator.generate_lab_report()


@pytest.fixture
def no_phi_clinical_note():
    """Provide a clinical note without PHI for testing."""
    return SyntheticDataGenerator.generate_clinical_note(include_phi=False)


@pytest.fixture
def test_hipaa_config():
    """Provide a test HIPAA configuration."""
    return MockHIPAAConfig.get_test_config()


@pytest.fixture
def file_manager():
    """Provide a test file manager with automatic cleanup."""
    manager = TestFileManager()
    yield manager
    manager.cleanup()


@pytest.fixture
def performance_tracker():
    """Provide a performance tracking utility."""
    return PerformanceTracker()


@pytest.fixture
def secure_temp_file():
    """Provide a secure temporary file."""
    path = TestSecurityUtils.create_secure_temp_file()
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


# Test decorators
def requires_phi_detection(func):
    """Decorator for tests that require PHI detection capability."""
    return pytest.mark.phi_test(func)


def requires_encryption(func):
    """Decorator for tests that require encryption capability."""
    return pytest.mark.encryption_test(func)


def hipaa_compliance_test(func):
    """Decorator for HIPAA compliance tests."""
    return pytest.mark.hipaa(func)


def performance_test(max_time: float):
    """Decorator for performance tests with time limits."""
    def decorator(func):
        return pytest.mark.performance(pytest.mark.timeout(max_time)(func))
    return decorator