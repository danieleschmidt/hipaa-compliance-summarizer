"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest
import yaml


@pytest.fixture(scope="session")
def test_config() -> dict:
    """Test configuration for HIPAA compliance testing."""
    return {
        "compliance": {
            "level": "strict",
            "audit_logging": True,
            "encryption_at_rest": True,
            "phi_detection_threshold": 0.95,
        },
        "redaction": {
            "method": "synthetic_replacement",
            "preserve_clinical_context": True,
            "maintain_document_structure": True,
        },
        "security": {
            "encryption_key_rotation": 90,
            "access_logging": True,
            "data_retention_policy": 2555,
            "secure_deletion": True,
        },
        "output": {
            "include_confidence_scores": True,
            "generate_audit_trail": True,
            "compliance_score": True,
            "redaction_summary": True,
        },
    }


@pytest.fixture
def temp_config_file(test_config: dict) -> Generator[str, None, None]:
    """Create a temporary HIPAA config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    try:
        os.unlink(config_path)
    except OSError:
        pass


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_documents(temp_dir: str) -> dict:
    """Create sample healthcare documents for testing."""
    documents = {}

    # Clinical note with PHI
    clinical_note = """
    Patient: John Doe
    DOB: 01/15/1980
    MRN: 123456789
    Date: March 15, 2024

    CHIEF COMPLAINT: Chest pain

    HISTORY OF PRESENT ILLNESS:
    45-year-old male presents with acute onset chest pain radiating to left arm.
    Patient reports pain started at 2:30 AM. No prior cardiac history.

    ASSESSMENT AND PLAN:
    1. Acute coronary syndrome - obtain EKG, troponins
    2. Admit to cardiac care unit for monitoring
    3. Aspirin 325mg, nitroglycerin PRN

    Dr. Smith
    """

    documents["clinical_note"] = Path(temp_dir) / "clinical_note.txt"
    documents["clinical_note"].write_text(clinical_note)

    # Lab report with PHI
    lab_report = """
    LABORATORY REPORT
    Patient: Jane Smith
    DOB: 05/22/1975
    SSN: 123-45-6789
    Account: LAB-789012

    Test Results:
    - Troponin I: 0.8 ng/mL (elevated)
    - CK-MB: 15 ng/mL (elevated)
    - Total CK: 450 U/L (elevated)

    Interpretation: Consistent with myocardial injury
    """

    documents["lab_report"] = Path(temp_dir) / "lab_report.txt"
    documents["lab_report"].write_text(lab_report)

    # Clean document without PHI
    clean_document = """
    MEDICAL RESEARCH SUMMARY

    This study evaluated the effectiveness of treatment protocols
    for acute coronary syndrome. Results showed improved outcomes
    with early intervention strategies.

    No patient identifiers included in this summary.
    """

    documents["clean_document"] = Path(temp_dir) / "clean_document.txt"
    documents["clean_document"].write_text(clean_document)

    return documents


@pytest.fixture
def mock_phi_detector():
    """Mock PHI detection service."""
    with patch("hipaa_compliance_summarizer.phi.PHIDetector") as mock:
        detector = Mock()
        detector.detect_phi.return_value = {
            "entities": [
                {"type": "PERSON", "text": "John Doe", "confidence": 0.98},
                {"type": "DATE", "text": "01/15/1980", "confidence": 0.95},
                {"type": "ID", "text": "123456789", "confidence": 0.99},
            ]
        }
        mock.return_value = detector
        yield detector


@pytest.fixture
def mock_redactor():
    """Mock redaction service."""
    with patch("hipaa_compliance_summarizer.phi.Redactor") as mock:
        redactor = Mock()
        redactor.redact_text.return_value = {
            "redacted_text": "Patient: [NAME] DOB: [DATE] MRN: [ID]",
            "redaction_count": 3,
            "confidence_score": 0.97,
        }
        mock.return_value = redactor
        yield redactor


@pytest.fixture
def mock_compliance_checker():
    """Mock compliance checking service."""
    with patch("hipaa_compliance_summarizer.security.ComplianceChecker") as mock:
        checker = Mock()
        checker.check_compliance.return_value = {
            "overall_score": 0.96,
            "hipaa_compliance": "COMPLIANT",
            "risk_level": "LOW",
            "violations": [],
            "recommendations": ["Document retention policy verified"],
        }
        mock.return_value = checker
        yield checker


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    original_env = os.environ.copy()

    # Set test environment variables
    os.environ.update(
        {
            "ENVIRONMENT": "test",
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
            "ENCRYPTION_KEY": "test-key-32-characters-long-123",
            "SECRET_KEY": "test-secret-key-for-testing",
            "DATABASE_URL": "sqlite:///:memory:",
            "CACHE_ENABLED": "false",
            "MOCK_EXTERNAL_APIS": "true",
        }
    )

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def performance_test_documents(temp_dir: str) -> list:
    """Generate multiple documents for performance testing."""
    documents = []
    base_content = """
    Patient: Test Patient {i}
    DOB: 01/{day:02d}/1980
    MRN: {mrn}
    Date: March {day}, 2024

    CHIEF COMPLAINT: Test complaint for performance testing

    ASSESSMENT: This is test document {i} for performance validation.
    """

    for i in range(100):  # Create 100 test documents
        day = (i % 28) + 1
        mrn = f"{1000000 + i:07d}"
        content = base_content.format(i=i, day=day, mrn=mrn)

        doc_path = Path(temp_dir) / f"perf_test_{i:03d}.txt"
        doc_path.write_text(content)
        documents.append(str(doc_path))

    return documents


@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "test_endpoints": ["http://localhost:8000/health"],
        "timeout": 30,
        "retry_attempts": 3,
        "expected_response_time": 2.0,
    }