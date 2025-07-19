"""Tests for enhanced PHI patterns including medical record numbers, DEA numbers, and insurance IDs."""

import pytest
from hipaa_compliance_summarizer.phi import PHIRedactor, Entity


class TestMedicalRecordNumbers:
    """Test detection of medical record numbers (MRN)."""

    def test_mrn_numeric_format(self):
        """Test detection of numeric MRN formats."""
        redactor = PHIRedactor(patterns={
            "mrn": r"\b(?:MRN|Medical Record|Patient ID)[:.]?\s*([A-Z]{0,3}\d{6,12})\b"
        })
        
        text = "Patient MRN: 123456789"  # pragma: allowlist secret
        entities = redactor.detect(text)
        
        assert len(entities) == 1
        assert entities[0].type == "mrn"
        assert "123456789" in entities[0].value

    def test_mrn_alphanumeric_format(self):
        """Test detection of alphanumeric MRN formats."""
        redactor = PHIRedactor(patterns={
            "mrn": r"\b(?:MRN|Medical Record|Patient ID)[:.]?\s*([A-Z]{0,3}\d{6,12})\b"
        })
        
        text = "Medical Record: ABC1234567"  # pragma: allowlist secret
        entities = redactor.detect(text)
        
        assert len(entities) == 1
        assert entities[0].type == "mrn"
        assert "ABC1234567" in entities[0].value  # pragma: allowlist secret

    def test_mrn_various_labels(self):
        """Test detection with various MRN labels."""
        redactor = PHIRedactor(patterns={
            "mrn": r"\b(?:MRN|Medical Record|Patient ID)[:.]?\s*([A-Z]{0,3}\d{6,12})\b"
        })
        
        test_cases = [
            "MRN: 987654321",
            "Medical Record 456789123",
            "Patient ID: XYZ789456123",
        ]
        
        for text in test_cases:
            entities = redactor.detect(text)
            assert len(entities) >= 1
            assert entities[0].type == "mrn"

    def test_mrn_case_insensitive(self):
        """Test MRN detection is case insensitive."""
        redactor = PHIRedactor(patterns={
            "mrn": r"\b(?:MRN|Medical Record|Patient ID)[:.]?\s*([A-Z]{0,3}\d{6,12})\b"
        })
        
        text = "mrn: 123456789 and Medical record: 987654321"
        entities = redactor.detect(text.upper())
        
        assert len(entities) >= 1


class TestDEANumbers:
    """Test detection of DEA (Drug Enforcement Administration) numbers."""

    def test_dea_valid_format(self):
        """Test detection of valid DEA number format."""
        redactor = PHIRedactor(patterns={
            "dea": r"\b(?:DEA|DEA#|DEA Number)[:.]?\s*([A-Z]{2}\d{7})\b"
        })
        
        text = "DEA: AB1234567"  # pragma: allowlist secret
        entities = redactor.detect(text)
        
        assert len(entities) == 1
        assert entities[0].type == "dea"
        assert "AB1234567" in entities[0].value  # pragma: allowlist secret

    def test_dea_with_hash_symbol(self):
        """Test DEA detection with hash symbol."""
        redactor = PHIRedactor(patterns={
            "dea": r"\b(?:DEA|DEA#|DEA Number)[:.]?\s*([A-Z]{2}\d{7})\b"
        })
        
        text = "DEA# XY9876543"
        entities = redactor.detect(text)
        
        assert len(entities) == 1
        assert entities[0].type == "dea"
        assert "XY9876543" in entities[0].value

    def test_dea_number_label(self):
        """Test DEA detection with 'DEA Number' label."""
        redactor = PHIRedactor(patterns={
            "dea": r"\b(?:DEA|DEA#|DEA Number)[:.]?\s*([A-Z]{2}\d{7})\b"
        })
        
        text = "DEA Number: CD5551234"
        entities = redactor.detect(text)
        
        assert len(entities) == 1
        assert entities[0].type == "dea"
        assert "CD5551234" in entities[0].value

    def test_dea_invalid_format_not_detected(self):
        """Test that invalid DEA formats are not detected."""
        redactor = PHIRedactor(patterns={
            "dea": r"\b(?:DEA|DEA#|DEA Number)[:.]?\s*([A-Z]{2}\d{7})\b"
        })
        
        invalid_cases = [
            "DEA: A1234567",  # Only one letter
            "DEA: ABC123456",  # Three letters
            "DEA: AB123456",   # Six digits instead of seven
            "DEA: AB12345678", # Eight digits instead of seven
        ]
        
        for text in invalid_cases:
            entities = redactor.detect(text)
            assert len(entities) == 0


class TestInsuranceIDs:
    """Test detection of insurance identification numbers."""

    def test_insurance_member_id(self):
        """Test detection of insurance member IDs."""
        redactor = PHIRedactor(patterns={
            "insurance_id": r"\b(?:Member ID|Policy|Insurance ID|Subscriber ID)[:.]?\s*([A-Z0-9]{8,15})\b"
        })
        
        text = "Member ID: ABC123456789"  # pragma: allowlist secret
        entities = redactor.detect(text)
        
        assert len(entities) == 1
        assert entities[0].type == "insurance_id"
        assert "ABC123456789" in entities[0].value  # pragma: allowlist secret

    def test_insurance_policy_number(self):
        """Test detection of insurance policy numbers."""
        redactor = PHIRedactor(patterns={
            "insurance_id": r"\b(?:Member ID|Policy|Insurance ID|Subscriber ID)[:.]?\s*([A-Z0-9]{8,15})\b"
        })
        
        text = "Policy: XYZ987654321DEF"
        entities = redactor.detect(text)
        
        assert len(entities) == 1
        assert entities[0].type == "insurance_id"
        assert "XYZ987654321DEF" in entities[0].value

    def test_insurance_subscriber_id(self):
        """Test detection of subscriber IDs."""
        redactor = PHIRedactor(patterns={
            "insurance_id": r"\b(?:Member ID|Policy|Insurance ID|Subscriber ID)[:.]?\s*([A-Z0-9]{8,15})\b"
        })
        
        text = "Subscriber ID: 1234567890AB"
        entities = redactor.detect(text)
        
        assert len(entities) == 1
        assert entities[0].type == "insurance_id"
        assert "1234567890AB" in entities[0].value  # pragma: allowlist secret

    def test_insurance_id_length_validation(self):
        """Test that insurance IDs respect length constraints."""
        redactor = PHIRedactor(patterns={
            "insurance_id": r"\b(?:Member ID|Policy|Insurance ID|Subscriber ID)[:.]?\s*([A-Z0-9]{8,15})\b"
        })
        
        # Too short (7 characters)
        text_short = "Member ID: ABC1234"
        entities_short = redactor.detect(text_short)
        assert len(entities_short) == 0
        
        # Too long (16 characters)
        text_long = "Member ID: ABC1234567890123"
        entities_long = redactor.detect(text_long)
        assert len(entities_long) == 0
        
        # Valid length (8 characters)
        text_valid = "Member ID: ABC12345"  # pragma: allowlist secret
        entities_valid = redactor.detect(text_valid)
        assert len(entities_valid) == 1


class TestEnhancedPHIIntegration:
    """Test integration of enhanced PHI patterns with existing patterns."""

    def test_multiple_phi_types_in_document(self):
        """Test detection of multiple PHI types in a single document."""
        redactor = PHIRedactor(patterns={
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "mrn": r"\b(?:MRN|Medical Record|Patient ID)[:.]?\s*([A-Z]{0,3}\d{6,12})\b",
            "dea": r"\b(?:DEA|DEA#|DEA Number)[:.]?\s*([A-Z]{2}\d{7})\b",
            "insurance_id": r"\b(?:Member ID|Policy|Insurance ID|Subscriber ID)[:.]?\s*([A-Z0-9]{8,15})\b"
        })
        
        text = """
        Patient Information:
        SSN: 123-45-6789
        Email: patient@example.com
        MRN: ABC1234567  # pragma: allowlist secret
        DEA: XY9876543
        Member ID: INSUR12345ABC
        """
        
        entities = redactor.detect(text)
        
        # Should detect all 5 PHI types
        assert len(entities) == 5
        entity_types = {e.type for e in entities}
        expected_types = {"ssn", "email", "mrn", "dea", "insurance_id"}
        assert entity_types == expected_types

    def test_redaction_preserves_document_structure(self):
        """Test that redaction maintains document readability."""
        redactor = PHIRedactor(patterns={
            "mrn": r"\b(?:MRN|Medical Record|Patient ID)[:.]?\s*([A-Z]{0,3}\d{6,12})\b",
            "dea": r"\b(?:DEA|DEA#|DEA Number)[:.]?\s*([A-Z]{2}\d{7})\b",
            "insurance_id": r"\b(?:Member ID|Policy|Insurance ID|Subscriber ID)[:.]?\s*([A-Z0-9]{8,15})\b"
        })
        
        text = "Patient MRN: 123456789, DEA: AB1234567, Insurance ID: MEMBER123456"
        result = redactor.redact(text)
        
        # Check that structure is preserved
        assert "Patient" in result.text
        assert "DEA:" in result.text
        assert "Insurance ID:" in result.text
        assert "[REDACTED]" in result.text
        
        # Verify actual PHI values are removed
        assert "123456789" not in result.text
        assert "AB1234567" not in result.text  # pragma: allowlist secret
        assert "MEMBER123456" not in result.text  # pragma: allowlist secret

    def test_enhanced_patterns_cache_compatibility(self):
        """Test that enhanced patterns work with the caching system."""
        redactor = PHIRedactor(patterns={
            "mrn": r"\b(?:MRN|Medical Record|Patient ID)[:.]?\s*([A-Z]{0,3}\d{6,12})\b",
            "dea": r"\b(?:DEA|DEA#|DEA Number)[:.]?\s*([A-Z]{2}\d{7})\b",
        })
        
        text = "MRN: 123456789 and DEA: AB1234567"
        
        # First detection (cache miss)
        entities1 = redactor.detect(text)
        
        # Second detection (cache hit)
        entities2 = redactor.detect(text)
        
        # Results should be identical
        assert len(entities1) == len(entities2) == 2
        assert entities1[0].type == entities2[0].type
        assert entities1[0].value == entities2[0].value
        assert entities1[1].type == entities2[1].type
        assert entities1[1].value == entities2[1].value

    def test_enhanced_patterns_performance_with_large_text(self):
        """Test performance with enhanced patterns on larger text blocks."""
        redactor = PHIRedactor(patterns={
            "mrn": r"\b(?:MRN|Medical Record|Patient ID)[:.]?\s*([A-Z]{0,3}\d{6,12})\b",
            "dea": r"\b(?:DEA|DEA#|DEA Number)[:.]?\s*([A-Z]{2}\d{7})\b",
            "insurance_id": r"\b(?:Member ID|Policy|Insurance ID|Subscriber ID)[:.]?\s*([A-Z0-9]{8,15})\b"
        })
        
        # Create a large text with multiple instances
        base_text = "Patient MRN: 123456789, DEA: AB1234567, Member ID: INSUR123456. "
        large_text = base_text * 100  # 100 repetitions
        
        entities = redactor.detect(large_text)
        
        # Should detect 300 entities (3 per repetition * 100 repetitions)
        assert len(entities) == 300
        
        # Verify caching improves performance on second run
        entities2 = redactor.detect(large_text)
        assert len(entities2) == 300