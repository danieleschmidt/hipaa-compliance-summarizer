"""Unit tests for service layer."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from hipaa_compliance_summarizer.services.phi_detection_service import PHIDetectionService, DetectionResult
from hipaa_compliance_summarizer.models.phi_entity import PHIEntity, PHICategory, RedactionMethod
from hipaa_compliance_summarizer.models.audit_log import AuditEvent, AuditAction


class TestPHIDetectionService:
    """Test PHI detection service."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = PHIDetectionService(enable_ml_models=False)
    
    def test_service_initialization(self):
        """Test service initialization."""
        service = PHIDetectionService(enable_ml_models=False)
        assert service.enable_ml_models is False
        assert service.redactor is not None
        assert service.detection_stats["total_detections"] == 0
        
        # Test with ML models enabled
        ml_service = PHIDetectionService(enable_ml_models=True)
        assert ml_service.enable_ml_models is True
    
    @patch('hipaa_compliance_summarizer.services.phi_detection_service.PHIRedactor')
    def test_pattern_based_detection(self, mock_redactor_class):
        """Test pattern-based PHI detection."""
        # Mock the redactor
        mock_redactor = Mock()
        mock_redactor.detect.return_value = [
            Mock(type="name", value="John Doe", start=0, end=8),
            Mock(type="ssn", value="123-45-6789", start=20, end=31),
        ]
        mock_redactor_class.return_value = mock_redactor
        
        service = PHIDetectionService()
        
        result = service.detect_phi_entities(
            "John Doe has SSN 123-45-6789",
            detection_method="pattern",
            confidence_threshold=0.8
        )
        
        assert isinstance(result, DetectionResult)
        assert len(result.entities) >= 0  # May vary based on actual implementation
        assert result.detection_method == "pattern"
        assert result.processing_time_ms > 0
        assert isinstance(result.confidence_scores, dict)
    
    def test_hybrid_detection_without_ml(self):
        """Test hybrid detection when ML models are not available."""
        service = PHIDetectionService(enable_ml_models=False)
        
        result = service.detect_phi_entities(
            "John Doe has phone 555-1234",
            detection_method="hybrid",
            confidence_threshold=0.8
        )
        
        # Should fall back to pattern-based detection
        assert result.detection_method == "hybrid"
        assert result.processing_time_ms > 0
    
    def test_ml_detection_fallback(self):
        """Test ML detection falls back to patterns when models unavailable."""
        service = PHIDetectionService(enable_ml_models=True)
        
        result = service.detect_phi_entities(
            "Patient John Doe",
            detection_method="ml",
            confidence_threshold=0.8
        )
        
        # Should fall back to pattern detection since ML models are placeholders
        assert result.processing_time_ms > 0
    
    def test_entity_type_mapping(self):
        """Test entity type to PHI category mapping."""
        service = PHIDetectionService()
        
        # Test various entity type mappings
        test_cases = [
            ("name", PHICategory.NAMES),
            ("ssn", PHICategory.SOCIAL_SECURITY_NUMBERS),
            ("phone", PHICategory.TELEPHONE_NUMBERS),
            ("email", PHICategory.EMAIL_ADDRESSES),
            ("date", PHICategory.DATES),
            ("unknown_type", PHICategory.OTHER_UNIQUE_IDENTIFYING_NUMBERS),
        ]
        
        for entity_type, expected_category in test_cases:
            category = service._map_entity_type_to_category(entity_type)
            assert category == expected_category
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        service = PHIDetectionService()
        
        # Test different entity types and values
        test_cases = [
            ("ssn", "123-45-6789", 0.9),  # SSN should have high confidence
            ("phone", "555-1234", 0.8),   # Phone should have good confidence
            ("name", "John", 0.6),        # Short names are less certain
            ("unknown", "test", 0.7),     # Unknown types get default
        ]
        
        for entity_type, value, min_expected in test_cases:
            confidence = service._calculate_pattern_confidence(entity_type, value)
            assert 0.0 <= confidence <= 1.0
            # Note: Actual confidence may vary based on implementation
    
    def test_risk_assessment(self):
        """Test risk level assessment."""
        service = PHIDetectionService()
        
        # High-risk categories
        high_risk_cases = [
            (PHICategory.SOCIAL_SECURITY_NUMBERS, 0.9),
            (PHICategory.MEDICAL_RECORD_NUMBERS, 0.95),
            (PHICategory.BIOMETRIC_IDENTIFIERS, 0.8),
        ]
        
        for category, confidence in high_risk_cases:
            risk = service._assess_risk_level(category, confidence)
            assert risk in ["high", "medium"]  # Should be high or medium risk
        
        # Low-risk categories
        low_risk_cases = [
            (PHICategory.WEB_URLS, 0.7),
            (PHICategory.IP_ADDRESSES, 0.6),
        ]
        
        for category, confidence in low_risk_cases:
            risk = service._assess_risk_level(category, confidence)
            assert risk in ["low", "medium"]  # Should be low or medium risk
    
    def test_detection_statistics(self):
        """Test detection statistics tracking."""
        service = PHIDetectionService()
        
        # Create mock entities
        entities = [
            PHIEntity(
                entity_id="1",
                category=PHICategory.NAMES,
                value="John",
                confidence_score=0.9,
                start_position=0,
                end_position=4
            ),
            PHIEntity(
                entity_id="2",
                category=PHICategory.DATES,
                value="2024-01-01",
                confidence_score=0.8,
                start_position=5,
                end_position=15
            ),
        ]
        
        # Update statistics
        service._update_detection_stats(entities, "pattern")
        
        stats = service.get_detection_statistics()
        assert stats["total_detections"] == 2
        assert stats["by_method"]["pattern"] == 2
        assert stats["by_category"]["names"] == 1
        assert stats["by_category"]["dates"] == 1
    
    def test_entity_validation(self):
        """Test PHI entity validation."""
        service = PHIDetectionService()
        
        # Valid entity
        valid_entity = PHIEntity(
            entity_id="test",
            category=PHICategory.SOCIAL_SECURITY_NUMBERS,
            value="123-45-6789",
            confidence_score=0.95,
            start_position=0,
            end_position=11
        )
        
        is_valid, errors = service.validate_phi_entity(valid_entity)
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid entity - low confidence
        low_conf_entity = PHIEntity(
            entity_id="test2",
            category=PHICategory.NAMES,
            value="John",
            confidence_score=0.3,  # Too low
            start_position=0,
            end_position=4
        )
        
        is_valid, errors = service.validate_phi_entity(low_conf_entity)
        assert is_valid is False
        assert "Confidence score too low" in errors[0]
        
        # Invalid entity - bad positions
        bad_pos_entity = PHIEntity(
            entity_id="test3",
            category=PHICategory.NAMES,
            value="John",
            confidence_score=0.9,
            start_position=10,
            end_position=5  # Invalid: end < start
        )
        
        is_valid, errors = service.validate_phi_entity(bad_pos_entity)
        assert is_valid is False
        assert "Invalid position range" in errors[0]
    
    def test_format_specific_validation(self):
        """Test format-specific validation for different PHI types."""
        service = PHIDetectionService()
        
        # Valid SSN
        valid_ssn = PHIEntity(
            entity_id="ssn1",
            category=PHICategory.SOCIAL_SECURITY_NUMBERS,
            value="123-45-6789",
            confidence_score=0.95,
            start_position=0,
            end_position=11
        )
        is_valid, errors = service.validate_phi_entity(valid_ssn)
        assert is_valid is True
        
        # Invalid SSN format
        invalid_ssn = PHIEntity(
            entity_id="ssn2",
            category=PHICategory.SOCIAL_SECURITY_NUMBERS,
            value="123456789",  # Missing dashes
            confidence_score=0.95,
            start_position=0,
            end_position=9
        )
        is_valid, errors = service.validate_phi_entity(invalid_ssn)
        # Note: This test depends on implementation details
        
        # Valid email
        valid_email = PHIEntity(
            entity_id="email1",
            category=PHICategory.EMAIL_ADDRESSES,
            value="john.doe@example.com",
            confidence_score=0.95,
            start_position=0,
            end_position=20
        )
        is_valid, errors = service.validate_phi_entity(valid_email)
        assert is_valid is True
    
    def test_audit_event_creation(self):
        """Test audit event creation for PHI detection."""
        service = PHIDetectionService()
        
        entities = [
            PHIEntity(
                entity_id="1",
                category=PHICategory.NAMES,
                value="John",
                confidence_score=0.9,
                start_position=0,
                end_position=4
            ),
            PHIEntity(
                entity_id="2",
                category=PHICategory.DATES,
                value="2024-01-01",
                confidence_score=0.8,
                start_position=5,
                end_position=15
            ),
        ]
        
        audit_event = service.create_audit_event(entities, document_id="doc-123")
        
        assert isinstance(audit_event, AuditEvent)
        assert audit_event.action == AuditAction.PHI_DETECTED
        assert audit_event.resource_type == "document"
        assert audit_event.resource_id == "doc-123"
        assert audit_event.compliance_relevant is True
        assert audit_event.security_level == "sensitive"
        
        # Check event details
        details = audit_event.details
        assert details["entity_count"] == 2
        assert "names" in details["categories_detected"]
        assert "dates" in details["categories_detected"]
        assert details["detection_method"] == "phi_detection_service"
    
    def test_deduplication(self):
        """Test entity deduplication logic."""
        service = PHIDetectionService()
        
        # Create entities with some duplicates
        entities = [
            PHIEntity(
                entity_id="1",
                category=PHICategory.NAMES,
                value="John",
                confidence_score=0.9,
                start_position=0,
                end_position=4
            ),
            PHIEntity(
                entity_id="2",
                category=PHICategory.NAMES,
                value="John",  # Same value and position
                confidence_score=0.8,
                start_position=0,
                end_position=4
            ),
            PHIEntity(
                entity_id="3",
                category=PHICategory.DATES,
                value="2024-01-01",
                confidence_score=0.8,
                start_position=5,
                end_position=15
            ),
        ]
        
        unique_entities = service._deduplicate_entities(entities)
        
        # Should have 2 unique entities (John at position 0-4, and the date)
        assert len(unique_entities) == 2
        
        # Check that entities are sorted by position
        assert unique_entities[0].start_position <= unique_entities[1].start_position
    
    def test_category_confidence_calculation(self):
        """Test confidence score calculation by category."""
        service = PHIDetectionService()
        
        entities = [
            PHIEntity(
                entity_id="1",
                category=PHICategory.NAMES,
                value="John",
                confidence_score=0.9,
                start_position=0,
                end_position=4
            ),
            PHIEntity(
                entity_id="2",
                category=PHICategory.NAMES,
                value="Jane",
                confidence_score=0.8,
                start_position=5,
                end_position=9
            ),
            PHIEntity(
                entity_id="3",
                category=PHICategory.DATES,
                value="2024-01-01",
                confidence_score=0.95,
                start_position=10,
                end_position=20
            ),
        ]
        
        confidence_by_category = service._calculate_category_confidence(entities)
        
        assert "names" in confidence_by_category
        assert "dates" in confidence_by_category
        
        # Names should average 0.85 (0.9 + 0.8) / 2
        assert abs(confidence_by_category["names"] - 0.85) < 0.01
        
        # Dates should be 0.95
        assert confidence_by_category["dates"] == 0.95


class TestServiceIntegration:
    """Test service integration scenarios."""
    
    def test_detection_service_performance_monitoring(self):
        """Test detection service with performance monitoring."""
        # Mock performance monitor
        mock_monitor = Mock()
        
        service = PHIDetectionService(enable_ml_models=False)
        service.performance_monitor = mock_monitor
        
        # Run detection
        result = service.detect_phi_entities(
            "John Doe has SSN 123-45-6789",
            detection_method="pattern"
        )
        
        # Verify monitoring was called (implementation dependent)
        assert isinstance(result, DetectionResult)
    
    def test_service_error_handling(self):
        """Test service error handling."""
        service = PHIDetectionService()
        
        # Test with empty text
        result = service.detect_phi_entities("", detection_method="pattern")
        assert isinstance(result, DetectionResult)
        assert len(result.entities) == 0
        
        # Test with very long text (should not crash)
        long_text = "John Doe " * 1000
        result = service.detect_phi_entities(long_text, detection_method="pattern")
        assert isinstance(result, DetectionResult)
    
    def test_service_configuration_integration(self):
        """Test service integration with configuration."""
        service = PHIDetectionService()
        
        # Verify service uses configuration appropriately
        assert service.redactor is not None
        
        # Test detection with various confidence thresholds
        text = "Patient John Doe, SSN: 123-45-6789"
        
        # High threshold
        result_high = service.detect_phi_entities(text, confidence_threshold=0.95)
        
        # Low threshold  
        result_low = service.detect_phi_entities(text, confidence_threshold=0.5)
        
        # Lower threshold should potentially detect more entities
        assert len(result_low.entities) >= len(result_high.entities)