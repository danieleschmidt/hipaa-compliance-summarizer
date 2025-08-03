"""Unit tests for data models."""

import pytest
from datetime import datetime, timedelta
import uuid

from hipaa_compliance_summarizer.models.phi_entity import PHIEntity, PHICategory, RedactionMethod
from hipaa_compliance_summarizer.models.audit_log import AuditEvent, AuditAction, AuditLog
from hipaa_compliance_summarizer.database.models import Document, PHIDetection, AuditRecord


class TestPHIEntity:
    """Test PHI entity model."""
    
    def test_phi_entity_creation(self):
        """Test creating a PHI entity."""
        entity = PHIEntity(
            entity_id="test-123",
            category=PHICategory.NAMES,
            value="John Doe",
            confidence_score=0.95,
            start_position=10,
            end_position=18
        )
        
        assert entity.entity_id == "test-123"
        assert entity.category == PHICategory.NAMES
        assert entity.value == "John Doe"
        assert entity.confidence_score == 0.95
        assert entity.start_position == 10
        assert entity.end_position == 18
        assert entity.detection_method == "pattern_matching"  # default
        assert entity.redaction_method == RedactionMethod.MASKING  # default
        assert entity.risk_level == "medium"  # default
        assert entity.requires_audit is True  # default
        assert entity.detected_at is not None
    
    def test_phi_entity_validation(self):
        """Test PHI entity validation."""
        # Test invalid confidence score
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
            PHIEntity(
                entity_id="test",
                category=PHICategory.NAMES,
                value="John Doe",
                confidence_score=1.5,  # Invalid
                start_position=10,
                end_position=18
            )
        
        # Test invalid positions
        with pytest.raises(ValueError, match="Start position must be less than end position"):
            PHIEntity(
                entity_id="test",
                category=PHICategory.NAMES,
                value="John Doe",
                confidence_score=0.95,
                start_position=18,
                end_position=10  # Invalid order
            )
    
    def test_risk_score_calculation(self):
        """Test risk score calculation."""
        # High-risk category
        ssn_entity = PHIEntity(
            entity_id="test-ssn",
            category=PHICategory.SOCIAL_SECURITY_NUMBERS,
            value="123-45-6789",
            confidence_score=0.98,
            start_position=0,
            end_position=11
        )
        risk_score = ssn_entity.calculate_risk_score()
        assert risk_score > 0.8  # Should be high risk
        
        # Low-risk category
        url_entity = PHIEntity(
            entity_id="test-url",
            category=PHICategory.WEB_URLS,
            value="http://example.com",
            confidence_score=0.7,
            start_position=0,
            end_position=18
        )
        risk_score = url_entity.calculate_risk_score()
        assert risk_score < 0.5  # Should be low risk
    
    def test_special_handling_detection(self):
        """Test special handling detection."""
        # High-risk category should require special handling
        ssn_entity = PHIEntity(
            entity_id="test-ssn",
            category=PHICategory.SOCIAL_SECURITY_NUMBERS,
            value="123-45-6789",
            confidence_score=0.95,
            start_position=0,
            end_position=11
        )
        assert ssn_entity.needs_special_handling() is True
        
        # High confidence should require special handling
        high_conf_entity = PHIEntity(
            entity_id="test-name",
            category=PHICategory.NAMES,
            value="John Doe",
            confidence_score=0.96,
            start_position=0,
            end_position=8
        )
        assert high_conf_entity.needs_special_handling() is True
        
        # Normal entity should not require special handling
        normal_entity = PHIEntity(
            entity_id="test-date",
            category=PHICategory.DATES,
            value="2024-01-01",
            confidence_score=0.85,
            start_position=0,
            end_position=10
        )
        assert normal_entity.needs_special_handling() is False
    
    def test_serialization(self):
        """Test PHI entity serialization."""
        entity = PHIEntity(
            entity_id="test-123",
            category=PHICategory.NAMES,
            value="John Doe",
            confidence_score=0.95,
            start_position=10,
            end_position=18
        )
        
        # Test to_dict
        data = entity.to_dict()
        assert data["entity_id"] == "test-123"
        assert data["category"] == "names"
        assert data["value"] == "John Doe"
        assert data["confidence_score"] == 0.95
        
        # Test from_dict
        restored_entity = PHIEntity.from_dict(data)
        assert restored_entity.entity_id == entity.entity_id
        assert restored_entity.category == entity.category
        assert restored_entity.value == entity.value
        assert restored_entity.confidence_score == entity.confidence_score


class TestAuditLog:
    """Test audit logging functionality."""
    
    def test_audit_event_creation(self):
        """Test creating an audit event."""
        event = AuditEvent(
            action=AuditAction.PHI_DETECTED,
            description="Detected PHI in document",
            resource_type="document",
            resource_id="doc-123",
            user_id="user-456"
        )
        
        assert event.action == AuditAction.PHI_DETECTED
        assert event.description == "Detected PHI in document"
        assert event.resource_type == "document"
        assert event.resource_id == "doc-123"
        assert event.user_id == "user-456"
        assert event.compliance_relevant is True  # default
        assert event.timestamp is not None
        assert event.event_id is not None
    
    def test_audit_event_security_classification(self):
        """Test automatic security classification."""
        # PHI detection should be classified as sensitive
        phi_event = AuditEvent(
            action=AuditAction.PHI_DETECTED,
            description="PHI detected"
        )
        assert phi_event.security_level == "sensitive"
        
        # Security violation should require investigation
        security_event = AuditEvent(
            action=AuditAction.SECURITY_VIOLATION,
            description="Unauthorized access attempt"
        )
        assert security_event.security_level == "sensitive"
        assert security_event.requires_investigation is True
        
        # Normal event should be normal security level
        normal_event = AuditEvent(
            action=AuditAction.USER_LOGIN,
            description="User logged in"
        )
        assert normal_event.security_level == "normal"
        assert normal_event.requires_investigation is False
    
    def test_audit_log_operations(self):
        """Test audit log operations."""
        audit_log = AuditLog()
        
        # Add events
        event1 = AuditEvent(action=AuditAction.PHI_DETECTED, description="PHI detected")
        event2 = AuditEvent(action=AuditAction.USER_LOGIN, description="User login")
        
        audit_log.add_event(event1)
        audit_log.add_event(event2)
        
        assert len(audit_log.events) == 2
        
        # Test filtering by action
        phi_events = audit_log.get_events_by_action(AuditAction.PHI_DETECTED)
        assert len(phi_events) == 1
        assert phi_events[0].action == AuditAction.PHI_DETECTED
        
        # Test security events
        security_events = audit_log.get_security_events()
        assert len(security_events) == 1  # PHI event is sensitive
        
        # Test compliance events
        compliance_events = audit_log.get_compliance_events()
        assert len(compliance_events) == 2  # Both are compliance relevant by default
    
    def test_audit_log_summary_report(self):
        """Test audit log summary report generation."""
        audit_log = AuditLog()
        
        # Add various events
        events = [
            AuditEvent(action=AuditAction.PHI_DETECTED, description="PHI detected", user_id="user1"),
            AuditEvent(action=AuditAction.PHI_DETECTED, description="PHI detected", user_id="user1"),
            AuditEvent(action=AuditAction.USER_LOGIN, description="User login", user_id="user2"),
            AuditEvent(action=AuditAction.SECURITY_VIOLATION, description="Security violation"),
        ]
        
        for event in events:
            audit_log.add_event(event)
        
        summary = audit_log.generate_summary_report()
        
        assert summary["log_summary"]["total_events"] == 4
        assert summary["action_breakdown"]["phi_detected"] == 2
        assert summary["action_breakdown"]["user_login"] == 1
        assert summary["user_activity"]["user1"] == 2
        assert summary["user_activity"]["user2"] == 1
        assert summary["security_metrics"]["security_events"] == 3  # 2 PHI + 1 security violation
        assert summary["security_metrics"]["investigation_required"] == 1  # security violation


class TestDatabaseModels:
    """Test database models."""
    
    def test_document_model(self):
        """Test Document model."""
        doc = Document(
            id="doc-123",
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024,
            file_hash="abc123",
            document_type="clinical_note",
            content_type="application/pdf",
            created_at=datetime.utcnow()
        )
        
        assert doc.id == "doc-123"
        assert doc.filename == "test.pdf"
        assert doc.processing_status == "pending"  # default
        assert doc.compliance_score == 0.0  # default
        assert doc.phi_entity_count == 0  # default
        
        # Test metadata operations
        metadata = {"source": "EHR", "department": "cardiology"}
        doc.set_metadata(metadata)
        retrieved_metadata = doc.get_metadata()
        assert retrieved_metadata == metadata
        
        # Test serialization
        doc_dict = doc.to_dict()
        assert doc_dict["id"] == "doc-123"
        assert doc_dict["filename"] == "test.pdf"
        assert doc_dict["metadata"] == metadata
    
    def test_phi_detection_model(self):
        """Test PHIDetection model."""
        detection = PHIDetection(
            id="phi-123",
            document_id="doc-123",
            entity_type="name",
            entity_category="names",
            entity_value="John Doe",
            redacted_value="[NAME]",
            start_position=10,
            end_position=18,
            confidence_score=0.95,
            created_at=datetime.utcnow()
        )
        
        assert detection.id == "phi-123"
        assert detection.document_id == "doc-123"
        assert detection.entity_type == "name"
        assert detection.confidence_score == 0.95
        assert detection.detection_method == "pattern_matching"  # default
        assert detection.requires_manual_review is False  # default
        
        # Test serialization
        detection_dict = detection.to_dict()
        assert detection_dict["id"] == "phi-123"
        assert detection_dict["entity_value"] == "John Doe"
        assert detection_dict["confidence_score"] == 0.95
    
    def test_audit_record_model(self):
        """Test AuditRecord model."""
        record = AuditRecord(
            id="audit-123",
            event_type="phi_detection",
            event_action="CREATE",
            event_description="PHI detected in document",
            resource_type="document",
            resource_id="doc-123",
            user_id="user-456",
            created_at=datetime.utcnow()
        )
        
        assert record.id == "audit-123"
        assert record.event_type == "phi_detection"
        assert record.event_action == "CREATE"
        assert record.resource_type == "document"
        assert record.success is True  # default
        assert record.security_level == "normal"  # default
        assert record.retention_period_days == 2555  # default (7 years)
        
        # Test event data operations
        event_data = {"phi_count": 5, "confidence": 0.95}
        record.set_event_data(event_data)
        retrieved_data = record.get_event_data()
        assert retrieved_data == event_data
        
        # Test serialization
        record_dict = record.to_dict()
        assert record_dict["id"] == "audit-123"
        assert record_dict["event_type"] == "phi_detection"
        assert record_dict["event_data"] == event_data


class TestModelValidation:
    """Test model validation and constraints."""
    
    def test_phi_entity_constraints(self):
        """Test PHI entity constraints."""
        # Test all HIPAA categories are supported
        for category in PHICategory:
            entity = PHIEntity(
                entity_id=f"test-{category.value}",
                category=category,
                value="test value",
                confidence_score=0.9,
                start_position=0,
                end_position=10
            )
            assert entity.category == category
        
        # Test all redaction methods are supported
        for method in RedactionMethod:
            entity = PHIEntity(
                entity_id="test",
                category=PHICategory.NAMES,
                value="test",
                confidence_score=0.9,
                start_position=0,
                end_position=4,
                redaction_method=method
            )
            assert entity.redaction_method == method
    
    def test_audit_actions_coverage(self):
        """Test that all audit actions are properly defined."""
        # Test a few key actions
        key_actions = [
            AuditAction.PHI_DETECTED,
            AuditAction.PHI_REDACTED,
            AuditAction.DOCUMENT_PROCESSED,
            AuditAction.SECURITY_VIOLATION,
            AuditAction.USER_LOGIN,
        ]
        
        for action in key_actions:
            event = AuditEvent(action=action, description="test")
            assert event.action == action
    
    def test_timestamp_handling(self):
        """Test timestamp handling across models."""
        now = datetime.utcnow()
        
        # Test PHI entity timestamp
        entity = PHIEntity(
            entity_id="test",
            category=PHICategory.NAMES,
            value="test",
            confidence_score=0.9,
            start_position=0,
            end_position=4,
            detected_at=now
        )
        assert entity.detected_at == now
        
        # Test audit event timestamp
        event = AuditEvent(
            action=AuditAction.PHI_DETECTED,
            description="test",
            timestamp=now
        )
        assert event.timestamp == now