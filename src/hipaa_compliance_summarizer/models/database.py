"""SQLAlchemy database models for HIPAA compliance system."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum as SqlEnum,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

from .audit_log import AuditAction

Base = declarative_base()


class ProcessingSession(Base):
    """Tracks individual document processing sessions for audit purposes."""
    
    __tablename__ = "processing_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    
    # Processing details
    started_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(50), nullable=False, default="processing")  # processing, completed, failed
    
    # Compliance settings
    compliance_level = Column(String(50), nullable=False)
    redaction_method = Column(String(100), nullable=False)
    
    # Audit trail
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    user_agent = Column(Text, nullable=True)
    
    # Metrics
    documents_processed = Column(Integer, default=0)
    phi_entities_detected = Column(Integer, default=0)
    phi_entities_redacted = Column(Integer, default=0)
    total_processing_time_ms = Column(Integer, default=0)
    
    # Relationships
    documents = relationship("ProcessedDocument", back_populates="session")
    phi_detections = relationship("PHIDetection", back_populates="session")
    audit_events = relationship("AuditEventRecord", back_populates="session")


class ProcessedDocument(Base):
    """Records details of each processed document."""
    
    __tablename__ = "processed_documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(String(255), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Document metadata
    original_filename = Column(String(500), nullable=False)
    document_type = Column(String(100), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False)  # SHA-256
    
    # Processing results
    processed_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    processing_time_ms = Column(Integer, nullable=False)
    compliance_score = Column(Float, nullable=False)
    
    # PHI analysis results
    phi_entities_count = Column(Integer, nullable=False, default=0)
    redaction_summary = Column(JSON, nullable=True)  # Count by PHI type
    confidence_scores = Column(JSON, nullable=True)  # Confidence by PHI type
    
    # Content summary (PHI-free)
    clinical_summary = Column(JSON, nullable=True)
    risk_assessment = Column(String(50), nullable=True)  # LOW, MEDIUM, HIGH, CRITICAL
    
    # HIPAA compliance
    hipaa_compliant = Column(Boolean, nullable=False, default=True)
    compliance_notes = Column(Text, nullable=True)
    retention_until = Column(DateTime(timezone=True), nullable=False)
    
    # Relationships
    session = relationship("ProcessingSession", back_populates="documents")
    phi_detections = relationship("PHIDetection", back_populates="document")


class PHIDetection(Base):
    """Individual PHI entity detections for detailed tracking."""
    
    __tablename__ = "phi_detections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # PHI details
    phi_type = Column(String(100), nullable=False, index=True)  # SSN, NAME, DATE_OF_BIRTH, etc.
    detected_text = Column(String(500), nullable=True)  # Encrypted/hashed original text
    redacted_text = Column(String(500), nullable=True)  # What it was replaced with
    
    # Detection metadata
    confidence_score = Column(Float, nullable=False)
    detection_method = Column(String(100), nullable=False)  # regex, ml_model, hybrid
    position_start = Column(Integer, nullable=True)
    position_end = Column(Integer, nullable=True)
    
    # Context preservation
    clinical_context = Column(String(1000), nullable=True)  # Surrounding non-PHI context
    preserved_for_analysis = Column(Boolean, default=False)
    
    # Audit
    detected_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    reviewed_by_human = Column(Boolean, default=False)
    false_positive = Column(Boolean, nullable=True)  # Null = not reviewed
    
    # Relationships
    document = relationship("ProcessedDocument", back_populates="phi_detections")
    session = relationship("ProcessingSession", back_populates="phi_detections")


class AuditEventRecord(Base):
    """Persistent storage for audit events."""
    
    __tablename__ = "audit_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Core event data
    action = Column(SqlEnum(AuditAction), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # User and session
    user_id = Column(String(255), nullable=True, index=True)
    session_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Resource information
    resource_type = Column(String(100), nullable=True, index=True)
    resource_id = Column(String(255), nullable=True, index=True)
    resource_path = Column(String(1000), nullable=True)
    
    # Event details
    description = Column(Text, nullable=True)
    details = Column(JSON, nullable=True)
    
    # Compliance and security
    compliance_relevant = Column(Boolean, default=True, index=True)
    retention_until = Column(DateTime(timezone=True), nullable=False)
    security_level = Column(String(50), nullable=False, default="normal", index=True)
    requires_investigation = Column(Boolean, default=False, index=True)
    
    # Relationships
    session = relationship("ProcessingSession", back_populates="audit_events")


class ComplianceReport(Base):
    """Generated compliance reports for audit periods."""
    
    __tablename__ = "compliance_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Report metadata
    generated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    generated_by = Column(String(255), nullable=True)
    report_period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    report_period_end = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Report data
    documents_processed = Column(Integer, nullable=False, default=0)
    phi_entities_detected = Column(Integer, nullable=False, default=0)
    phi_entities_redacted = Column(Integer, nullable=False, default=0)
    
    # Compliance metrics
    overall_compliance_rate = Column(Float, nullable=False)
    average_confidence_score = Column(Float, nullable=False)
    violations_detected = Column(Integer, nullable=False, default=0)
    false_positive_rate = Column(Float, nullable=True)
    
    # Risk assessment
    high_risk_documents = Column(Integer, nullable=False, default=0)
    medium_risk_documents = Column(Integer, nullable=False, default=0)
    low_risk_documents = Column(Integer, nullable=False, default=0)
    
    # Performance metrics
    average_processing_time_ms = Column(Integer, nullable=False)
    cache_hit_rate = Column(Float, nullable=True)
    
    # Report content
    summary = Column(Text, nullable=True)
    recommendations = Column(JSON, nullable=True)
    detailed_metrics = Column(JSON, nullable=True)
    
    # Retention
    retention_until = Column(DateTime(timezone=True), nullable=False)


class SystemConfiguration(Base):
    """System configuration changes for audit trail."""
    
    __tablename__ = "system_configurations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Configuration metadata
    changed_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    changed_by = Column(String(255), nullable=True)
    change_reason = Column(Text, nullable=True)
    
    # Configuration data
    configuration_key = Column(String(255), nullable=False, index=True)
    previous_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=False)
    
    # Impact tracking
    affected_systems = Column(JSON, nullable=True)
    requires_restart = Column(Boolean, default=False)
    rollback_available = Column(Boolean, default=True)
    
    # Approval workflow
    approved_by = Column(String(255), nullable=True)
    approval_timestamp = Column(DateTime(timezone=True), nullable=True)
    
    # Compliance impact
    compliance_impact_assessment = Column(Text, nullable=True)
    security_review_required = Column(Boolean, default=False)


# Database utility functions and session management

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False  # Set to True for SQL logging
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def health_check(self) -> bool:
        """Perform database health check."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception:
            return False


# Auto-update retention dates on audit events
@event.listens_for(AuditEventRecord, 'before_insert')
def set_audit_retention(mapper, connection, target):
    """Automatically set retention date based on compliance requirements."""
    if target.retention_until is None:
        retention_days = 2555  # 7 years for HIPAA compliance
        target.retention_until = datetime.utcnow().replace(
            year=target.timestamp.year + 7
        )


@event.listens_for(ProcessedDocument, 'before_insert')
def set_document_retention(mapper, connection, target):
    """Set document retention period based on HIPAA requirements."""
    if target.retention_until is None:
        # 7 years from processing date
        target.retention_until = target.processed_at.replace(
            year=target.processed_at.year + 7
        )


@event.listens_for(ComplianceReport, 'before_insert') 
def set_report_retention(mapper, connection, target):
    """Set compliance report retention period."""
    if target.retention_until is None:
        # 10 years for compliance reports
        target.retention_until = target.generated_at.replace(
            year=target.generated_at.year + 10
        )