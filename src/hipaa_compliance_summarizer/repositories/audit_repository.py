"""Repository for audit and compliance data operations."""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session

from ..models.audit_log import AuditAction, AuditEvent
from ..models.database import (
    AuditEventRecord,
    ComplianceReport,
    PHIDetection,
    ProcessedDocument,
    ProcessingSession,
)


class AuditRepository:
    """Repository for audit trail and compliance data operations."""

    def __init__(self, session: Session):
        self.session = session

    # Processing Session Operations
    def create_processing_session(
        self,
        user_id: Optional[str] = None,
        compliance_level: str = "standard",
        redaction_method: str = "synthetic_replacement",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> ProcessingSession:
        """Create a new processing session."""
        session = ProcessingSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            compliance_level=compliance_level,
            redaction_method=redaction_method,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        self.session.add(session)
        self.session.commit()
        self.session.refresh(session)
        return session

    def complete_processing_session(
        self,
        session_id: str,
        status: str = "completed"
    ) -> Optional[ProcessingSession]:
        """Mark a processing session as completed."""
        session = self.session.query(ProcessingSession).filter(
            ProcessingSession.session_id == session_id
        ).first()

        if session:
            session.completed_at = datetime.utcnow()
            session.status = status
            self.session.commit()
            self.session.refresh(session)

        return session

    def get_processing_session(self, session_id: str) -> Optional[ProcessingSession]:
        """Retrieve a processing session by ID."""
        return self.session.query(ProcessingSession).filter(
            ProcessingSession.session_id == session_id
        ).first()

    # Document Operations
    def record_processed_document(
        self,
        session_id: str,
        document_id: str,
        original_filename: str,
        document_type: str,
        file_size_bytes: int,
        content_hash: str,
        processing_time_ms: int,
        compliance_score: float,
        phi_entities_count: int = 0,
        redaction_summary: Optional[Dict[str, Any]] = None,
        confidence_scores: Optional[Dict[str, Any]] = None,
        clinical_summary: Optional[Dict[str, Any]] = None,
        risk_assessment: str = "LOW",
        hipaa_compliant: bool = True,
        compliance_notes: Optional[str] = None,
    ) -> ProcessedDocument:
        """Record a processed document in the database."""

        # Get session record
        processing_session = self.get_processing_session(session_id)
        if not processing_session:
            raise ValueError(f"Processing session {session_id} not found")

        document = ProcessedDocument(
            document_id=document_id,
            session_id=processing_session.id,
            original_filename=original_filename,
            document_type=document_type,
            file_size_bytes=file_size_bytes,
            content_hash=content_hash,
            processing_time_ms=processing_time_ms,
            compliance_score=compliance_score,
            phi_entities_count=phi_entities_count,
            redaction_summary=redaction_summary or {},
            confidence_scores=confidence_scores or {},
            clinical_summary=clinical_summary or {},
            risk_assessment=risk_assessment,
            hipaa_compliant=hipaa_compliant,
            compliance_notes=compliance_notes,
        )

        self.session.add(document)

        # Update session metrics
        processing_session.documents_processed += 1
        processing_session.phi_entities_detected += phi_entities_count
        processing_session.total_processing_time_ms += processing_time_ms

        self.session.commit()
        self.session.refresh(document)
        return document

    # PHI Detection Operations
    def record_phi_detection(
        self,
        document_id: str,
        session_id: str,
        phi_type: str,
        confidence_score: float,
        detection_method: str = "regex",
        detected_text: Optional[str] = None,
        redacted_text: Optional[str] = None,
        position_start: Optional[int] = None,
        position_end: Optional[int] = None,
        clinical_context: Optional[str] = None,
        preserved_for_analysis: bool = False,
    ) -> PHIDetection:
        """Record a PHI detection event."""

        # Get processing session
        processing_session = self.get_processing_session(session_id)
        if not processing_session:
            raise ValueError(f"Processing session {session_id} not found")

        # Get document record
        document = self.session.query(ProcessedDocument).filter(
            ProcessedDocument.document_id == document_id,
            ProcessedDocument.session_id == processing_session.id
        ).first()

        if not document:
            raise ValueError(f"Document {document_id} not found in session {session_id}")

        phi_detection = PHIDetection(
            document_id=document.id,
            session_id=processing_session.id,
            phi_type=phi_type,
            detected_text=detected_text,  # Should be encrypted/hashed
            redacted_text=redacted_text,
            confidence_score=confidence_score,
            detection_method=detection_method,
            position_start=position_start,
            position_end=position_end,
            clinical_context=clinical_context,
            preserved_for_analysis=preserved_for_analysis,
        )

        self.session.add(phi_detection)

        # Update session PHI count
        processing_session.phi_entities_redacted += 1

        self.session.commit()
        self.session.refresh(phi_detection)
        return phi_detection

    # Audit Event Operations
    def record_audit_event(self, audit_event: AuditEvent) -> AuditEventRecord:
        """Record an audit event in the database."""

        # Map session_id to database session if it exists
        session_uuid = None
        if audit_event.session_id:
            processing_session = self.get_processing_session(audit_event.session_id)
            if processing_session:
                session_uuid = processing_session.id

        audit_record = AuditEventRecord(
            event_id=audit_event.event_id,
            action=audit_event.action,
            timestamp=audit_event.timestamp,
            user_id=audit_event.user_id,
            session_id=session_uuid,
            ip_address=audit_event.ip_address,
            user_agent=audit_event.user_agent,
            resource_type=audit_event.resource_type,
            resource_id=audit_event.resource_id,
            resource_path=audit_event.resource_path,
            description=audit_event.description,
            details=audit_event.details,
            compliance_relevant=audit_event.compliance_relevant,
            security_level=audit_event.security_level,
            requires_investigation=audit_event.requires_investigation,
        )

        self.session.add(audit_record)
        self.session.commit()
        self.session.refresh(audit_record)
        return audit_record

    def get_audit_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        security_level: Optional[str] = None,
        requires_investigation: Optional[bool] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[AuditEventRecord]:
        """Query audit events with filters."""

        query = self.session.query(AuditEventRecord)

        # Apply filters
        if start_time:
            query = query.filter(AuditEventRecord.timestamp >= start_time)
        if end_time:
            query = query.filter(AuditEventRecord.timestamp <= end_time)
        if user_id:
            query = query.filter(AuditEventRecord.user_id == user_id)
        if action:
            query = query.filter(AuditEventRecord.action == action)
        if security_level:
            query = query.filter(AuditEventRecord.security_level == security_level)
        if requires_investigation is not None:
            query = query.filter(AuditEventRecord.requires_investigation == requires_investigation)

        return query.order_by(desc(AuditEventRecord.timestamp)).offset(offset).limit(limit).all()

    # Compliance Report Operations
    def create_compliance_report(
        self,
        report_period_start: datetime,
        report_period_end: datetime,
        generated_by: Optional[str] = None,
    ) -> ComplianceReport:
        """Generate and store a compliance report."""

        # Calculate metrics from the database
        session_query = self.session.query(ProcessingSession).filter(
            and_(
                ProcessingSession.started_at >= report_period_start,
                ProcessingSession.started_at <= report_period_end,
                ProcessingSession.status == "completed"
            )
        )

        document_query = self.session.query(ProcessedDocument).join(ProcessingSession).filter(
            and_(
                ProcessingSession.started_at >= report_period_start,
                ProcessingSession.started_at <= report_period_end
            )
        )

        # Calculate summary metrics
        total_documents = document_query.count()
        total_phi_detected = self.session.query(func.sum(ProcessedDocument.phi_entities_count)).join(ProcessingSession).filter(
            and_(
                ProcessingSession.started_at >= report_period_start,
                ProcessingSession.started_at <= report_period_end
            )
        ).scalar() or 0

        avg_compliance_score = self.session.query(func.avg(ProcessedDocument.compliance_score)).join(ProcessingSession).filter(
            and_(
                ProcessingSession.started_at >= report_period_start,
                ProcessingSession.started_at <= report_period_end
            )
        ).scalar() or 0.0

        avg_processing_time = self.session.query(func.avg(ProcessedDocument.processing_time_ms)).join(ProcessingSession).filter(
            and_(
                ProcessingSession.started_at >= report_period_start,
                ProcessingSession.started_at <= report_period_end
            )
        ).scalar() or 0

        # Risk distribution
        high_risk_count = document_query.filter(ProcessedDocument.risk_assessment == "HIGH").count()
        medium_risk_count = document_query.filter(ProcessedDocument.risk_assessment == "MEDIUM").count()
        low_risk_count = document_query.filter(ProcessedDocument.risk_assessment == "LOW").count()

        # Violations (non-compliant documents)
        violations = document_query.filter(ProcessedDocument.hipaa_compliant == False).count()

        report = ComplianceReport(
            report_id=f"compliance-report-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            generated_by=generated_by,
            report_period_start=report_period_start,
            report_period_end=report_period_end,
            documents_processed=total_documents,
            phi_entities_detected=total_phi_detected,
            phi_entities_redacted=total_phi_detected,  # Assuming all detected PHI is redacted
            overall_compliance_rate=avg_compliance_score,
            average_confidence_score=avg_compliance_score,
            violations_detected=violations,
            high_risk_documents=high_risk_count,
            medium_risk_documents=medium_risk_count,
            low_risk_documents=low_risk_count,
            average_processing_time_ms=int(avg_processing_time),
        )

        self.session.add(report)
        self.session.commit()
        self.session.refresh(report)
        return report

    def get_compliance_reports(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50
    ) -> List[ComplianceReport]:
        """Retrieve compliance reports."""

        query = self.session.query(ComplianceReport)

        if start_date:
            query = query.filter(ComplianceReport.report_period_start >= start_date)
        if end_date:
            query = query.filter(ComplianceReport.report_period_end <= end_date)

        return query.order_by(desc(ComplianceReport.generated_at)).limit(limit).all()

    # Analytics and Reporting
    def get_processing_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get processing statistics for analytics."""

        base_query = self.session.query(ProcessingSession)
        if start_date:
            base_query = base_query.filter(ProcessingSession.started_at >= start_date)
        if end_date:
            base_query = base_query.filter(ProcessingSession.started_at <= end_date)

        completed_sessions = base_query.filter(ProcessingSession.status == "completed").all()

        if not completed_sessions:
            return {
                "total_sessions": 0,
                "total_documents": 0,
                "total_phi_detected": 0,
                "average_processing_time": 0,
                "compliance_levels": {},
                "document_types": {},
                "risk_distribution": {},
            }

        # Calculate statistics
        total_sessions = len(completed_sessions)
        total_documents = sum(s.documents_processed for s in completed_sessions)
        total_phi = sum(s.phi_entities_detected for s in completed_sessions)
        avg_time = sum(s.total_processing_time_ms for s in completed_sessions) / total_sessions if total_sessions > 0 else 0

        # Compliance level distribution
        compliance_levels = {}
        for session in completed_sessions:
            level = session.compliance_level
            compliance_levels[level] = compliance_levels.get(level, 0) + 1

        # Document type distribution
        document_type_query = self.session.query(
            ProcessedDocument.document_type,
            func.count(ProcessedDocument.id)
        ).join(ProcessingSession)

        if start_date:
            document_type_query = document_type_query.filter(ProcessingSession.started_at >= start_date)
        if end_date:
            document_type_query = document_type_query.filter(ProcessingSession.started_at <= end_date)

        document_types = dict(document_type_query.group_by(ProcessedDocument.document_type).all())

        # Risk distribution
        risk_query = self.session.query(
            ProcessedDocument.risk_assessment,
            func.count(ProcessedDocument.id)
        ).join(ProcessingSession)

        if start_date:
            risk_query = risk_query.filter(ProcessingSession.started_at >= start_date)
        if end_date:
            risk_query = risk_query.filter(ProcessingSession.started_at <= end_date)

        risk_distribution = dict(risk_query.group_by(ProcessedDocument.risk_assessment).all())

        return {
            "total_sessions": total_sessions,
            "total_documents": total_documents,
            "total_phi_detected": total_phi,
            "average_processing_time_ms": avg_time,
            "compliance_levels": compliance_levels,
            "document_types": document_types,
            "risk_distribution": risk_distribution,
        }

    # Data Retention and Cleanup
    def cleanup_expired_records(self) -> Dict[str, int]:
        """Clean up records that have exceeded retention period."""

        current_time = datetime.utcnow()

        # Clean up expired audit events
        expired_audit_count = self.session.query(AuditEventRecord).filter(
            AuditEventRecord.retention_until <= current_time
        ).delete()

        # Clean up expired documents
        expired_docs_count = self.session.query(ProcessedDocument).filter(
            ProcessedDocument.retention_until <= current_time
        ).delete()

        # Clean up expired reports
        expired_reports_count = self.session.query(ComplianceReport).filter(
            ComplianceReport.retention_until <= current_time
        ).delete()

        self.session.commit()

        return {
            "expired_audit_events": expired_audit_count,
            "expired_documents": expired_docs_count,
            "expired_reports": expired_reports_count,
        }

    def get_retention_status(self) -> Dict[str, Any]:
        """Get data retention status and upcoming expirations."""

        current_time = datetime.utcnow()
        next_week = current_time + timedelta(days=7)
        next_month = current_time + timedelta(days=30)

        # Count records expiring soon
        audit_expiring_week = self.session.query(AuditEventRecord).filter(
            and_(
                AuditEventRecord.retention_until <= next_week,
                AuditEventRecord.retention_until > current_time
            )
        ).count()

        docs_expiring_week = self.session.query(ProcessedDocument).filter(
            and_(
                ProcessedDocument.retention_until <= next_week,
                ProcessedDocument.retention_until > current_time
            )
        ).count()

        return {
            "current_time": current_time.isoformat(),
            "audit_events_total": self.session.query(AuditEventRecord).count(),
            "documents_total": self.session.query(ProcessedDocument).count(),
            "reports_total": self.session.query(ComplianceReport).count(),
            "expiring_this_week": {
                "audit_events": audit_expiring_week,
                "documents": docs_expiring_week,
            }
        }
