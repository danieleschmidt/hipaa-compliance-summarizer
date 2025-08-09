"""Audit logging models for HIPAA compliance tracking."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AuditAction(str, Enum):
    """Types of auditable actions in the system."""

    # Document processing actions
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_PROCESSED = "document_processed"
    DOCUMENT_DELETED = "document_deleted"

    # PHI-related actions
    PHI_DETECTED = "phi_detected"
    PHI_REDACTED = "phi_redacted"
    PHI_ACCESSED = "phi_accessed"

    # System actions
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"

    # Compliance actions
    COMPLIANCE_REPORT_GENERATED = "compliance_report_generated"
    AUDIT_LOG_ACCESSED = "audit_log_accessed"
    CONFIGURATION_CHANGED = "configuration_changed"

    # Security actions
    UNAUTHORIZED_ACCESS_ATTEMPT = "unauthorized_access_attempt"
    SECURITY_VIOLATION = "security_violation"
    ENCRYPTION_KEY_ROTATED = "encryption_key_rotated"


@dataclass
class AuditEvent:
    """Represents a single auditable event in the system."""

    # Core event identification
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action: AuditAction = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # User and session information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Resource information
    resource_type: Optional[str] = None  # document, phi_entity, system, etc.
    resource_id: Optional[str] = None
    resource_path: Optional[str] = None

    # Event details
    description: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    # Compliance tracking
    compliance_relevant: bool = True
    retention_period_days: int = 2555  # 7 years for HIPAA

    # Security classification
    security_level: str = "normal"  # normal, sensitive, critical
    requires_investigation: bool = False

    def __post_init__(self):
        """Validate and normalize audit event data."""
        if self.action is None:
            raise ValueError("Audit action is required")

        # Ensure timestamp is UTC
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=datetime.now().astimezone().tzinfo)

        # Set security level based on action
        if self.action in [
            AuditAction.PHI_DETECTED,
            AuditAction.PHI_REDACTED,
            AuditAction.SECURITY_VIOLATION,
            AuditAction.UNAUTHORIZED_ACCESS_ATTEMPT
        ]:
            self.security_level = "sensitive"

        if self.action in [
            AuditAction.SECURITY_VIOLATION,
            AuditAction.UNAUTHORIZED_ACCESS_ATTEMPT
        ]:
            self.requires_investigation = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "action": self.action.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "resource_path": self.resource_path,
            "description": self.description,
            "details": self.details,
            "compliance_relevant": self.compliance_relevant,
            "retention_period_days": self.retention_period_days,
            "security_level": self.security_level,
            "requires_investigation": self.requires_investigation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Create AuditEvent from dictionary."""
        action = AuditAction(data["action"])
        timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            event_id=data["event_id"],
            action=action,
            timestamp=timestamp,
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            resource_path=data.get("resource_path"),
            description=data.get("description"),
            details=data.get("details", {}),
            compliance_relevant=data.get("compliance_relevant", True),
            retention_period_days=data.get("retention_period_days", 2555),
            security_level=data.get("security_level", "normal"),
            requires_investigation=data.get("requires_investigation", False),
        )


@dataclass
class AuditLog:
    """Container for audit events with query and filtering capabilities."""

    events: List[AuditEvent] = field(default_factory=list)
    log_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_event(self, event: AuditEvent) -> None:
        """Add an audit event to the log."""
        self.events.append(event)

    def get_events_by_action(self, action: AuditAction) -> List[AuditEvent]:
        """Get all events matching a specific action."""
        return [event for event in self.events if event.action == action]

    def get_events_by_user(self, user_id: str) -> List[AuditEvent]:
        """Get all events for a specific user."""
        return [event for event in self.events if event.user_id == user_id]

    def get_events_by_resource(self, resource_type: str, resource_id: str = None) -> List[AuditEvent]:
        """Get all events for a specific resource."""
        events = [event for event in self.events if event.resource_type == resource_type]
        if resource_id:
            events = [event for event in events if event.resource_id == resource_id]
        return events

    def get_security_events(self) -> List[AuditEvent]:
        """Get all security-related events."""
        return [event for event in self.events if event.security_level in ["sensitive", "critical"]]

    def get_events_requiring_investigation(self) -> List[AuditEvent]:
        """Get all events that require investigation."""
        return [event for event in self.events if event.requires_investigation]

    def get_events_in_timeframe(self, start_time: datetime, end_time: datetime) -> List[AuditEvent]:
        """Get all events within a specific timeframe."""
        return [
            event for event in self.events
            if start_time <= event.timestamp <= end_time
        ]

    def get_compliance_events(self) -> List[AuditEvent]:
        """Get all compliance-relevant events."""
        return [event for event in self.events if event.compliance_relevant]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "log_id": self.log_id,
            "created_at": self.created_at.isoformat(),
            "event_count": len(self.events),
            "events": [event.to_dict() for event in self.events]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditLog":
        """Create AuditLog from dictionary."""
        log = cls(
            log_id=data["log_id"],
            created_at=datetime.fromisoformat(data["created_at"])
        )

        for event_data in data.get("events", []):
            event = AuditEvent.from_dict(event_data)
            log.add_event(event)

        return log

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of audit log contents."""
        action_counts = {}
        user_counts = {}
        security_event_count = 0
        investigation_required_count = 0

        for event in self.events:
            # Count actions
            action_counts[event.action.value] = action_counts.get(event.action.value, 0) + 1

            # Count users
            if event.user_id:
                user_counts[event.user_id] = user_counts.get(event.user_id, 0) + 1

            # Count security events
            if event.security_level in ["sensitive", "critical"]:
                security_event_count += 1

            # Count events requiring investigation
            if event.requires_investigation:
                investigation_required_count += 1

        return {
            "log_summary": {
                "log_id": self.log_id,
                "total_events": len(self.events),
                "created_at": self.created_at.isoformat(),
                "time_span": {
                    "earliest_event": min(event.timestamp for event in self.events).isoformat() if self.events else None,
                    "latest_event": max(event.timestamp for event in self.events).isoformat() if self.events else None,
                }
            },
            "action_breakdown": action_counts,
            "user_activity": user_counts,
            "security_metrics": {
                "security_events": security_event_count,
                "investigation_required": investigation_required_count,
                "compliance_events": len(self.get_compliance_events()),
            }
        }
