"""Comprehensive audit logging system for HIPAA compliance tracking."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from .constants import SECURITY_LIMITS

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    DOCUMENT_PROCESSING = "document_processing"
    PHI_DETECTION = "phi_detection"
    DATA_ACCESS = "data_access"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"
    CONFIGURATION_CHANGE = "configuration_change"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    ERROR_EVENT = "error_event"


class AuditLevel(str, Enum):
    """Audit event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Structured audit event for HIPAA compliance."""

    event_id: str
    timestamp: str
    event_type: AuditEventType
    level: AuditLevel
    source: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    document_id: Optional[str] = None
    phi_detected: Optional[int] = None
    compliance_score: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    duration_ms: Optional[float] = None
    status: str = "success"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert audit event to JSON string."""
        return json.dumps(self.to_dict(), default=str, separators=(',', ':'))


class AuditLogger:
    """Centralized audit logging system for HIPAA compliance."""

    def __init__(self, log_file: Optional[Union[str, Path]] = None,
                 max_log_size: int = SECURITY_LIMITS.MAX_LOG_FILE_SIZE,
                 backup_count: int = 5):
        """Initialize audit logger.
        
        Args:
            log_file: Path to audit log file (defaults to hipaa_audit.log)
            max_log_size: Maximum log file size in bytes
            backup_count: Number of backup log files to retain
        """
        self.log_file = Path(log_file) if log_file else Path("hipaa_audit.log")
        self.max_log_size = max_log_size
        self.backup_count = backup_count

        # Session tracking
        self.session_id = str(uuid4())
        self.session_start_time = datetime.utcnow()

        # Event tracking
        self.event_count = 0
        self.events_by_type: Dict[AuditEventType, int] = {}
        self.events_by_level: Dict[AuditLevel, int] = {}

        # Setup file logging
        self._setup_file_logging()

        # Log session start
        self.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            level=AuditLevel.INFO,
            operation="audit_session_start",
            details={"session_id": self.session_id}
        )

    def _setup_file_logging(self):
        """Setup rotating file logging for audit events."""
        try:
            # Ensure log directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            # Setup rotating file handler
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_log_size,
                backupCount=self.backup_count
            )

            # JSON formatter for structured logging
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)

            # Create dedicated audit logger
            self.audit_file_logger = logging.getLogger(f"{__name__}.audit_file")
            self.audit_file_logger.setLevel(logging.INFO)
            self.audit_file_logger.addHandler(file_handler)
            self.audit_file_logger.propagate = False

            logger.info(f"Audit logging configured: {self.log_file}")

        except Exception as e:
            logger.error(f"Failed to setup audit file logging: {e}")
            self.audit_file_logger = None

    def log_event(self,
                  event_type: AuditEventType,
                  level: AuditLevel = AuditLevel.INFO,
                  operation: str = "unknown",
                  user_id: Optional[str] = None,
                  document_id: Optional[str] = None,
                  phi_detected: Optional[int] = None,
                  compliance_score: Optional[float] = None,
                  details: Optional[Dict[str, Any]] = None,
                  ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  duration_ms: Optional[float] = None,
                  status: str = "success",
                  error_message: Optional[str] = None) -> str:
        """Log an audit event.
        
        Returns:
            Event ID for reference
        """
        event_id = str(uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Create audit event
        event = AuditEvent(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            level=level,
            source="hipaa_compliance_summarizer",
            operation=operation,
            user_id=user_id,
            session_id=self.session_id,
            document_id=document_id,
            phi_detected=phi_detected,
            compliance_score=compliance_score,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            duration_ms=duration_ms,
            status=status,
            error_message=error_message
        )

        # Update statistics
        self.event_count += 1
        self.events_by_type[event_type] = self.events_by_type.get(event_type, 0) + 1
        self.events_by_level[level] = self.events_by_level.get(level, 0) + 1

        # Log to file
        if self.audit_file_logger:
            try:
                self.audit_file_logger.info(event.to_json())
            except Exception as e:
                logger.error(f"Failed to write audit event to file: {e}")

        # Log to standard logger based on level
        log_message = f"AUDIT: {event_type.value} - {operation}"
        if level == AuditLevel.CRITICAL:
            logger.critical(log_message, extra={"audit_event": event.to_dict()})
        elif level == AuditLevel.ERROR:
            logger.error(log_message, extra={"audit_event": event.to_dict()})
        elif level == AuditLevel.WARNING:
            logger.warning(log_message, extra={"audit_event": event.to_dict()})
        else:
            logger.info(log_message, extra={"audit_event": event.to_dict()})

        return event_id

    def log_document_processing(self,
                               document_id: str,
                               operation: str,
                               duration_ms: float,
                               phi_detected: int = 0,
                               compliance_score: float = 1.0,
                               status: str = "success",
                               error_message: Optional[str] = None,
                               user_id: Optional[str] = None) -> str:
        """Log document processing event."""
        return self.log_event(
            event_type=AuditEventType.DOCUMENT_PROCESSING,
            level=AuditLevel.ERROR if status != "success" else AuditLevel.INFO,
            operation=operation,
            user_id=user_id,
            document_id=document_id,
            phi_detected=phi_detected,
            compliance_score=compliance_score,
            duration_ms=duration_ms,
            status=status,
            error_message=error_message,
            details={
                "processing_duration_ms": duration_ms,
                "phi_entities_detected": phi_detected,
                "compliance_score": compliance_score
            }
        )

    def log_phi_detection(self,
                         document_id: str,
                         phi_type: str,
                         confidence: float,
                         redaction_method: str,
                         user_id: Optional[str] = None) -> str:
        """Log PHI detection event."""
        return self.log_event(
            event_type=AuditEventType.PHI_DETECTION,
            level=AuditLevel.INFO,
            operation="phi_detected",
            user_id=user_id,
            document_id=document_id,
            details={
                "phi_type": phi_type,
                "confidence_score": confidence,
                "redaction_method": redaction_method
            }
        )

    def log_security_event(self,
                          event_description: str,
                          severity: AuditLevel = AuditLevel.WARNING,
                          ip_address: Optional[str] = None,
                          user_agent: Optional[str] = None,
                          user_id: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None) -> str:
        """Log security-related event."""
        return self.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            level=severity,
            operation="security_event",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={"description": event_description, **(details or {})}
        )

    def log_compliance_check(self,
                           document_id: str,
                           compliance_level: str,
                           score: float,
                           violations: List[str],
                           user_id: Optional[str] = None) -> str:
        """Log compliance check event."""
        level = AuditLevel.ERROR if violations else AuditLevel.INFO
        return self.log_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            level=level,
            operation="compliance_verification",
            user_id=user_id,
            document_id=document_id,
            compliance_score=score,
            details={
                "compliance_level": compliance_level,
                "violations_found": violations,
                "violation_count": len(violations)
            }
        )

    def log_configuration_change(self,
                                setting_name: str,
                                old_value: Any,
                                new_value: Any,
                                user_id: Optional[str] = None) -> str:
        """Log configuration change event."""
        return self.log_event(
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            level=AuditLevel.WARNING,
            operation="config_change",
            user_id=user_id,
            details={
                "setting_name": setting_name,
                "old_value": str(old_value),
                "new_value": str(new_value)
            }
        )

    def log_user_action(self,
                       action: str,
                       user_id: str,
                       resource: Optional[str] = None,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None) -> str:
        """Log user action event."""
        return self.log_event(
            event_type=AuditEventType.USER_ACTION,
            level=AuditLevel.INFO,
            operation=action,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={"resource": resource, **(details or {})}
        )

    def log_error(self,
                 error_type: str,
                 error_message: str,
                 operation: str,
                 user_id: Optional[str] = None,
                 document_id: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> str:
        """Log error event."""
        return self.log_event(
            event_type=AuditEventType.ERROR_EVENT,
            level=AuditLevel.ERROR,
            operation=operation,
            user_id=user_id,
            document_id=document_id,
            status="error",
            error_message=error_message,
            details={"error_type": error_type, **(details or {})}
        )

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current audit session."""
        session_duration = (datetime.utcnow() - self.session_start_time).total_seconds()

        return {
            "session_id": self.session_id,
            "session_start_time": self.session_start_time.isoformat() + "Z",
            "session_duration_seconds": session_duration,
            "total_events": self.event_count,
            "events_by_type": {k.value: v for k, v in self.events_by_type.items()},
            "events_by_level": {k.value: v for k, v in self.events_by_level.items()},
            "log_file": str(self.log_file)
        }

    def export_audit_trail(self,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          event_types: Optional[List[AuditEventType]] = None,
                          output_file: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Export audit trail for specified time range and event types."""
        if not self.log_file.exists():
            logger.warning(f"Audit log file not found: {self.log_file}")
            return []

        events = []
        try:
            with open(self.log_file) as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())

                        # Filter by time range
                        if start_time or end_time:
                            event_time = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                            if start_time and event_time < start_time:
                                continue
                            if end_time and event_time > end_time:
                                continue

                        # Filter by event type
                        if event_types and event['event_type'] not in [et.value for et in event_types]:
                            continue

                        events.append(event)

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse audit log line: {e}")
                        continue

            # Export to file if specified
            if output_file:
                output_path = Path(output_file)
                with open(output_path, 'w') as f:
                    json.dump(events, f, indent=2, default=str)
                logger.info(f"Exported {len(events)} audit events to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export audit trail: {e}")
            raise

        return events

    def close_session(self):
        """Close the current audit session."""
        self.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            level=AuditLevel.INFO,
            operation="audit_session_end",
            details=self.get_session_summary()
        )


# Global audit logger instance
_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()
    return _global_audit_logger


def initialize_audit_logging(log_file: Optional[Union[str, Path]] = None,
                            max_log_size: int = SECURITY_LIMITS.MAX_LOG_FILE_SIZE,
                            backup_count: int = 5) -> AuditLogger:
    """Initialize global audit logging."""
    global _global_audit_logger
    _global_audit_logger = AuditLogger(log_file, max_log_size, backup_count)
    return _global_audit_logger


__all__ = [
    "AuditEventType",
    "AuditLevel",
    "AuditEvent",
    "AuditLogger",
    "get_audit_logger",
    "initialize_audit_logging"
]
