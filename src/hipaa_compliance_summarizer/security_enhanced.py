"""Enhanced security features for HIPAA compliance system."""

import hashlib
import hmac
import logging
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event for audit logging."""

    event_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    source_ip: Optional[str]
    user_id: Optional[str]
    resource: Optional[str]
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: secrets.token_hex(16))


@dataclass
class AccessAttempt:
    """Track access attempts for rate limiting."""

    ip_address: str
    attempts: int
    first_attempt: datetime
    last_attempt: datetime
    blocked: bool = False


class SecurityManager:
    """Enhanced security manager for HIPAA compliance."""

    def __init__(self):
        self.access_attempts: Dict[str, AccessAttempt] = {}
        self.blocked_ips: Set[str] = set()
        self.security_events: List[SecurityEvent] = []
        self.session_tokens: Dict[str, datetime] = {}
        self.max_attempts = 5
        self.block_duration = timedelta(minutes=15)
        self.session_timeout = timedelta(hours=1)

    def log_security_event(self,
                          event_type: str,
                          severity: str,
                          source_ip: Optional[str] = None,
                          user_id: Optional[str] = None,
                          resource: Optional[str] = None,
                          **details) -> str:
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            resource=resource,
            details=details
        )

        self.security_events.append(event)

        # Log to system logger based on severity
        log_func = {
            'LOW': logger.info,
            'MEDIUM': logger.warning,
            'HIGH': logger.error,
            'CRITICAL': logger.critical
        }.get(severity, logger.info)

        log_func(f"Security Event [{event.event_id}]: {event_type} - {details}")

        return event.event_id

    def track_access_attempt(self, ip_address: str, success: bool = True) -> bool:
        """Track access attempt and apply rate limiting."""
        now = datetime.utcnow()

        # Check if IP is already blocked
        if ip_address in self.blocked_ips:
            self.log_security_event(
                "blocked_ip_access_attempt",
                "MEDIUM",
                source_ip=ip_address,
                reason="IP blocked due to excessive attempts"
            )
            return False

        # Get or create access attempt record
        if ip_address not in self.access_attempts:
            self.access_attempts[ip_address] = AccessAttempt(
                ip_address=ip_address,
                attempts=0,
                first_attempt=now,
                last_attempt=now
            )

        attempt = self.access_attempts[ip_address]

        # Reset counter if block duration has passed
        if now - attempt.last_attempt > self.block_duration:
            attempt.attempts = 0
            attempt.first_attempt = now
            attempt.blocked = False

        # Increment attempts for failed access
        if not success:
            attempt.attempts += 1
            attempt.last_attempt = now

            # Block IP if too many attempts
            if attempt.attempts >= self.max_attempts:
                attempt.blocked = True
                self.blocked_ips.add(ip_address)

                self.log_security_event(
                    "ip_blocked",
                    "HIGH",
                    source_ip=ip_address,
                    attempts=attempt.attempts,
                    block_duration=str(self.block_duration)
                )
                return False
            else:
                self.log_security_event(
                    "failed_access_attempt",
                    "MEDIUM",
                    source_ip=ip_address,
                    attempts=attempt.attempts,
                    remaining=self.max_attempts - attempt.attempts
                )
        else:
            # Reset on successful access
            if attempt.attempts > 0:
                self.log_security_event(
                    "successful_access_after_failures",
                    "LOW",
                    source_ip=ip_address,
                    previous_attempts=attempt.attempts
                )
            attempt.attempts = 0

        return True

    def generate_session_token(self, user_id: str) -> str:
        """Generate secure session token."""
        token = secrets.token_urlsafe(32)
        self.session_tokens[token] = datetime.utcnow()

        self.log_security_event(
            "session_created",
            "LOW",
            user_id=user_id,
            token_prefix=token[:8] + "..."
        )

        return token

    def validate_session_token(self, token: str) -> bool:
        """Validate session token."""
        if token not in self.session_tokens:
            self.log_security_event(
                "invalid_session_token",
                "MEDIUM",
                token_prefix=token[:8] + "..." if len(token) > 8 else token
            )
            return False

        # Check if token has expired
        if datetime.utcnow() - self.session_tokens[token] > self.session_timeout:
            del self.session_tokens[token]
            self.log_security_event(
                "expired_session_token",
                "LOW",
                token_prefix=token[:8] + "..."
            )
            return False

        # Refresh token timestamp
        self.session_tokens[token] = datetime.utcnow()
        return True

    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash sensitive data with salt."""
        if salt is None:
            salt = secrets.token_hex(16)

        # Use PBKDF2 for key derivation
        hashed = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return hashed.hex(), salt

    def verify_hash(self, data: str, hashed_data: str, salt: str) -> bool:
        """Verify hashed data."""
        test_hash, _ = self.hash_sensitive_data(data, salt)
        return hmac.compare_digest(test_hash, hashed_data)

    @contextmanager
    def security_context(self, operation: str, user_id: Optional[str] = None):
        """Security context manager for operations."""
        start_time = time.time()
        event_id = self.log_security_event(
            f"operation_start_{operation}",
            "LOW",
            user_id=user_id
        )

        try:
            yield

            # Log successful operation
            duration = time.time() - start_time
            self.log_security_event(
                f"operation_success_{operation}",
                "LOW",
                user_id=user_id,
                duration_seconds=duration,
                related_event=event_id
            )

        except Exception as e:
            # Log failed operation
            duration = time.time() - start_time
            self.log_security_event(
                f"operation_failed_{operation}",
                "HIGH",
                user_id=user_id,
                error=str(e),
                duration_seconds=duration,
                related_event=event_id
            )
            raise

    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data."""
        now = datetime.utcnow()
        recent_events = [
            event for event in self.security_events
            if now - event.timestamp < timedelta(hours=24)
        ]

        severity_counts = {}
        event_type_counts = {}

        for event in recent_events:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1

        return {
            "timestamp": now.isoformat(),
            "active_sessions": len(self.session_tokens),
            "blocked_ips": len(self.blocked_ips),
            "recent_events_24h": len(recent_events),
            "severity_breakdown": severity_counts,
            "top_event_types": dict(sorted(event_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "failed_attempts_tracking": len(self.access_attempts),
            "system_status": "SECURE" if not any(e.severity == "CRITICAL" for e in recent_events[-10:]) else "ALERT"
        }

    def cleanup_old_events(self, retention_days: int = 90):
        """Clean up old security events."""
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        before_count = len(self.security_events)

        self.security_events = [
            event for event in self.security_events
            if event.timestamp > cutoff
        ]

        removed_count = before_count - len(self.security_events)
        if removed_count > 0:
            self.log_security_event(
                "security_events_cleanup",
                "LOW",
                events_removed=removed_count,
                retention_days=retention_days
            )


# Global security manager instance
_security_manager = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def log_security_event(event_type: str, severity: str, **details) -> str:
    """Log security event using global manager."""
    return get_security_manager().log_security_event(event_type, severity, **details)


def security_context(operation: str, user_id: Optional[str] = None):
    """Security context decorator/manager."""
    return get_security_manager().security_context(operation, user_id)


__all__ = [
    'SecurityEvent',
    'AccessAttempt',
    'SecurityManager',
    'get_security_manager',
    'log_security_event',
    'security_context'
]
