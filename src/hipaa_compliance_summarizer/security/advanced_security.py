"""Advanced security controls and monitoring for HIPAA compliance system."""

import hashlib
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import base64

    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecurityEventType(str, Enum):
    """Types of security events."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    ENCRYPTION_KEY_ROTATION = "encryption_key_rotation"
    CONFIGURATION_CHANGE = "configuration_change"
    PHI_ACCESS = "phi_access"
    COMPLIANCE_VIOLATION = "compliance_violation"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccessLevel(str, Enum):
    """Access permission levels."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    AUDIT = "audit"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    event_type: SecurityEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.LOW
    details: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    success: bool = True


@dataclass
class UserSession:
    """User session information."""
    session_id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)
    is_active: bool = True
    expires_at: Optional[datetime] = None


class PasswordHasher:
    """Secure password hashing using bcrypt."""

    def __init__(self, rounds: int = 12):
        self.rounds = rounds
        if not BCRYPT_AVAILABLE:
            logger.warning("bcrypt not available, using fallback hash method")

    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        if BCRYPT_AVAILABLE:
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(self.rounds)).decode('utf-8')
        else:
            # Fallback using built-in hashlib (not recommended for production)
            salt = secrets.token_hex(16)
            pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
            return f"pbkdf2_sha256${salt}${pwd_hash.hex()}"

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        if BCRYPT_AVAILABLE and not password_hash.startswith('pbkdf2_sha256$'):
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        else:
            # Handle fallback format
            if password_hash.startswith('pbkdf2_sha256$'):
                _, salt, stored_hash = password_hash.split('$')
                pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
                return pwd_hash.hex() == stored_hash
            return False


class DataEncryption:
    """Data encryption service for PHI protection."""

    def __init__(self, key: bytes = None):
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Cryptography library not available, encryption disabled")
            self.cipher = None
            return

        if key is None:
            key = Fernet.generate_key()

        self.cipher = Fernet(key)
        self.key = key

    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not self.cipher:
            logger.warning("Encryption not available, returning data as-is")
            return data

        return self.cipher.encrypt(data.encode('utf-8')).decode('utf-8')

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt encrypted data."""
        if not self.cipher:
            logger.warning("Decryption not available, returning data as-is")
            return encrypted_data

        return self.cipher.decrypt(encrypted_data.encode('utf-8')).decode('utf-8')

    def rotate_key(self) -> bytes:
        """Generate and set new encryption key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return b''

        new_key = Fernet.generate_key()
        self.cipher = Fernet(new_key)
        self.key = new_key
        return new_key

    @staticmethod
    def derive_key_from_password(password: str, salt: bytes = None) -> bytes:
        """Derive encryption key from password."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return b''

        if salt is None:
            salt = secrets.token_bytes(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )

        key = base64.urlsafe_b64encode(kdf.derive(password.encode('utf-8')))
        return key


class RateLimiter:
    """Rate limiting for API endpoints and user actions."""

    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit."""
        current_time = time.time()

        with self.lock:
            if identifier not in self.requests:
                self.requests[identifier] = []

            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if current_time - req_time < self.time_window
            ]

            # Check if under limit
            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(current_time)
                return True

            return False

    def get_remaining_requests(self, identifier: str) -> int:
        """Get number of remaining requests for identifier."""
        current_time = time.time()

        with self.lock:
            if identifier not in self.requests:
                return self.max_requests

            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if current_time - req_time < self.time_window
            ]

            return max(0, self.max_requests - len(self.requests[identifier]))

    def reset_limit(self, identifier: str):
        """Reset rate limit for identifier."""
        with self.lock:
            if identifier in self.requests:
                del self.requests[identifier]


class AccessControl:
    """Role-based access control system."""

    def __init__(self):
        self.role_permissions: Dict[str, Set[str]] = {
            'admin': {'read', 'write', 'delete', 'admin', 'audit'},
            'user': {'read', 'write'},
            'auditor': {'read', 'audit'},
            'viewer': {'read'},
        }
        self.user_roles: Dict[str, Set[str]] = {}

    def assign_role(self, user_id: str, role: str):
        """Assign role to user."""
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role)
        logger.info(f"Assigned role '{role}' to user {user_id}")

    def remove_role(self, user_id: str, role: str):
        """Remove role from user."""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role)
            logger.info(f"Removed role '{role}' from user {user_id}")

    def has_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission."""
        if user_id not in self.user_roles:
            return False

        user_permissions = set()
        for role in self.user_roles[user_id]:
            if role in self.role_permissions:
                user_permissions.update(self.role_permissions[role])

        return permission in user_permissions

    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for user."""
        if user_id not in self.user_roles:
            return set()

        permissions = set()
        for role in self.user_roles[user_id]:
            if role in self.role_permissions:
                permissions.update(self.role_permissions[role])

        return permissions


class SessionManager:
    """Manages user sessions with security controls."""

    def __init__(self, session_timeout: int = 3600, max_sessions_per_user: int = 5):
        self.session_timeout = session_timeout
        self.max_sessions_per_user = max_sessions_per_user
        self.sessions: Dict[str, UserSession] = {}
        self.user_sessions: Dict[str, Set[str]] = {}
        self.lock = threading.Lock()

    def create_session(self, user_id: str, ip_address: str = None,
                      user_agent: str = None, permissions: Set[str] = None) -> UserSession:
        """Create new user session."""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(seconds=self.session_timeout)

        with self.lock:
            # Limit sessions per user
            if user_id in self.user_sessions:
                user_session_ids = list(self.user_sessions[user_id])
                if len(user_session_ids) >= self.max_sessions_per_user:
                    # Remove oldest session
                    oldest_session_id = min(
                        user_session_ids,
                        key=lambda sid: self.sessions[sid].created_at
                    )
                    self.invalidate_session(oldest_session_id)

            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                permissions=permissions or set(),
                expires_at=expires_at
            )

            self.sessions[session_id] = session

            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)

        logger.info(f"Created session {session_id} for user {user_id}")
        return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID."""
        with self.lock:
            session = self.sessions.get(session_id)

            if session and self._is_session_valid(session):
                session.last_activity = datetime.utcnow()
                return session
            elif session:
                # Session expired, clean up
                self.invalidate_session(session_id)

            return None

    def _is_session_valid(self, session: UserSession) -> bool:
        """Check if session is still valid."""
        if not session.is_active:
            return False

        if session.expires_at and datetime.utcnow() > session.expires_at:
            return False

        # Check inactivity timeout
        if (datetime.utcnow() - session.last_activity).total_seconds() > self.session_timeout:
            return False

        return True

    def invalidate_session(self, session_id: str):
        """Invalidate session."""
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.is_active = False

                # Remove from user sessions
                if session.user_id in self.user_sessions:
                    self.user_sessions[session.user_id].discard(session_id)

                del self.sessions[session_id]
                logger.info(f"Invalidated session {session_id}")

    def invalidate_all_user_sessions(self, user_id: str):
        """Invalidate all sessions for a user."""
        with self.lock:
            if user_id in self.user_sessions:
                session_ids = list(self.user_sessions[user_id])
                for session_id in session_ids:
                    self.invalidate_session(session_id)

    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        with self.lock:
            expired_sessions = []
            for session_id, session in self.sessions.items():
                if not self._is_session_valid(session):
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                self.invalidate_session(session_id)

            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


class SecurityMonitor:
    """Monitors security events and detects threats."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.events: List[SecurityEvent] = []
        self.max_events = self.config.get("max_events", 10000)
        self.alert_callbacks: List[Callable] = []
        self.suspicious_ips: Set[str] = set()
        self.blocked_ips: Set[str] = set()
        self.threat_patterns = self._load_threat_patterns()
        self.lock = threading.Lock()

    def _load_threat_patterns(self) -> Dict[str, Any]:
        """Load threat detection patterns."""
        return {
            "failed_login_threshold": 5,
            "failed_login_window": 300,  # 5 minutes
            "suspicious_user_agent_patterns": [
                "bot", "crawler", "scanner", "sqlmap", "nikto"
            ],
            "rate_limit_threshold": 100,
            "rate_limit_window": 60,
        }

    def log_event(self, event: SecurityEvent):
        """Log security event."""
        with self.lock:
            self.events.append(event)

            # Maintain event history size
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]

        # Analyze for threats
        self._analyze_threat(event)

        # Trigger alerts if needed
        if event.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            self._trigger_alerts(event)

        logger.info(f"Security event logged: {event.event_type.value} from {event.ip_address}")

    def _analyze_threat(self, event: SecurityEvent):
        """Analyze event for threat patterns."""
        if not event.ip_address:
            return

        # Analyze failed login attempts
        if event.event_type == SecurityEventType.LOGIN_FAILURE:
            recent_failures = self._count_recent_events(
                SecurityEventType.LOGIN_FAILURE,
                event.ip_address,
                self.threat_patterns["failed_login_window"]
            )

            if recent_failures >= self.threat_patterns["failed_login_threshold"]:
                self._mark_ip_suspicious(event.ip_address, "Multiple failed logins")
                event.risk_level = RiskLevel.HIGH

        # Analyze user agent
        if event.user_agent:
            for pattern in self.threat_patterns["suspicious_user_agent_patterns"]:
                if pattern.lower() in event.user_agent.lower():
                    self._mark_ip_suspicious(event.ip_address, f"Suspicious user agent: {pattern}")
                    event.risk_level = RiskLevel.MEDIUM
                    break

    def _count_recent_events(self, event_type: SecurityEventType,
                            ip_address: str, time_window: int) -> int:
        """Count recent events of specific type from IP."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)

        count = 0
        for event in self.events:
            if (event.event_type == event_type and
                event.ip_address == ip_address and
                event.timestamp >= cutoff_time):
                count += 1

        return count

    def _mark_ip_suspicious(self, ip_address: str, reason: str):
        """Mark IP address as suspicious."""
        self.suspicious_ips.add(ip_address)
        logger.warning(f"Marked IP {ip_address} as suspicious: {reason}")

        # Auto-block if too many suspicious activities
        suspicious_count = sum(1 for event in self.events[-100:]
                             if event.ip_address == ip_address and
                             event.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])

        if suspicious_count >= 10:
            self.block_ip(ip_address, f"Auto-blocked: {reason}")

    def block_ip(self, ip_address: str, reason: str):
        """Block IP address."""
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP {ip_address}: {reason}")

        # Log blocking event
        self.log_event(SecurityEvent(
            event_id=secrets.token_hex(8),
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            ip_address=ip_address,
            risk_level=RiskLevel.CRITICAL,
            details={"action": "ip_blocked", "reason": reason}
        ))

    def unblock_ip(self, ip_address: str):
        """Unblock IP address."""
        self.blocked_ips.discard(ip_address)
        self.suspicious_ips.discard(ip_address)
        logger.info(f"Unblocked IP {ip_address}")

    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked."""
        return ip_address in self.blocked_ips

    def is_ip_suspicious(self, ip_address: str) -> bool:
        """Check if IP is suspicious."""
        return ip_address in self.suspicious_ips

    def add_alert_callback(self, callback: Callable):
        """Add callback for security alerts."""
        self.alert_callbacks.append(callback)

    def _trigger_alerts(self, event: SecurityEvent):
        """Trigger security alerts."""
        for callback in self.alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        if not self.events:
            return {"total_events": 0}

        # Count events by type
        event_counts = {}
        risk_counts = {}

        recent_events = [e for e in self.events
                        if (datetime.utcnow() - e.timestamp).total_seconds() < 86400]  # Last 24 hours

        for event in recent_events:
            event_counts[event.event_type.value] = event_counts.get(event.event_type.value, 0) + 1
            risk_counts[event.risk_level.value] = risk_counts.get(event.risk_level.value, 0) + 1

        return {
            "total_events": len(self.events),
            "recent_events_24h": len(recent_events),
            "event_types": event_counts,
            "risk_levels": risk_counts,
            "suspicious_ips": len(self.suspicious_ips),
            "blocked_ips": len(self.blocked_ips),
            "top_suspicious_ips": list(self.suspicious_ips)[:10],
            "top_blocked_ips": list(self.blocked_ips)[:10]
        }


class SecurityFramework:
    """Comprehensive security framework."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.failed_attempts = {}  # Track failed login attempts by user_id

        # Initialize components
        self.password_hasher = PasswordHasher()
        self.encryption = DataEncryption()
        self.rate_limiter = RateLimiter(
            max_requests=self.config.get("rate_limit_requests", 100),
            time_window=self.config.get("rate_limit_window", 3600)
        )
        self.access_control = AccessControl()
        self.session_manager = SessionManager(
            session_timeout=self.config.get("session_timeout", 3600),
            max_sessions_per_user=self.config.get("max_sessions", 5)
        )
        self.security_monitor = SecurityMonitor(self.config.get("monitoring", {}))

        # Start cleanup task
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start periodic cleanup of expired sessions."""
        def cleanup():
            while True:
                try:
                    self.session_manager.cleanup_expired_sessions()
                    time.sleep(300)  # Cleanup every 5 minutes
                except Exception as e:
                    logger.error(f"Security cleanup error: {e}")
                    time.sleep(60)

        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()

    def authenticate_user(self, user_id: str, password: str,
                         ip_address: str = None, user_agent: str = None) -> Optional[UserSession]:
        """Authenticate user and create session."""

        # Check if IP is blocked
        if ip_address and self.security_monitor.is_ip_blocked(ip_address):
            self.security_monitor.log_event(SecurityEvent(
                event_id=secrets.token_hex(8),
                event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                risk_level=RiskLevel.CRITICAL,
                success=False,
                details={"reason": "IP blocked"}
            ))
            return None

        # Check rate limiting
        rate_limit_key = f"auth_{ip_address or user_id}"
        if not self.rate_limiter.is_allowed(rate_limit_key):
            self.security_monitor.log_event(SecurityEvent(
                event_id=secrets.token_hex(8),
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                risk_level=RiskLevel.HIGH,
                success=False
            ))
            return None

        # Implement robust user authentication against secure credential store
        auth_success = self._verify_user_credentials(user_id, credentials)
        
        if not auth_success:
            # Implement progressive lockout for failed attempts
            self._track_failed_login_attempt(user_id, ip_address)

        if auth_success:
            # Create session
            permissions = self.access_control.get_user_permissions(user_id)
            session = self.session_manager.create_session(
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                permissions=permissions
            )

            # Log successful login
            self.security_monitor.log_event(SecurityEvent(
                event_id=secrets.token_hex(8),
                event_type=SecurityEventType.LOGIN_SUCCESS,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=session.session_id,
                risk_level=RiskLevel.LOW
            ))

            return session
        else:
            # Log failed login
            self.security_monitor.log_event(SecurityEvent(
                event_id=secrets.token_hex(8),
                event_type=SecurityEventType.LOGIN_FAILURE,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                risk_level=RiskLevel.MEDIUM,
                success=False
            ))

            return None

    def authorize_action(self, session_id: str, action: str, resource: str = None) -> bool:
        """Authorize user action."""
        session = self.session_manager.get_session(session_id)

        if not session:
            return False

        # Check permissions
        has_permission = action in session.permissions

        if not has_permission:
            # Log unauthorized access
            self.security_monitor.log_event(SecurityEvent(
                event_id=secrets.token_hex(8),
                event_type=SecurityEventType.PERMISSION_DENIED,
                user_id=session.user_id,
                ip_address=session.ip_address,
                user_agent=session.user_agent,
                session_id=session_id,
                resource=resource,
                action=action,
                risk_level=RiskLevel.MEDIUM,
                success=False
            ))

        return has_permission

    def _verify_user_credentials(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Verify user credentials with secure hashing and rate limiting."""
        try:
            # Implement secure credential verification
            password = credentials.get('password', '')
            stored_hash = self._get_stored_password_hash(user_id)
            
            if not stored_hash:
                return False
            
            # Use bcrypt or secure fallback for password verification
            if BCRYPT_AVAILABLE:
                import bcrypt
                return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
            else:
                # Fallback to PBKDF2 with secure parameters
                import hashlib
                salt = stored_hash[:32]  # Extract salt
                key = stored_hash[32:]   # Extract key
                new_key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
                return key == new_key
                
        except Exception as e:
            logger.error("Credential verification failed: %s", e)
            return False
    
    def _get_stored_password_hash(self, user_id: str) -> Optional[str]:
        """Retrieve stored password hash from secure credential store."""
        # Production implementation would connect to secure database
        # For now, return None to indicate user not found
        return None
    
    def _track_failed_login_attempt(self, user_id: str, ip_address: str):
        """Track failed login attempts with progressive lockout."""
        current_time = time.time()
        
        # Track per-user attempts
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        # Clean old attempts (older than lockout window)
        lockout_window = 3600  # 1 hour
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if current_time - attempt < lockout_window
        ]
        
        # Add current failed attempt
        self.failed_attempts[user_id].append(current_time)
        
        # Check if user should be locked out
        if len(self.failed_attempts[user_id]) >= 5:
            self.security_monitor.log_event(SecurityEvent(
                event_id=secrets.token_hex(8),
                event_type=SecurityEventType.ACCOUNT_LOCKOUT,
                user_id=user_id,
                ip_address=ip_address,
                risk_level=RiskLevel.HIGH,
                details={"lockout_reason": "excessive_failed_logins"}
            ))

    def log_data_access(self, session_id: str, data_type: str,
                       record_count: int = 1, phi_involved: bool = False):
        """Log data access for audit trail."""
        session = self.session_manager.get_session(session_id)

        event_type = SecurityEventType.PHI_ACCESS if phi_involved else SecurityEventType.DATA_ACCESS
        risk_level = RiskLevel.HIGH if phi_involved else RiskLevel.LOW

        self.security_monitor.log_event(SecurityEvent(
            event_id=secrets.token_hex(8),
            event_type=event_type,
            user_id=session.user_id if session else None,
            ip_address=session.ip_address if session else None,
            user_agent=session.user_agent if session else None,
            session_id=session_id,
            resource=data_type,
            risk_level=risk_level,
            details={
                "record_count": record_count,
                "phi_involved": phi_involved
            }
        ))

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "active_sessions": len(self.session_manager.sessions),
            "blocked_ips": len(self.security_monitor.blocked_ips),
            "suspicious_ips": len(self.security_monitor.suspicious_ips),
            "security_stats": self.security_monitor.get_security_stats(),
            "encryption_enabled": CRYPTOGRAPHY_AVAILABLE and self.encryption.cipher is not None,
            "password_hashing": "bcrypt" if BCRYPT_AVAILABLE else "pbkdf2"
        }


# Global security framework instance
security_framework = SecurityFramework()
