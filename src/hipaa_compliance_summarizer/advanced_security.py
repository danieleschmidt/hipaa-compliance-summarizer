"""Advanced security monitoring and threat detection for HIPAA compliance system.

This module provides comprehensive security monitoring including:
- Real-time threat detection
- Behavioral anomaly detection  
- Advanced intrusion detection
- Security event correlation
- Automated threat response
"""

import logging
import time
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from threading import Lock, Thread
from contextlib import contextmanager
import ipaddress

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Represents a security-related event."""
    
    event_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    timestamp: datetime
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    file_path: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    threat_indicators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "source_ip": self.source_ip,
            "user_id": self.user_id,
            "file_path": self.file_path,
            "description": self.description,
            "metadata": self.metadata,
            "threat_indicators": self.threat_indicators,
        }


@dataclass
class ThreatProfile:
    """Profile of potential threats and attack patterns."""
    
    ip_address: str
    failed_attempts: int = 0
    suspicious_files: List[str] = field(default_factory=list)
    anomalous_behavior_score: float = 0.0
    first_seen: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    blocked: bool = False
    threat_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL


class SecurityMonitor:
    """Advanced security monitoring and threat detection system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security monitor with configuration."""
        self.config = config or {}
        self._events_lock = Lock()
        self._profiles_lock = Lock()
        
        # Event storage (in-memory for this implementation)
        self._security_events: deque = deque(maxlen=10000)
        self._threat_profiles: Dict[str, ThreatProfile] = {}
        
        # Configuration
        self.max_failed_attempts = self.config.get('max_failed_attempts', 5)
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.7)
        self.monitoring_window_hours = self.config.get('monitoring_window_hours', 24)
        self.auto_block_enabled = self.config.get('auto_block_enabled', True)
        
        # Threat detection patterns
        self._suspicious_patterns = [
            r'\.\./',           # Path traversal
            r'<script',         # XSS attempts
            r'javascript:',     # JavaScript injection
            r'eval\s*\(',      # Code execution
            r'exec\s*\(',      # Command execution
            r'/etc/passwd',    # System file access
            r'SELECT.*FROM',   # SQL injection
            r'UNION.*SELECT',  # SQL injection
            r'DROP\s+TABLE',   # Database attacks
            r'<\?php',         # PHP injection
        ]
        
        # Known malicious IP ranges (example - should be loaded from threat intel)
        self._malicious_ip_ranges = [
            '10.0.0.0/8',    # Example - replace with real threat intel
            '192.168.1.0/24' # Example - replace with real threat intel
        ]
        
        # Start monitoring thread
        self._monitoring_active = True
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Advanced security monitor initialized")
    
    def log_security_event(self, event_type: str, severity: str, 
                          description: str, **kwargs) -> SecurityEvent:
        """Log a security event for monitoring and analysis."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            description=description,
            **kwargs
        )
        
        with self._events_lock:
            self._security_events.append(event)
        
        # Real-time threat analysis
        self._analyze_event(event)
        
        # Log to standard logging system
        log_level = getattr(logging, severity, logging.INFO)
        logger.log(log_level, f"Security Event: {event.to_dict()}")
        
        return event
    
    def _analyze_event(self, event: SecurityEvent) -> None:
        """Analyze security event for threats and anomalies."""
        if not event.source_ip:
            return
            
        with self._profiles_lock:
            profile = self._threat_profiles.get(
                event.source_ip, 
                ThreatProfile(ip_address=event.source_ip)
            )
            
            # Update threat profile based on event
            self._update_threat_profile(profile, event)
            self._threat_profiles[event.source_ip] = profile
            
            # Check for auto-blocking conditions
            if self._should_auto_block(profile):
                self._auto_block_ip(profile)
    
    def _update_threat_profile(self, profile: ThreatProfile, event: SecurityEvent) -> None:
        """Update threat profile with new security event."""
        profile.last_activity = event.timestamp
        
        # Track failed attempts
        if event.event_type == 'authentication_failure':
            profile.failed_attempts += 1
        
        # Track suspicious file access
        if event.file_path and self._is_suspicious_file(event.file_path):
            if event.file_path not in profile.suspicious_files:
                profile.suspicious_files.append(event.file_path)
        
        # Update anomaly score
        anomaly_score = self._calculate_anomaly_score(event)
        profile.anomalous_behavior_score = max(
            profile.anomalous_behavior_score, 
            anomaly_score
        )
        
        # Update threat level
        profile.threat_level = self._calculate_threat_level(profile)
    
    def _is_suspicious_file(self, file_path: str) -> bool:
        """Check if file path indicates suspicious activity."""
        suspicious_paths = [
            '/etc/', '/proc/', '/sys/', '/root/',
            'password', 'shadow', 'passwd', 'config'
        ]
        
        file_path_lower = file_path.lower()
        return any(suspicious in file_path_lower for suspicious in suspicious_paths)
    
    def _calculate_anomaly_score(self, event: SecurityEvent) -> float:
        """Calculate anomaly score for security event."""
        score = 0.0
        
        # Check for malicious patterns
        text_content = f"{event.description} {event.file_path or ''}"
        for pattern in self._suspicious_patterns:
            import re
            if re.search(pattern, text_content, re.IGNORECASE):
                score += 0.3
        
        # Check for malicious IP ranges
        if event.source_ip and self._is_malicious_ip(event.source_ip):
            score += 0.5
        
        # Time-based anomalies (activity outside normal hours)
        if event.timestamp.hour < 6 or event.timestamp.hour > 22:
            score += 0.1
        
        # High-severity events increase score
        severity_weights = {
            'CRITICAL': 1.0,
            'HIGH': 0.7,
            'MEDIUM': 0.4,
            'LOW': 0.1,
            'INFO': 0.0
        }
        score += severity_weights.get(event.severity, 0.0)
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _is_malicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is in known malicious ranges."""
        try:
            ip = ipaddress.ip_address(ip_address)
            for range_str in self._malicious_ip_ranges:
                network = ipaddress.ip_network(range_str, strict=False)
                if ip in network:
                    return True
        except ValueError:
            logger.warning(f"Invalid IP address format: {ip_address}")
        
        return False
    
    def _calculate_threat_level(self, profile: ThreatProfile) -> str:
        """Calculate overall threat level for a profile."""
        if profile.failed_attempts >= 10 or profile.anomalous_behavior_score >= 0.9:
            return "CRITICAL"
        elif profile.failed_attempts >= 5 or profile.anomalous_behavior_score >= 0.7:
            return "HIGH"  
        elif profile.failed_attempts >= 3 or profile.anomalous_behavior_score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _should_auto_block(self, profile: ThreatProfile) -> bool:
        """Determine if IP should be automatically blocked."""
        if not self.auto_block_enabled or profile.blocked:
            return False
            
        return (
            profile.failed_attempts >= self.max_failed_attempts or
            profile.anomalous_behavior_score >= self.anomaly_threshold or
            profile.threat_level == "CRITICAL"
        )
    
    def _auto_block_ip(self, profile: ThreatProfile) -> None:
        """Automatically block suspicious IP address."""
        profile.blocked = True
        
        # Log blocking action
        self.log_security_event(
            event_type="ip_auto_blocked",
            severity="HIGH",
            description=f"Automatically blocked IP {profile.ip_address} due to suspicious activity",
            source_ip=profile.ip_address,
            metadata={
                "failed_attempts": profile.failed_attempts,
                "anomaly_score": profile.anomalous_behavior_score,
                "threat_level": profile.threat_level,
                "suspicious_files": profile.suspicious_files
            }
        )
        
        # In a real implementation, would integrate with firewall/WAF
        logger.critical(f"AUTO-BLOCKED IP: {profile.ip_address}")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop for continuous threat analysis."""
        while self._monitoring_active:
            try:
                # Cleanup old events and profiles
                self._cleanup_old_data()
                
                # Generate threat intelligence reports
                self._generate_threat_intel()
                
                # Sleep before next iteration
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in security monitoring loop: {e}")
                time.sleep(60)  # Wait before retry
    
    def _cleanup_old_data(self) -> None:
        """Clean up old security events and threat profiles."""
        cutoff_time = datetime.now() - timedelta(hours=self.monitoring_window_hours)
        
        with self._profiles_lock:
            # Remove old threat profiles
            expired_ips = [
                ip for ip, profile in self._threat_profiles.items()
                if profile.last_activity < cutoff_time and not profile.blocked
            ]
            
            for ip in expired_ips:
                del self._threat_profiles[ip]
            
            if expired_ips:
                logger.info(f"Cleaned up {len(expired_ips)} expired threat profiles")
    
    def _generate_threat_intel(self) -> None:
        """Generate threat intelligence reports."""
        with self._profiles_lock:
            high_risk_ips = [
                profile for profile in self._threat_profiles.values()
                if profile.threat_level in ["HIGH", "CRITICAL"]
            ]
            
            if high_risk_ips:
                logger.warning(f"Threat Intelligence: {len(high_risk_ips)} high-risk IPs detected")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        with self._events_lock, self._profiles_lock:
            # Event statistics
            recent_events = [
                e for e in self._security_events
                if e.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            event_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            
            for event in recent_events:
                event_counts[event.event_type] += 1
                severity_counts[event.severity] += 1
            
            # Threat profile statistics  
            threat_stats = defaultdict(int)
            blocked_ips = []
            
            for profile in self._threat_profiles.values():
                threat_stats[profile.threat_level] += 1
                if profile.blocked:
                    blocked_ips.append(profile.ip_address)
            
            return {
                "monitoring_status": "active",
                "events_last_24h": len(recent_events),
                "event_types": dict(event_counts),
                "severity_distribution": dict(severity_counts),
                "threat_levels": dict(threat_stats),
                "blocked_ips": blocked_ips,
                "total_threat_profiles": len(self._threat_profiles),
                "auto_blocking_enabled": self.auto_block_enabled,
                "last_updated": datetime.now().isoformat()
            }
    
    def get_threat_profile(self, ip_address: str) -> Optional[ThreatProfile]:
        """Get threat profile for specific IP address."""
        with self._profiles_lock:
            return self._threat_profiles.get(ip_address)
    
    def manually_block_ip(self, ip_address: str, reason: str) -> bool:
        """Manually block an IP address."""
        with self._profiles_lock:
            profile = self._threat_profiles.get(
                ip_address,
                ThreatProfile(ip_address=ip_address)
            )
            profile.blocked = True
            self._threat_profiles[ip_address] = profile
        
        self.log_security_event(
            event_type="ip_manually_blocked",
            severity="HIGH", 
            description=f"Manually blocked IP {ip_address}: {reason}",
            source_ip=ip_address,
            metadata={"reason": reason}
        )
        
        logger.warning(f"MANUALLY BLOCKED IP: {ip_address} - {reason}")
        return True
    
    def unblock_ip(self, ip_address: str, reason: str) -> bool:
        """Unblock a previously blocked IP address."""
        with self._profiles_lock:
            if ip_address in self._threat_profiles:
                self._threat_profiles[ip_address].blocked = False
                
                self.log_security_event(
                    event_type="ip_unblocked",
                    severity="INFO",
                    description=f"Unblocked IP {ip_address}: {reason}",
                    source_ip=ip_address,
                    metadata={"reason": reason}
                )
                
                logger.info(f"UNBLOCKED IP: {ip_address} - {reason}")
                return True
        
        return False
    
    def stop_monitoring(self) -> None:
        """Stop the security monitoring system."""
        self._monitoring_active = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        logger.info("Security monitoring stopped")


@contextmanager
def security_context(monitor: SecurityMonitor, operation: str, **kwargs):
    """Context manager for security-aware operations."""
    start_time = time.time()
    
    monitor.log_security_event(
        event_type="operation_started",
        severity="INFO",
        description=f"Started security-monitored operation: {operation}",
        metadata={"operation": operation, **kwargs}
    )
    
    try:
        yield
        
        monitor.log_security_event(
            event_type="operation_completed",
            severity="INFO", 
            description=f"Completed operation: {operation}",
            metadata={
                "operation": operation,
                "duration_seconds": time.time() - start_time,
                **kwargs
            }
        )
        
    except Exception as e:
        monitor.log_security_event(
            event_type="operation_failed",
            severity="HIGH",
            description=f"Operation failed: {operation} - {str(e)}",
            metadata={
                "operation": operation,
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                **kwargs
            }
        )
        raise


# Global security monitor instance
_global_monitor: Optional[SecurityMonitor] = None
_monitor_lock = Lock()


def get_security_monitor(config: Optional[Dict[str, Any]] = None) -> SecurityMonitor:
    """Get or create global security monitor instance."""
    global _global_monitor
    
    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = SecurityMonitor(config)
        return _global_monitor


def initialize_security_monitoring(config: Optional[Dict[str, Any]] = None) -> SecurityMonitor:
    """Initialize global security monitoring system."""
    return get_security_monitor(config)


# Convenience functions for common security operations
def log_security_event(event_type: str, severity: str, description: str, **kwargs) -> SecurityEvent:
    """Log security event using global monitor."""
    monitor = get_security_monitor()
    return monitor.log_security_event(event_type, severity, description, **kwargs)


def get_security_dashboard() -> Dict[str, Any]:
    """Get security dashboard data."""
    monitor = get_security_monitor()
    return monitor.get_security_dashboard()


def block_suspicious_ip(ip_address: str, reason: str) -> bool:
    """Block suspicious IP address."""
    monitor = get_security_monitor()
    return monitor.manually_block_ip(ip_address, reason)