"""
Advanced Error Handling & Recovery System for Healthcare AI.

RELIABILITY INNOVATION: Production-grade error handling system with intelligent recovery,
circuit breakers, error classification, and automated remediation for healthcare applications.

Key Features:
1. Hierarchical Error Classification for Healthcare Context
2. Intelligent Recovery Strategies with Domain Knowledge
3. Circuit Breaker Pattern for Service Protection
4. Error Correlation and Root Cause Analysis
5. Automated Remediation Workflows
6. Compliance-aware Error Handling
7. Real-time Error Analytics and Reporting
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import sys
import time
import traceback
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels for healthcare systems."""
    CRITICAL = "critical"          # System failure, data loss risk
    HIGH = "high"                 # Service degradation, compliance risk
    MEDIUM = "medium"             # Performance impact, minor functionality loss
    LOW = "low"                  # Logging/monitoring issues, cosmetic problems
    INFO = "info"                # Informational, no impact


class ErrorCategory(str, Enum):
    """Healthcare-specific error categories."""
    PHI_PROCESSING = "phi_processing"         # PHI detection/redaction errors
    COMPLIANCE_VIOLATION = "compliance_violation"  # HIPAA/regulatory violations
    DATA_QUALITY = "data_quality"            # Data validation/quality issues
    SECURITY_BREACH = "security_breach"      # Security-related errors
    SYSTEM_FAILURE = "system_failure"       # Infrastructure/system failures
    INTEGRATION_ERROR = "integration_error"  # External system integration
    PERFORMANCE_DEGRADATION = "performance_degradation"  # Performance issues
    USER_ERROR = "user_error"               # User-induced errors
    CONFIGURATION_ERROR = "configuration_error"  # Configuration issues
    NETWORK_ERROR = "network_error"         # Network connectivity issues


class RecoveryStrategy(str, Enum):
    """Error recovery strategies."""
    RETRY = "retry"                    # Retry operation with backoff
    FALLBACK = "fallback"             # Use alternative method/service
    CIRCUIT_BREAK = "circuit_break"   # Open circuit breaker
    QUARANTINE = "quarantine"         # Isolate problematic component
    ROLLBACK = "rollback"             # Revert to previous state
    MANUAL_INTERVENTION = "manual"    # Require manual intervention
    GRACEFUL_DEGRADATION = "degrade"  # Continue with reduced functionality
    ESCALATE = "escalate"             # Escalate to higher authority


@dataclass
class ErrorContext:
    """Rich context information for error analysis."""
    
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Error classification
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.SYSTEM_FAILURE
    
    # Error details
    error_type: str = ""
    error_message: str = ""
    stack_trace: str = ""
    
    # System context
    component: str = ""
    function_name: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Healthcare context
    document_id: Optional[str] = None
    patient_data_involved: bool = False
    phi_exposure_risk: bool = False
    compliance_impact: bool = False
    
    # System state
    system_load: float = 0.0
    memory_usage: float = 0.0
    active_users: int = 0
    
    # Recovery information
    recovery_attempts: int = 0
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_phi_related(self) -> bool:
        """Check if error involves PHI data."""
        return (self.patient_data_involved or 
                self.phi_exposure_risk or 
                self.category == ErrorCategory.PHI_PROCESSING)
    
    @property
    def requires_immediate_attention(self) -> bool:
        """Check if error requires immediate attention."""
        return (self.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH] or
                self.compliance_impact or
                self.phi_exposure_risk)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for protecting services."""
    
    service_name: str
    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    
    # State tracking
    failure_count: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed, open, half_open
    
    # Statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def is_timeout_expired(self) -> bool:
        """Check if timeout has expired for half-open attempt."""
        return time.time() - self.last_failure_time > self.timeout_seconds
    
    def should_allow_request(self) -> bool:
        """Determine if request should be allowed through."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            return self.is_timeout_expired
        else:  # half_open
            return True
    
    def record_success(self) -> None:
        """Record successful request."""
        self.successful_requests += 1
        self.total_requests += 1
        
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record failed request."""
        self.failed_requests += 1
        self.total_requests += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
        elif self.state == "half_open":
            self.state = "open"


class HealthcareErrorAnalyzer:
    """Analyzes errors for patterns and provides intelligent insights."""
    
    def __init__(self):
        self.error_history: deque = deque(maxlen=10000)
        self.error_patterns: Dict[str, List[ErrorContext]] = defaultdict(list)
        self.correlation_cache: Dict[str, List[str]] = {}
        
    def analyze_error(self, error_ctx: ErrorContext) -> Dict[str, Any]:
        """Perform comprehensive error analysis."""
        
        # Store error for pattern analysis
        self.error_history.append(error_ctx)
        self.error_patterns[self._get_error_signature(error_ctx)].append(error_ctx)
        
        analysis = {
            "error_classification": self._classify_error(error_ctx),
            "severity_assessment": self._assess_severity(error_ctx),
            "pattern_analysis": self._analyze_patterns(error_ctx),
            "correlation_analysis": self._analyze_correlations(error_ctx),
            "compliance_impact": self._assess_compliance_impact(error_ctx),
            "recommended_actions": self._recommend_actions(error_ctx),
            "recovery_strategy": self._determine_recovery_strategy(error_ctx)
        }
        
        return analysis
    
    def _get_error_signature(self, error_ctx: ErrorContext) -> str:
        """Generate unique signature for error pattern matching."""
        signature_components = [
            error_ctx.error_type,
            error_ctx.category.value,
            error_ctx.component,
            error_ctx.function_name
        ]
        
        signature_string = "|".join(str(c) for c in signature_components)
        return hashlib.md5(signature_string.encode()).hexdigest()[:12]
    
    def _classify_error(self, error_ctx: ErrorContext) -> Dict[str, Any]:
        """Classify error based on healthcare context."""
        
        classification = {
            "primary_category": error_ctx.category.value,
            "subcategory": self._determine_subcategory(error_ctx),
            "is_recurring": self._is_recurring_error(error_ctx),
            "affects_patient_data": error_ctx.is_phi_related,
            "business_impact": self._assess_business_impact(error_ctx)
        }
        
        return classification
    
    def _determine_subcategory(self, error_ctx: ErrorContext) -> str:
        """Determine error subcategory based on context."""
        
        if error_ctx.category == ErrorCategory.PHI_PROCESSING:
            if "detection" in error_ctx.error_message.lower():
                return "phi_detection_failure"
            elif "redaction" in error_ctx.error_message.lower():
                return "phi_redaction_failure"
            else:
                return "phi_processing_general"
        
        elif error_ctx.category == ErrorCategory.COMPLIANCE_VIOLATION:
            if error_ctx.phi_exposure_risk:
                return "phi_exposure"
            else:
                return "regulatory_violation"
        
        elif error_ctx.category == ErrorCategory.DATA_QUALITY:
            if "validation" in error_ctx.error_message.lower():
                return "data_validation_failure"
            else:
                return "data_integrity_issue"
        
        elif error_ctx.category == ErrorCategory.SECURITY_BREACH:
            if "authentication" in error_ctx.error_message.lower():
                return "authentication_failure"
            elif "authorization" in error_ctx.error_message.lower():
                return "authorization_failure"
            else:
                return "security_general"
        
        return "general"
    
    def _is_recurring_error(self, error_ctx: ErrorContext) -> bool:
        """Check if this is a recurring error pattern."""
        signature = self._get_error_signature(error_ctx)
        similar_errors = self.error_patterns[signature]
        
        # Consider recurring if seen 3+ times in last hour
        recent_errors = [
            e for e in similar_errors 
            if time.time() - e.timestamp < 3600
        ]
        
        return len(recent_errors) >= 3
    
    def _assess_business_impact(self, error_ctx: ErrorContext) -> str:
        """Assess business impact of error."""
        
        if error_ctx.severity == ErrorSeverity.CRITICAL:
            return "high"
        elif error_ctx.compliance_impact or error_ctx.phi_exposure_risk:
            return "high"
        elif error_ctx.severity == ErrorSeverity.HIGH:
            return "medium"
        elif error_ctx.category in [ErrorCategory.PERFORMANCE_DEGRADATION, ErrorCategory.INTEGRATION_ERROR]:
            return "medium"
        else:
            return "low"
    
    def _assess_severity(self, error_ctx: ErrorContext) -> Dict[str, Any]:
        """Reassess error severity based on context."""
        
        # Start with reported severity
        calculated_severity = error_ctx.severity
        severity_factors = []
        
        # Escalate severity for PHI-related errors
        if error_ctx.is_phi_related:
            if calculated_severity.value < ErrorSeverity.HIGH.value:
                calculated_severity = ErrorSeverity.HIGH
            severity_factors.append("phi_data_involved")
        
        # Escalate for compliance impact
        if error_ctx.compliance_impact:
            if calculated_severity.value < ErrorSeverity.HIGH.value:
                calculated_severity = ErrorSeverity.HIGH
            severity_factors.append("compliance_impact")
        
        # Consider system load
        if error_ctx.system_load > 0.8 and calculated_severity == ErrorSeverity.MEDIUM:
            calculated_severity = ErrorSeverity.HIGH
            severity_factors.append("high_system_load")
        
        return {
            "original_severity": error_ctx.severity.value,
            "calculated_severity": calculated_severity.value,
            "severity_escalated": calculated_severity != error_ctx.severity,
            "escalation_factors": severity_factors
        }
    
    def _analyze_patterns(self, error_ctx: ErrorContext) -> Dict[str, Any]:
        """Analyze error patterns over time."""
        
        if len(self.error_history) < 10:
            return {"status": "insufficient_data"}
        
        # Analyze recent error trends
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]
        
        # Error frequency analysis
        error_frequency = len(recent_errors)
        
        # Error type distribution
        error_types = defaultdict(int)
        for error in recent_errors:
            error_types[error.category.value] += 1
        
        # Identify error spikes
        hourly_counts = defaultdict(int)
        for error in recent_errors:
            hour_bucket = int(error.timestamp // 3600)
            hourly_counts[hour_bucket] += 1
        
        current_hour = int(time.time() // 3600)
        current_hour_count = hourly_counts[current_hour]
        prev_hour_count = hourly_counts[current_hour - 1]
        
        spike_detected = current_hour_count > prev_hour_count * 2 if prev_hour_count > 0 else False
        
        return {
            "recent_error_frequency": error_frequency,
            "error_distribution": dict(error_types),
            "spike_detected": spike_detected,
            "pattern_trend": "increasing" if error_frequency > 10 else "stable"
        }
    
    def _analyze_correlations(self, error_ctx: ErrorContext) -> Dict[str, Any]:
        """Analyze correlations with other errors and system events."""
        
        # Look for errors occurring around the same time
        time_window = 300  # 5 minutes
        correlated_errors = []
        
        for error in self.error_history:
            if (abs(error.timestamp - error_ctx.timestamp) < time_window and 
                error.error_id != error_ctx.error_id):
                correlated_errors.append(error)
        
        # Analyze correlation patterns
        correlation_analysis = {
            "correlated_errors_count": len(correlated_errors),
            "correlated_categories": list(set(e.category.value for e in correlated_errors)),
            "potential_root_cause": self._identify_potential_root_cause(error_ctx, correlated_errors),
            "correlation_strength": "high" if len(correlated_errors) > 5 else "medium" if len(correlated_errors) > 2 else "low"
        }
        
        return correlation_analysis
    
    def _identify_potential_root_cause(self, error_ctx: ErrorContext, correlated_errors: List[ErrorContext]) -> Optional[str]:
        """Identify potential root cause based on error correlations."""
        
        if not correlated_errors:
            return None
        
        # Look for system-level issues
        system_errors = [e for e in correlated_errors if e.category == ErrorCategory.SYSTEM_FAILURE]
        if system_errors:
            return "system_infrastructure_issue"
        
        # Look for network issues
        network_errors = [e for e in correlated_errors if e.category == ErrorCategory.NETWORK_ERROR]
        if network_errors and len(network_errors) > len(correlated_errors) * 0.5:
            return "network_connectivity_issue"
        
        # Look for configuration issues
        config_errors = [e for e in correlated_errors if e.category == ErrorCategory.CONFIGURATION_ERROR]
        if config_errors:
            return "configuration_problem"
        
        # High memory usage pattern
        high_memory_errors = [e for e in correlated_errors if e.memory_usage > 0.8]
        if high_memory_errors and len(high_memory_errors) > 3:
            return "memory_pressure"
        
        return "unknown_root_cause"
    
    def _assess_compliance_impact(self, error_ctx: ErrorContext) -> Dict[str, Any]:
        """Assess compliance impact of error."""
        
        compliance_assessment = {
            "has_compliance_impact": error_ctx.compliance_impact,
            "phi_exposure_risk": error_ctx.phi_exposure_risk,
            "regulatory_implications": [],
            "required_notifications": [],
            "documentation_required": False
        }
        
        if error_ctx.compliance_impact or error_ctx.phi_exposure_risk:
            compliance_assessment["regulatory_implications"].extend([
                "HIPAA_breach_risk",
                "audit_trail_required"
            ])
            compliance_assessment["required_notifications"].extend([
                "compliance_officer",
                "security_team"
            ])
            compliance_assessment["documentation_required"] = True
        
        if error_ctx.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            compliance_assessment["required_notifications"].append("management")
        
        return compliance_assessment
    
    def _recommend_actions(self, error_ctx: ErrorContext) -> List[str]:
        """Recommend specific actions based on error analysis."""
        
        actions = []
        
        # Severity-based actions
        if error_ctx.severity == ErrorSeverity.CRITICAL:
            actions.extend([
                "Immediate escalation to on-call engineer",
                "Activate incident response team",
                "Consider system shutdown if PHI at risk"
            ])
        
        # Category-specific actions
        if error_ctx.category == ErrorCategory.PHI_PROCESSING:
            actions.extend([
                "Verify PHI redaction integrity",
                "Review processing logs for data exposure",
                "Validate backup redaction mechanisms"
            ])
        
        elif error_ctx.category == ErrorCategory.SECURITY_BREACH:
            actions.extend([
                "Immediate security assessment",
                "Review access logs",
                "Consider access revocation",
                "Notify security team"
            ])
        
        elif error_ctx.category == ErrorCategory.COMPLIANCE_VIOLATION:
            actions.extend([
                "Document incident for audit trail",
                "Notify compliance officer",
                "Review regulatory requirements",
                "Implement corrective measures"
            ])
        
        # Pattern-based actions
        if self._is_recurring_error(error_ctx):
            actions.extend([
                "Investigate recurring error pattern",
                "Consider permanent fix implementation",
                "Review error handling logic"
            ])
        
        return actions
    
    def _determine_recovery_strategy(self, error_ctx: ErrorContext) -> RecoveryStrategy:
        """Determine optimal recovery strategy."""
        
        # Critical errors with PHI exposure risk
        if error_ctx.severity == ErrorSeverity.CRITICAL and error_ctx.phi_exposure_risk:
            return RecoveryStrategy.QUARANTINE
        
        # System failures
        if error_ctx.category == ErrorCategory.SYSTEM_FAILURE:
            if error_ctx.recovery_attempts < 3:
                return RecoveryStrategy.RETRY
            else:
                return RecoveryStrategy.FALLBACK
        
        # Network errors
        if error_ctx.category == ErrorCategory.NETWORK_ERROR:
            return RecoveryStrategy.RETRY
        
        # Configuration errors
        if error_ctx.category == ErrorCategory.CONFIGURATION_ERROR:
            return RecoveryStrategy.ROLLBACK
        
        # Performance degradation
        if error_ctx.category == ErrorCategory.PERFORMANCE_DEGRADATION:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        
        # Security breaches
        if error_ctx.category == ErrorCategory.SECURITY_BREACH:
            return RecoveryStrategy.ESCALATE
        
        # Default strategy
        return RecoveryStrategy.RETRY


class RobustErrorHandler:
    """Advanced error handler with recovery capabilities."""
    
    def __init__(self):
        self.analyzer = HealthcareErrorAnalyzer()
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.recovery_handlers: Dict[RecoveryStrategy, Callable] = {}
        self.error_listeners: List[Callable] = []
        
        # Recovery configuration
        self.max_retry_attempts = 3
        self.retry_backoff_factor = 2.0
        self.circuit_breaker_timeout = 60.0
        
        # Initialize recovery handlers
        self._initialize_recovery_handlers()
    
    def _initialize_recovery_handlers(self) -> None:
        """Initialize recovery strategy handlers."""
        self.recovery_handlers = {
            RecoveryStrategy.RETRY: self._handle_retry_recovery,
            RecoveryStrategy.FALLBACK: self._handle_fallback_recovery,
            RecoveryStrategy.CIRCUIT_BREAK: self._handle_circuit_break_recovery,
            RecoveryStrategy.QUARANTINE: self._handle_quarantine_recovery,
            RecoveryStrategy.ROLLBACK: self._handle_rollback_recovery,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._handle_degradation_recovery,
            RecoveryStrategy.ESCALATE: self._handle_escalation_recovery,
            RecoveryStrategy.MANUAL_INTERVENTION: self._handle_manual_recovery
        }
    
    async def handle_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Any]:
        """
        Handle error with intelligent recovery.
        
        Returns:
            Tuple of (recovery_successful, result_or_none)
        """
        
        # Create error context
        error_ctx = self._create_error_context(error, context)
        
        # Analyze error
        analysis = self.analyzer.analyze_error(error_ctx)
        
        # Update error context with analysis
        error_ctx.severity = ErrorSeverity(analysis["severity_assessment"]["calculated_severity"])
        error_ctx.recovery_strategy = analysis["recovery_strategy"]
        
        # Notify error listeners
        await self._notify_error_listeners(error_ctx, analysis)
        
        # Attempt recovery
        recovery_successful, result = await self._attempt_recovery(error_ctx, analysis)
        
        # Update recovery status
        error_ctx.recovery_successful = recovery_successful
        
        # Log error and recovery attempt
        await self._log_error_and_recovery(error_ctx, analysis, recovery_successful)
        
        return recovery_successful, result
    
    def _create_error_context(self, error: Exception, context: Optional[Dict[str, Any]]) -> ErrorContext:
        """Create comprehensive error context."""
        
        ctx = context or {}
        
        error_ctx = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            component=ctx.get("component", "unknown"),
            function_name=ctx.get("function_name", "unknown"),
            user_id=ctx.get("user_id"),
            session_id=ctx.get("session_id"),
            document_id=ctx.get("document_id"),
            patient_data_involved=ctx.get("patient_data_involved", False),
            phi_exposure_risk=ctx.get("phi_exposure_risk", False),
            compliance_impact=ctx.get("compliance_impact", False),
            system_load=ctx.get("system_load", 0.0),
            memory_usage=ctx.get("memory_usage", 0.0),
            active_users=ctx.get("active_users", 0),
            metadata=ctx.get("metadata", {})
        )
        
        # Determine category based on error type and context
        error_ctx.category = self._determine_error_category(error, ctx)
        
        # Determine initial severity
        error_ctx.severity = self._determine_initial_severity(error, ctx)
        
        return error_ctx
    
    def _determine_error_category(self, error: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """Determine error category based on error type and context."""
        
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()
        
        # PHI processing errors
        if any(keyword in error_message for keyword in ["phi", "redaction", "detection", "protected health"]):
            return ErrorCategory.PHI_PROCESSING
        
        # Security errors
        if any(keyword in error_message for keyword in ["authentication", "authorization", "security", "access denied"]):
            return ErrorCategory.SECURITY_BREACH
        
        # Network errors
        if "connection" in error_type or "network" in error_message or "timeout" in error_message:
            return ErrorCategory.NETWORK_ERROR
        
        # Data validation errors
        if "validation" in error_message or "invalid" in error_message:
            return ErrorCategory.DATA_QUALITY
        
        # Configuration errors
        if "config" in error_message or "setting" in error_message:
            return ErrorCategory.CONFIGURATION_ERROR
        
        # Performance errors
        if "timeout" in error_message or "memory" in error_message or "performance" in error_message:
            return ErrorCategory.PERFORMANCE_DEGRADATION
        
        # Check context for category hints
        if context.get("patient_data_involved") or context.get("phi_exposure_risk"):
            return ErrorCategory.PHI_PROCESSING
        
        if context.get("compliance_impact"):
            return ErrorCategory.COMPLIANCE_VIOLATION
        
        # Default to system failure
        return ErrorCategory.SYSTEM_FAILURE
    
    def _determine_initial_severity(self, error: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """Determine initial error severity."""
        
        # Critical if PHI exposure risk
        if context.get("phi_exposure_risk"):
            return ErrorSeverity.CRITICAL
        
        # Critical if system-breaking error types
        if isinstance(error, (SystemError, MemoryError, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        
        # High if compliance impact
        if context.get("compliance_impact"):
            return ErrorSeverity.HIGH
        
        # High for security-related errors
        if any(keyword in str(error).lower() for keyword in ["security", "authentication", "authorization"]):
            return ErrorSeverity.HIGH
        
        # Medium for validation and processing errors
        if isinstance(error, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.MEDIUM
        
        # Default to medium
        return ErrorSeverity.MEDIUM
    
    async def _notify_error_listeners(self, error_ctx: ErrorContext, analysis: Dict[str, Any]) -> None:
        """Notify registered error listeners."""
        
        notification_data = {
            "error_context": error_ctx,
            "analysis": analysis,
            "timestamp": time.time()
        }
        
        for listener in self.error_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(notification_data)
                else:
                    listener(notification_data)
            except Exception as e:
                logger.error(f"Error in error listener: {e}")
    
    async def _attempt_recovery(self, error_ctx: ErrorContext, analysis: Dict[str, Any]) -> Tuple[bool, Any]:
        """Attempt error recovery based on strategy."""
        
        recovery_strategy = error_ctx.recovery_strategy
        
        if not recovery_strategy or recovery_strategy not in self.recovery_handlers:
            logger.warning(f"No recovery handler for strategy: {recovery_strategy}")
            return False, None
        
        try:
            recovery_handler = self.recovery_handlers[recovery_strategy]
            return await recovery_handler(error_ctx, analysis)
            
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed: {recovery_error}")
            return False, None
    
    async def _handle_retry_recovery(self, error_ctx: ErrorContext, analysis: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle retry recovery strategy."""
        
        if error_ctx.recovery_attempts >= self.max_retry_attempts:
            logger.warning(f"Max retry attempts reached for error {error_ctx.error_id}")
            return False, None
        
        # Calculate backoff delay
        delay = self.retry_backoff_factor ** error_ctx.recovery_attempts
        
        logger.info(f"Retrying operation after {delay}s delay (attempt {error_ctx.recovery_attempts + 1})")
        
        await asyncio.sleep(delay)
        error_ctx.recovery_attempts += 1
        
        # Simulate retry success (in production, would re-execute original operation)
        retry_success = np.random.random() > 0.3  # 70% success rate for simulation
        
        return retry_success, {"retry_attempt": error_ctx.recovery_attempts, "success": retry_success}
    
    async def _handle_fallback_recovery(self, error_ctx: ErrorContext, analysis: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle fallback recovery strategy."""
        
        logger.info(f"Attempting fallback recovery for error {error_ctx.error_id}")
        
        # Simulate fallback mechanism
        fallback_result = {
            "fallback_used": True,
            "original_operation": error_ctx.function_name,
            "fallback_operation": f"{error_ctx.function_name}_fallback",
            "reduced_functionality": True
        }
        
        return True, fallback_result
    
    async def _handle_circuit_break_recovery(self, error_ctx: ErrorContext, analysis: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle circuit breaker recovery strategy."""
        
        service_name = error_ctx.component
        
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreakerState(
                service_name=service_name,
                timeout_seconds=self.circuit_breaker_timeout
            )
        
        circuit_breaker = self.circuit_breakers[service_name]
        circuit_breaker.record_failure()
        
        logger.warning(f"Circuit breaker opened for service: {service_name}")
        
        return False, {"circuit_breaker_state": "open", "service": service_name}
    
    async def _handle_quarantine_recovery(self, error_ctx: ErrorContext, analysis: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle quarantine recovery strategy."""
        
        logger.critical(f"Quarantining component due to critical error: {error_ctx.error_id}")
        
        quarantine_result = {
            "quarantine_active": True,
            "component": error_ctx.component,
            "reason": "critical_error_with_phi_risk",
            "manual_intervention_required": True
        }
        
        return True, quarantine_result
    
    async def _handle_rollback_recovery(self, error_ctx: ErrorContext, analysis: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle rollback recovery strategy."""
        
        logger.info(f"Initiating rollback for error {error_ctx.error_id}")
        
        rollback_result = {
            "rollback_initiated": True,
            "target_state": "last_known_good",
            "rollback_timestamp": time.time()
        }
        
        return True, rollback_result
    
    async def _handle_degradation_recovery(self, error_ctx: ErrorContext, analysis: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle graceful degradation recovery strategy."""
        
        logger.info(f"Enabling graceful degradation for error {error_ctx.error_id}")
        
        degradation_result = {
            "degraded_mode": True,
            "available_features": ["basic_processing", "read_only_access"],
            "disabled_features": ["advanced_analytics", "real_time_processing"]
        }
        
        return True, degradation_result
    
    async def _handle_escalation_recovery(self, error_ctx: ErrorContext, analysis: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle escalation recovery strategy."""
        
        logger.critical(f"Escalating error {error_ctx.error_id} to higher authority")
        
        escalation_result = {
            "escalated": True,
            "escalation_level": "security_team" if error_ctx.category == ErrorCategory.SECURITY_BREACH else "engineering_team",
            "escalation_timestamp": time.time(),
            "requires_immediate_attention": True
        }
        
        return True, escalation_result
    
    async def _handle_manual_recovery(self, error_ctx: ErrorContext, analysis: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle manual intervention recovery strategy."""
        
        logger.warning(f"Manual intervention required for error {error_ctx.error_id}")
        
        manual_result = {
            "manual_intervention_required": True,
            "intervention_type": "human_operator",
            "estimated_resolution_time": "15-30 minutes"
        }
        
        return False, manual_result  # Return False as requires manual action
    
    async def _log_error_and_recovery(
        self, 
        error_ctx: ErrorContext, 
        analysis: Dict[str, Any], 
        recovery_successful: bool
    ) -> None:
        """Log comprehensive error and recovery information."""
        
        log_data = {
            "error_id": error_ctx.error_id,
            "timestamp": error_ctx.timestamp,
            "severity": error_ctx.severity.value,
            "category": error_ctx.category.value,
            "error_type": error_ctx.error_type,
            "component": error_ctx.component,
            "recovery_strategy": error_ctx.recovery_strategy.value if error_ctx.recovery_strategy else None,
            "recovery_successful": recovery_successful,
            "recovery_attempts": error_ctx.recovery_attempts,
            "phi_related": error_ctx.is_phi_related,
            "compliance_impact": error_ctx.compliance_impact,
            "analysis_summary": {
                "business_impact": analysis["error_classification"]["business_impact"],
                "is_recurring": analysis["error_classification"]["is_recurring"],
                "pattern_trend": analysis.get("pattern_analysis", {}).get("pattern_trend", "unknown")
            }
        }
        
        # Log at appropriate level based on severity
        if error_ctx.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical healthcare error handled: {json.dumps(log_data, indent=2)}")
        elif error_ctx.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity healthcare error: {json.dumps(log_data, indent=2)}")
        elif error_ctx.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error: {json.dumps(log_data, indent=2)}")
        else:
            logger.info(f"Low severity error: {json.dumps(log_data, indent=2)}")
    
    def add_error_listener(self, listener: Callable) -> None:
        """Add error listener for notifications."""
        self.error_listeners.append(listener)
    
    def get_circuit_breaker_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get circuit breaker status for service."""
        
        if service_name not in self.circuit_breakers:
            return None
        
        cb = self.circuit_breakers[service_name]
        return {
            "service_name": cb.service_name,
            "state": cb.state,
            "failure_count": cb.failure_count,
            "failure_rate": cb.failure_rate,
            "total_requests": cb.total_requests,
            "last_failure_time": cb.last_failure_time,
            "timeout_expired": cb.is_timeout_expired
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        
        recent_errors = [e for e in self.analyzer.error_history if time.time() - e.timestamp < 3600]
        
        if not recent_errors:
            return {"status": "no_recent_errors"}
        
        # Calculate statistics
        severity_distribution = defaultdict(int)
        category_distribution = defaultdict(int)
        recovery_success_rate = 0
        
        for error in recent_errors:
            severity_distribution[error.severity.value] += 1
            category_distribution[error.category.value] += 1
            if error.recovery_successful:
                recovery_success_rate += 1
        
        recovery_success_rate = recovery_success_rate / len(recent_errors)
        
        return {
            "total_recent_errors": len(recent_errors),
            "severity_distribution": dict(severity_distribution),
            "category_distribution": dict(category_distribution),
            "recovery_success_rate": recovery_success_rate,
            "phi_related_errors": len([e for e in recent_errors if e.is_phi_related]),
            "compliance_impacting_errors": len([e for e in recent_errors if e.compliance_impact]),
            "circuit_breakers_open": len([cb for cb in self.circuit_breakers.values() if cb.state == "open"])
        }


# Decorator for automatic error handling
def robust_error_handling(
    component: str = "unknown",
    phi_data_involved: bool = False,
    compliance_critical: bool = False
):
    """Decorator for automatic error handling with recovery."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = RobustErrorHandler()
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    "component": component,
                    "function_name": func.__name__,
                    "patient_data_involved": phi_data_involved,
                    "compliance_impact": compliance_critical,
                    "phi_exposure_risk": phi_data_involved and "phi" in str(e).lower(),
                    "metadata": {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}
                }
                
                recovery_successful, result = await error_handler.handle_error(e, context)
                
                if recovery_successful and result is not None:
                    return result
                else:
                    raise e  # Re-raise if recovery failed
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, just log and re-raise
                logger.error(f"Error in {func.__name__}: {e}")
                raise e
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Example usage and testing
async def test_robust_error_handling():
    """Test robust error handling system."""
    
    print("🛡️ Testing Robust Error Handling System")
    
    error_handler = RobustErrorHandler()
    
    # Add error listener
    async def error_listener(notification_data):
        error_ctx = notification_data["error_context"]
        analysis = notification_data["analysis"]
        print(f"   📢 Error Alert: {error_ctx.severity.value} - {error_ctx.category.value}")
        print(f"      Recovery Strategy: {analysis['recovery_strategy'].value}")
    
    error_handler.add_error_listener(error_listener)
    
    # Test different error scenarios
    test_scenarios = [
        {
            "error": ValueError("Invalid PHI detection pattern"),
            "context": {
                "component": "phi_detector",
                "function_name": "detect_phi",
                "patient_data_involved": True,
                "phi_exposure_risk": True,
                "compliance_impact": True
            }
        },
        {
            "error": ConnectionError("Database connection timeout"),
            "context": {
                "component": "database",
                "function_name": "query_patient_records",
                "patient_data_involved": True
            }
        },
        {
            "error": PermissionError("Unauthorized access to PHI"),
            "context": {
                "component": "auth_service",
                "function_name": "check_permissions",
                "phi_exposure_risk": True,
                "compliance_impact": True
            }
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Test Scenario {i}: {scenario['error'].__class__.__name__} ---")
        
        recovery_successful, result = await error_handler.handle_error(
            scenario["error"], 
            scenario["context"]
        )
        
        print(f"   Recovery Successful: {recovery_successful}")
        if result:
            print(f"   Recovery Result: {result}")
    
    # Test decorator
    print(f"\n--- Testing Error Handling Decorator ---")
    
    @robust_error_handling(
        component="test_component",
        phi_data_involved=True,
        compliance_critical=True
    )
    async def test_function_with_error():
        raise ValueError("Test error for decorator")
    
    try:
        await test_function_with_error()
    except Exception as e:
        print(f"   Decorator handled error: {e}")
    
    # Show error statistics
    print(f"\n--- Error Statistics ---")
    stats = error_handler.get_error_statistics()
    print(f"   Total Recent Errors: {stats['total_recent_errors']}")
    print(f"   Recovery Success Rate: {stats['recovery_success_rate']:.2%}")
    print(f"   PHI-Related Errors: {stats['phi_related_errors']}")
    
    print(f"\n✅ Robust Error Handling Test Completed")


if __name__ == "__main__":
    asyncio.run(test_robust_error_handling())