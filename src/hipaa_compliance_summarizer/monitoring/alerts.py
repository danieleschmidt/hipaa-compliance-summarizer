"""Alert management system for HIPAA compliance monitoring."""

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status enumeration."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure."""

    id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    details: Dict[str, Any]
    source: str
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_by": self.acknowledged_by,
            "tags": self.tags
        }


class AlertRule:
    """Base class for alert rules."""

    def __init__(self, name: str, severity: AlertSeverity,
                 cooldown_minutes: int = 5, tags: Dict[str, str] = None):
        """Initialize alert rule.
        
        Args:
            name: Alert rule name
            severity: Alert severity level
            cooldown_minutes: Minimum time between identical alerts
            tags: Optional tags for the alert
        """
        self.name = name
        self.severity = severity
        self.cooldown_minutes = cooldown_minutes
        self.tags = tags or {}
        self.last_triggered = {}  # Track last trigger time by condition

    def should_trigger(self, condition_key: str) -> bool:
        """Check if alert should trigger based on cooldown."""
        now = datetime.utcnow()
        last_trigger = self.last_triggered.get(condition_key)

        if not last_trigger:
            return True

        return (now - last_trigger).total_seconds() >= (self.cooldown_minutes * 60)

    def mark_triggered(self, condition_key: str):
        """Mark alert as triggered for cooldown tracking."""
        self.last_triggered[condition_key] = datetime.utcnow()

    def evaluate(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate metrics and return alerts if conditions are met.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            List of alerts to trigger
        """
        raise NotImplementedError


class ThresholdAlertRule(AlertRule):
    """Alert rule based on metric thresholds."""

    def __init__(self, name: str, severity: AlertSeverity, metric_name: str,
                 threshold: float, operator: str = "greater_than",
                 cooldown_minutes: int = 5, tags: Dict[str, str] = None):
        """Initialize threshold alert rule.
        
        Args:
            name: Alert rule name
            severity: Alert severity level
            metric_name: Name of the metric to monitor
            threshold: Threshold value
            operator: Comparison operator (greater_than, less_than, equals)
            cooldown_minutes: Cooldown period
            tags: Optional tags
        """
        super().__init__(name, severity, cooldown_minutes, tags)
        self.metric_name = metric_name
        self.threshold = threshold
        self.operator = operator

    def evaluate(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate threshold condition."""
        alerts = []

        if self.metric_name not in metrics:
            return alerts

        current_value = metrics[self.metric_name]
        condition_met = False

        # Evaluate condition based on operator
        if self.operator == "greater_than" and current_value > self.threshold:
            condition_met = True
        elif self.operator == "less_than" and current_value < self.threshold:
            condition_met = True
        elif self.operator == "equals" and current_value == self.threshold:
            condition_met = True
        elif self.operator == "greater_or_equal" and current_value >= self.threshold:
            condition_met = True
        elif self.operator == "less_or_equal" and current_value <= self.threshold:
            condition_met = True

        if condition_met:
            condition_key = f"{self.metric_name}_{self.operator}_{self.threshold}"

            if self.should_trigger(condition_key):
                alert = Alert(
                    id=f"{self.name}_{int(time.time())}",
                    name=self.name,
                    severity=self.severity,
                    status=AlertStatus.ACTIVE,
                    message=f"{self.metric_name} is {current_value} (threshold: {self.operator} {self.threshold})",
                    details={
                        "metric_name": self.metric_name,
                        "current_value": current_value,
                        "threshold": self.threshold,
                        "operator": self.operator
                    },
                    source="threshold_monitor",
                    created_at=datetime.utcnow(),
                    tags=self.tags
                )

                alerts.append(alert)
                self.mark_triggered(condition_key)

        return alerts


class ComplianceViolationAlertRule(AlertRule):
    """Alert rule for HIPAA compliance violations."""

    def __init__(self, max_violations_per_hour: int = 5,
                 cooldown_minutes: int = 10):
        """Initialize compliance violation alert rule.
        
        Args:
            max_violations_per_hour: Maximum violations before alert
            cooldown_minutes: Cooldown period
        """
        super().__init__(
            name="compliance_violations_threshold",
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=cooldown_minutes,
            tags={"type": "compliance", "category": "hipaa"}
        )
        self.max_violations_per_hour = max_violations_per_hour

    def evaluate(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate compliance violation threshold."""
        alerts = []

        violations_count = metrics.get("compliance_violations_last_hour", 0)

        if violations_count >= self.max_violations_per_hour:
            condition_key = f"violations_{violations_count}"

            if self.should_trigger(condition_key):
                alert = Alert(
                    id=f"compliance_violation_{int(time.time())}",
                    name=self.name,
                    severity=self.severity,
                    status=AlertStatus.ACTIVE,
                    message=f"High number of compliance violations: {violations_count} in the last hour",
                    details={
                        "violations_count": violations_count,
                        "threshold": self.max_violations_per_hour,
                        "period": "1 hour"
                    },
                    source="compliance_monitor",
                    created_at=datetime.utcnow(),
                    tags=self.tags
                )

                alerts.append(alert)
                self.mark_triggered(condition_key)

        return alerts


class HealthCheckAlertRule(AlertRule):
    """Alert rule for health check failures."""

    def __init__(self, cooldown_minutes: int = 3):
        """Initialize health check alert rule.
        
        Args:
            cooldown_minutes: Cooldown period
        """
        super().__init__(
            name="health_check_failure",
            severity=AlertSeverity.ERROR,
            cooldown_minutes=cooldown_minutes,
            tags={"type": "health", "category": "system"}
        )

    def evaluate(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate health check status."""
        alerts = []

        health_status = metrics.get("health_status", {})

        for check_name, check_result in health_status.items():
            if isinstance(check_result, dict) and check_result.get("status") == "unhealthy":
                condition_key = f"health_{check_name}"

                if self.should_trigger(condition_key):
                    alert = Alert(
                        id=f"health_check_{check_name}_{int(time.time())}",
                        name=f"{self.name}_{check_name}",
                        severity=self.severity,
                        status=AlertStatus.ACTIVE,
                        message=f"Health check failed: {check_name} - {check_result.get('message', 'Unknown error')}",
                        details={
                            "check_name": check_name,
                            "check_result": check_result,
                            "response_time_ms": check_result.get("response_time_ms", 0)
                        },
                        source="health_monitor",
                        created_at=datetime.utcnow(),
                        tags={**self.tags, "health_check": check_name}
                    )

                    alerts.append(alert)
                    self.mark_triggered(condition_key)

        return alerts


class AlertManager:
    """Manager for alert rules and notifications."""

    def __init__(self, notification_service=None):
        """Initialize alert manager.
        
        Args:
            notification_service: Optional notification service for sending alerts
        """
        self.notification_service = notification_service
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_history_size = int(os.getenv("ALERT_HISTORY_SIZE", "1000"))

        # Alert routing configuration
        self.severity_routes = {
            AlertSeverity.INFO: ["log"],
            AlertSeverity.WARNING: ["log", "email"],
            AlertSeverity.ERROR: ["log", "email", "slack"],
            AlertSeverity.CRITICAL: ["log", "email", "slack", "webhook"]
        }

    def register_rule(self, rule: AlertRule):
        """Register an alert rule.
        
        Args:
            rule: Alert rule to register
        """
        self.alert_rules.append(rule)
        logger.info(f"Registered alert rule: {rule.name}")

    def evaluate_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate all alert rules against current metrics.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            List of new alerts
        """
        new_alerts = []

        for rule in self.alert_rules:
            try:
                rule_alerts = rule.evaluate(metrics)
                new_alerts.extend(rule_alerts)
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")

        # Process new alerts
        for alert in new_alerts:
            self._process_alert(alert)

        return new_alerts

    def _process_alert(self, alert: Alert):
        """Process a new alert."""
        # Add to active alerts
        self.active_alerts[alert.id] = alert

        # Add to history
        self.alert_history.append(alert)

        # Trim history if needed
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]

        # Send notifications
        self._send_notifications(alert)

        logger.info(f"Alert triggered: {alert.name} - {alert.message}")

    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        if not self.notification_service:
            return

        routes = self.severity_routes.get(alert.severity, ["log"])

        for route in routes:
            try:
                if route == "email" and hasattr(self.notification_service, 'send_compliance_violation_alert'):
                    # Send email notification
                    recipients = os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",")
                    if recipients and recipients[0]:
                        self.notification_service.send_compliance_violation_alert(
                            to_addresses=recipients,
                            document_id=alert.details.get("document_id", "unknown"),
                            violation_type=alert.name,
                            risk_level=alert.severity.value,
                            violation_details=alert.message
                        )

                elif route == "slack" and hasattr(self.notification_service, 'send_system_alert'):
                    # Send Slack notification
                    self.notification_service.send_system_alert(
                        alert_type=alert.name,
                        severity=alert.severity.value,
                        component=alert.source,
                        description=alert.message
                    )

                elif route == "webhook" and hasattr(self.notification_service, 'send_webhook'):
                    # Send webhook notification
                    webhook_payload = {
                        "alert": alert.to_dict(),
                        "event_type": "alert_triggered"
                    }
                    self.notification_service.send_webhook("monitoring", webhook_payload)

            except Exception as e:
                logger.error(f"Failed to send {route} notification for alert {alert.id}: {e}")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: User who acknowledged the alert
            
        Returns:
            True if alert was acknowledged successfully
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True

        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert.
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if alert was resolved successfully
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()

            # Remove from active alerts
            del self.active_alerts[alert_id]

            logger.info(f"Alert {alert_id} resolved")
            return True

        return False

    def get_active_alerts(self, severity: AlertSeverity = None,
                         tags: Dict[str, str] = None) -> List[Alert]:
        """Get active alerts with optional filtering.
        
        Args:
            severity: Optional severity filter
            tags: Optional tag filters
            
        Returns:
            List of matching active alerts
        """
        alerts = list(self.active_alerts.values())

        # Filter by severity
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        # Filter by tags
        if tags:
            def matches_tags(alert: Alert) -> bool:
                return all(alert.tags.get(k) == v for k, v in tags.items())

            alerts = [a for a in alerts if matches_tags(a)]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        active_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1

        # Recent alerts (last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_alerts = [a for a in self.alert_history if a.created_at >= cutoff]

        return {
            "active_alerts": len(self.active_alerts),
            "active_by_severity": dict(active_by_severity),
            "recent_alerts_24h": len(recent_alerts),
            "total_rules": len(self.alert_rules),
            "last_evaluation": datetime.utcnow().isoformat()
        }

    def setup_default_rules(self):
        """Setup default alert rules for HIPAA compliance system."""
        # Performance threshold rules
        self.register_rule(ThresholdAlertRule(
            name="high_cpu_usage",
            severity=AlertSeverity.WARNING,
            metric_name="cpu_usage_percent",
            threshold=80.0,
            operator="greater_than",
            tags={"type": "performance", "component": "system"}
        ))

        self.register_rule(ThresholdAlertRule(
            name="critical_cpu_usage",
            severity=AlertSeverity.CRITICAL,
            metric_name="cpu_usage_percent",
            threshold=95.0,
            operator="greater_than",
            tags={"type": "performance", "component": "system"}
        ))

        self.register_rule(ThresholdAlertRule(
            name="high_memory_usage",
            severity=AlertSeverity.WARNING,
            metric_name="memory_usage_percent",
            threshold=80.0,
            operator="greater_than",
            tags={"type": "performance", "component": "system"}
        ))

        self.register_rule(ThresholdAlertRule(
            name="high_error_rate",
            severity=AlertSeverity.ERROR,
            metric_name="error_rate",
            threshold=0.05,  # 5%
            operator="greater_than",
            tags={"type": "application", "component": "api"}
        ))

        self.register_rule(ThresholdAlertRule(
            name="slow_response_time",
            severity=AlertSeverity.WARNING,
            metric_name="avg_response_time_ms",
            threshold=5000.0,  # 5 seconds
            operator="greater_than",
            tags={"type": "performance", "component": "api"}
        ))

        # Compliance-specific rules
        self.register_rule(ComplianceViolationAlertRule(
            max_violations_per_hour=5,
            cooldown_minutes=10
        ))

        # Health check rules
        self.register_rule(HealthCheckAlertRule(cooldown_minutes=3))

        logger.info(f"Setup {len(self.alert_rules)} default alert rules")
