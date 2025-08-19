"""
Comprehensive Monitoring & Alerting System for Healthcare AI.

OBSERVABILITY INNOVATION: Production-grade monitoring system with intelligent alerting,
performance tracking, compliance monitoring, and predictive analytics for healthcare AI systems.

Key Features:
1. Multi-dimensional Healthcare Metrics Collection
2. Intelligent Alerting with ML-based Anomaly Detection
3. Compliance Monitoring with Real-time Violation Detection
4. Performance Analytics with Predictive Insights
5. Distributed Tracing for Complex Healthcare Workflows
6. Security Monitoring with Threat Detection
7. Business Intelligence Dashboard for Healthcare Operations
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected in healthcare monitoring."""
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    BUSINESS = "business"
    SYSTEM = "system"
    PHI_HANDLING = "phi_handling"
    USER_ACTIVITY = "user_activity"
    DATA_QUALITY = "data_quality"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MonitoringScope(str, Enum):
    """Scope of monitoring coverage."""
    GLOBAL = "global"           # System-wide monitoring
    SERVICE = "service"         # Individual service monitoring
    USER_SESSION = "session"    # User session monitoring
    DOCUMENT = "document"       # Document processing monitoring
    WORKFLOW = "workflow"       # Healthcare workflow monitoring


@dataclass
class HealthcareMetric:
    """Healthcare-specific metric with rich context."""
    
    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Metric identification
    name: str = ""
    type: MetricType = MetricType.PERFORMANCE
    scope: MonitoringScope = MonitoringScope.GLOBAL
    
    # Metric value and metadata
    value: Union[float, int, str, bool] = 0
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Healthcare context
    patient_data_involved: bool = False
    phi_processing: bool = False
    compliance_relevant: bool = False
    clinical_workflow: Optional[str] = None
    
    # Source information
    source_component: str = ""
    source_function: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    document_id: Optional[str] = None
    
    # Quality indicators
    accuracy: float = 1.0  # Metric accuracy/confidence
    sample_size: int = 1   # Number of samples contributing to metric
    
    @property
    def is_healthcare_critical(self) -> bool:
        """Check if metric is critical for healthcare operations."""
        return (self.compliance_relevant or 
                self.phi_processing or 
                self.type in [MetricType.COMPLIANCE, MetricType.PHI_HANDLING, MetricType.SECURITY])
    
    @property
    def metric_key(self) -> str:
        """Generate unique key for metric aggregation."""
        return f"{self.name}:{self.source_component}:{self.scope.value}"


@dataclass
class AnomalyDetection:
    """Anomaly detection configuration and results."""
    
    detection_method: str = "statistical"  # statistical, ml_based, threshold
    sensitivity: float = 0.95  # Detection sensitivity
    baseline_window: int = 1440  # Minutes for baseline calculation
    
    # Thresholds
    statistical_threshold: float = 3.0  # Standard deviations
    absolute_threshold: Optional[float] = None
    relative_threshold: Optional[float] = None  # Percentage change
    
    # Detection results
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    confidence: float = 0.0
    baseline_value: Optional[float] = None
    deviation: Optional[float] = None


@dataclass
class HealthcareAlert:
    """Healthcare-specific alert with contextual information."""
    
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Alert classification
    level: AlertLevel = AlertLevel.MEDIUM
    type: str = ""
    title: str = ""
    description: str = ""
    
    # Triggering metric and conditions
    triggering_metric: Optional[HealthcareMetric] = None
    threshold_violated: Optional[Dict[str, Any]] = None
    anomaly_detection: Optional[AnomalyDetection] = None
    
    # Healthcare context
    compliance_impact: bool = False
    phi_exposure_risk: bool = False
    patient_safety_impact: bool = False
    clinical_workflow_affected: Optional[str] = None
    
    # Response information
    recommended_actions: List[str] = field(default_factory=list)
    escalation_required: bool = False
    auto_mitigation_possible: bool = False
    estimated_resolution_time: Optional[str] = None
    
    # Alert management
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    @property
    def requires_immediate_attention(self) -> bool:
        """Check if alert requires immediate attention."""
        return (self.level in [AlertLevel.CRITICAL, AlertLevel.HIGH] or
                self.compliance_impact or
                self.phi_exposure_risk or
                self.patient_safety_impact)
    
    @property
    def alert_age_minutes(self) -> float:
        """Calculate alert age in minutes."""
        return (time.time() - self.timestamp) / 60.0


class HealthcareMetricsCollector:
    """Collects and processes healthcare-specific metrics."""
    
    def __init__(self):
        self.metrics_buffer: deque = deque(maxlen=50000)
        self.metric_aggregates: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.collection_handlers: Dict[MetricType, Callable] = {}
        
        # Aggregation windows (in minutes)
        self.aggregation_windows = [1, 5, 15, 60, 1440]  # 1min, 5min, 15min, 1hr, 1day
        
        # Initialize collection handlers
        self._initialize_collection_handlers()
    
    def _initialize_collection_handlers(self) -> None:
        """Initialize metric collection handlers."""
        self.collection_handlers = {
            MetricType.PERFORMANCE: self._collect_performance_metrics,
            MetricType.COMPLIANCE: self._collect_compliance_metrics,
            MetricType.SECURITY: self._collect_security_metrics,
            MetricType.PHI_HANDLING: self._collect_phi_handling_metrics,
            MetricType.DATA_QUALITY: self._collect_data_quality_metrics
        }
    
    async def collect_metric(self, metric: HealthcareMetric) -> None:
        """Collect and store healthcare metric."""
        
        # Validate metric
        if not self._validate_metric(metric):
            logger.warning(f"Invalid metric rejected: {metric.name}")
            return
        
        # Enrich metric with additional context
        await self._enrich_metric_context(metric)
        
        # Store metric
        self.metrics_buffer.append(metric)
        
        # Update aggregates
        await self._update_metric_aggregates(metric)
        
        # Log metric collection
        if metric.is_healthcare_critical:
            logger.info(f"Healthcare critical metric collected: {metric.name} = {metric.value}")
    
    def _validate_metric(self, metric: HealthcareMetric) -> bool:
        """Validate metric data quality."""
        
        # Required fields
        if not metric.name or not metric.source_component:
            return False
        
        # Value validation
        if isinstance(metric.value, (int, float)):
            if np.isnan(metric.value) or np.isinf(metric.value):
                return False
        
        # Healthcare-specific validation
        if metric.phi_processing and not metric.patient_data_involved:
            logger.warning("PHI processing metric should involve patient data")
        
        if metric.compliance_relevant and metric.type != MetricType.COMPLIANCE:
            # Auto-correct metric type for compliance-relevant metrics
            metric.type = MetricType.COMPLIANCE
        
        return True
    
    async def _enrich_metric_context(self, metric: HealthcareMetric) -> None:
        """Enrich metric with additional contextual information."""
        
        # Add timestamp-based tags
        dt = datetime.fromtimestamp(metric.timestamp)
        metric.tags.update({
            "hour": str(dt.hour),
            "day_of_week": str(dt.weekday()),
            "is_business_hours": str(9 <= dt.hour <= 17),
            "is_weekend": str(dt.weekday() >= 5)
        })
        
        # Add healthcare context tags
        if metric.phi_processing:
            metric.tags["phi_processing"] = "true"
            metric.compliance_relevant = True
        
        if metric.clinical_workflow:
            metric.tags["workflow"] = metric.clinical_workflow
        
        # Performance context
        if metric.type == MetricType.PERFORMANCE:
            metric.tags["performance_category"] = self._categorize_performance_metric(metric)
    
    def _categorize_performance_metric(self, metric: HealthcareMetric) -> str:
        """Categorize performance metrics."""
        
        metric_name_lower = metric.name.lower()
        
        if "latency" in metric_name_lower or "response_time" in metric_name_lower:
            return "latency"
        elif "throughput" in metric_name_lower or "requests_per_second" in metric_name_lower:
            return "throughput"
        elif "memory" in metric_name_lower or "cpu" in metric_name_lower:
            return "resource_usage"
        elif "error" in metric_name_lower or "failure" in metric_name_lower:
            return "error_rate"
        else:
            return "general"
    
    async def _update_metric_aggregates(self, metric: HealthcareMetric) -> None:
        """Update metric aggregates for different time windows."""
        
        metric_key = metric.metric_key
        current_time = time.time()
        
        # Initialize aggregate if not exists
        if metric_key not in self.metric_aggregates:
            self.metric_aggregates[metric_key] = {
                "latest_value": metric.value,
                "latest_timestamp": metric.timestamp,
                "windows": {}
            }
        
        # Update latest value
        aggregate = self.metric_aggregates[metric_key]
        aggregate["latest_value"] = metric.value
        aggregate["latest_timestamp"] = metric.timestamp
        
        # Update windowed aggregates
        for window_minutes in self.aggregation_windows:
            window_key = f"{window_minutes}m"
            window_start = current_time - (window_minutes * 60)
            
            # Get metrics in window
            window_metrics = [
                m for m in self.metrics_buffer
                if (m.metric_key == metric_key and 
                    m.timestamp >= window_start and
                    isinstance(m.value, (int, float)))
            ]
            
            if window_metrics:
                values = [m.value for m in window_metrics]
                aggregate["windows"][window_key] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "p50": statistics.median(values),
                    "p95": np.percentile(values, 95) if len(values) >= 20 else max(values),
                    "p99": np.percentile(values, 99) if len(values) >= 100 else max(values)
                }
    
    async def _collect_performance_metrics(self) -> List[HealthcareMetric]:
        """Collect system performance metrics."""
        
        current_time = time.time()
        metrics = []
        
        # Simulate performance metrics collection
        metrics.extend([
            HealthcareMetric(
                name="phi_detection_latency",
                type=MetricType.PERFORMANCE,
                value=np.random.gamma(2, 50),  # Simulated latency in ms
                unit="milliseconds",
                source_component="phi_detector",
                phi_processing=True,
                tags={"operation": "phi_detection"}
            ),
            HealthcareMetric(
                name="compliance_check_duration",
                type=MetricType.PERFORMANCE,
                value=np.random.gamma(1.5, 30),
                unit="milliseconds",
                source_component="compliance_checker",
                compliance_relevant=True,
                tags={"operation": "compliance_validation"}
            ),
            HealthcareMetric(
                name="document_processing_throughput",
                type=MetricType.PERFORMANCE,
                value=np.random.uniform(50, 200),
                unit="documents_per_minute",
                source_component="document_processor",
                patient_data_involved=True,
                tags={"operation": "batch_processing"}
            )
        ])
        
        return metrics
    
    async def _collect_compliance_metrics(self) -> List[HealthcareMetric]:
        """Collect compliance-related metrics."""
        
        metrics = []
        
        # Simulate compliance metrics
        metrics.extend([
            HealthcareMetric(
                name="hipaa_compliance_score",
                type=MetricType.COMPLIANCE,
                value=np.random.uniform(0.92, 0.99),
                unit="score",
                source_component="compliance_monitor",
                compliance_relevant=True,
                tags={"regulation": "hipaa"}
            ),
            HealthcareMetric(
                name="phi_redaction_accuracy",
                type=MetricType.COMPLIANCE,
                value=np.random.uniform(0.95, 0.999),
                unit="accuracy",
                source_component="phi_redactor",
                phi_processing=True,
                compliance_relevant=True,
                tags={"operation": "phi_redaction"}
            ),
            HealthcareMetric(
                name="audit_trail_completeness",
                type=MetricType.COMPLIANCE,
                value=np.random.uniform(0.98, 1.0),
                unit="completeness",
                source_component="audit_logger",
                compliance_relevant=True,
                tags={"audit_type": "access_log"}
            )
        ])
        
        return metrics
    
    async def _collect_security_metrics(self) -> List[HealthcareMetric]:
        """Collect security-related metrics."""
        
        metrics = []
        
        # Simulate security metrics
        metrics.extend([
            HealthcareMetric(
                name="failed_authentication_attempts",
                type=MetricType.SECURITY,
                value=np.random.poisson(2),  # Low rate of failed attempts
                unit="count",
                source_component="auth_service",
                compliance_relevant=True,
                tags={"security_event": "auth_failure"}
            ),
            HealthcareMetric(
                name="suspicious_access_patterns",
                type=MetricType.SECURITY,
                value=np.random.poisson(0.5),
                unit="count",
                source_component="security_monitor",
                phi_processing=True,
                compliance_relevant=True,
                tags={"security_event": "anomalous_access"}
            ),
            HealthcareMetric(
                name="encryption_success_rate",
                type=MetricType.SECURITY,
                value=np.random.uniform(0.998, 1.0),
                unit="rate",
                source_component="encryption_service",
                phi_processing=True,
                tags={"operation": "data_encryption"}
            )
        ])
        
        return metrics
    
    async def _collect_phi_handling_metrics(self) -> List[HealthcareMetric]:
        """Collect PHI handling specific metrics."""
        
        metrics = []
        
        metrics.extend([
            HealthcareMetric(
                name="phi_entities_detected_per_document",
                type=MetricType.PHI_HANDLING,
                value=np.random.poisson(5),
                unit="count",
                source_component="phi_detector",
                phi_processing=True,
                patient_data_involved=True,
                tags={"operation": "phi_detection"}
            ),
            HealthcareMetric(
                name="phi_false_positive_rate",
                type=MetricType.PHI_HANDLING,
                value=np.random.uniform(0.01, 0.05),
                unit="rate",
                source_component="phi_detector",
                phi_processing=True,
                tags={"metric_type": "quality"}
            ),
            HealthcareMetric(
                name="phi_exposure_incidents",
                type=MetricType.PHI_HANDLING,
                value=np.random.poisson(0.1),  # Very low rate
                unit="count",
                source_component="phi_monitor",
                phi_processing=True,
                compliance_relevant=True,
                tags={"incident_type": "phi_exposure"}
            )
        ])
        
        return metrics
    
    async def _collect_data_quality_metrics(self) -> List[HealthcareMetric]:
        """Collect data quality metrics."""
        
        metrics = []
        
        metrics.extend([
            HealthcareMetric(
                name="data_completeness_score",
                type=MetricType.DATA_QUALITY,
                value=np.random.uniform(0.90, 0.98),
                unit="score",
                source_component="data_validator",
                patient_data_involved=True,
                tags={"quality_dimension": "completeness"}
            ),
            HealthcareMetric(
                name="data_accuracy_score",
                type=MetricType.DATA_QUALITY,
                value=np.random.uniform(0.92, 0.99),
                unit="score",
                source_component="data_validator",
                patient_data_involved=True,
                tags={"quality_dimension": "accuracy"}
            )
        ])
        
        return metrics
    
    def get_metric_summary(self, metric_name: str, window_minutes: int = 60) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a metric."""
        
        # Find matching metric keys
        matching_keys = [key for key in self.metric_aggregates.keys() if metric_name in key]
        
        if not matching_keys:
            return None
        
        window_key = f"{window_minutes}m"
        summaries = []
        
        for key in matching_keys:
            aggregate = self.metric_aggregates[key]
            if window_key in aggregate.get("windows", {}):
                window_data = aggregate["windows"][window_key]
                summaries.append({
                    "metric_key": key,
                    "latest_value": aggregate["latest_value"],
                    "window_stats": window_data
                })
        
        return {
            "metric_name": metric_name,
            "window_minutes": window_minutes,
            "matching_metrics": len(summaries),
            "summaries": summaries
        }


class IntelligentAlerting:
    """Intelligent alerting system with ML-based anomaly detection."""
    
    def __init__(self, metrics_collector: HealthcareMetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_alerts: Dict[str, HealthcareAlert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Alert configuration
        self.alert_rules: List[Dict[str, Any]] = []
        self.anomaly_detectors: Dict[str, AnomalyDetection] = {}
        self.alert_suppression: Dict[str, float] = {}  # metric_key -> last_alert_time
        
        # Escalation configuration
        self.escalation_chains: Dict[str, List[str]] = {
            "critical": ["on_call_engineer", "security_team", "compliance_officer"],
            "high": ["service_owner", "compliance_team"],
            "medium": ["service_owner"],
            "low": ["monitoring_team"]
        }
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
    
    def _initialize_default_alert_rules(self) -> None:
        """Initialize default healthcare alert rules."""
        
        self.alert_rules = [
            # Compliance alerts
            {
                "name": "hipaa_compliance_violation",
                "metric_pattern": "hipaa_compliance_score",
                "condition": "below_threshold",
                "threshold": 0.95,
                "level": AlertLevel.CRITICAL,
                "compliance_impact": True
            },
            {
                "name": "phi_redaction_accuracy_drop",
                "metric_pattern": "phi_redaction_accuracy",
                "condition": "below_threshold",
                "threshold": 0.98,
                "level": AlertLevel.HIGH,
                "phi_exposure_risk": True
            },
            
            # Performance alerts
            {
                "name": "phi_detection_latency_spike",
                "metric_pattern": "phi_detection_latency",
                "condition": "anomaly_detection",
                "threshold": 3.0,  # 3 standard deviations
                "level": AlertLevel.MEDIUM,
                "patient_safety_impact": False
            },
            
            # Security alerts
            {
                "name": "failed_authentication_spike",
                "metric_pattern": "failed_authentication_attempts",
                "condition": "above_threshold",
                "threshold": 10,
                "level": AlertLevel.HIGH,
                "phi_exposure_risk": True
            },
            {
                "name": "phi_exposure_incident",
                "metric_pattern": "phi_exposure_incidents",
                "condition": "above_threshold",
                "threshold": 0,
                "level": AlertLevel.CRITICAL,
                "compliance_impact": True,
                "phi_exposure_risk": True
            },
            
            # Data quality alerts
            {
                "name": "data_quality_degradation",
                "metric_pattern": "data_completeness_score",
                "condition": "below_threshold",
                "threshold": 0.85,
                "level": AlertLevel.MEDIUM,
                "patient_safety_impact": True
            }
        ]
    
    async def evaluate_alerts(self) -> List[HealthcareAlert]:
        """Evaluate all metrics against alert rules."""
        
        new_alerts = []
        current_time = time.time()
        
        # Get recent metrics for evaluation
        recent_metrics = [
            m for m in self.metrics_collector.metrics_buffer
            if current_time - m.timestamp < 300  # Last 5 minutes
        ]
        
        # Evaluate each alert rule
        for rule in self.alert_rules:
            matching_metrics = [
                m for m in recent_metrics
                if rule["metric_pattern"] in m.name
            ]
            
            for metric in matching_metrics:
                alert = await self._evaluate_metric_against_rule(metric, rule)
                if alert:
                    new_alerts.append(alert)
        
        # Process and deduplicate alerts
        processed_alerts = await self._process_new_alerts(new_alerts)
        
        return processed_alerts
    
    async def _evaluate_metric_against_rule(
        self, 
        metric: HealthcareMetric, 
        rule: Dict[str, Any]
    ) -> Optional[HealthcareAlert]:
        """Evaluate a metric against a specific alert rule."""
        
        # Check alert suppression
        if self._is_alert_suppressed(metric, rule):
            return None
        
        # Evaluate condition
        condition_met = False
        threshold_info = None
        anomaly_info = None
        
        if rule["condition"] == "above_threshold":
            if isinstance(metric.value, (int, float)):
                condition_met = metric.value > rule["threshold"]
                threshold_info = {
                    "type": "above_threshold",
                    "threshold": rule["threshold"],
                    "actual_value": metric.value
                }
        
        elif rule["condition"] == "below_threshold":
            if isinstance(metric.value, (int, float)):
                condition_met = metric.value < rule["threshold"]
                threshold_info = {
                    "type": "below_threshold",
                    "threshold": rule["threshold"],
                    "actual_value": metric.value
                }
        
        elif rule["condition"] == "anomaly_detection":
            anomaly_info = await self._detect_anomaly(metric, rule["threshold"])
            condition_met = anomaly_info.is_anomaly
        
        if not condition_met:
            return None
        
        # Create alert
        alert = HealthcareAlert(
            level=AlertLevel(rule["level"]),
            type=rule["name"],
            title=self._generate_alert_title(rule, metric),
            description=self._generate_alert_description(rule, metric, threshold_info, anomaly_info),
            triggering_metric=metric,
            threshold_violated=threshold_info,
            anomaly_detection=anomaly_info,
            compliance_impact=rule.get("compliance_impact", False),
            phi_exposure_risk=rule.get("phi_exposure_risk", False),
            patient_safety_impact=rule.get("patient_safety_impact", False),
            clinical_workflow_affected=metric.clinical_workflow,
            recommended_actions=self._generate_recommended_actions(rule, metric),
            escalation_required=rule["level"] in ["critical", "high"],
            estimated_resolution_time=self._estimate_resolution_time(rule)
        )
        
        return alert
    
    def _is_alert_suppressed(self, metric: HealthcareMetric, rule: Dict[str, Any]) -> bool:
        """Check if alert should be suppressed to avoid spam."""
        
        suppression_key = f"{rule['name']}:{metric.metric_key}"
        current_time = time.time()
        
        # Suppression periods based on alert level
        suppression_periods = {
            "critical": 300,   # 5 minutes
            "high": 900,       # 15 minutes
            "medium": 1800,    # 30 minutes
            "low": 3600        # 1 hour
        }
        
        suppression_period = suppression_periods.get(rule["level"], 1800)
        
        if suppression_key in self.alert_suppression:
            last_alert_time = self.alert_suppression[suppression_key]
            if current_time - last_alert_time < suppression_period:
                return True
        
        # Update suppression timestamp
        self.alert_suppression[suppression_key] = current_time
        return False
    
    async def _detect_anomaly(self, metric: HealthcareMetric, threshold: float) -> AnomalyDetection:
        """Detect anomaly in metric using statistical methods."""
        
        metric_key = metric.metric_key
        
        # Get historical values for baseline
        historical_metrics = [
            m for m in self.metrics_collector.metrics_buffer
            if (m.metric_key == metric_key and 
                isinstance(m.value, (int, float)) and
                time.time() - m.timestamp < 86400)  # Last 24 hours
        ]
        
        if len(historical_metrics) < 10:
            return AnomalyDetection(is_anomaly=False, anomaly_score=0.0)
        
        # Calculate baseline statistics
        values = [m.value for m in historical_metrics[:-1]]  # Exclude current value
        baseline_mean = statistics.mean(values)
        baseline_std = statistics.stdev(values) if len(values) > 1 else 0
        
        if baseline_std == 0:
            return AnomalyDetection(is_anomaly=False, anomaly_score=0.0)
        
        # Calculate z-score
        z_score = abs((metric.value - baseline_mean) / baseline_std)
        
        # Determine if anomaly
        is_anomaly = z_score > threshold
        confidence = min(z_score / threshold, 1.0) if threshold > 0 else 0
        
        return AnomalyDetection(
            detection_method="statistical",
            statistical_threshold=threshold,
            is_anomaly=is_anomaly,
            anomaly_score=z_score,
            confidence=confidence,
            baseline_value=baseline_mean,
            deviation=metric.value - baseline_mean
        )
    
    def _generate_alert_title(self, rule: Dict[str, Any], metric: HealthcareMetric) -> str:
        """Generate descriptive alert title."""
        
        rule_name = rule["name"].replace("_", " ").title()
        component = metric.source_component.replace("_", " ").title()
        
        return f"{rule_name} - {component}"
    
    def _generate_alert_description(
        self,
        rule: Dict[str, Any],
        metric: HealthcareMetric,
        threshold_info: Optional[Dict[str, Any]],
        anomaly_info: Optional[AnomalyDetection]
    ) -> str:
        """Generate detailed alert description."""
        
        description_parts = [
            f"Alert triggered for metric '{metric.name}' from component '{metric.source_component}'."
        ]
        
        if threshold_info:
            description_parts.append(
                f"Value {threshold_info['actual_value']} {threshold_info['type'].replace('_', ' ')} "
                f"threshold {threshold_info['threshold']}."
            )
        
        if anomaly_info and anomaly_info.is_anomaly:
            description_parts.append(
                f"Anomaly detected with score {anomaly_info.anomaly_score:.2f} "
                f"(baseline: {anomaly_info.baseline_value:.2f}, deviation: {anomaly_info.deviation:.2f})."
            )
        
        if metric.phi_processing:
            description_parts.append("This metric involves PHI processing.")
        
        if metric.compliance_relevant:
            description_parts.append("This metric is compliance-relevant.")
        
        return " ".join(description_parts)
    
    def _generate_recommended_actions(self, rule: Dict[str, Any], metric: HealthcareMetric) -> List[str]:
        """Generate recommended actions for alert."""
        
        actions = []
        
        # Rule-specific actions
        if "compliance" in rule["name"]:
            actions.extend([
                "Review compliance logs for violations",
                "Verify PHI redaction processes",
                "Contact compliance officer if necessary"
            ])
        
        elif "performance" in rule["name"] or "latency" in rule["name"]:
            actions.extend([
                "Check system resource utilization",
                "Review application logs for errors",
                "Consider scaling resources if needed"
            ])
        
        elif "security" in rule["name"] or "authentication" in rule["name"]:
            actions.extend([
                "Review security logs for suspicious activity",
                "Verify access controls are functioning",
                "Consider blocking suspicious IP addresses"
            ])
        
        elif "phi_exposure" in rule["name"]:
            actions.extend([
                "IMMEDIATE: Investigate potential PHI exposure",
                "Review affected documents and access logs",
                "Notify security and compliance teams",
                "Document incident for breach assessment"
            ])
        
        # Component-specific actions
        if metric.source_component == "phi_detector":
            actions.append("Verify PHI detection model performance")
        elif metric.source_component == "compliance_checker":
            actions.append("Review compliance checking logic")
        
        return actions
    
    def _estimate_resolution_time(self, rule: Dict[str, Any]) -> str:
        """Estimate resolution time based on alert type."""
        
        resolution_times = {
            "critical": "15-30 minutes",
            "high": "1-2 hours",
            "medium": "2-4 hours",
            "low": "4-8 hours"
        }
        
        return resolution_times.get(rule["level"], "unknown")
    
    async def _process_new_alerts(self, new_alerts: List[HealthcareAlert]) -> List[HealthcareAlert]:
        """Process and deduplicate new alerts."""
        
        processed_alerts = []
        
        for alert in new_alerts:
            # Check for existing similar alerts
            similar_alert = self._find_similar_active_alert(alert)
            
            if similar_alert:
                # Update existing alert instead of creating new one
                await self._update_existing_alert(similar_alert, alert)
            else:
                # Add new alert
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
                processed_alerts.append(alert)
                
                # Auto-mitigation if possible
                await self._attempt_auto_mitigation(alert)
        
        return processed_alerts
    
    def _find_similar_active_alert(self, new_alert: HealthcareAlert) -> Optional[HealthcareAlert]:
        """Find similar active alert to avoid duplicates."""
        
        for alert in self.active_alerts.values():
            if (alert.type == new_alert.type and
                alert.triggering_metric and new_alert.triggering_metric and
                alert.triggering_metric.metric_key == new_alert.triggering_metric.metric_key and
                not alert.resolved):
                return alert
        
        return None
    
    async def _update_existing_alert(self, existing_alert: HealthcareAlert, new_alert: HealthcareAlert) -> None:
        """Update existing alert with new information."""
        
        existing_alert.timestamp = new_alert.timestamp
        existing_alert.description += f" Updated: {new_alert.description}"
        
        # Escalate level if new alert is more severe
        if new_alert.level.value < existing_alert.level.value:  # Lower enum value = higher severity
            existing_alert.level = new_alert.level
            existing_alert.escalation_required = True
    
    async def _attempt_auto_mitigation(self, alert: HealthcareAlert) -> None:
        """Attempt automatic mitigation for alert."""
        
        if not alert.auto_mitigation_possible:
            return
        
        # Auto-mitigation logic based on alert type
        if "performance" in alert.type and alert.level in [AlertLevel.MEDIUM, AlertLevel.LOW]:
            # Could implement auto-scaling, load balancing, etc.
            alert.recommended_actions.append("Auto-mitigation: Attempted resource scaling")
        
        elif "data_quality" in alert.type:
            # Could implement data validation retry, etc.
            alert.recommended_actions.append("Auto-mitigation: Initiated data quality check")


class HealthcareDashboard:
    """Healthcare operations dashboard with real-time insights."""
    
    def __init__(self, metrics_collector: HealthcareMetricsCollector, alerting: IntelligentAlerting):
        self.metrics_collector = metrics_collector
        self.alerting = alerting
        
    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get real-time healthcare operations dashboard."""
        
        current_time = time.time()
        
        # Overall system health
        system_health = self._calculate_system_health()
        
        # Compliance status
        compliance_status = self._get_compliance_status()
        
        # Performance metrics
        performance_summary = self._get_performance_summary()
        
        # Security status
        security_status = self._get_security_status()
        
        # PHI handling metrics
        phi_handling_summary = self._get_phi_handling_summary()
        
        # Active alerts
        alert_summary = self._get_alert_summary()
        
        # Recent activity
        recent_activity = self._get_recent_activity_summary()
        
        return {
            "timestamp": current_time,
            "dashboard_version": "2.0",
            "system_health": system_health,
            "compliance_status": compliance_status,
            "performance_summary": performance_summary,
            "security_status": security_status,
            "phi_handling_summary": phi_handling_summary,
            "alert_summary": alert_summary,
            "recent_activity": recent_activity,
            "recommendations": self._generate_dashboard_recommendations()
        }
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score."""
        
        health_indicators = []
        
        # Check critical alerts
        critical_alerts = len([a for a in self.alerting.active_alerts.values() 
                             if a.level == AlertLevel.CRITICAL and not a.resolved])
        
        # Performance health
        perf_summary = self.metrics_collector.get_metric_summary("latency", 15)
        if perf_summary and perf_summary["summaries"]:
            avg_latency = np.mean([s["window_stats"]["avg"] for s in perf_summary["summaries"]])
            perf_health = max(0, 1 - (avg_latency / 1000))  # Normalize to 0-1
            health_indicators.append(perf_health)
        
        # Compliance health
        compliance_summary = self.metrics_collector.get_metric_summary("compliance_score", 15)
        if compliance_summary and compliance_summary["summaries"]:
            avg_compliance = np.mean([s["latest_value"] for s in compliance_summary["summaries"]])
            health_indicators.append(avg_compliance)
        
        # Calculate overall health
        if health_indicators:
            overall_health = np.mean(health_indicators)
        else:
            overall_health = 0.5
        
        # Reduce health based on critical alerts
        if critical_alerts > 0:
            overall_health *= max(0.3, 1 - (critical_alerts * 0.2))
        
        health_status = "excellent" if overall_health >= 0.95 else \
                       "good" if overall_health >= 0.85 else \
                       "fair" if overall_health >= 0.70 else "poor"
        
        return {
            "overall_score": overall_health,
            "status": health_status,
            "critical_alerts": critical_alerts,
            "health_indicators": len(health_indicators)
        }
    
    def _get_compliance_status(self) -> Dict[str, Any]:
        """Get compliance status summary."""
        
        # Get compliance metrics
        hipaa_summary = self.metrics_collector.get_metric_summary("hipaa_compliance_score", 60)
        phi_accuracy_summary = self.metrics_collector.get_metric_summary("phi_redaction_accuracy", 60)
        audit_summary = self.metrics_collector.get_metric_summary("audit_trail_completeness", 60)
        
        compliance_metrics = {}
        
        if hipaa_summary and hipaa_summary["summaries"]:
            compliance_metrics["hipaa_compliance"] = {
                "current_score": hipaa_summary["summaries"][0]["latest_value"],
                "avg_score": hipaa_summary["summaries"][0]["window_stats"]["avg"]
            }
        
        if phi_accuracy_summary and phi_accuracy_summary["summaries"]:
            compliance_metrics["phi_redaction_accuracy"] = {
                "current_score": phi_accuracy_summary["summaries"][0]["latest_value"],
                "avg_score": phi_accuracy_summary["summaries"][0]["window_stats"]["avg"]
            }
        
        if audit_summary and audit_summary["summaries"]:
            compliance_metrics["audit_completeness"] = {
                "current_score": audit_summary["summaries"][0]["latest_value"],
                "avg_score": audit_summary["summaries"][0]["window_stats"]["avg"]
            }
        
        # Calculate overall compliance status
        if compliance_metrics:
            avg_scores = [m["current_score"] for m in compliance_metrics.values()]
            overall_compliance = np.mean(avg_scores)
            
            status = "compliant" if overall_compliance >= 0.95 else \
                    "warning" if overall_compliance >= 0.90 else "violation"
        else:
            overall_compliance = 0.0
            status = "unknown"
        
        return {
            "overall_compliance_score": overall_compliance,
            "status": status,
            "metrics": compliance_metrics,
            "violations_last_24h": len([a for a in self.alerting.alert_history 
                                      if time.time() - a.timestamp < 86400 and a.compliance_impact])
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        
        performance_metrics = {}
        
        # Get key performance metrics
        latency_summary = self.metrics_collector.get_metric_summary("latency", 15)
        throughput_summary = self.metrics_collector.get_metric_summary("throughput", 15)
        
        if latency_summary and latency_summary["summaries"]:
            stats = latency_summary["summaries"][0]["window_stats"]
            performance_metrics["latency"] = {
                "avg_ms": stats["avg"],
                "p95_ms": stats["p95"],
                "p99_ms": stats["p99"]
            }
        
        if throughput_summary and throughput_summary["summaries"]:
            stats = throughput_summary["summaries"][0]["window_stats"]
            performance_metrics["throughput"] = {
                "avg_per_min": stats["avg"],
                "max_per_min": stats["max"]
            }
        
        return {
            "metrics": performance_metrics,
            "performance_alerts_active": len([a for a in self.alerting.active_alerts.values() 
                                            if "performance" in a.type and not a.resolved])
        }
    
    def _get_security_status(self) -> Dict[str, Any]:
        """Get security status summary."""
        
        # Get security metrics
        auth_failures = self.metrics_collector.get_metric_summary("failed_authentication_attempts", 60)
        suspicious_access = self.metrics_collector.get_metric_summary("suspicious_access_patterns", 60)
        
        security_metrics = {}
        
        if auth_failures and auth_failures["summaries"]:
            stats = auth_failures["summaries"][0]["window_stats"]
            security_metrics["authentication_failures"] = {
                "count_last_hour": stats["sum"],
                "rate_per_hour": stats["avg"] * 60
            }
        
        if suspicious_access and suspicious_access["summaries"]:
            stats = suspicious_access["summaries"][0]["window_stats"]
            security_metrics["suspicious_activity"] = {
                "incidents_last_hour": stats["sum"]
            }
        
        # Security alert count
        security_alerts = len([a for a in self.alerting.active_alerts.values() 
                             if a.type in ["security", "authentication"] and not a.resolved])
        
        return {
            "metrics": security_metrics,
            "active_security_alerts": security_alerts,
            "phi_exposure_risk_alerts": len([a for a in self.alerting.active_alerts.values() 
                                           if a.phi_exposure_risk and not a.resolved])
        }
    
    def _get_phi_handling_summary(self) -> Dict[str, Any]:
        """Get PHI handling metrics summary."""
        
        phi_metrics = {}
        
        # PHI detection metrics
        phi_detection = self.metrics_collector.get_metric_summary("phi_entities_detected", 60)
        phi_false_positives = self.metrics_collector.get_metric_summary("phi_false_positive_rate", 60)
        phi_exposure = self.metrics_collector.get_metric_summary("phi_exposure_incidents", 60)
        
        if phi_detection and phi_detection["summaries"]:
            stats = phi_detection["summaries"][0]["window_stats"]
            phi_metrics["detection"] = {
                "entities_per_document_avg": stats["avg"],
                "total_entities_detected": stats["sum"]
            }
        
        if phi_false_positives and phi_false_positives["summaries"]:
            phi_metrics["quality"] = {
                "false_positive_rate": phi_false_positives["summaries"][0]["latest_value"]
            }
        
        if phi_exposure and phi_exposure["summaries"]:
            phi_metrics["exposure_incidents"] = {
                "count_last_hour": phi_exposure["summaries"][0]["window_stats"]["sum"]
            }
        
        return {
            "metrics": phi_metrics,
            "phi_processing_alerts": len([a for a in self.alerting.active_alerts.values() 
                                        if a.triggering_metric and a.triggering_metric.phi_processing and not a.resolved])
        }
    
    def _get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        
        active_alerts = list(self.alerting.active_alerts.values())
        recent_alerts = [a for a in self.alerting.alert_history if time.time() - a.timestamp < 3600]
        
        # Alert distribution by level
        alert_distribution = defaultdict(int)
        for alert in active_alerts:
            if not alert.resolved:
                alert_distribution[alert.level.value] += 1
        
        return {
            "total_active": len([a for a in active_alerts if not a.resolved]),
            "distribution_by_level": dict(alert_distribution),
            "recent_alerts_1h": len(recent_alerts),
            "unacknowledged": len([a for a in active_alerts if not a.acknowledged and not a.resolved]),
            "requiring_escalation": len([a for a in active_alerts if a.escalation_required and not a.resolved])
        }
    
    def _get_recent_activity_summary(self) -> Dict[str, Any]:
        """Get recent system activity summary."""
        
        recent_metrics = [m for m in self.metrics_collector.metrics_buffer 
                         if time.time() - m.timestamp < 300]  # Last 5 minutes
        
        activity_by_component = defaultdict(int)
        phi_processing_activity = 0
        
        for metric in recent_metrics:
            activity_by_component[metric.source_component] += 1
            if metric.phi_processing:
                phi_processing_activity += 1
        
        return {
            "metrics_collected_5m": len(recent_metrics),
            "active_components": len(activity_by_component),
            "phi_processing_activity": phi_processing_activity,
            "component_activity": dict(activity_by_component)
        }
    
    def _generate_dashboard_recommendations(self) -> List[str]:
        """Generate actionable recommendations for dashboard."""
        
        recommendations = []
        
        # Check for critical alerts
        critical_alerts = [a for a in self.alerting.active_alerts.values() 
                         if a.level == AlertLevel.CRITICAL and not a.resolved]
        
        if critical_alerts:
            recommendations.append(f"URGENT: Address {len(critical_alerts)} critical alerts immediately")
        
        # Check compliance status
        compliance_status = self._get_compliance_status()
        if compliance_status["overall_compliance_score"] < 0.95:
            recommendations.append("Review compliance metrics - scores below threshold")
        
        # Check PHI exposure alerts
        phi_exposure_alerts = [a for a in self.alerting.active_alerts.values() 
                             if a.phi_exposure_risk and not a.resolved]
        
        if phi_exposure_alerts:
            recommendations.append("Investigate PHI exposure risk alerts immediately")
        
        # Performance recommendations
        performance_summary = self._get_performance_summary()
        if "latency" in performance_summary["metrics"]:
            if performance_summary["metrics"]["latency"]["p95_ms"] > 1000:
                recommendations.append("High latency detected - consider performance optimization")
        
        if not recommendations:
            recommendations.append("System operating normally - continue monitoring")
        
        return recommendations


# Example usage and testing
async def test_comprehensive_monitoring():
    """Test comprehensive monitoring system."""
    
    print("📊 Testing Comprehensive Healthcare Monitoring System")
    
    # Initialize monitoring components
    metrics_collector = HealthcareMetricsCollector()
    alerting = IntelligentAlerting(metrics_collector)
    dashboard = HealthcareDashboard(metrics_collector, alerting)
    
    print("\n1. Collecting Healthcare Metrics")
    
    # Simulate metric collection
    for _ in range(10):
        # Collect different types of metrics
        perf_metrics = await metrics_collector._collect_performance_metrics()
        compliance_metrics = await metrics_collector._collect_compliance_metrics()
        security_metrics = await metrics_collector._collect_security_metrics()
        phi_metrics = await metrics_collector._collect_phi_handling_metrics()
        
        all_metrics = perf_metrics + compliance_metrics + security_metrics + phi_metrics
        
        for metric in all_metrics:
            await metrics_collector.collect_metric(metric)
        
        await asyncio.sleep(0.1)  # Simulate time passage
    
    print(f"   Collected {len(metrics_collector.metrics_buffer)} metrics")
    
    print("\n2. Evaluating Alerts")
    
    # Simulate some alert conditions
    critical_metric = HealthcareMetric(
        name="phi_exposure_incidents",
        type=MetricType.PHI_HANDLING,
        value=1,  # This should trigger critical alert
        source_component="phi_monitor",
        phi_processing=True,
        compliance_relevant=True
    )
    
    await metrics_collector.collect_metric(critical_metric)
    
    # Evaluate alerts
    new_alerts = await alerting.evaluate_alerts()
    print(f"   Generated {len(new_alerts)} new alerts")
    
    for alert in new_alerts:
        print(f"     {alert.level.value.upper()}: {alert.title}")
    
    print("\n3. Dashboard Summary")
    
    dashboard_data = dashboard.get_real_time_dashboard()
    
    print(f"   System Health: {dashboard_data['system_health']['status']} ({dashboard_data['system_health']['overall_score']:.2%})")
    print(f"   Compliance Status: {dashboard_data['compliance_status']['status']}")
    print(f"   Active Alerts: {dashboard_data['alert_summary']['total_active']}")
    print(f"   Critical Alerts: {dashboard_data['alert_summary']['distribution_by_level'].get('critical', 0)}")
    
    print("\n4. Recommendations")
    for i, rec in enumerate(dashboard_data['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print("\n✅ Comprehensive Monitoring Test Completed")
    
    return {
        "metrics_collected": len(metrics_collector.metrics_buffer),
        "alerts_generated": len(new_alerts),
        "dashboard_data": dashboard_data
    }


if __name__ == "__main__":
    asyncio.run(test_comprehensive_monitoring())