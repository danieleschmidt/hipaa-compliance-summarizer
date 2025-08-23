"""Intelligent auto-scaling system for HIPAA compliance infrastructure."""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"
    MIGRATE = "migrate"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    WORKERS = "workers"


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling policies."""
    resource_type: ResourceType
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    cooldown_seconds: int = 300
    min_instances: int = 1
    max_instances: int = 10
    scaling_factor: float = 1.5
    enabled: bool = True


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    network_io_percent: float
    active_workers: int
    queue_size: int
    response_time_ms: float
    error_rate: float


@dataclass
class ScalingEvent:
    """Record of a scaling action."""
    timestamp: float
    action: ScalingAction
    resource_type: ResourceType
    reason: str
    previous_capacity: int
    new_capacity: int
    success: bool
    duration_seconds: float = 0.0


class PredictiveScaler:
    """ML-based predictive scaling using time series analysis."""

    def __init__(self):
        self.metrics_history = []
        self.prediction_models = {}
        self.prediction_window = 3600  # 1 hour
        self.min_data_points = 50

    def record_metrics(self, metrics: ResourceMetrics):
        """Record resource metrics for prediction."""
        self.metrics_history.append(metrics)

        # Keep only recent data (last 24 hours)
        cutoff = time.time() - 86400
        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp > cutoff
        ]

    def predict_resource_demand(self, resource_type: ResourceType,
                              lookahead_seconds: int = 1800) -> Dict[str, float]:
        """Predict resource demand using simple trend analysis."""
        if len(self.metrics_history) < self.min_data_points:
            return {"predicted_utilization": 0.5, "confidence": 0.0}

        # Extract relevant metric based on resource type
        if resource_type == ResourceType.CPU:
            values = [m.cpu_percent for m in self.metrics_history[-self.min_data_points:]]
        elif resource_type == ResourceType.MEMORY:
            values = [m.memory_percent for m in self.metrics_history[-self.min_data_points:]]
        elif resource_type == ResourceType.WORKERS:
            values = [m.active_workers for m in self.metrics_history[-self.min_data_points:]]
        else:
            values = [m.cpu_percent for m in self.metrics_history[-self.min_data_points:]]

        # Simple trend-based prediction
        values = np.array(values)

        # Calculate trend (linear regression slope)
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]  # Slope of linear fit

        # Predict future value
        future_periods = lookahead_seconds / 300  # Assume metrics every 5 minutes
        current_value = values[-1] if len(values) > 0 else 50.0
        predicted_value = current_value + (trend * future_periods)

        # Calculate confidence based on data variance
        variance = np.var(values) if len(values) > 1 else 100.0
        confidence = max(0.1, min(0.9, 1.0 / (1.0 + variance / 100.0)))

        return {
            "predicted_utilization": max(0.0, min(100.0, predicted_value)),
            "confidence": confidence,
            "trend": trend,
            "current_value": current_value
        }


class IntelligentAutoScaler:
    """Intelligent auto-scaling system with ML-driven decisions."""

    def __init__(self):
        self.scaling_policies: Dict[ResourceType, ScalingPolicy] = {}
        self.current_capacity: Dict[ResourceType, int] = {}
        self.scaling_history: List[ScalingEvent] = []
        self.last_scaling_action: Dict[ResourceType, float] = {}
        self.predictor = PredictiveScaler()
        self.lock = threading.Lock()
        self.monitoring_active = False
        self.scaling_callbacks: Dict[ResourceType, Callable] = {}

        # Initialize default policies
        self._setup_default_policies()

    def _setup_default_policies(self):
        """Setup default scaling policies."""
        self.scaling_policies[ResourceType.CPU] = ScalingPolicy(
            resource_type=ResourceType.CPU,
            scale_up_threshold=0.75,
            scale_down_threshold=0.25,
            cooldown_seconds=300,
            min_instances=1,
            max_instances=8
        )

        self.scaling_policies[ResourceType.MEMORY] = ScalingPolicy(
            resource_type=ResourceType.MEMORY,
            scale_up_threshold=0.80,
            scale_down_threshold=0.30,
            cooldown_seconds=300,
            min_instances=1,
            max_instances=6
        )

        self.scaling_policies[ResourceType.WORKERS] = ScalingPolicy(
            resource_type=ResourceType.WORKERS,
            scale_up_threshold=0.85,
            scale_down_threshold=0.20,
            cooldown_seconds=180,
            min_instances=2,
            max_instances=20,
            scaling_factor=2.0
        )

        # Initialize current capacity
        self.current_capacity = {
            ResourceType.CPU: 2,
            ResourceType.MEMORY: 4,  # GB
            ResourceType.WORKERS: 4
        }

    def register_scaling_callback(self, resource_type: ResourceType,
                                callback: Callable[[ScalingAction, int], bool]):
        """Register callback for executing scaling actions."""
        self.scaling_callbacks[resource_type] = callback

    def update_policy(self, resource_type: ResourceType, policy: ScalingPolicy):
        """Update scaling policy for a resource type."""
        with self.lock:
            self.scaling_policies[resource_type] = policy

    def process_metrics(self, metrics: ResourceMetrics) -> List[ScalingEvent]:
        """Process metrics and make scaling decisions."""
        self.predictor.record_metrics(metrics)

        scaling_events = []

        with self.lock:
            for resource_type, policy in self.scaling_policies.items():
                if not policy.enabled:
                    continue

                # Check cooldown period
                last_action_time = self.last_scaling_action.get(resource_type, 0)
                if time.time() - last_action_time < policy.cooldown_seconds:
                    continue

                # Get current utilization based on resource type
                current_utilization = self._get_utilization_for_resource(
                    metrics, resource_type
                )

                # Get predictive insights
                prediction = self.predictor.predict_resource_demand(
                    resource_type, lookahead_seconds=1800
                )

                # Make scaling decision
                scaling_action = self._make_scaling_decision(
                    resource_type, policy, current_utilization, prediction
                )

                if scaling_action != ScalingAction.NO_ACTION:
                    event = self._execute_scaling_action(
                        resource_type, scaling_action, policy,
                        current_utilization, prediction
                    )
                    if event:
                        scaling_events.append(event)
                        self.scaling_history.append(event)
                        self.last_scaling_action[resource_type] = time.time()

        return scaling_events

    def _get_utilization_for_resource(self, metrics: ResourceMetrics,
                                    resource_type: ResourceType) -> float:
        """Get utilization percentage for specific resource type."""
        if resource_type == ResourceType.CPU:
            return metrics.cpu_percent / 100.0
        elif resource_type == ResourceType.MEMORY:
            return metrics.memory_percent / 100.0
        elif resource_type == ResourceType.WORKERS:
            # Worker utilization based on queue size and active workers
            if metrics.active_workers == 0:
                return 0.0
            utilization = metrics.queue_size / max(1, metrics.active_workers)
            return min(1.0, utilization / 10.0)  # Normalize to 0-1
        else:
            return metrics.cpu_percent / 100.0  # Default to CPU

    def _make_scaling_decision(self, resource_type: ResourceType,
                             policy: ScalingPolicy,
                             current_utilization: float,
                             prediction: Dict[str, float]) -> ScalingAction:
        """Make intelligent scaling decision."""
        predicted_utilization = prediction["predicted_utilization"] / 100.0
        confidence = prediction["confidence"]
        trend = prediction.get("trend", 0)

        current_capacity = self.current_capacity.get(resource_type, 1)

        # Determine if scaling is needed based on current + predicted demand
        effective_utilization = (
            current_utilization * 0.7 + predicted_utilization * 0.3
        )

        # Include trend information
        if trend > 0.5 and confidence > 0.6:  # Strong upward trend
            effective_utilization += 0.1
        elif trend < -0.5 and confidence > 0.6:  # Strong downward trend
            effective_utilization -= 0.1

        # Scale up conditions
        if (effective_utilization > policy.scale_up_threshold and
            current_capacity < policy.max_instances):
            return ScalingAction.SCALE_UP

        # Scale down conditions
        if (effective_utilization < policy.scale_down_threshold and
            current_capacity > policy.min_instances):
            return ScalingAction.SCALE_DOWN

        return ScalingAction.NO_ACTION

    def _execute_scaling_action(self, resource_type: ResourceType,
                              action: ScalingAction,
                              policy: ScalingPolicy,
                              current_utilization: float,
                              prediction: Dict[str, float]) -> Optional[ScalingEvent]:
        """Execute scaling action."""
        start_time = time.time()
        current_capacity = self.current_capacity.get(resource_type, 1)

        if action == ScalingAction.SCALE_UP:
            new_capacity = min(
                int(current_capacity * policy.scaling_factor),
                policy.max_instances
            )
        elif action == ScalingAction.SCALE_DOWN:
            new_capacity = max(
                int(current_capacity / policy.scaling_factor),
                policy.min_instances
            )
        else:
            return None

        if new_capacity == current_capacity:
            return None

        # Execute via callback if available
        success = True
        if resource_type in self.scaling_callbacks:
            try:
                success = self.scaling_callbacks[resource_type](action, new_capacity)
            except Exception as e:
                logger.error(f"Scaling callback failed: {e}")
                success = False

        if success:
            self.current_capacity[resource_type] = new_capacity

        reason = f"Utilization: {current_utilization:.2f}, Predicted: {prediction['predicted_utilization']:.1f}%"

        return ScalingEvent(
            timestamp=start_time,
            action=action,
            resource_type=resource_type,
            reason=reason,
            previous_capacity=current_capacity,
            new_capacity=new_capacity if success else current_capacity,
            success=success,
            duration_seconds=time.time() - start_time
        )

    def get_scaling_insights(self) -> Dict[str, Any]:
        """Get insights about scaling behavior."""
        with self.lock:
            recent_events = [
                e for e in self.scaling_history
                if time.time() - e.timestamp < 3600  # Last hour
            ]

            # Calculate scaling statistics
            total_scale_ups = len([e for e in recent_events if e.action == ScalingAction.SCALE_UP])
            total_scale_downs = len([e for e in recent_events if e.action == ScalingAction.SCALE_DOWN])
            successful_actions = len([e for e in recent_events if e.success])

            # Resource utilization trends
            resource_trends = {}
            for resource_type in self.scaling_policies.keys():
                prediction = self.predictor.predict_resource_demand(resource_type)
                resource_trends[resource_type.value] = {
                    "current_capacity": self.current_capacity.get(resource_type, 0),
                    "predicted_utilization": prediction.get("predicted_utilization", 0),
                    "trend": prediction.get("trend", 0),
                    "confidence": prediction.get("confidence", 0)
                }

            return {
                "current_capacity": {rt.value: cap for rt, cap in self.current_capacity.items()},
                "recent_scaling_events": len(recent_events),
                "scale_ups": total_scale_ups,
                "scale_downs": total_scale_downs,
                "success_rate": successful_actions / max(1, len(recent_events)),
                "resource_trends": resource_trends,
                "policies": {
                    rt.value: {
                        "scale_up_threshold": policy.scale_up_threshold,
                        "scale_down_threshold": policy.scale_down_threshold,
                        "min_instances": policy.min_instances,
                        "max_instances": policy.max_instances,
                        "enabled": policy.enabled
                    }
                    for rt, policy in self.scaling_policies.items()
                },
                "prediction_data_points": len(self.predictor.metrics_history)
            }

    def start_monitoring(self, metrics_source: Callable[[], ResourceMetrics],
                        interval_seconds: int = 60):
        """Start monitoring and auto-scaling."""
        if self.monitoring_active:
            return

        self.monitoring_active = True

        def monitoring_loop():
            while self.monitoring_active:
                try:
                    metrics = metrics_source()
                    events = self.process_metrics(metrics)

                    if events:
                        logger.info(f"Executed {len(events)} scaling actions")
                        for event in events:
                            logger.info(
                                f"Scaled {event.resource_type.value} "
                                f"{event.action.value}: "
                                f"{event.previous_capacity} -> {event.new_capacity}"
                            )

                    time.sleep(interval_seconds)

                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                    time.sleep(interval_seconds)

        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        logger.info("Auto-scaling monitoring started")

    def stop_monitoring(self):
        """Stop monitoring and auto-scaling."""
        self.monitoring_active = False
        logger.info("Auto-scaling monitoring stopped")


# Global auto-scaler instance
auto_scaler = IntelligentAutoScaler()


def configure_scaling_policy(resource_type: str, **kwargs) -> bool:
    """Configure scaling policy for a resource type."""
    try:
        rt = ResourceType(resource_type)
        current_policy = auto_scaler.scaling_policies.get(rt, ScalingPolicy(rt))

        # Update policy with provided kwargs
        for key, value in kwargs.items():
            if hasattr(current_policy, key):
                setattr(current_policy, key, value)

        auto_scaler.update_policy(rt, current_policy)
        return True

    except (ValueError, AttributeError) as e:
        logger.error(f"Failed to configure scaling policy: {e}")
        return False


def get_current_scaling_status() -> Dict[str, Any]:
    """Get current auto-scaling status and insights."""
    return auto_scaler.get_scaling_insights()


# Healthcare-specific scaling configurations
def setup_hipaa_scaling_policies():
    """Setup HIPAA-compliant scaling policies with security considerations."""

    # Conservative scaling for PHI processing workloads
    phi_cpu_policy = ScalingPolicy(
        resource_type=ResourceType.CPU,
        scale_up_threshold=0.60,  # Lower threshold for PHI processing
        scale_down_threshold=0.20,
        cooldown_seconds=600,  # Longer cooldown for stability
        min_instances=2,  # Always maintain minimum for availability
        max_instances=6,  # Controlled maximum for security
        scaling_factor=1.3  # Conservative scaling factor
    )

    phi_memory_policy = ScalingPolicy(
        resource_type=ResourceType.MEMORY,
        scale_up_threshold=0.70,
        scale_down_threshold=0.25,
        cooldown_seconds=600,
        min_instances=4,  # Minimum memory allocation
        max_instances=16,
        scaling_factor=1.5
    )

    phi_workers_policy = ScalingPolicy(
        resource_type=ResourceType.WORKERS,
        scale_up_threshold=0.70,
        scale_down_threshold=0.15,
        cooldown_seconds=300,
        min_instances=3,  # Always maintain minimum worker pool
        max_instances=15,  # Controlled for resource management
        scaling_factor=1.8
    )

    auto_scaler.update_policy(ResourceType.CPU, phi_cpu_policy)
    auto_scaler.update_policy(ResourceType.MEMORY, phi_memory_policy)
    auto_scaler.update_policy(ResourceType.WORKERS, phi_workers_policy)

    logger.info("HIPAA-compliant scaling policies configured")


# Initialize HIPAA policies on import
setup_hipaa_scaling_policies()
