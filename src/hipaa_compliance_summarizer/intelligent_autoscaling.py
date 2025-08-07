"""Intelligent auto-scaling system for HIPAA compliance processing.

This module provides advanced auto-scaling capabilities including:
- Predictive scaling based on historical patterns
- Machine learning-driven resource optimization
- Cost-aware scaling decisions
- Multi-dimensional scaling metrics
- Integration with cloud platforms
- Proactive capacity planning
"""

import asyncio
import logging
import time
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from threading import Lock, Thread
from contextlib import contextmanager
from enum import Enum
import statistics


logger = logging.getLogger(__name__)


class ScalingDecision(Enum):
    """Auto-scaling decisions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ScalingTrigger(Enum):
    """Types of scaling triggers."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    PREDICTIVE = "predictive"
    COST_OPTIMIZATION = "cost_optimization"


@dataclass
class ScalingMetric:
    """Individual scaling metric measurement."""
    
    metric_type: ScalingTrigger
    timestamp: datetime
    value: float
    threshold_low: float
    threshold_high: float
    weight: float = 1.0
    
    def is_above_threshold(self) -> bool:
        """Check if metric is above high threshold."""
        return self.value > self.threshold_high
    
    def is_below_threshold(self) -> bool:
        """Check if metric is below low threshold."""
        return self.value < self.threshold_low
    
    def get_scaling_pressure(self) -> float:
        """Calculate scaling pressure (-1 to 1, where 1 means scale up urgently)."""
        if self.value > self.threshold_high:
            # Scale up pressure
            max_pressure = min(self.threshold_high * 2, 100.0)  # Cap at 2x threshold or 100%
            pressure = (self.value - self.threshold_high) / (max_pressure - self.threshold_high)
            return min(pressure, 1.0) * self.weight
        elif self.value < self.threshold_low:
            # Scale down pressure
            pressure = (self.threshold_low - self.value) / self.threshold_low
            return -min(pressure, 1.0) * self.weight
        else:
            # Within acceptable range
            return 0.0


@dataclass
class ScalingEvent:
    """Record of a scaling action."""
    
    timestamp: datetime
    decision: ScalingDecision
    trigger: ScalingTrigger
    instances_before: int
    instances_after: int
    metrics_snapshot: Dict[str, float]
    cost_impact: float = 0.0
    effectiveness_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/analysis."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "decision": self.decision.value,
            "trigger": self.trigger.value,
            "instances_before": self.instances_before,
            "instances_after": self.instances_after,
            "metrics_snapshot": self.metrics_snapshot,
            "cost_impact": self.cost_impact,
            "effectiveness_score": self.effectiveness_score
        }


@dataclass
class ResourcePrediction:
    """Prediction for future resource requirements."""
    
    prediction_time: datetime
    for_time: datetime
    predicted_cpu_utilization: float
    predicted_memory_utilization: float
    predicted_queue_length: int
    predicted_throughput: float
    confidence: float
    recommended_instances: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prediction_time": self.prediction_time.isoformat(),
            "for_time": self.for_time.isoformat(),
            "predicted_cpu_utilization": self.predicted_cpu_utilization,
            "predicted_memory_utilization": self.predicted_memory_utilization,
            "predicted_queue_length": self.predicted_queue_length,
            "predicted_throughput": self.predicted_throughput,
            "confidence": self.confidence,
            "recommended_instances": self.recommended_instances
        }


class PatternAnalyzer:
    """Analyzes historical patterns for predictive scaling."""
    
    def __init__(self, history_hours: int = 168):  # 1 week
        """Initialize pattern analyzer."""
        self.history_hours = history_hours
        self._metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    def add_metric_point(self, metric_type: str, timestamp: datetime, value: float) -> None:
        """Add a metric data point."""
        self._metric_history[metric_type].append({
            "timestamp": timestamp,
            "value": value,
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "day_of_month": timestamp.day
        })
    
    def detect_patterns(self, metric_type: str) -> Dict[str, Any]:
        """Detect patterns in metric data."""
        history = list(self._metric_history[metric_type])
        if len(history) < 24:  # Need at least 24 hours of data
            return {"patterns_detected": False, "reason": "insufficient_data"}
        
        # Hourly patterns
        hourly_averages = defaultdict(list)
        for point in history:
            hourly_averages[point["hour_of_day"]].append(point["value"])
        
        hourly_pattern = {
            hour: statistics.mean(values) if values else 0.0
            for hour, values in hourly_averages.items()
        }
        
        # Daily patterns  
        daily_averages = defaultdict(list)
        for point in history:
            daily_averages[point["day_of_week"]].append(point["value"])
        
        daily_pattern = {
            day: statistics.mean(values) if values else 0.0
            for day, values in daily_averages.items()
        }
        
        # Trend analysis
        recent_values = [p["value"] for p in history[-24:]]  # Last 24 hours
        older_values = [p["value"] for p in history[-48:-24]]  # Previous 24 hours
        
        trend = "stable"
        if len(recent_values) > 0 and len(older_values) > 0:
            recent_avg = statistics.mean(recent_values)
            older_avg = statistics.mean(older_values)
            change_percent = (recent_avg - older_avg) / older_avg * 100 if older_avg > 0 else 0
            
            if change_percent > 20:
                trend = "increasing"
            elif change_percent < -20:
                trend = "decreasing"
        
        return {
            "patterns_detected": True,
            "hourly_pattern": hourly_pattern,
            "daily_pattern": daily_pattern,
            "trend": trend,
            "peak_hour": max(hourly_pattern.items(), key=lambda x: x[1])[0] if hourly_pattern else None,
            "low_hour": min(hourly_pattern.items(), key=lambda x: x[1])[0] if hourly_pattern else None,
            "data_points": len(history)
        }
    
    def predict_future_value(self, metric_type: str, future_time: datetime) -> Tuple[float, float]:
        """Predict future metric value with confidence score."""
        patterns = self.detect_patterns(metric_type)
        
        if not patterns["patterns_detected"]:
            # No patterns, return current average with low confidence
            history = list(self._metric_history[metric_type])
            if history:
                recent_avg = statistics.mean([p["value"] for p in history[-12:]])
                return recent_avg, 0.3
            else:
                return 0.0, 0.0
        
        # Use hourly pattern for prediction
        hour = future_time.hour
        base_prediction = patterns["hourly_pattern"].get(hour, 0.0)
        
        # Apply trend
        trend_multiplier = 1.0
        if patterns["trend"] == "increasing":
            trend_multiplier = 1.1
        elif patterns["trend"] == "decreasing":
            trend_multiplier = 0.9
        
        prediction = base_prediction * trend_multiplier
        
        # Confidence based on data consistency
        hourly_values = [p["value"] for p in self._metric_history[metric_type] if p["hour_of_day"] == hour]
        if len(hourly_values) > 3:
            variance = statistics.variance(hourly_values)
            mean_val = statistics.mean(hourly_values)
            cv = (math.sqrt(variance) / mean_val) if mean_val > 0 else 1.0  # Coefficient of variation
            confidence = max(0.1, min(0.9, 1.0 - cv))  # Lower variance = higher confidence
        else:
            confidence = 0.3
        
        return max(0.0, prediction), confidence


class CostOptimizer:
    """Optimizes scaling decisions based on cost considerations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cost optimizer."""
        self.config = config or {}
        
        # Cost parameters (example values - should be configured)
        self.cost_per_instance_hour = self.config.get('cost_per_instance_hour', 0.10)
        self.sla_violation_cost = self.config.get('sla_violation_cost', 100.0)
        self.scaling_overhead_minutes = self.config.get('scaling_overhead_minutes', 3)
        
    def calculate_scaling_cost(self, instances_before: int, instances_after: int, 
                             duration_hours: float) -> float:
        """Calculate the cost impact of a scaling decision."""
        instance_diff = instances_after - instances_before
        
        # Direct compute cost
        compute_cost = instance_diff * self.cost_per_instance_hour * duration_hours
        
        # Scaling overhead (time when resources are not fully utilized)
        overhead_cost = abs(instance_diff) * (self.scaling_overhead_minutes / 60.0) * self.cost_per_instance_hour
        
        return compute_cost + overhead_cost
    
    def estimate_sla_violation_risk(self, current_metrics: Dict[str, float], 
                                   predicted_load: float) -> float:
        """Estimate risk of SLA violations if not scaling."""
        cpu_util = current_metrics.get('cpu_utilization', 0.0)
        queue_length = current_metrics.get('queue_length', 0.0)
        
        # Simple risk calculation (would be more sophisticated in practice)
        risk_factors = []
        
        if cpu_util > 80:
            risk_factors.append((cpu_util - 80) / 20)  # 0-1 scale above 80%
        
        if queue_length > 10:
            risk_factors.append(min(queue_length / 50, 1.0))  # 0-1 scale up to 50 items
        
        if predicted_load > 1.2:  # 20% increase predicted
            risk_factors.append((predicted_load - 1.2) / 0.8)
        
        return min(statistics.mean(risk_factors) if risk_factors else 0.0, 1.0)
    
    def should_scale_for_cost(self, scaling_cost: float, sla_risk: float) -> bool:
        """Determine if scaling is cost-effective."""
        expected_sla_cost = sla_risk * self.sla_violation_cost
        return scaling_cost < expected_sla_cost


class IntelligentAutoScaler:
    """Intelligent auto-scaling system with predictive capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize intelligent auto-scaler."""
        self.config = config or {}
        self._lock = Lock()
        
        # Scaling configuration
        self.min_instances = self.config.get('min_instances', 1)
        self.max_instances = self.config.get('max_instances', 20)
        self.current_instances = self.min_instances
        
        # Scaling behavior
        self.cooldown_period = self.config.get('cooldown_period_minutes', 5)
        self.scale_up_threshold = self.config.get('scale_up_threshold', 0.7)
        self.scale_down_threshold = self.config.get('scale_down_threshold', 0.3)
        
        # Components
        self.pattern_analyzer = PatternAnalyzer()
        self.cost_optimizer = CostOptimizer(self.config.get('cost_optimization', {}))
        
        # History tracking
        self._scaling_events: deque = deque(maxlen=1000)
        self._current_metrics: Dict[str, ScalingMetric] = {}
        self._predictions: deque = deque(maxlen=100)
        
        # Background processing
        self._autoscaler_active = True
        self._last_scaling_action = datetime.now() - timedelta(minutes=10)  # Allow immediate first action
        
        self._start_autoscaler_thread()
        
        logger.info("Intelligent auto-scaler initialized")
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update current metrics for scaling decisions."""
        timestamp = datetime.now()
        
        with self._lock:
            # Update scaling metrics
            self._current_metrics = {
                ScalingTrigger.CPU_UTILIZATION.value: ScalingMetric(
                    ScalingTrigger.CPU_UTILIZATION, timestamp,
                    metrics.get('cpu_utilization', 0.0),
                    threshold_low=30.0, threshold_high=70.0, weight=1.0
                ),
                ScalingTrigger.MEMORY_UTILIZATION.value: ScalingMetric(
                    ScalingTrigger.MEMORY_UTILIZATION, timestamp,
                    metrics.get('memory_utilization', 0.0),
                    threshold_low=40.0, threshold_high=80.0, weight=0.8
                ),
                ScalingTrigger.QUEUE_LENGTH.value: ScalingMetric(
                    ScalingTrigger.QUEUE_LENGTH, timestamp,
                    metrics.get('queue_length', 0.0),
                    threshold_low=5.0, threshold_high=20.0, weight=1.2
                ),
                ScalingTrigger.RESPONSE_TIME.value: ScalingMetric(
                    ScalingTrigger.RESPONSE_TIME, timestamp,
                    metrics.get('response_time_ms', 0.0),
                    threshold_low=500.0, threshold_high=2000.0, weight=1.1
                )
            }
        
        # Add to pattern analyzer
        for metric_name, value in metrics.items():
            self.pattern_analyzer.add_metric_point(metric_name, timestamp, value)
    
    def _start_autoscaler_thread(self) -> None:
        """Start background auto-scaling thread."""
        
        def autoscaler_loop():
            while self._autoscaler_active:
                try:
                    # Check if cooldown period has passed
                    if datetime.now() - self._last_scaling_action < timedelta(minutes=self.cooldown_period):
                        time.sleep(30)  # Check again in 30 seconds
                        continue
                    
                    # Make scaling decision
                    decision, trigger, recommended_instances = self._make_scaling_decision()
                    
                    if decision != ScalingDecision.MAINTAIN:
                        self._execute_scaling_decision(decision, trigger, recommended_instances)
                    
                    # Generate predictions
                    self._generate_predictions()
                    
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Auto-scaler loop error: {e}")
                    time.sleep(120)
        
        self._autoscaler_thread = Thread(target=autoscaler_loop, daemon=True)
        self._autoscaler_thread.start()
    
    def _make_scaling_decision(self) -> Tuple[ScalingDecision, ScalingTrigger, int]:
        """Make intelligent scaling decision based on multiple factors."""
        with self._lock:
            current_metrics = dict(self._current_metrics)
        
        if not current_metrics:
            return ScalingDecision.MAINTAIN, ScalingTrigger.CPU_UTILIZATION, self.current_instances
        
        # Calculate overall scaling pressure
        total_pressure = 0.0
        primary_trigger = ScalingTrigger.CPU_UTILIZATION
        max_pressure = 0.0
        
        for metric_name, metric in current_metrics.items():
            pressure = metric.get_scaling_pressure()
            total_pressure += pressure
            
            if abs(pressure) > abs(max_pressure):
                max_pressure = pressure
                primary_trigger = metric.metric_type
        
        # Predictive scaling check
        prediction_pressure = self._get_predictive_pressure()
        if abs(prediction_pressure) > abs(max_pressure):
            max_pressure = prediction_pressure
            primary_trigger = ScalingTrigger.PREDICTIVE
        
        # Cost-aware decision
        recommended_instances = self._calculate_recommended_instances(total_pressure)
        
        if recommended_instances > self.current_instances:
            # Scale up decision
            scaling_cost = self.cost_optimizer.calculate_scaling_cost(
                self.current_instances, recommended_instances, 1.0  # 1 hour assumption
            )
            
            current_metric_values = {name: metric.value for name, metric in current_metrics.items()}
            sla_risk = self.cost_optimizer.estimate_sla_violation_risk(
                current_metric_values, 1.0 + max_pressure
            )
            
            if self.cost_optimizer.should_scale_for_cost(scaling_cost, sla_risk):
                return ScalingDecision.SCALE_UP, primary_trigger, recommended_instances
            else:
                logger.info("Scale-up blocked by cost optimization")
                return ScalingDecision.MAINTAIN, primary_trigger, self.current_instances
        
        elif recommended_instances < self.current_instances:
            # Scale down decision - less strict cost checking
            return ScalingDecision.SCALE_DOWN, primary_trigger, recommended_instances
        
        return ScalingDecision.MAINTAIN, primary_trigger, self.current_instances
    
    def _get_predictive_pressure(self) -> float:
        """Get scaling pressure based on predictive analysis."""
        future_time = datetime.now() + timedelta(minutes=15)  # Look ahead 15 minutes
        
        # Predict CPU utilization
        cpu_prediction, cpu_confidence = self.pattern_analyzer.predict_future_value('cpu_utilization', future_time)
        
        if cpu_confidence < 0.5:
            return 0.0  # Low confidence, don't use prediction
        
        # Convert prediction to scaling pressure
        if cpu_prediction > 80:
            return min((cpu_prediction - 80) / 20, 1.0) * cpu_confidence
        elif cpu_prediction < 20:
            return -min((20 - cpu_prediction) / 20, 1.0) * cpu_confidence
        else:
            return 0.0
    
    def _calculate_recommended_instances(self, total_pressure: float) -> int:
        """Calculate recommended number of instances based on pressure."""
        if total_pressure > 0.5:
            # Scale up
            scale_factor = 1.0 + min(total_pressure, 2.0)  # Cap at 3x current
            recommended = math.ceil(self.current_instances * scale_factor)
        elif total_pressure < -0.5:
            # Scale down
            scale_factor = 1.0 + max(total_pressure, -0.5)  # Max 50% reduction
            recommended = math.floor(self.current_instances * scale_factor)
        else:
            recommended = self.current_instances
        
        # Enforce limits
        return max(self.min_instances, min(self.max_instances, recommended))
    
    def _execute_scaling_decision(self, decision: ScalingDecision, trigger: ScalingTrigger, 
                                target_instances: int) -> None:
        """Execute a scaling decision."""
        old_instances = self.current_instances
        
        # Record scaling event
        with self._lock:
            current_metric_values = {
                name: metric.value for name, metric in self._current_metrics.items()
            }
        
        scaling_event = ScalingEvent(
            timestamp=datetime.now(),
            decision=decision,
            trigger=trigger,
            instances_before=old_instances,
            instances_after=target_instances,
            metrics_snapshot=current_metric_values,
            cost_impact=self.cost_optimizer.calculate_scaling_cost(old_instances, target_instances, 1.0)
        )
        
        # Execute the scaling (in practice, would integrate with cloud APIs)
        if self._simulate_scaling(target_instances):
            self.current_instances = target_instances
            self._last_scaling_action = datetime.now()
            
            with self._lock:
                self._scaling_events.append(scaling_event)
            
            logger.info(f"Scaling decision executed: {decision.value} from {old_instances} to {target_instances} instances (trigger: {trigger.value})")
        else:
            logger.error(f"Failed to execute scaling decision: {decision.value}")
    
    def _simulate_scaling(self, target_instances: int) -> bool:
        """Simulate scaling operation (replace with actual cloud integration)."""
        # In a real implementation, this would:
        # 1. Call cloud APIs to provision/terminate instances
        # 2. Update load balancer configuration
        # 3. Monitor health of new instances
        # 4. Handle any failures gracefully
        
        logger.info(f"Simulating scaling to {target_instances} instances...")
        time.sleep(1)  # Simulate API call delay
        return True
    
    def _generate_predictions(self) -> None:
        """Generate future resource predictions."""
        future_times = [
            datetime.now() + timedelta(minutes=15),
            datetime.now() + timedelta(hours=1),
            datetime.now() + timedelta(hours=4),
            datetime.now() + timedelta(hours=24)
        ]
        
        for future_time in future_times:
            cpu_pred, cpu_conf = self.pattern_analyzer.predict_future_value('cpu_utilization', future_time)
            mem_pred, mem_conf = self.pattern_analyzer.predict_future_value('memory_utilization', future_time)
            queue_pred, queue_conf = self.pattern_analyzer.predict_future_value('queue_length', future_time)
            
            # Calculate recommended instances for predicted load
            predicted_pressure = 0.0
            if cpu_pred > 70:
                predicted_pressure += (cpu_pred - 70) / 30
            elif cpu_pred < 30:
                predicted_pressure -= (30 - cpu_pred) / 30
            
            recommended = self._calculate_recommended_instances(predicted_pressure)
            
            prediction = ResourcePrediction(
                prediction_time=datetime.now(),
                for_time=future_time,
                predicted_cpu_utilization=cpu_pred,
                predicted_memory_utilization=mem_pred,
                predicted_queue_length=int(queue_pred),
                predicted_throughput=max(0.0, 100.0 - cpu_pred),  # Simplified throughput prediction
                confidence=min(cpu_conf, mem_conf, queue_conf),
                recommended_instances=recommended
            )
            
            with self._lock:
                self._predictions.append(prediction)
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status and metrics."""
        with self._lock:
            current_metrics = {
                name: {
                    "value": metric.value,
                    "threshold_low": metric.threshold_low,
                    "threshold_high": metric.threshold_high,
                    "scaling_pressure": metric.get_scaling_pressure()
                }
                for name, metric in self._current_metrics.items()
            }
            
            recent_events = [
                event.to_dict() for event in list(self._scaling_events)[-10:]
            ]
            
            recent_predictions = [
                pred.to_dict() for pred in list(self._predictions)[-5:]
            ]
        
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "cooldown_remaining_minutes": max(0, (
                self._last_scaling_action + timedelta(minutes=self.cooldown_period) - datetime.now()
            ).total_seconds() / 60),
            "current_metrics": current_metrics,
            "recent_scaling_events": recent_events,
            "predictions": recent_predictions,
            "autoscaler_active": self._autoscaler_active,
            "last_updated": datetime.now().isoformat()
        }
    
    def force_scaling_decision(self, target_instances: int, reason: str = "manual") -> bool:
        """Manually force a scaling decision."""
        if target_instances < self.min_instances or target_instances > self.max_instances:
            logger.error(f"Target instances {target_instances} outside allowed range [{self.min_instances}, {self.max_instances}]")
            return False
        
        if target_instances == self.current_instances:
            logger.info("Target instances same as current, no scaling needed")
            return True
        
        decision = ScalingDecision.SCALE_UP if target_instances > self.current_instances else ScalingDecision.SCALE_DOWN
        
        # Create manual scaling event
        scaling_event = ScalingEvent(
            timestamp=datetime.now(),
            decision=decision,
            trigger=ScalingTrigger.COST_OPTIMIZATION,  # Use as manual trigger
            instances_before=self.current_instances,
            instances_after=target_instances,
            metrics_snapshot={"manual_reason": reason},
            cost_impact=self.cost_optimizer.calculate_scaling_cost(self.current_instances, target_instances, 1.0)
        )
        
        if self._simulate_scaling(target_instances):
            self.current_instances = target_instances
            self._last_scaling_action = datetime.now()
            
            with self._lock:
                self._scaling_events.append(scaling_event)
            
            logger.info(f"Manual scaling executed: {decision.value} to {target_instances} instances (reason: {reason})")
            return True
        
        return False
    
    def shutdown(self) -> None:
        """Shutdown the auto-scaler."""
        self._autoscaler_active = False
        
        if hasattr(self, '_autoscaler_thread'):
            self._autoscaler_thread.join(timeout=5)
        
        logger.info("Intelligent auto-scaler shutdown completed")


# Global auto-scaler instance
_global_autoscaler: Optional[IntelligentAutoScaler] = None
_autoscaler_lock = None  # Will be created when needed


def get_intelligent_autoscaler(config: Optional[Dict[str, Any]] = None) -> IntelligentAutoScaler:
    """Get or create global intelligent auto-scaler."""
    global _global_autoscaler, _autoscaler_lock
    
    if _autoscaler_lock is None:
        import threading
        _autoscaler_lock = threading.Lock()
    
    with _autoscaler_lock:
        if _global_autoscaler is None:
            _global_autoscaler = IntelligentAutoScaler(config)
        return _global_autoscaler


def initialize_intelligent_autoscaling(config: Optional[Dict[str, Any]] = None) -> IntelligentAutoScaler:
    """Initialize intelligent auto-scaling system."""
    return get_intelligent_autoscaler(config)


def update_autoscaling_metrics(metrics: Dict[str, float]) -> None:
    """Update metrics for auto-scaling decisions."""
    autoscaler = get_intelligent_autoscaler()
    autoscaler.update_metrics(metrics)


def get_autoscaling_status() -> Dict[str, Any]:
    """Get current auto-scaling status."""
    autoscaler = get_intelligent_autoscaler()
    return autoscaler.get_scaling_status()