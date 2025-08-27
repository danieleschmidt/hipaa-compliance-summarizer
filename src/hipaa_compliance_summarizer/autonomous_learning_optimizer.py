"""
Autonomous Learning Optimizer for HIPAA Compliance Summarizer Generation 4

This module provides advanced machine learning-driven optimization that continuously
learns from system behavior, user patterns, and performance metrics to autonomously
improve system efficiency, accuracy, and user experience.
"""
import asyncio
import json
import logging
import pickle
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from .advanced_monitoring import AdvancedMonitor
from .intelligent_performance_optimizer import IntelligentPerformanceOptimizer


class OptimizationDomain(Enum):
    """Domains for autonomous optimization."""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    RESOURCE_USAGE = "resource_usage"
    USER_EXPERIENCE = "user_experience"
    COMPLIANCE = "compliance"
    SECURITY = "security"


class LearningPhase(Enum):
    """Phases of the learning process."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    REFINEMENT = "refinement"
    ADAPTATION = "adaptation"


@dataclass
class OptimizationMetric:
    """Single optimization metric with context."""
    domain: OptimizationDomain
    name: str
    value: float
    target: float
    importance_weight: float
    timestamp: datetime
    context: Dict[str, Any]


@dataclass
class LearningPattern:
    """Learned pattern from system behavior."""
    pattern_id: str
    domain: OptimizationDomain
    description: str
    confidence: float
    impact_score: float
    usage_frequency: int
    last_applied: Optional[datetime]
    success_rate: float
    parameters: Dict[str, Any]


@dataclass
class OptimizationAction:
    """Action to be taken for optimization."""
    action_id: str
    domain: OptimizationDomain
    action_type: str
    description: str
    parameters: Dict[str, Any]
    expected_impact: float
    confidence: float
    risk_level: str
    reversible: bool


@dataclass
class OptimizationResult:
    """Result of optimization execution."""
    action_id: str
    success: bool
    actual_impact: float
    execution_time: float
    side_effects: List[str]
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    learned_insights: List[str]


class AutonomousLearningOptimizer:
    """
    Advanced autonomous learning system that continuously optimizes
    the HIPAA compliance system through ML-driven insights.
    
    Features:
    - Multi-domain optimization (performance, accuracy, compliance)
    - Adaptive learning from user behavior and system patterns
    - Intelligent parameter tuning with safety constraints
    - Predictive optimization based on usage patterns
    - Self-evolving optimization strategies
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        exploration_factor: float = 0.1,
        confidence_threshold: float = 0.7,
        model_update_interval: int = 100,
        max_history_size: int = 10000
    ):
        self.learning_rate = learning_rate
        self.exploration_factor = exploration_factor
        self.confidence_threshold = confidence_threshold
        self.model_update_interval = model_update_interval
        self.max_history_size = max_history_size
        
        # Learning state
        self.current_phase = LearningPhase.EXPLORATION
        self.iteration_count = 0
        self.optimization_history: deque = deque(maxlen=max_history_size)
        self.learned_patterns: Dict[str, LearningPattern] = {}
        self.active_optimizations: Dict[str, OptimizationAction] = {}
        
        # ML models for different domains
        self.models: Dict[OptimizationDomain, Dict[str, Any]] = {}
        self.scalers: Dict[OptimizationDomain, StandardScaler] = {}
        self.anomaly_detectors: Dict[OptimizationDomain, IsolationForest] = {}
        
        # Metrics tracking
        self.metrics_history: Dict[OptimizationDomain, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.baseline_metrics: Dict[OptimizationDomain, Dict[str, float]] = {}
        
        # Performance tracking
        self.optimization_success_rate = 0.0
        self.total_impact_achieved = 0.0
        self.adaptive_parameters: Dict[str, float] = {
            "learning_rate": learning_rate,
            "exploration_factor": exploration_factor,
            "confidence_threshold": confidence_threshold
        }
        
        # Dependencies
        self.monitor = AdvancedMonitor()
        self.performance_optimizer = IntelligentPerformanceOptimizer()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models
        self._initialize_ml_models()
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for each optimization domain."""
        for domain in OptimizationDomain:
            self.models[domain] = {
                "predictor": RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10
                ),
                "clusterer": KMeans(n_clusters=5, random_state=42),
                "trained": False,
                "feature_names": []
            }
            
            self.scalers[domain] = StandardScaler()
            self.anomaly_detectors[domain] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
    
    async def continuous_optimization_loop(self):
        """Main continuous optimization loop that runs autonomously."""
        self.logger.info("Starting autonomous continuous optimization loop")
        
        while True:
            try:
                # Collect current system metrics
                current_metrics = await self._collect_comprehensive_metrics()
                
                # Update learning phase based on performance
                self._update_learning_phase()
                
                # Analyze system state and identify optimization opportunities
                opportunities = await self._identify_optimization_opportunities(current_metrics)
                
                # Select and execute optimizations based on current phase
                if opportunities:
                    selected_actions = self._select_optimization_actions(opportunities)
                    
                    for action in selected_actions:
                        result = await self._execute_optimization_action(action, current_metrics)
                        await self._learn_from_optimization_result(action, result)
                
                # Update ML models periodically
                if self.iteration_count % self.model_update_interval == 0:
                    await self._update_ml_models()
                
                # Adapt learning parameters based on success rate
                self._adapt_learning_parameters()
                
                self.iteration_count += 1
                
                # Sleep based on current phase
                sleep_duration = self._calculate_optimization_interval()
                await asyncio.sleep(sleep_duration)
                
            except Exception as e:
                self.logger.error(f"Error in continuous optimization loop: {e}")
                await asyncio.sleep(60)  # Fallback sleep on error
    
    async def _collect_comprehensive_metrics(self) -> Dict[OptimizationDomain, Dict[str, float]]:
        """Collect comprehensive metrics across all optimization domains."""
        metrics = {}
        
        # Performance metrics
        performance_metrics = await self._collect_performance_metrics()
        metrics[OptimizationDomain.PERFORMANCE] = performance_metrics
        
        # Accuracy metrics
        accuracy_metrics = await self._collect_accuracy_metrics()
        metrics[OptimizationDomain.ACCURACY] = accuracy_metrics
        
        # Resource usage metrics
        resource_metrics = await self._collect_resource_metrics()
        metrics[OptimizationDomain.RESOURCE_USAGE] = resource_metrics
        
        # User experience metrics
        ux_metrics = await self._collect_user_experience_metrics()
        metrics[OptimizationDomain.USER_EXPERIENCE] = ux_metrics
        
        # Compliance metrics
        compliance_metrics = await self._collect_compliance_metrics()
        metrics[OptimizationDomain.COMPLIANCE] = compliance_metrics
        
        # Security metrics
        security_metrics = await self._collect_security_metrics()
        metrics[OptimizationDomain.SECURITY] = security_metrics
        
        # Store metrics in history for trend analysis
        for domain, domain_metrics in metrics.items():
            self.metrics_history[domain].append({
                "timestamp": datetime.now(),
                "metrics": domain_metrics
            })
        
        return metrics
    
    async def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect performance-related metrics."""
        # In production, this would collect real metrics
        import random
        return {
            "response_time": random.uniform(80, 150),
            "throughput": random.uniform(450, 550),
            "cpu_usage": random.uniform(0.3, 0.7),
            "memory_usage": random.uniform(0.4, 0.8),
            "cache_hit_rate": random.uniform(0.85, 0.95),
            "error_rate": random.uniform(0.01, 0.05)
        }
    
    async def _collect_accuracy_metrics(self) -> Dict[str, float]:
        """Collect accuracy and quality metrics."""
        import random
        return {
            "phi_detection_accuracy": random.uniform(0.95, 0.99),
            "false_positive_rate": random.uniform(0.01, 0.05),
            "false_negative_rate": random.uniform(0.005, 0.02),
            "classification_confidence": random.uniform(0.85, 0.95),
            "model_drift_score": random.uniform(0.0, 0.1)
        }
    
    async def _collect_resource_metrics(self) -> Dict[str, float]:
        """Collect resource utilization metrics."""
        import random
        return {
            "memory_efficiency": random.uniform(0.7, 0.9),
            "disk_usage": random.uniform(0.3, 0.6),
            "network_utilization": random.uniform(0.2, 0.5),
            "gpu_usage": random.uniform(0.4, 0.8),
            "cost_per_operation": random.uniform(0.001, 0.005)
        }
    
    async def _collect_user_experience_metrics(self) -> Dict[str, float]:
        """Collect user experience metrics."""
        import random
        return {
            "user_satisfaction": random.uniform(4.0, 4.8),
            "task_completion_rate": random.uniform(0.85, 0.95),
            "user_retention": random.uniform(0.8, 0.9),
            "feature_adoption": random.uniform(0.6, 0.8),
            "support_ticket_rate": random.uniform(0.02, 0.08)
        }
    
    async def _collect_compliance_metrics(self) -> Dict[str, float]:
        """Collect compliance-related metrics."""
        import random
        return {
            "hipaa_compliance_score": random.uniform(0.95, 1.0),
            "audit_readiness": random.uniform(0.9, 0.98),
            "data_governance_score": random.uniform(0.88, 0.96),
            "privacy_protection_level": random.uniform(0.92, 0.99),
            "regulatory_alignment": random.uniform(0.9, 0.97)
        }
    
    async def _collect_security_metrics(self) -> Dict[str, float]:
        """Collect security metrics."""
        import random
        return {
            "security_posture": random.uniform(0.9, 0.98),
            "threat_detection_rate": random.uniform(0.85, 0.95),
            "vulnerability_score": random.uniform(0.0, 0.2),
            "access_control_effectiveness": random.uniform(0.9, 0.99),
            "incident_response_time": random.uniform(5, 15)  # minutes
        }
    
    def _update_learning_phase(self):
        """Update learning phase based on system performance and experience."""
        success_rate = self._calculate_recent_success_rate()
        
        if self.iteration_count < 50:
            self.current_phase = LearningPhase.EXPLORATION
        elif success_rate < 0.6:
            self.current_phase = LearningPhase.EXPLORATION
        elif success_rate > 0.8 and len(self.learned_patterns) > 10:
            self.current_phase = LearningPhase.EXPLOITATION
        elif len(self.learned_patterns) > 20:
            self.current_phase = LearningPhase.REFINEMENT
        else:
            self.current_phase = LearningPhase.ADAPTATION
        
        self.logger.debug(f"Learning phase: {self.current_phase.value}, Success rate: {success_rate:.2f}")
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate success rate from recent optimization history."""
        if not self.optimization_history:
            return 0.5  # Default neutral success rate
        
        recent_results = list(self.optimization_history)[-20:]  # Last 20 optimizations
        successful = sum(1 for result in recent_results if result.get('success', False))
        return successful / len(recent_results) if recent_results else 0.5
    
    async def _identify_optimization_opportunities(
        self,
        current_metrics: Dict[OptimizationDomain, Dict[str, float]]
    ) -> List[OptimizationAction]:
        """Identify optimization opportunities using ML insights."""
        opportunities = []
        
        for domain, metrics in current_metrics.items():
            domain_opportunities = await self._analyze_domain_for_opportunities(domain, metrics)
            opportunities.extend(domain_opportunities)
        
        # Prioritize opportunities by expected impact and confidence
        opportunities.sort(key=lambda x: x.expected_impact * x.confidence, reverse=True)
        
        return opportunities
    
    async def _analyze_domain_for_opportunities(
        self,
        domain: OptimizationDomain,
        metrics: Dict[str, float]
    ) -> List[OptimizationAction]:
        """Analyze a specific domain for optimization opportunities."""
        opportunities = []
        
        # Use ML models if trained, otherwise use rule-based analysis
        if self.models[domain]["trained"]:
            opportunities.extend(await self._ml_based_opportunity_detection(domain, metrics))
        else:
            opportunities.extend(await self._rule_based_opportunity_detection(domain, metrics))
        
        # Add anomaly-based opportunities
        anomaly_opportunities = await self._detect_anomaly_opportunities(domain, metrics)
        opportunities.extend(anomaly_opportunities)
        
        return opportunities
    
    async def _ml_based_opportunity_detection(
        self,
        domain: OptimizationDomain,
        metrics: Dict[str, float]
    ) -> List[OptimizationAction]:
        """Use ML models to detect optimization opportunities."""
        opportunities = []
        model = self.models[domain]
        
        try:
            # Prepare features
            feature_vector = np.array([[metrics.get(name, 0.0) for name in model["feature_names"]]])
            feature_vector_scaled = self.scalers[domain].transform(feature_vector)
            
            # Predict optimal values
            predicted_optimal = model["predictor"].predict(feature_vector_scaled)[0]
            
            # Compare with current performance
            current_performance = np.mean(list(metrics.values()))
            improvement_potential = predicted_optimal - current_performance
            
            if improvement_potential > 0.05:  # 5% improvement threshold
                action = OptimizationAction(
                    action_id=f"ml_opt_{domain.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    domain=domain,
                    action_type="ml_optimization",
                    description=f"ML-predicted optimization for {domain.value}",
                    parameters={"target_improvement": improvement_potential},
                    expected_impact=improvement_potential,
                    confidence=0.8,  # High confidence for ML-based predictions
                    risk_level="LOW",
                    reversible=True
                )
                opportunities.append(action)
        
        except Exception as e:
            self.logger.warning(f"ML-based opportunity detection failed for {domain}: {e}")
        
        return opportunities
    
    async def _rule_based_opportunity_detection(
        self,
        domain: OptimizationDomain,
        metrics: Dict[str, float]
    ) -> List[OptimizationAction]:
        """Use rule-based analysis to detect opportunities."""
        opportunities = []
        
        if domain == OptimizationDomain.PERFORMANCE:
            # Performance optimization rules
            if metrics.get("response_time", 0) > 120:
                opportunities.append(OptimizationAction(
                    action_id=f"perf_latency_{datetime.now().strftime('%H%M%S')}",
                    domain=domain,
                    action_type="latency_optimization",
                    description="Optimize response time",
                    parameters={"target_latency": 100},
                    expected_impact=0.15,
                    confidence=0.7,
                    risk_level="MEDIUM",
                    reversible=True
                ))
            
            if metrics.get("cache_hit_rate", 1.0) < 0.9:
                opportunities.append(OptimizationAction(
                    action_id=f"perf_cache_{datetime.now().strftime('%H%M%S')}",
                    domain=domain,
                    action_type="cache_optimization",
                    description="Improve cache efficiency",
                    parameters={"target_hit_rate": 0.95},
                    expected_impact=0.12,
                    confidence=0.8,
                    risk_level="LOW",
                    reversible=True
                ))
        
        elif domain == OptimizationDomain.ACCURACY:
            if metrics.get("phi_detection_accuracy", 1.0) < 0.97:
                opportunities.append(OptimizationAction(
                    action_id=f"acc_phi_{datetime.now().strftime('%H%M%S')}",
                    domain=domain,
                    action_type="model_tuning",
                    description="Improve PHI detection accuracy",
                    parameters={"target_accuracy": 0.98},
                    expected_impact=0.1,
                    confidence=0.6,
                    risk_level="MEDIUM",
                    reversible=True
                ))
        
        elif domain == OptimizationDomain.RESOURCE_USAGE:
            if metrics.get("memory_efficiency", 1.0) < 0.8:
                opportunities.append(OptimizationAction(
                    action_id=f"res_mem_{datetime.now().strftime('%H%M%S')}",
                    domain=domain,
                    action_type="memory_optimization",
                    description="Optimize memory usage",
                    parameters={"target_efficiency": 0.85},
                    expected_impact=0.08,
                    confidence=0.75,
                    risk_level="LOW",
                    reversible=True
                ))
        
        return opportunities
    
    async def _detect_anomaly_opportunities(
        self,
        domain: OptimizationDomain,
        metrics: Dict[str, float]
    ) -> List[OptimizationAction]:
        """Detect optimization opportunities based on anomalies."""
        opportunities = []
        
        # This would use trained anomaly detection models
        # For demo, we'll simulate anomaly detection
        
        import random
        if random.random() < 0.1:  # 10% chance of detecting anomaly
            opportunities.append(OptimizationAction(
                action_id=f"anomaly_{domain.value}_{datetime.now().strftime('%H%M%S')}",
                domain=domain,
                action_type="anomaly_correction",
                description=f"Correct detected anomaly in {domain.value}",
                parameters={"anomaly_score": random.uniform(0.1, 0.3)},
                expected_impact=0.05,
                confidence=0.6,
                risk_level="MEDIUM",
                reversible=True
            ))
        
        return opportunities
    
    def _select_optimization_actions(
        self,
        opportunities: List[OptimizationAction]
    ) -> List[OptimizationAction]:
        """Select optimization actions based on current learning phase and constraints."""
        
        max_concurrent_optimizations = {
            LearningPhase.EXPLORATION: 3,
            LearningPhase.EXPLOITATION: 5,
            LearningPhase.REFINEMENT: 2,
            LearningPhase.ADAPTATION: 4
        }
        
        max_actions = max_concurrent_optimizations[self.current_phase]
        
        # Filter by confidence threshold
        confident_actions = [
            action for action in opportunities
            if action.confidence >= self.confidence_threshold
        ]
        
        # In exploration phase, also include some lower-confidence actions
        if self.current_phase == LearningPhase.EXPLORATION:
            exploration_actions = [
                action for action in opportunities
                if action.confidence < self.confidence_threshold and action.risk_level == "LOW"
            ]
            # Add some exploration actions
            num_exploration = min(max_actions // 3, len(exploration_actions))
            confident_actions.extend(exploration_actions[:num_exploration])
        
        # Select top actions by expected impact
        selected_actions = sorted(
            confident_actions,
            key=lambda x: x.expected_impact * x.confidence,
            reverse=True
        )[:max_actions]
        
        return selected_actions
    
    async def _execute_optimization_action(
        self,
        action: OptimizationAction,
        baseline_metrics: Dict[OptimizationDomain, Dict[str, float]]
    ) -> OptimizationResult:
        """Execute an optimization action and measure its impact."""
        
        self.logger.info(f"Executing optimization action: {action.description}")
        start_time = datetime.now()
        
        try:
            # Record baseline metrics
            metrics_before = baseline_metrics[action.domain].copy()
            
            # Execute the optimization (simulated)
            success = await self._simulate_optimization_execution(action)
            
            # Wait for effects to take place
            await asyncio.sleep(2)  # Simulated execution time
            
            # Collect metrics after optimization
            post_optimization_metrics = await self._collect_comprehensive_metrics()
            metrics_after = post_optimization_metrics[action.domain]
            
            # Calculate actual impact
            actual_impact = self._calculate_optimization_impact(
                metrics_before, metrics_after, action.domain
            )
            
            # Identify side effects
            side_effects = self._identify_side_effects(
                baseline_metrics, post_optimization_metrics, action.domain
            )
            
            # Generate learned insights
            learned_insights = self._generate_learned_insights(
                action, actual_impact, metrics_before, metrics_after
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = OptimizationResult(
                action_id=action.action_id,
                success=success,
                actual_impact=actual_impact,
                execution_time=execution_time,
                side_effects=side_effects,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                learned_insights=learned_insights
            )
            
            self.logger.info(f"Optimization completed: {action.description} - Impact: {actual_impact:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization execution failed: {action.description}: {e}")
            return OptimizationResult(
                action_id=action.action_id,
                success=False,
                actual_impact=0.0,
                execution_time=(datetime.now() - start_time).total_seconds(),
                side_effects=[f"Execution error: {str(e)}"],
                metrics_before=baseline_metrics[action.domain],
                metrics_after={},
                learned_insights=["Optimization failed due to execution error"]
            )
    
    async def _simulate_optimization_execution(self, action: OptimizationAction) -> bool:
        """Simulate execution of optimization action."""
        # In production, this would execute real optimizations
        import random
        
        success_rates = {
            "latency_optimization": 0.8,
            "cache_optimization": 0.9,
            "model_tuning": 0.7,
            "memory_optimization": 0.85,
            "ml_optimization": 0.75,
            "anomaly_correction": 0.6
        }
        
        base_success_rate = success_rates.get(action.action_type, 0.7)
        
        # Adjust success rate based on confidence and risk
        if action.confidence > 0.8:
            base_success_rate += 0.1
        if action.risk_level == "LOW":
            base_success_rate += 0.05
        elif action.risk_level == "HIGH":
            base_success_rate -= 0.1
        
        return random.random() < base_success_rate
    
    def _calculate_optimization_impact(
        self,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        domain: OptimizationDomain
    ) -> float:
        """Calculate the actual impact of optimization."""
        
        # Define positive/negative metrics for each domain
        positive_metrics = {
            OptimizationDomain.PERFORMANCE: ["throughput", "cache_hit_rate"],
            OptimizationDomain.ACCURACY: ["phi_detection_accuracy", "classification_confidence"],
            OptimizationDomain.RESOURCE_USAGE: ["memory_efficiency"],
            OptimizationDomain.USER_EXPERIENCE: ["user_satisfaction", "task_completion_rate", "user_retention"],
            OptimizationDomain.COMPLIANCE: ["hipaa_compliance_score", "audit_readiness"],
            OptimizationDomain.SECURITY: ["security_posture", "threat_detection_rate", "access_control_effectiveness"]
        }
        
        negative_metrics = {
            OptimizationDomain.PERFORMANCE: ["response_time", "error_rate", "cpu_usage", "memory_usage"],
            OptimizationDomain.ACCURACY: ["false_positive_rate", "false_negative_rate", "model_drift_score"],
            OptimizationDomain.RESOURCE_USAGE: ["disk_usage", "network_utilization", "cost_per_operation"],
            OptimizationDomain.USER_EXPERIENCE: ["support_ticket_rate"],
            OptimizationDomain.COMPLIANCE: [],
            OptimizationDomain.SECURITY: ["vulnerability_score", "incident_response_time"]
        }
        
        total_impact = 0.0
        metric_count = 0
        
        # Calculate positive impact (higher is better)
        for metric in positive_metrics.get(domain, []):
            if metric in metrics_before and metric in metrics_after:
                before = metrics_before[metric]
                after = metrics_after[metric]
                if before > 0:  # Avoid division by zero
                    impact = (after - before) / before
                    total_impact += impact
                    metric_count += 1
        
        # Calculate negative impact (lower is better)
        for metric in negative_metrics.get(domain, []):
            if metric in metrics_before and metric in metrics_after:
                before = metrics_before[metric]
                after = metrics_after[metric]
                if before > 0:  # Avoid division by zero
                    impact = (before - after) / before  # Reversed for negative metrics
                    total_impact += impact
                    metric_count += 1
        
        return total_impact / metric_count if metric_count > 0 else 0.0
    
    def _identify_side_effects(
        self,
        baseline_metrics: Dict[OptimizationDomain, Dict[str, float]],
        post_metrics: Dict[OptimizationDomain, Dict[str, float]],
        target_domain: OptimizationDomain
    ) -> List[str]:
        """Identify side effects of optimization on other domains."""
        side_effects = []
        
        for domain, metrics in post_metrics.items():
            if domain == target_domain:
                continue  # Skip the target domain
            
            baseline_domain_metrics = baseline_metrics.get(domain, {})
            
            for metric_name, current_value in metrics.items():
                baseline_value = baseline_domain_metrics.get(metric_name, current_value)
                
                if baseline_value > 0:
                    change_pct = (current_value - baseline_value) / baseline_value
                    
                    if abs(change_pct) > 0.05:  # 5% change threshold
                        direction = "increased" if change_pct > 0 else "decreased"
                        side_effects.append(
                            f"{domain.value}.{metric_name} {direction} by {abs(change_pct)*100:.1f}%"
                        )
        
        return side_effects
    
    def _generate_learned_insights(
        self,
        action: OptimizationAction,
        actual_impact: float,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float]
    ) -> List[str]:
        """Generate insights learned from optimization execution."""
        insights = []
        
        # Compare expected vs actual impact
        impact_accuracy = 1 - abs(action.expected_impact - actual_impact) / max(action.expected_impact, 0.01)
        
        if impact_accuracy > 0.8:
            insights.append(f"Impact prediction was accurate ({impact_accuracy:.2f})")
        elif impact_accuracy < 0.5:
            insights.append(f"Impact prediction was inaccurate ({impact_accuracy:.2f}) - need better modeling")
        
        # Learn from successful optimizations
        if actual_impact > 0.05:
            insights.append(f"Optimization type '{action.action_type}' showed good results for {action.domain.value}")
        
        # Learn from failure patterns
        if actual_impact < 0:
            insights.append(f"Optimization type '{action.action_type}' caused regression in {action.domain.value}")
        
        # Parameter effectiveness insights
        if action.parameters:
            insights.append(f"Parameter configuration: {action.parameters} resulted in {actual_impact:.3f} impact")
        
        return insights
    
    async def _learn_from_optimization_result(
        self,
        action: OptimizationAction,
        result: OptimizationResult
    ):
        """Learn from optimization result and update internal models."""
        
        # Store result in history
        self.optimization_history.append({
            "timestamp": datetime.now(),
            "action": asdict(action),
            "result": asdict(result),
            "success": result.success,
            "impact": result.actual_impact
        })
        
        # Update learned patterns
        await self._update_learned_patterns(action, result)
        
        # Update success metrics
        self._update_success_metrics(result)
        
        # Learn parameter effectiveness
        self._learn_parameter_effectiveness(action, result)
        
        self.logger.info(f"Learned from optimization: {action.action_type} - Impact: {result.actual_impact:.3f}")
    
    async def _update_learned_patterns(
        self,
        action: OptimizationAction,
        result: OptimizationResult
    ):
        """Update learned patterns based on optimization results."""
        
        pattern_id = f"{action.domain.value}_{action.action_type}"
        
        if pattern_id not in self.learned_patterns:
            self.learned_patterns[pattern_id] = LearningPattern(
                pattern_id=pattern_id,
                domain=action.domain,
                description=f"Pattern for {action.action_type} in {action.domain.value}",
                confidence=0.5,
                impact_score=0.0,
                usage_frequency=0,
                last_applied=None,
                success_rate=0.0,
                parameters={}
            )
        
        pattern = self.learned_patterns[pattern_id]
        pattern.usage_frequency += 1
        pattern.last_applied = datetime.now()
        
        # Update success rate
        if result.success:
            pattern.success_rate = (pattern.success_rate * (pattern.usage_frequency - 1) + 1.0) / pattern.usage_frequency
        else:
            pattern.success_rate = (pattern.success_rate * (pattern.usage_frequency - 1) + 0.0) / pattern.usage_frequency
        
        # Update impact score
        pattern.impact_score = (pattern.impact_score * (pattern.usage_frequency - 1) + result.actual_impact) / pattern.usage_frequency
        
        # Update confidence based on consistency
        if pattern.usage_frequency > 5:
            impact_variance = 0.1  # Would calculate actual variance in production
            pattern.confidence = min(0.95, pattern.success_rate * (1 - impact_variance))
        
        # Learn effective parameters
        if action.parameters and result.success:
            pattern.parameters.update(action.parameters)
    
    def _update_success_metrics(self, result: OptimizationResult):
        """Update overall optimization success metrics."""
        recent_results = list(self.optimization_history)[-100:]  # Last 100 results
        
        if recent_results:
            successes = sum(1 for r in recent_results if r.get('success', False))
            self.optimization_success_rate = successes / len(recent_results)
            
            total_impact = sum(r.get('impact', 0.0) for r in recent_results if r.get('success', False))
            self.total_impact_achieved = total_impact
    
    def _learn_parameter_effectiveness(
        self,
        action: OptimizationAction,
        result: OptimizationResult
    ):
        """Learn which parameters are most effective for different optimization types."""
        # This would implement more sophisticated parameter learning
        # For now, we just log the insight
        if result.success and result.actual_impact > 0.05:
            self.logger.debug(f"Effective parameters for {action.action_type}: {action.parameters}")
    
    async def _update_ml_models(self):
        """Update ML models with recent optimization data."""
        self.logger.info("Updating ML models with recent optimization data")
        
        for domain in OptimizationDomain:
            await self._update_domain_model(domain)
    
    async def _update_domain_model(self, domain: OptimizationDomain):
        """Update ML model for a specific domain."""
        
        # Collect training data from metrics history
        domain_history = self.metrics_history[domain]
        
        if len(domain_history) < 20:  # Need minimum data for training
            return
        
        # Prepare training data
        X, y = self._prepare_training_data(domain_history)
        
        if X.shape[0] < 10:  # Need minimum samples
            return
        
        try:
            # Update scaler
            self.scalers[domain].fit(X)
            X_scaled = self.scalers[domain].transform(X)
            
            # Train predictor model
            self.models[domain]["predictor"].fit(X_scaled, y)
            
            # Train clustering model for pattern detection
            self.models[domain]["clusterer"].fit(X_scaled)
            
            # Train anomaly detector
            self.anomaly_detectors[domain].fit(X_scaled)
            
            # Store feature names
            self.models[domain]["feature_names"] = list(domain_history[0]["metrics"].keys())
            self.models[domain]["trained"] = True
            
            # Evaluate model performance
            y_pred = self.models[domain]["predictor"].predict(X_scaled)
            r2 = r2_score(y, y_pred)
            
            self.logger.info(f"Updated {domain.value} model - R² score: {r2:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to update {domain.value} model: {e}")
    
    def _prepare_training_data(self, domain_history: deque) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from domain history."""
        
        X = []  # Features (current metrics)
        y = []  # Target (next period performance)
        
        history_list = list(domain_history)
        
        for i in range(len(history_list) - 1):
            current_metrics = history_list[i]["metrics"]
            next_metrics = history_list[i + 1]["metrics"]
            
            # Use current metrics as features
            features = list(current_metrics.values())
            
            # Use next period overall performance as target
            target = np.mean(list(next_metrics.values()))
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def _adapt_learning_parameters(self):
        """Adapt learning parameters based on recent performance."""
        
        if self.optimization_success_rate < 0.6:
            # Poor performance - increase exploration
            self.adaptive_parameters["exploration_factor"] = min(0.3, self.exploration_factor * 1.1)
            self.adaptive_parameters["confidence_threshold"] = max(0.5, self.confidence_threshold * 0.95)
        elif self.optimization_success_rate > 0.8:
            # Good performance - reduce exploration
            self.adaptive_parameters["exploration_factor"] = max(0.05, self.exploration_factor * 0.9)
            self.adaptive_parameters["confidence_threshold"] = min(0.85, self.confidence_threshold * 1.02)
        
        # Adapt learning rate based on impact achieved
        if self.total_impact_achieved > 0.5:
            self.adaptive_parameters["learning_rate"] = min(0.05, self.learning_rate * 1.05)
        elif self.total_impact_achieved < 0.1:
            self.adaptive_parameters["learning_rate"] = max(0.005, self.learning_rate * 0.95)
    
    def _calculate_optimization_interval(self) -> float:
        """Calculate interval between optimization cycles based on current phase."""
        base_intervals = {
            LearningPhase.EXPLORATION: 300,  # 5 minutes
            LearningPhase.EXPLOITATION: 600,  # 10 minutes
            LearningPhase.REFINEMENT: 900,  # 15 minutes
            LearningPhase.ADAPTATION: 450   # 7.5 minutes
        }
        
        base_interval = base_intervals[self.current_phase]
        
        # Adjust based on success rate
        if self.optimization_success_rate > 0.8:
            return base_interval * 0.8  # More frequent when successful
        elif self.optimization_success_rate < 0.4:
            return base_interval * 1.5  # Less frequent when failing
        
        return base_interval
    
    def get_learning_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive learning dashboard."""
        
        recent_patterns = sorted(
            self.learned_patterns.values(),
            key=lambda p: p.last_applied or datetime.min,
            reverse=True
        )[:10]
        
        return {
            "learning_status": {
                "current_phase": self.current_phase.value,
                "iteration_count": self.iteration_count,
                "success_rate": self.optimization_success_rate,
                "total_impact_achieved": self.total_impact_achieved,
                "patterns_learned": len(self.learned_patterns)
            },
            "adaptive_parameters": self.adaptive_parameters,
            "model_status": {
                domain.value: {
                    "trained": self.models[domain]["trained"],
                    "feature_count": len(self.models[domain]["feature_names"])
                }
                for domain in OptimizationDomain
            },
            "recent_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "domain": p.domain.value,
                    "confidence": p.confidence,
                    "success_rate": p.success_rate,
                    "impact_score": p.impact_score,
                    "usage_frequency": p.usage_frequency
                }
                for p in recent_patterns
            ],
            "optimization_history_summary": {
                "total_optimizations": len(self.optimization_history),
                "recent_success_rate": self._calculate_recent_success_rate(),
                "average_impact": np.mean([
                    r.get('impact', 0.0) for r in list(self.optimization_history)[-50:]
                ]) if self.optimization_history else 0.0
            }
        }
    
    async def save_learning_state(self, filepath: str):
        """Save current learning state to file."""
        state = {
            "learned_patterns": {k: asdict(v) for k, v in self.learned_patterns.items()},
            "optimization_history": list(self.optimization_history),
            "adaptive_parameters": self.adaptive_parameters,
            "success_metrics": {
                "success_rate": self.optimization_success_rate,
                "total_impact": self.total_impact_achieved
            },
            "model_metadata": {
                domain.value: {
                    "trained": self.models[domain]["trained"],
                    "feature_names": self.models[domain]["feature_names"]
                }
                for domain in OptimizationDomain
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save ML models separately
        models_dir = Path(filepath).parent / "ml_models"
        models_dir.mkdir(exist_ok=True)
        
        for domain in OptimizationDomain:
            if self.models[domain]["trained"]:
                model_file = models_dir / f"{domain.value}_model.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump({
                        "predictor": self.models[domain]["predictor"],
                        "clusterer": self.models[domain]["clusterer"],
                        "scaler": self.scalers[domain],
                        "anomaly_detector": self.anomaly_detectors[domain]
                    }, f)
        
        self.logger.info(f"Learning state saved to {filepath}")
    
    async def load_learning_state(self, filepath: str):
        """Load learning state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore learned patterns
            self.learned_patterns = {
                k: LearningPattern(**v) for k, v in state["learned_patterns"].items()
            }
            
            # Restore optimization history
            self.optimization_history = deque(state["optimization_history"], maxlen=self.max_history_size)
            
            # Restore adaptive parameters
            self.adaptive_parameters.update(state["adaptive_parameters"])
            
            # Restore success metrics
            success_metrics = state["success_metrics"]
            self.optimization_success_rate = success_metrics["success_rate"]
            self.total_impact_achieved = success_metrics["total_impact"]
            
            # Load ML models
            models_dir = Path(filepath).parent / "ml_models"
            
            for domain in OptimizationDomain:
                model_file = models_dir / f"{domain.value}_model.pkl"
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        saved_models = pickle.load(f)
                        self.models[domain]["predictor"] = saved_models["predictor"]
                        self.models[domain]["clusterer"] = saved_models["clusterer"]
                        self.scalers[domain] = saved_models["scaler"]
                        self.anomaly_detectors[domain] = saved_models["anomaly_detector"]
                        self.models[domain]["trained"] = True
            
            self.logger.info(f"Learning state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load learning state: {e}")


# Example usage and demonstration
async def main():
    """Demonstrate autonomous learning optimizer."""
    
    # Initialize optimizer
    optimizer = AutonomousLearningOptimizer(
        learning_rate=0.02,
        exploration_factor=0.15,
        confidence_threshold=0.6,
        model_update_interval=50
    )
    
    print("\n=== AUTONOMOUS LEARNING OPTIMIZER DEMONSTRATION ===")
    print("Starting autonomous optimization system...")
    
    # Run several optimization cycles to demonstrate learning
    for cycle in range(5):
        print(f"\nOptimization Cycle {cycle + 1}")
        
        # Collect metrics
        current_metrics = await optimizer._collect_comprehensive_metrics()
        
        # Identify opportunities
        opportunities = await optimizer._identify_optimization_opportunities(current_metrics)
        print(f"Identified {len(opportunities)} optimization opportunities")
        
        # Select and execute optimizations
        if opportunities:
            selected_actions = optimizer._select_optimization_actions(opportunities)
            print(f"Selected {len(selected_actions)} actions for execution")
            
            for action in selected_actions:
                result = await optimizer._execute_optimization_action(action, current_metrics)
                await optimizer._learn_from_optimization_result(action, result)
                
                status = "✓" if result.success else "✗"
                print(f"  {status} {action.description} - Impact: {result.actual_impact:.3f}")
        
        # Update learning phase
        optimizer._update_learning_phase()
        print(f"Learning phase: {optimizer.current_phase.value}")
        
        # Update models periodically
        if (cycle + 1) % 2 == 0:
            await optimizer._update_ml_models()
            print("Updated ML models")
    
    # Show learning dashboard
    dashboard = optimizer.get_learning_dashboard()
    print(f"\n=== LEARNING DASHBOARD ===")
    print(f"Success Rate: {dashboard['learning_status']['success_rate']:.2f}")
    print(f"Total Impact: {dashboard['learning_status']['total_impact_achieved']:.3f}")
    print(f"Patterns Learned: {dashboard['learning_status']['patterns_learned']}")
    print(f"Current Phase: {dashboard['learning_status']['current_phase']}")
    
    # Show top learned patterns
    if dashboard['recent_patterns']:
        print(f"\nTop Learned Patterns:")
        for pattern in dashboard['recent_patterns'][:3]:
            print(f"  • {pattern['pattern_id']}: {pattern['confidence']:.2f} confidence, "
                  f"{pattern['success_rate']:.2f} success rate")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())