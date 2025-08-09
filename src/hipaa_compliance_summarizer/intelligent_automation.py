"""Intelligent Automation System for HIPAA Compliance.

This module provides AI-driven automation capabilities including adaptive
workflows, intelligent decision making, predictive analytics, and self-optimizing
systems for healthcare compliance processing.
"""

from __future__ import annotations

import json
import logging
import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable, Awaitable
from pathlib import Path

import numpy as np
from .monitoring.tracing import trace_operation
from .ml_integration import MLModelManager, ModelType, PredictionResult
from .error_handling import HIPAAError, ErrorSeverity
from .performance import PerformanceOptimizer

logger = logging.getLogger(__name__)


class AutomationLevel(str, Enum):
    """Levels of automation sophistication."""
    MANUAL = "manual"
    ASSISTED = "assisted" 
    CONDITIONAL = "conditional"
    HIGH = "high"
    FULL = "full"


class DecisionType(str, Enum):
    """Types of automated decisions."""
    PROCESSING_PATH = "processing_path"
    QUALITY_THRESHOLD = "quality_threshold"
    RESOURCE_ALLOCATION = "resource_allocation"
    ESCALATION_TRIGGER = "escalation_trigger"
    OPTIMIZATION_STRATEGY = "optimization_strategy"


@dataclass
class AutomationContext:
    """Context for automated decision making."""
    document_metadata: Dict[str, Any]
    processing_history: List[Dict[str, Any]]
    system_metrics: Dict[str, float]
    user_preferences: Dict[str, Any]
    compliance_requirements: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class AutomatedDecision:
    """Result of automated decision making."""
    decision_type: DecisionType
    decision_value: Any
    confidence: float
    reasoning: List[str]
    alternative_options: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]
    execution_plan: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class WorkflowStep:
    """Individual step in an automated workflow."""
    step_id: str
    name: str
    description: str
    step_type: str
    configuration: Dict[str, Any]
    dependencies: List[str]
    conditions: Dict[str, Any]
    retry_policy: Dict[str, Any]
    timeout_seconds: int = 300
    
    def is_ready_to_execute(self, completed_steps: List[str]) -> bool:
        """Check if step dependencies are satisfied."""
        return all(dep in completed_steps for dep in self.dependencies)


@dataclass
class AdaptiveWorkflow:
    """Self-adapting workflow definition."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    adaptation_rules: List[Dict[str, Any]]
    success_criteria: Dict[str, Any]
    failure_handling: Dict[str, Any]
    performance_targets: Dict[str, float]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()


class BaseDecisionEngine(ABC):
    """Base class for automated decision engines."""
    
    def __init__(self, engine_name: str, config: Dict[str, Any] = None):
        self.engine_name = engine_name
        self.config = config or {}
        self.decision_history: List[AutomatedDecision] = []
        self.performance_metrics: Dict[str, float] = {}
        
    @abstractmethod
    async def make_decision(
        self, 
        decision_type: DecisionType,
        context: AutomationContext
    ) -> AutomatedDecision:
        """Make an automated decision based on context."""
        pass
    
    def learn_from_outcome(self, decision: AutomatedDecision, outcome: Dict[str, Any]) -> None:
        """Learn from decision outcomes to improve future decisions."""
        # Update performance metrics based on outcome
        success_rate = outcome.get("success", False)
        
        if "success_rate" not in self.performance_metrics:
            self.performance_metrics["success_rate"] = 0.0
            self.performance_metrics["decision_count"] = 0
        
        count = self.performance_metrics["decision_count"]
        current_rate = self.performance_metrics["success_rate"]
        
        # Update running average
        new_rate = (current_rate * count + (1.0 if success_rate else 0.0)) / (count + 1)
        self.performance_metrics["success_rate"] = new_rate
        self.performance_metrics["decision_count"] = count + 1
        
        # Store decision with outcome for learning
        decision.metadata["outcome"] = outcome
        self.decision_history.append(decision)
        
        # Keep only last 1000 decisions for memory efficiency
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]


class IntelligentProcessingDecisionEngine(BaseDecisionEngine):
    """AI-driven decision engine for processing path optimization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("intelligent_processing", config)
        self.model_manager: Optional[MLModelManager] = None
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.8)
        
    def set_model_manager(self, model_manager: MLModelManager) -> None:
        """Set ML model manager for intelligent decisions."""
        self.model_manager = model_manager
        
    @trace_operation("intelligent_decision_making")
    async def make_decision(
        self, 
        decision_type: DecisionType,
        context: AutomationContext
    ) -> AutomatedDecision:
        """Make intelligent processing decisions."""
        
        if decision_type == DecisionType.PROCESSING_PATH:
            return await self._decide_processing_path(context)
        elif decision_type == DecisionType.QUALITY_THRESHOLD:
            return await self._decide_quality_threshold(context)
        elif decision_type == DecisionType.RESOURCE_ALLOCATION:
            return await self._decide_resource_allocation(context)
        else:
            # Default decision
            return AutomatedDecision(
                decision_type=decision_type,
                decision_value="default",
                confidence=0.5,
                reasoning=["Default fallback decision"],
                alternative_options=[],
                risk_assessment={"low": 0.8, "medium": 0.2, "high": 0.0},
                execution_plan=[],
                metadata={}
            )
    
    async def _decide_processing_path(self, context: AutomationContext) -> AutomatedDecision:
        """Decide optimal processing path based on document characteristics."""
        doc_metadata = context.document_metadata
        
        # Analyze document characteristics
        document_size = doc_metadata.get("size_bytes", 0)
        document_type = doc_metadata.get("type", "unknown")
        complexity_score = doc_metadata.get("complexity_score", 0.5)
        
        # Historical performance analysis
        historical_performance = self._analyze_historical_performance(context)
        
        # AI-driven path selection
        processing_paths = {
            "fast_track": {
                "conditions": lambda: document_size < 50000 and complexity_score < 0.3,
                "confidence": 0.9,
                "reasoning": "Small document with low complexity suitable for fast processing"
            },
            "standard": {
                "conditions": lambda: document_size < 500000 and complexity_score < 0.7,
                "confidence": 0.8,
                "reasoning": "Medium-sized document requires standard processing pipeline"
            },
            "intensive": {
                "conditions": lambda: document_size >= 500000 or complexity_score >= 0.7,
                "confidence": 0.85,
                "reasoning": "Large or complex document requires intensive processing with ML models"
            },
            "adaptive": {
                "conditions": lambda: True,  # Always available
                "confidence": 0.7,
                "reasoning": "Adaptive processing path with real-time optimization"
            }
        }
        
        # Select best path based on conditions and historical performance
        selected_path = "standard"  # Default
        max_confidence = 0.0
        reasoning_list = []
        
        for path_name, path_config in processing_paths.items():
            if path_config["conditions"]():
                historical_success = historical_performance.get(path_name, {}).get("success_rate", 0.5)
                adjusted_confidence = path_config["confidence"] * (0.5 + 0.5 * historical_success)
                
                if adjusted_confidence > max_confidence:
                    max_confidence = adjusted_confidence
                    selected_path = path_name
                    reasoning_list = [
                        path_config["reasoning"],
                        f"Historical success rate: {historical_success:.2f}",
                        f"Adjusted confidence: {adjusted_confidence:.2f}"
                    ]
        
        # Generate alternative options
        alternatives = [
            {
                "path": path_name,
                "confidence": config["confidence"],
                "description": config["reasoning"]
            }
            for path_name, config in processing_paths.items()
            if path_name != selected_path and config["conditions"]()
        ]
        
        # Risk assessment
        risk_assessment = self._assess_processing_risk(selected_path, context)
        
        # Execution plan
        execution_plan = self._generate_execution_plan(selected_path, context)
        
        return AutomatedDecision(
            decision_type=DecisionType.PROCESSING_PATH,
            decision_value=selected_path,
            confidence=max_confidence,
            reasoning=reasoning_list,
            alternative_options=alternatives,
            risk_assessment=risk_assessment,
            execution_plan=execution_plan,
            metadata={
                "document_size": document_size,
                "document_type": document_type,
                "complexity_score": complexity_score,
                "selection_method": "ai_driven_optimization"
            }
        )
    
    async def _decide_quality_threshold(self, context: AutomationContext) -> AutomatedDecision:
        """Intelligently adjust quality thresholds based on context."""
        compliance_reqs = context.compliance_requirements
        system_metrics = context.system_metrics
        
        # Base thresholds
        base_phi_confidence = 0.8
        base_compliance_score = 0.9
        
        # Adaptive adjustments
        phi_threshold = base_phi_confidence
        compliance_threshold = base_compliance_score
        
        reasoning = []
        
        # Adjust based on compliance level
        compliance_level = compliance_reqs.get("level", "standard")
        if compliance_level == "strict":
            phi_threshold += 0.1
            compliance_threshold += 0.05
            reasoning.append("Strict compliance mode: raised thresholds")
        elif compliance_level == "minimal":
            phi_threshold -= 0.1
            compliance_threshold -= 0.1
            reasoning.append("Minimal compliance mode: lowered thresholds")
        
        # Adjust based on system performance
        cpu_usage = system_metrics.get("cpu_usage", 0.5)
        memory_usage = system_metrics.get("memory_usage", 0.5)
        
        if cpu_usage > 0.8 or memory_usage > 0.8:
            phi_threshold -= 0.05
            compliance_threshold -= 0.03
            reasoning.append("High system load: slightly lowered thresholds")
        
        # Adjust based on error rates
        recent_error_rate = system_metrics.get("recent_error_rate", 0.0)
        if recent_error_rate > 0.1:
            phi_threshold += 0.05
            compliance_threshold += 0.02
            reasoning.append("High error rate detected: increased thresholds for safety")
        
        # Ensure thresholds stay within reasonable bounds
        phi_threshold = max(0.5, min(0.95, phi_threshold))
        compliance_threshold = max(0.7, min(0.98, compliance_threshold))
        
        thresholds = {
            "phi_confidence_threshold": phi_threshold,
            "compliance_score_threshold": compliance_threshold
        }
        
        return AutomatedDecision(
            decision_type=DecisionType.QUALITY_THRESHOLD,
            decision_value=thresholds,
            confidence=0.85,
            reasoning=reasoning,
            alternative_options=[
                {"thresholds": {"phi_confidence_threshold": base_phi_confidence, 
                               "compliance_score_threshold": base_compliance_score},
                 "description": "Default baseline thresholds"}
            ],
            risk_assessment={"threshold_too_low": 0.2, "threshold_too_high": 0.3, "optimal": 0.5},
            execution_plan=[
                {"action": "update_phi_detector_threshold", "value": phi_threshold},
                {"action": "update_compliance_threshold", "value": compliance_threshold}
            ],
            metadata={
                "base_phi_threshold": base_phi_confidence,
                "base_compliance_threshold": base_compliance_score,
                "adjustment_factors": {
                    "compliance_level": compliance_level,
                    "system_load": {"cpu": cpu_usage, "memory": memory_usage},
                    "error_rate": recent_error_rate
                }
            }
        )
    
    async def _decide_resource_allocation(self, context: AutomationContext) -> AutomatedDecision:
        """Intelligently allocate computational resources."""
        system_metrics = context.system_metrics
        doc_metadata = context.document_metadata
        
        # Current resource utilization
        cpu_usage = system_metrics.get("cpu_usage", 0.5)
        memory_usage = system_metrics.get("memory_usage", 0.5)
        queue_length = system_metrics.get("processing_queue_length", 0)
        
        # Document characteristics affecting resource needs
        document_size = doc_metadata.get("size_bytes", 0)
        estimated_complexity = doc_metadata.get("complexity_score", 0.5)
        
        # Base resource allocation
        base_cpu_cores = 2
        base_memory_gb = 4
        base_timeout = 300
        
        # Intelligent scaling decisions
        cpu_cores = base_cpu_cores
        memory_gb = base_memory_gb
        timeout_seconds = base_timeout
        
        reasoning = []
        
        # Scale up for large documents
        if document_size > 1000000:  # >1MB
            cpu_cores = min(8, cpu_cores * 2)
            memory_gb = min(16, memory_gb * 1.5)
            timeout_seconds = int(timeout_seconds * 1.5)
            reasoning.append("Large document detected: scaled up resources")
        
        # Scale up for complex processing
        if estimated_complexity > 0.7:
            cpu_cores = min(8, cpu_cores + 1)
            memory_gb = min(16, memory_gb + 2)
            reasoning.append("High complexity processing: added extra resources")
        
        # Scale down if system is under high load
        if cpu_usage > 0.8:
            cpu_cores = max(1, cpu_cores - 1)
            reasoning.append("High CPU usage: reduced core allocation")
        
        if memory_usage > 0.8:
            memory_gb = max(2, memory_gb - 1)
            reasoning.append("High memory usage: reduced memory allocation")
        
        # Priority adjustment based on queue length
        priority = "normal"
        if queue_length > 100:
            priority = "low"
            timeout_seconds = int(timeout_seconds * 0.8)
            reasoning.append("Long processing queue: reduced priority and timeout")
        elif queue_length < 10:
            priority = "high"
            reasoning.append("Short processing queue: increased priority")
        
        allocation = {
            "cpu_cores": cpu_cores,
            "memory_gb": memory_gb,
            "timeout_seconds": timeout_seconds,
            "priority": priority,
            "max_concurrent_operations": min(4, cpu_cores)
        }
        
        return AutomatedDecision(
            decision_type=DecisionType.RESOURCE_ALLOCATION,
            decision_value=allocation,
            confidence=0.8,
            reasoning=reasoning,
            alternative_options=[
                {
                    "allocation": {
                        "cpu_cores": base_cpu_cores,
                        "memory_gb": base_memory_gb,
                        "timeout_seconds": base_timeout,
                        "priority": "normal"
                    },
                    "description": "Default resource allocation"
                }
            ],
            risk_assessment={"under_provisioned": 0.3, "over_provisioned": 0.2, "optimal": 0.5},
            execution_plan=[
                {"action": "allocate_cpu_cores", "value": cpu_cores},
                {"action": "allocate_memory", "value": f"{memory_gb}GB"},
                {"action": "set_timeout", "value": timeout_seconds},
                {"action": "set_priority", "value": priority}
            ],
            metadata={
                "system_state": {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "queue_length": queue_length
                },
                "document_factors": {
                    "size_bytes": document_size,
                    "complexity": estimated_complexity
                }
            }
        )
    
    def _analyze_historical_performance(self, context: AutomationContext) -> Dict[str, Dict[str, float]]:
        """Analyze historical performance of different processing paths."""
        processing_history = context.processing_history
        
        performance_by_path = {}
        
        for record in processing_history[-100:]:  # Last 100 records
            path = record.get("processing_path", "unknown")
            success = record.get("success", False)
            processing_time = record.get("processing_time_ms", 0)
            
            if path not in performance_by_path:
                performance_by_path[path] = {
                    "success_count": 0,
                    "total_count": 0,
                    "total_time": 0,
                    "success_rate": 0.0,
                    "avg_time": 0.0
                }
            
            stats = performance_by_path[path]
            stats["total_count"] += 1
            stats["total_time"] += processing_time
            
            if success:
                stats["success_count"] += 1
            
            # Update rates
            stats["success_rate"] = stats["success_count"] / stats["total_count"]
            stats["avg_time"] = stats["total_time"] / stats["total_count"]
        
        return performance_by_path
    
    def _assess_processing_risk(self, processing_path: str, context: AutomationContext) -> Dict[str, float]:
        """Assess risk levels for chosen processing path."""
        doc_metadata = context.document_metadata
        document_size = doc_metadata.get("size_bytes", 0)
        complexity = doc_metadata.get("complexity_score", 0.5)
        
        # Risk factors
        size_risk = min(1.0, document_size / 10000000)  # Risk increases with size
        complexity_risk = complexity
        path_risk = {
            "fast_track": 0.1,
            "standard": 0.2,
            "intensive": 0.4,
            "adaptive": 0.3
        }.get(processing_path, 0.5)
        
        # Combined risk assessment
        overall_risk = (size_risk + complexity_risk + path_risk) / 3
        
        return {
            "low": max(0, 1 - overall_risk * 1.5),
            "medium": min(1, overall_risk * 2) if overall_risk < 0.7 else max(0, 1 - overall_risk),
            "high": max(0, overall_risk - 0.3) if overall_risk > 0.3 else 0
        }
    
    def _generate_execution_plan(self, processing_path: str, context: AutomationContext) -> List[Dict[str, Any]]:
        """Generate detailed execution plan for processing path."""
        base_steps = [
            {"step": "initialize_processors", "estimated_time_ms": 100},
            {"step": "load_document", "estimated_time_ms": 500},
            {"step": "preprocess_content", "estimated_time_ms": 1000}
        ]
        
        path_specific_steps = {
            "fast_track": [
                {"step": "quick_phi_scan", "estimated_time_ms": 2000},
                {"step": "basic_redaction", "estimated_time_ms": 1000},
                {"step": "simple_compliance_check", "estimated_time_ms": 500}
            ],
            "standard": [
                {"step": "comprehensive_phi_detection", "estimated_time_ms": 5000},
                {"step": "contextual_redaction", "estimated_time_ms": 3000},
                {"step": "compliance_analysis", "estimated_time_ms": 2000},
                {"step": "quality_assurance", "estimated_time_ms": 1500}
            ],
            "intensive": [
                {"step": "ml_phi_detection", "estimated_time_ms": 10000},
                {"step": "advanced_redaction", "estimated_time_ms": 5000},
                {"step": "comprehensive_compliance_analysis", "estimated_time_ms": 4000},
                {"step": "multi_model_validation", "estimated_time_ms": 3000},
                {"step": "detailed_quality_assurance", "estimated_time_ms": 2000}
            ],
            "adaptive": [
                {"step": "dynamic_path_selection", "estimated_time_ms": 200},
                {"step": "adaptive_processing", "estimated_time_ms": 8000},
                {"step": "real_time_optimization", "estimated_time_ms": 1000},
                {"step": "adaptive_quality_control", "estimated_time_ms": 1500}
            ]
        }
        
        path_steps = path_specific_steps.get(processing_path, path_specific_steps["standard"])
        
        return base_steps + path_steps + [
            {"step": "generate_summary", "estimated_time_ms": 1000},
            {"step": "finalize_output", "estimated_time_ms": 500}
        ]


class AdaptiveWorkflowEngine:
    """Engine for executing and adapting workflows based on performance."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.workflows: Dict[str, AdaptiveWorkflow] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.decision_engine: Optional[BaseDecisionEngine] = None
        self.performance_optimizer = PerformanceOptimizer()
        
    def register_decision_engine(self, engine: BaseDecisionEngine) -> None:
        """Register decision engine for workflow adaptation."""
        self.decision_engine = engine
        
    def register_workflow(self, workflow: AdaptiveWorkflow) -> None:
        """Register an adaptive workflow."""
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered adaptive workflow: {workflow.workflow_id}")
    
    @trace_operation("adaptive_workflow_execution")
    async def execute_workflow(
        self, 
        workflow_id: str, 
        context: AutomationContext,
        input_data: Any
    ) -> Dict[str, Any]:
        """Execute adaptive workflow with intelligent decision making."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        start_time = time.perf_counter()
        
        execution_log = {
            "workflow_id": workflow_id,
            "start_time": datetime.utcnow().isoformat(),
            "steps_completed": [],
            "steps_failed": [],
            "decisions_made": [],
            "adaptations_applied": [],
            "performance_metrics": {}
        }
        
        try:
            # Execute workflow steps with adaptation
            completed_steps = []
            current_data = input_data
            
            # Pre-execution optimization
            if self.decision_engine:
                optimization_decision = await self.decision_engine.make_decision(
                    DecisionType.OPTIMIZATION_STRATEGY,
                    context
                )
                execution_log["decisions_made"].append(asdict(optimization_decision))
            
            for step in workflow.steps:
                if not step.is_ready_to_execute(completed_steps):
                    continue
                
                # Check if step should be adapted based on current context
                adapted_step = await self._adapt_step_if_needed(step, context, execution_log)
                
                # Execute step with monitoring
                step_start = time.perf_counter()
                try:
                    step_result = await self._execute_workflow_step(
                        adapted_step, current_data, context
                    )
                    
                    step_duration = (time.perf_counter() - step_start) * 1000
                    
                    completed_steps.append(step.step_id)
                    execution_log["steps_completed"].append({
                        "step_id": step.step_id,
                        "duration_ms": step_duration,
                        "result_size": len(str(step_result)),
                        "adaptations": step != adapted_step
                    })
                    
                    current_data = step_result  # Pass result to next step
                    
                except Exception as e:
                    logger.error(f"Step {step.step_id} failed: {e}")
                    execution_log["steps_failed"].append({
                        "step_id": step.step_id,
                        "error": str(e),
                        "duration_ms": (time.perf_counter() - step_start) * 1000
                    })
                    
                    # Apply failure handling
                    if not await self._handle_step_failure(step, e, context, execution_log):
                        raise  # Re-raise if can't recover
            
            total_duration = (time.perf_counter() - start_time) * 1000
            
            # Post-execution analysis and learning
            execution_log["end_time"] = datetime.utcnow().isoformat()
            execution_log["total_duration_ms"] = total_duration
            execution_log["success"] = len(execution_log["steps_failed"]) == 0
            execution_log["performance_metrics"] = await self._calculate_workflow_metrics(execution_log)
            
            # Learn from execution for future adaptations
            await self._learn_from_execution(workflow_id, execution_log, context)
            
            # Store execution history
            self.execution_history.append(execution_log)
            
            return {
                "result": current_data,
                "execution_log": execution_log,
                "success": execution_log["success"]
            }
            
        except Exception as e:
            execution_log["end_time"] = datetime.utcnow().isoformat()
            execution_log["total_duration_ms"] = (time.perf_counter() - start_time) * 1000
            execution_log["success"] = False
            execution_log["error"] = str(e)
            
            self.execution_history.append(execution_log)
            
            logger.error(f"Workflow {workflow_id} execution failed: {e}")
            raise
    
    async def _adapt_step_if_needed(
        self, 
        step: WorkflowStep, 
        context: AutomationContext,
        execution_log: Dict[str, Any]
    ) -> WorkflowStep:
        """Adapt workflow step based on current context and performance history."""
        
        # Check adaptation rules
        workflow = self._get_workflow_by_step(step.step_id)
        if not workflow:
            return step
        
        adaptation_needed = False
        adaptations = []
        
        # Analyze performance history for this step
        step_performance = self._get_step_performance_history(step.step_id)
        
        # Rule 1: Timeout adaptation based on historical performance
        if step_performance.get("avg_duration_ms", 0) > step.timeout_seconds * 1000 * 0.8:
            new_timeout = int(step.timeout_seconds * 1.5)
            adaptations.append(f"Increased timeout from {step.timeout_seconds}s to {new_timeout}s")
            step.timeout_seconds = new_timeout
            adaptation_needed = True
        
        # Rule 2: Configuration adaptation based on system load
        system_metrics = context.system_metrics
        if system_metrics.get("cpu_usage", 0) > 0.8:
            # Reduce computational complexity
            if "batch_size" in step.configuration:
                old_batch_size = step.configuration["batch_size"]
                new_batch_size = max(1, old_batch_size // 2)
                step.configuration["batch_size"] = new_batch_size
                adaptations.append(f"Reduced batch size from {old_batch_size} to {new_batch_size}")
                adaptation_needed = True
        
        # Rule 3: Quality vs Speed trade-off adaptation
        if step_performance.get("failure_rate", 0) > 0.1:
            # Increase quality measures
            if "quality_checks" in step.configuration:
                step.configuration["quality_checks"] = True
                adaptations.append("Enabled additional quality checks due to high failure rate")
                adaptation_needed = True
        
        if adaptation_needed:
            execution_log["adaptations_applied"].append({
                "step_id": step.step_id,
                "adaptations": adaptations,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"Adapted step {step.step_id}: {', '.join(adaptations)}")
        
        return step
    
    async def _execute_workflow_step(
        self, 
        step: WorkflowStep, 
        input_data: Any, 
        context: AutomationContext
    ) -> Any:
        """Execute individual workflow step."""
        
        # Simulate step execution based on step type
        step_type = step.step_type
        
        if step_type == "phi_detection":
            # Simulate PHI detection
            await asyncio.sleep(0.1)  # Simulate processing time
            return {
                "phi_entities": [
                    {"text": "John Doe", "type": "name", "confidence": 0.95},
                    {"text": "123-45-6789", "type": "ssn", "confidence": 0.98}
                ],
                "processing_time_ms": 150
            }
        
        elif step_type == "compliance_analysis":
            # Simulate compliance analysis
            await asyncio.sleep(0.05)
            return {
                "compliance_score": 0.94,
                "violations": [],
                "recommendations": ["Consider additional review for high-risk content"]
            }
        
        elif step_type == "quality_assurance":
            # Simulate quality checks
            await asyncio.sleep(0.02)
            return {
                "quality_score": 0.92,
                "issues_found": 0,
                "validation_passed": True
            }
        
        else:
            # Generic step execution
            await asyncio.sleep(0.01)
            return {"step_completed": True, "output": input_data}
    
    async def _handle_step_failure(
        self, 
        step: WorkflowStep, 
        error: Exception, 
        context: AutomationContext,
        execution_log: Dict[str, Any]
    ) -> bool:
        """Handle step failure and attempt recovery."""
        
        retry_policy = step.retry_policy
        max_retries = retry_policy.get("max_retries", 3)
        retry_delay = retry_policy.get("delay_seconds", 5)
        
        # Attempt retries with exponential backoff
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(retry_delay * (2 ** attempt))
                logger.info(f"Retrying step {step.step_id}, attempt {attempt + 1}/{max_retries}")
                
                # Try executing step again (simplified)
                result = await self._execute_workflow_step(step, None, context)
                
                execution_log["adaptations_applied"].append({
                    "step_id": step.step_id,
                    "adaptation": f"Successfully recovered after {attempt + 1} retries",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                return True
                
            except Exception as retry_error:
                logger.warning(f"Retry {attempt + 1} failed for step {step.step_id}: {retry_error}")
                continue
        
        # All retries failed
        logger.error(f"Step {step.step_id} failed after {max_retries} retries")
        return False
    
    def _get_workflow_by_step(self, step_id: str) -> Optional[AdaptiveWorkflow]:
        """Find workflow containing the given step."""
        for workflow in self.workflows.values():
            if any(step.step_id == step_id for step in workflow.steps):
                return workflow
        return None
    
    def _get_step_performance_history(self, step_id: str) -> Dict[str, float]:
        """Get performance metrics for a specific step."""
        step_executions = []
        
        for execution in self.execution_history[-50:]:  # Last 50 executions
            for completed_step in execution.get("steps_completed", []):
                if completed_step.get("step_id") == step_id:
                    step_executions.append(completed_step)
            
            for failed_step in execution.get("steps_failed", []):
                if failed_step.get("step_id") == step_id:
                    step_executions.append({**failed_step, "failed": True})
        
        if not step_executions:
            return {}
        
        # Calculate metrics
        total_executions = len(step_executions)
        failed_executions = sum(1 for exec in step_executions if exec.get("failed", False))
        successful_executions = total_executions - failed_executions
        
        avg_duration = 0
        if successful_executions > 0:
            durations = [exec.get("duration_ms", 0) for exec in step_executions if not exec.get("failed", False)]
            avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "failure_rate": failed_executions / total_executions,
            "success_rate": successful_executions / total_executions,
            "avg_duration_ms": avg_duration
        }
    
    async def _calculate_workflow_metrics(self, execution_log: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive workflow performance metrics."""
        completed_steps = execution_log.get("steps_completed", [])
        failed_steps = execution_log.get("steps_failed", [])
        
        total_steps = len(completed_steps) + len(failed_steps)
        success_rate = len(completed_steps) / total_steps if total_steps > 0 else 0
        
        avg_step_duration = 0
        if completed_steps:
            durations = [step.get("duration_ms", 0) for step in completed_steps]
            avg_step_duration = sum(durations) / len(durations)
        
        return {
            "success_rate": success_rate,
            "total_steps": total_steps,
            "completed_steps": len(completed_steps),
            "failed_steps": len(failed_steps),
            "avg_step_duration_ms": avg_step_duration,
            "total_duration_ms": execution_log.get("total_duration_ms", 0),
            "adaptations_count": len(execution_log.get("adaptations_applied", []))
        }
    
    async def _learn_from_execution(
        self, 
        workflow_id: str, 
        execution_log: Dict[str, Any], 
        context: AutomationContext
    ) -> None:
        """Learn from workflow execution to improve future performance."""
        
        # Extract learning signals
        success = execution_log.get("success", False)
        duration = execution_log.get("total_duration_ms", 0)
        adaptations = len(execution_log.get("adaptations_applied", []))
        
        # Update workflow adaptation rules based on performance
        workflow = self.workflows.get(workflow_id)
        if workflow:
            # Learn about optimal timeouts
            for step_result in execution_log.get("steps_completed", []):
                step_id = step_result.get("step_id")
                duration_ms = step_result.get("duration_ms", 0)
                
                # Find corresponding step and update timeout if needed
                for step in workflow.steps:
                    if step.step_id == step_id:
                        optimal_timeout = int(duration_ms / 1000 * 1.2)  # 20% buffer
                        if optimal_timeout < step.timeout_seconds * 0.5:
                            # Timeout can be reduced
                            step.timeout_seconds = max(30, optimal_timeout)
        
        # Provide feedback to decision engine
        if self.decision_engine:
            outcome = {
                "success": success,
                "duration_ms": duration,
                "efficiency_score": 1.0 - (adaptations / 10.0),  # Fewer adaptations = more efficient
                "context": asdict(context)
            }
            
            # Learn from all decisions made during execution
            for decision_dict in execution_log.get("decisions_made", []):
                decision = AutomatedDecision(**decision_dict)
                self.decision_engine.learn_from_outcome(decision, outcome)


def create_default_hipaa_workflow() -> AdaptiveWorkflow:
    """Create default HIPAA compliance processing workflow."""
    
    steps = [
        WorkflowStep(
            step_id="document_validation",
            name="Document Validation",
            description="Validate input document format and security",
            step_type="validation",
            configuration={"strict_validation": True, "security_scan": True},
            dependencies=[],
            conditions={},
            retry_policy={"max_retries": 2, "delay_seconds": 1},
            timeout_seconds=60
        ),
        WorkflowStep(
            step_id="phi_detection",
            name="PHI Detection",
            description="Detect protected health information using ML models",
            step_type="phi_detection",
            configuration={"confidence_threshold": 0.8, "use_context": True},
            dependencies=["document_validation"],
            conditions={},
            retry_policy={"max_retries": 3, "delay_seconds": 5},
            timeout_seconds=300
        ),
        WorkflowStep(
            step_id="phi_redaction",
            name="PHI Redaction",
            description="Redact identified PHI based on compliance requirements",
            step_type="redaction",
            configuration={"redaction_method": "contextual", "preserve_meaning": True},
            dependencies=["phi_detection"],
            conditions={},
            retry_policy={"max_retries": 2, "delay_seconds": 3},
            timeout_seconds=180
        ),
        WorkflowStep(
            step_id="compliance_analysis",
            name="Compliance Analysis",
            description="Analyze document for HIPAA compliance",
            step_type="compliance_analysis",
            configuration={"compliance_level": "standard", "generate_report": True},
            dependencies=["phi_redaction"],
            conditions={},
            retry_policy={"max_retries": 2, "delay_seconds": 2},
            timeout_seconds=120
        ),
        WorkflowStep(
            step_id="quality_assurance",
            name="Quality Assurance",
            description="Perform final quality checks on processed document",
            step_type="quality_assurance",
            configuration={"comprehensive_check": True, "audit_trail": True},
            dependencies=["compliance_analysis"],
            conditions={},
            retry_policy={"max_retries": 1, "delay_seconds": 1},
            timeout_seconds=60
        )
    ]
    
    adaptation_rules = [
        {
            "condition": "high_failure_rate",
            "threshold": 0.1,
            "action": "increase_quality_checks",
            "description": "Enable additional quality checks when failure rate exceeds 10%"
        },
        {
            "condition": "high_processing_time",
            "threshold": 600000,  # 10 minutes in ms
            "action": "reduce_batch_size",
            "description": "Reduce batch size when processing time exceeds 10 minutes"
        },
        {
            "condition": "low_confidence_scores",
            "threshold": 0.7,
            "action": "increase_confidence_threshold",
            "description": "Raise confidence thresholds when average scores are low"
        }
    ]
    
    return AdaptiveWorkflow(
        workflow_id="default_hipaa_processing",
        name="Default HIPAA Compliance Processing",
        description="Standard workflow for HIPAA-compliant document processing with adaptive optimization",
        steps=steps,
        adaptation_rules=adaptation_rules,
        success_criteria={
            "min_success_rate": 0.95,
            "max_processing_time_ms": 600000,
            "min_compliance_score": 0.9
        },
        failure_handling={
            "max_retries": 3,
            "escalation_threshold": 0.1,
            "fallback_workflow": "simple_hipaa_processing"
        },
        performance_targets={
            "target_throughput_per_hour": 100,
            "target_success_rate": 0.98,
            "target_avg_processing_time_ms": 45000
        }
    )


async def initialize_intelligent_automation(config: Dict[str, Any] = None) -> Tuple[IntelligentProcessingDecisionEngine, AdaptiveWorkflowEngine]:
    """Initialize intelligent automation system."""
    
    # Initialize decision engine
    decision_engine = IntelligentProcessingDecisionEngine(
        config.get("decision_engine", {}) if config else {}
    )
    
    # Initialize workflow engine
    workflow_engine = AdaptiveWorkflowEngine(
        config.get("workflow_engine", {}) if config else {}
    )
    
    # Register decision engine with workflow engine
    workflow_engine.register_decision_engine(decision_engine)
    
    # Register default workflow
    default_workflow = create_default_hipaa_workflow()
    workflow_engine.register_workflow(default_workflow)
    
    logger.info("Intelligent automation system initialized successfully")
    logger.info(f"Registered workflows: {list(workflow_engine.workflows.keys())}")
    
    return decision_engine, workflow_engine