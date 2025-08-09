"""Comprehensive tests for intelligent automation components."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.hipaa_compliance_summarizer.intelligent_automation import (
    IntelligentProcessingDecisionEngine,
    AdaptiveWorkflowEngine,
    AutomationContext,
    AutomatedDecision,
    WorkflowStep,
    AdaptiveWorkflow,
    DecisionType,
    AutomationLevel,
    create_default_hipaa_workflow,
    initialize_intelligent_automation
)


@pytest.fixture
def automation_context():
    """Create automation context for testing."""
    return AutomationContext(
        document_metadata={
            "size_bytes": 25000,
            "type": "clinical_note",
            "complexity_score": 0.4
        },
        processing_history=[
            {
                "processing_path": "standard",
                "success": True,
                "processing_time_ms": 5000
            },
            {
                "processing_path": "fast_track", 
                "success": False,
                "processing_time_ms": 1500
            }
        ],
        system_metrics={
            "cpu_usage": 0.6,
            "memory_usage": 0.4,
            "processing_queue_length": 15,
            "recent_error_rate": 0.05
        },
        user_preferences={
            "quality_over_speed": True,
            "compliance_level": "strict"
        },
        compliance_requirements={
            "level": "strict",
            "frameworks": ["HIPAA"]
        }
    )


@pytest.fixture
def decision_engine():
    """Create decision engine for testing."""
    config = {
        "learning_rate": 0.1,
        "confidence_threshold": 0.8
    }
    return IntelligentProcessingDecisionEngine(config)


@pytest.fixture
def workflow_engine():
    """Create workflow engine for testing."""
    return AdaptiveWorkflowEngine()


class TestIntelligentProcessingDecisionEngine:
    """Test cases for Intelligent Processing Decision Engine."""
    
    def test_initialization(self, decision_engine):
        """Test decision engine initialization."""
        assert decision_engine.engine_name == "intelligent_processing"
        assert decision_engine.learning_rate == 0.1
        assert decision_engine.confidence_threshold == 0.8
        assert len(decision_engine.decision_history) == 0
        assert len(decision_engine.performance_metrics) == 0
    
    @pytest.mark.asyncio
    async def test_decide_processing_path(self, decision_engine, automation_context):
        """Test processing path decision making."""
        decision = await decision_engine.make_decision(
            DecisionType.PROCESSING_PATH,
            automation_context
        )
        
        assert isinstance(decision, AutomatedDecision)
        assert decision.decision_type == DecisionType.PROCESSING_PATH
        assert decision.decision_value in ["fast_track", "standard", "intensive", "adaptive"]
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.reasoning) > 0
        assert isinstance(decision.alternative_options, list)
        assert isinstance(decision.risk_assessment, dict)
        assert isinstance(decision.execution_plan, list)
    
    @pytest.mark.asyncio
    async def test_decide_processing_path_small_document(self, decision_engine):
        """Test processing path decision for small document."""
        context = AutomationContext(
            document_metadata={
                "size_bytes": 5000,  # Small document
                "type": "clinical_note",
                "complexity_score": 0.2  # Low complexity
            },
            processing_history=[],
            system_metrics={"cpu_usage": 0.3, "memory_usage": 0.2},
            user_preferences={},
            compliance_requirements={"level": "standard"}
        )
        
        decision = await decision_engine.make_decision(
            DecisionType.PROCESSING_PATH,
            context
        )
        
        # Should choose fast_track for small, simple document
        assert decision.decision_value == "fast_track"
        assert decision.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_decide_processing_path_large_document(self, decision_engine):
        """Test processing path decision for large document."""
        context = AutomationContext(
            document_metadata={
                "size_bytes": 1000000,  # Large document
                "type": "medical_record",
                "complexity_score": 0.8  # High complexity
            },
            processing_history=[],
            system_metrics={"cpu_usage": 0.4, "memory_usage": 0.3},
            user_preferences={},
            compliance_requirements={"level": "standard"}
        )
        
        decision = await decision_engine.make_decision(
            DecisionType.PROCESSING_PATH,
            context
        )
        
        # Should choose intensive for large, complex document
        assert decision.decision_value == "intensive"
    
    @pytest.mark.asyncio
    async def test_decide_quality_threshold(self, decision_engine, automation_context):
        """Test quality threshold decision making."""
        decision = await decision_engine.make_decision(
            DecisionType.QUALITY_THRESHOLD,
            automation_context
        )
        
        assert decision.decision_type == DecisionType.QUALITY_THRESHOLD
        assert isinstance(decision.decision_value, dict)
        assert "phi_confidence_threshold" in decision.decision_value
        assert "compliance_score_threshold" in decision.decision_value
        
        # Values should be within reasonable bounds
        phi_threshold = decision.decision_value["phi_confidence_threshold"]
        compliance_threshold = decision.decision_value["compliance_score_threshold"]
        
        assert 0.5 <= phi_threshold <= 0.95
        assert 0.7 <= compliance_threshold <= 0.98
    
    @pytest.mark.asyncio
    async def test_decide_quality_threshold_strict_compliance(self, decision_engine):
        """Test quality threshold decision with strict compliance."""
        context = AutomationContext(
            document_metadata={},
            processing_history=[],
            system_metrics={"cpu_usage": 0.3, "memory_usage": 0.2},
            user_preferences={},
            compliance_requirements={"level": "strict"}
        )
        
        decision = await decision_engine.make_decision(
            DecisionType.QUALITY_THRESHOLD,
            context
        )
        
        # Strict compliance should raise thresholds
        phi_threshold = decision.decision_value["phi_confidence_threshold"]
        compliance_threshold = decision.decision_value["compliance_score_threshold"]
        
        assert phi_threshold > 0.8  # Should be higher for strict mode
        assert compliance_threshold > 0.9
    
    @pytest.mark.asyncio
    async def test_decide_resource_allocation(self, decision_engine, automation_context):
        """Test resource allocation decision making."""
        decision = await decision_engine.make_decision(
            DecisionType.RESOURCE_ALLOCATION,
            automation_context
        )
        
        assert decision.decision_type == DecisionType.RESOURCE_ALLOCATION
        assert isinstance(decision.decision_value, dict)
        
        allocation = decision.decision_value
        assert "cpu_cores" in allocation
        assert "memory_gb" in allocation
        assert "timeout_seconds" in allocation
        assert "priority" in allocation
        assert "max_concurrent_operations" in allocation
        
        # Values should be reasonable
        assert allocation["cpu_cores"] >= 1
        assert allocation["memory_gb"] >= 2
        assert allocation["timeout_seconds"] > 0
        assert allocation["priority"] in ["low", "normal", "high"]
    
    @pytest.mark.asyncio
    async def test_decide_resource_allocation_high_load(self, decision_engine):
        """Test resource allocation under high system load."""
        context = AutomationContext(
            document_metadata={"size_bytes": 50000},
            processing_history=[],
            system_metrics={
                "cpu_usage": 0.9,  # High CPU usage
                "memory_usage": 0.85,  # High memory usage
                "processing_queue_length": 150  # Long queue
            },
            user_preferences={},
            compliance_requirements={}
        )
        
        decision = await decision_engine.make_decision(
            DecisionType.RESOURCE_ALLOCATION,
            context
        )
        
        allocation = decision.decision_value
        
        # Should allocate fewer resources under high load
        assert allocation["cpu_cores"] <= 2  # Conservative CPU allocation
        assert allocation["memory_gb"] <= 4  # Conservative memory allocation
        assert allocation["priority"] == "low"  # Lower priority due to long queue
    
    def test_analyze_historical_performance(self, decision_engine, automation_context):
        """Test historical performance analysis."""
        performance = decision_engine._analyze_historical_performance(automation_context)
        
        assert isinstance(performance, dict)
        
        # Should have entries for paths in history
        assert "standard" in performance
        assert "fast_track" in performance
        
        # Check performance metrics structure
        standard_perf = performance["standard"]
        assert "success_rate" in standard_perf
        assert "avg_time" in standard_perf
        assert standard_perf["success_rate"] == 1.0  # From test data
        
        fast_track_perf = performance["fast_track"]
        assert fast_track_perf["success_rate"] == 0.0  # From test data (failed)
    
    def test_assess_processing_risk(self, decision_engine, automation_context):
        """Test processing risk assessment."""
        risk = decision_engine._assess_processing_risk("standard", automation_context)
        
        assert isinstance(risk, dict)
        assert "low" in risk
        assert "medium" in risk
        assert "high" in risk
        
        # Risk probabilities should sum to approximately 1.0
        total_risk = sum(risk.values())
        assert 0.9 <= total_risk <= 1.1
        
        # All risk values should be non-negative
        assert all(r >= 0 for r in risk.values())
    
    def test_generate_execution_plan(self, decision_engine, automation_context):
        """Test execution plan generation."""
        plan = decision_engine._generate_execution_plan("standard", automation_context)
        
        assert isinstance(plan, list)
        assert len(plan) > 0
        
        # Check plan structure
        for step in plan:
            assert isinstance(step, dict)
            assert "step" in step
            assert "estimated_time_ms" in step
            assert step["estimated_time_ms"] > 0
        
        # Should have common steps
        step_names = [step["step"] for step in plan]
        assert "initialize_processors" in step_names
        assert "load_document" in step_names
        assert "finalize_output" in step_names
    
    def test_learn_from_outcome(self, decision_engine):
        """Test learning from decision outcomes."""
        # Create a decision
        decision = AutomatedDecision(
            decision_type=DecisionType.PROCESSING_PATH,
            decision_value="standard",
            confidence=0.8,
            reasoning=["Test decision"],
            alternative_options=[],
            risk_assessment={"low": 0.7, "medium": 0.3, "high": 0.0},
            execution_plan=[],
            metadata={}
        )
        
        # Test successful outcome
        outcome = {"success": True, "processing_time_ms": 5000}
        decision_engine.learn_from_outcome(decision, outcome)
        
        # Check that metrics were updated
        assert "success_rate" in decision_engine.performance_metrics
        assert "decision_count" in decision_engine.performance_metrics
        assert decision_engine.performance_metrics["success_rate"] == 1.0
        assert decision_engine.performance_metrics["decision_count"] == 1
        
        # Test failed outcome
        failed_outcome = {"success": False, "error": "Processing failed"}
        decision_engine.learn_from_outcome(decision, failed_outcome)
        
        # Success rate should decrease
        assert decision_engine.performance_metrics["success_rate"] == 0.5  # 1 success, 1 failure
        assert decision_engine.performance_metrics["decision_count"] == 2


class TestAdaptiveWorkflowEngine:
    """Test cases for Adaptive Workflow Engine."""
    
    def test_initialization(self, workflow_engine):
        """Test workflow engine initialization."""
        assert isinstance(workflow_engine.workflows, dict)
        assert len(workflow_engine.workflows) == 0
        assert isinstance(workflow_engine.execution_history, list)
        assert len(workflow_engine.execution_history) == 0
    
    def test_register_workflow(self, workflow_engine):
        """Test workflow registration."""
        workflow = create_default_hipaa_workflow()
        workflow_engine.register_workflow(workflow)
        
        assert workflow.workflow_id in workflow_engine.workflows
        assert workflow_engine.workflows[workflow.workflow_id] == workflow
    
    def test_register_decision_engine(self, workflow_engine, decision_engine):
        """Test decision engine registration."""
        workflow_engine.register_decision_engine(decision_engine)
        assert workflow_engine.decision_engine == decision_engine
    
    @pytest.mark.asyncio
    async def test_execute_workflow_simple(self, workflow_engine, automation_context):
        """Test simple workflow execution."""
        # Create a simple workflow for testing
        steps = [
            WorkflowStep(
                step_id="test_step_1",
                name="Test Step 1",
                description="First test step",
                step_type="validation",
                configuration={},
                dependencies=[],
                conditions={},
                retry_policy={"max_retries": 1, "delay_seconds": 1},
                timeout_seconds=30
            ),
            WorkflowStep(
                step_id="test_step_2",
                name="Test Step 2", 
                description="Second test step",
                step_type="processing",
                configuration={},
                dependencies=["test_step_1"],
                conditions={},
                retry_policy={"max_retries": 1, "delay_seconds": 1},
                timeout_seconds=30
            )
        ]
        
        workflow = AdaptiveWorkflow(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="Simple test workflow",
            steps=steps,
            adaptation_rules=[],
            success_criteria={},
            failure_handling={},
            performance_targets={}
        )
        
        workflow_engine.register_workflow(workflow)
        
        input_data = {"test_input": "test_value"}
        result = await workflow_engine.execute_workflow(
            "test_workflow",
            automation_context,
            input_data
        )
        
        assert isinstance(result, dict)
        assert "result" in result
        assert "execution_log" in result
        assert "success" in result
        
        execution_log = result["execution_log"]
        assert execution_log["workflow_id"] == "test_workflow"
        assert len(execution_log["steps_completed"]) == 2  # Both steps should complete
        assert execution_log["success"] is True
    
    @pytest.mark.asyncio
    async def test_execute_workflow_with_decision_engine(self, workflow_engine, decision_engine, automation_context):
        """Test workflow execution with decision engine."""
        workflow_engine.register_decision_engine(decision_engine)
        
        workflow = create_default_hipaa_workflow()
        workflow_engine.register_workflow(workflow)
        
        input_data = {"document": "test document content"}
        result = await workflow_engine.execute_workflow(
            workflow.workflow_id,
            automation_context,
            input_data
        )
        
        execution_log = result["execution_log"]
        
        # Should have made optimization decisions
        assert len(execution_log.get("decisions_made", [])) > 0
        
        # Check decision structure
        decision = execution_log["decisions_made"][0]
        assert "decision_type" in decision
        assert "decision_value" in decision
        assert "confidence" in decision
    
    @pytest.mark.asyncio
    async def test_step_adaptation(self, workflow_engine):
        """Test workflow step adaptation."""
        # Create workflow with step that needs adaptation
        step = WorkflowStep(
            step_id="adaptable_step",
            name="Adaptable Step",
            description="Step that can be adapted",
            step_type="phi_detection",
            configuration={"batch_size": 10, "quality_checks": False},
            dependencies=[],
            conditions={},
            retry_policy={"max_retries": 2, "delay_seconds": 5},
            timeout_seconds=60
        )
        
        # Create context that should trigger adaptation
        high_load_context = AutomationContext(
            document_metadata={},
            processing_history=[],
            system_metrics={"cpu_usage": 0.9, "memory_usage": 0.8},  # High load
            user_preferences={},
            compliance_requirements={}
        )
        
        execution_log = {"adaptations_applied": []}
        
        adapted_step = await workflow_engine._adapt_step_if_needed(
            step, high_load_context, execution_log
        )
        
        # Should have adapted the step
        assert len(execution_log["adaptations_applied"]) > 0
        
        # Batch size should be reduced for high load
        assert adapted_step.configuration["batch_size"] < step.configuration["batch_size"]
    
    @pytest.mark.asyncio
    async def test_step_failure_handling(self, workflow_engine):
        """Test step failure handling and recovery."""
        step = WorkflowStep(
            step_id="failing_step",
            name="Failing Step",
            description="Step that might fail",
            step_type="test_failure",
            configuration={},
            dependencies=[],
            conditions={},
            retry_policy={"max_retries": 2, "delay_seconds": 0.1},
            timeout_seconds=5
        )
        
        context = AutomationContext(
            document_metadata={},
            processing_history=[],
            system_metrics={},
            user_preferences={},
            compliance_requirements={}
        )
        
        execution_log = {"adaptations_applied": []}
        
        # Mock the step execution to fail initially
        with patch.object(workflow_engine, '_execute_workflow_step') as mock_execute:
            mock_execute.side_effect = [
                Exception("First attempt failed"),
                {"step_completed": True}  # Second attempt succeeds
            ]
            
            recovery_success = await workflow_engine._handle_step_failure(
                step, Exception("Test failure"), context, execution_log
            )
        
        assert recovery_success is True
        assert len(execution_log["adaptations_applied"]) > 0
    
    def test_get_step_performance_history(self, workflow_engine):
        """Test step performance history retrieval."""
        # Add some fake execution history
        workflow_engine.execution_history = [
            {
                "steps_completed": [
                    {"step_id": "test_step", "duration_ms": 1000},
                    {"step_id": "other_step", "duration_ms": 500}
                ],
                "steps_failed": []
            },
            {
                "steps_completed": [
                    {"step_id": "test_step", "duration_ms": 1200}
                ],
                "steps_failed": [
                    {"step_id": "test_step", "duration_ms": 800, "error": "Test error"}
                ]
            }
        ]
        
        performance = workflow_engine._get_step_performance_history("test_step")
        
        assert isinstance(performance, dict)
        assert "total_executions" in performance
        assert "successful_executions" in performance
        assert "failed_executions" in performance
        assert "failure_rate" in performance
        assert "success_rate" in performance
        assert "avg_duration_ms" in performance
        
        # Should have 3 executions total (2 successful, 1 failed)
        assert performance["total_executions"] == 3
        assert performance["successful_executions"] == 2
        assert performance["failed_executions"] == 1
        assert performance["failure_rate"] == 1/3
        assert performance["success_rate"] == 2/3
    
    @pytest.mark.asyncio
    async def test_calculate_workflow_metrics(self, workflow_engine):
        """Test workflow metrics calculation."""
        execution_log = {
            "steps_completed": [
                {"step_id": "step1", "duration_ms": 1000},
                {"step_id": "step2", "duration_ms": 1500},
                {"step_id": "step3", "duration_ms": 800}
            ],
            "steps_failed": [
                {"step_id": "step4", "duration_ms": 500, "error": "Test error"}
            ],
            "total_duration_ms": 5000,
            "adaptations_applied": [
                {"adaptation": "timeout_increase"}
            ]
        }
        
        metrics = await workflow_engine._calculate_workflow_metrics(execution_log)
        
        assert isinstance(metrics, dict)
        assert metrics["success_rate"] == 3/4  # 3 completed, 1 failed
        assert metrics["total_steps"] == 4
        assert metrics["completed_steps"] == 3
        assert metrics["failed_steps"] == 1
        assert metrics["avg_step_duration_ms"] == (1000 + 1500 + 800) / 3
        assert metrics["total_duration_ms"] == 5000
        assert metrics["adaptations_count"] == 1


class TestWorkflowStep:
    """Test cases for Workflow Step."""
    
    def test_workflow_step_creation(self):
        """Test workflow step creation."""
        step = WorkflowStep(
            step_id="test_step",
            name="Test Step",
            description="A test step",
            step_type="validation",
            configuration={"param1": "value1"},
            dependencies=["dep1", "dep2"],
            conditions={"condition1": True},
            retry_policy={"max_retries": 3, "delay_seconds": 5},
            timeout_seconds=120
        )
        
        assert step.step_id == "test_step"
        assert step.name == "Test Step"
        assert step.description == "A test step"
        assert step.step_type == "validation"
        assert step.configuration["param1"] == "value1"
        assert step.dependencies == ["dep1", "dep2"]
        assert step.conditions["condition1"] is True
        assert step.retry_policy["max_retries"] == 3
        assert step.timeout_seconds == 120
    
    def test_is_ready_to_execute(self):
        """Test step execution readiness check."""
        step = WorkflowStep(
            step_id="test_step",
            name="Test Step", 
            description="A test step",
            step_type="validation",
            configuration={},
            dependencies=["step1", "step2"],
            conditions={},
            retry_policy={},
            timeout_seconds=60
        )
        
        # Should not be ready if dependencies not completed
        assert not step.is_ready_to_execute([])
        assert not step.is_ready_to_execute(["step1"])
        
        # Should be ready if all dependencies completed
        assert step.is_ready_to_execute(["step1", "step2"])
        assert step.is_ready_to_execute(["step1", "step2", "step3"])  # Extra completed steps OK
    
    def test_no_dependencies(self):
        """Test step with no dependencies."""
        step = WorkflowStep(
            step_id="initial_step",
            name="Initial Step",
            description="Step with no dependencies",
            step_type="initialization",
            configuration={},
            dependencies=[],  # No dependencies
            conditions={},
            retry_policy={},
            timeout_seconds=30
        )
        
        # Should always be ready to execute
        assert step.is_ready_to_execute([])
        assert step.is_ready_to_execute(["other_step"])


class TestAdaptiveWorkflow:
    """Test cases for Adaptive Workflow."""
    
    def test_adaptive_workflow_creation(self):
        """Test adaptive workflow creation."""
        steps = [
            WorkflowStep(
                step_id="step1",
                name="Step 1",
                description="First step",
                step_type="validation",
                configuration={},
                dependencies=[],
                conditions={},
                retry_policy={},
                timeout_seconds=60
            )
        ]
        
        workflow = AdaptiveWorkflow(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="A test workflow",
            steps=steps,
            adaptation_rules=[{"rule": "test_rule"}],
            success_criteria={"min_success_rate": 0.95},
            failure_handling={"max_retries": 3},
            performance_targets={"target_throughput": 100}
        )
        
        assert workflow.workflow_id == "test_workflow"
        assert workflow.name == "Test Workflow"
        assert workflow.description == "A test workflow"
        assert len(workflow.steps) == 1
        assert workflow.steps[0].step_id == "step1"
        assert len(workflow.adaptation_rules) == 1
        assert workflow.success_criteria["min_success_rate"] == 0.95
        assert workflow.failure_handling["max_retries"] == 3
        assert workflow.performance_targets["target_throughput"] == 100
        assert workflow.created_at is not None


class TestDefaultWorkflow:
    """Test cases for default HIPAA workflow."""
    
    def test_create_default_hipaa_workflow(self):
        """Test default HIPAA workflow creation."""
        workflow = create_default_hipaa_workflow()
        
        assert isinstance(workflow, AdaptiveWorkflow)
        assert workflow.workflow_id == "default_hipaa_processing"
        assert workflow.name == "Default HIPAA Compliance Processing"
        assert len(workflow.steps) > 0
        assert len(workflow.adaptation_rules) > 0
        assert len(workflow.success_criteria) > 0
        assert len(workflow.failure_handling) > 0
        assert len(workflow.performance_targets) > 0
        
        # Check that steps are properly ordered with dependencies
        step_ids = [step.step_id for step in workflow.steps]
        assert "document_validation" in step_ids
        assert "phi_detection" in step_ids
        assert "phi_redaction" in step_ids
        assert "compliance_analysis" in step_ids
        assert "quality_assurance" in step_ids
        
        # Check dependency chain
        phi_detection_step = next(s for s in workflow.steps if s.step_id == "phi_detection")
        assert "document_validation" in phi_detection_step.dependencies
        
        phi_redaction_step = next(s for s in workflow.steps if s.step_id == "phi_redaction") 
        assert "phi_detection" in phi_redaction_step.dependencies


class TestIntegrationScenarios:
    """Integration test scenarios for intelligent automation."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_intelligent_processing(self):
        """Test complete intelligent processing pipeline."""
        decision_engine, workflow_engine = await initialize_intelligent_automation()
        
        # Create processing context
        context = AutomationContext(
            document_metadata={
                "size_bytes": 75000,
                "type": "medical_record",
                "complexity_score": 0.6
            },
            processing_history=[],
            system_metrics={
                "cpu_usage": 0.4,
                "memory_usage": 0.3,
                "processing_queue_length": 25
            },
            user_preferences={"quality_over_speed": True},
            compliance_requirements={"level": "strict"}
        )
        
        # Test decision making
        path_decision = await decision_engine.make_decision(
            DecisionType.PROCESSING_PATH,
            context
        )
        
        assert path_decision.success is True
        assert path_decision.decision_value in ["fast_track", "standard", "intensive", "adaptive"]
        
        # Test workflow execution with decision
        input_data = {"document_content": "Test medical record content"}
        workflow_result = await workflow_engine.execute_workflow(
            "default_hipaa_processing",
            context,
            input_data
        )
        
        assert workflow_result["success"] is True
        assert "execution_log" in workflow_result
        
        # Should have made optimization decisions
        execution_log = workflow_result["execution_log"]
        assert len(execution_log.get("decisions_made", [])) > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_learning_cycle(self):
        """Test adaptive learning across multiple processing cycles."""
        decision_engine, workflow_engine = await initialize_intelligent_automation()
        
        # Simulate multiple processing cycles with feedback
        contexts = [
            AutomationContext(
                document_metadata={"size_bytes": 10000, "complexity_score": 0.2},
                processing_history=[],
                system_metrics={"cpu_usage": 0.3},
                user_preferences={},
                compliance_requirements={"level": "standard"}
            ),
            AutomationContext(
                document_metadata={"size_bytes": 100000, "complexity_score": 0.8},
                processing_history=[],
                system_metrics={"cpu_usage": 0.7},
                user_preferences={},
                compliance_requirements={"level": "strict"}
            ),
            AutomationContext(
                document_metadata={"size_bytes": 50000, "complexity_score": 0.5},
                processing_history=[],
                system_metrics={"cpu_usage": 0.5},
                user_preferences={},
                compliance_requirements={"level": "standard"}
            )
        ]
        
        initial_metrics = decision_engine.performance_metrics.copy()
        
        for i, context in enumerate(contexts):
            # Make decision
            decision = await decision_engine.make_decision(
                DecisionType.PROCESSING_PATH,
                context
            )
            
            # Simulate execution outcome
            outcome = {
                "success": i % 2 == 0,  # Alternate success/failure
                "processing_time_ms": 5000 + i * 1000,
                "quality_score": 0.9 - i * 0.1
            }
            
            # Learn from outcome
            decision_engine.learn_from_outcome(decision, outcome)
        
        # Performance metrics should be updated
        assert len(decision_engine.performance_metrics) > len(initial_metrics)
        assert "success_rate" in decision_engine.performance_metrics
        assert "decision_count" in decision_engine.performance_metrics
        assert decision_engine.performance_metrics["decision_count"] == len(contexts)
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self):
        """Test concurrent execution of multiple workflows."""
        decision_engine, workflow_engine = await initialize_intelligent_automation()
        
        contexts = [
            AutomationContext(
                document_metadata={"size_bytes": 20000, "type": "clinical_note"},
                processing_history=[],
                system_metrics={"cpu_usage": 0.4},
                user_preferences={},
                compliance_requirements={}
            )
            for _ in range(3)
        ]
        
        input_data = {"document": f"Test document {i}"} for i in range(3)
        
        # Create concurrent execution tasks
        tasks = [
            workflow_engine.execute_workflow(
                "default_hipaa_processing",
                context,
                data
            )
            for context, data in zip(contexts, input_data)
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert result["success"] is True
            assert "execution_log" in result
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        decision_engine, workflow_engine = await initialize_intelligent_automation()
        
        # Test with context that might cause issues
        problematic_context = AutomationContext(
            document_metadata={
                "size_bytes": 0,  # Empty document
                "type": "unknown",
                "complexity_score": 1.0  # Maximum complexity
            },
            processing_history=[],
            system_metrics={
                "cpu_usage": 0.95,  # Near capacity
                "memory_usage": 0.9,
                "processing_queue_length": 500  # Very long queue
            },
            user_preferences={},
            compliance_requirements={"level": "strict"}
        )
        
        # Decision making should still work
        decision = await decision_engine.make_decision(
            DecisionType.RESOURCE_ALLOCATION,
            problematic_context
        )
        
        assert decision.success is True
        # Should allocate minimal resources under high load
        allocation = decision.decision_value
        assert allocation["cpu_cores"] <= 2
        assert allocation["priority"] == "low"
        
        # Workflow execution should handle the challenging conditions
        try:
            result = await workflow_engine.execute_workflow(
                "default_hipaa_processing",
                problematic_context,
                {"document": ""}
            )
            # Even if it fails, it should fail gracefully
            assert "execution_log" in result
        except Exception as e:
            # Should be a controlled failure, not a crash
            assert isinstance(e, (ValueError, RuntimeError))


@pytest.mark.asyncio
async def test_initialize_intelligent_automation():
    """Test intelligent automation initialization."""
    config = {
        "decision_engine": {
            "learning_rate": 0.15,
            "confidence_threshold": 0.85
        },
        "workflow_engine": {
            "max_concurrent_workflows": 10,
            "performance_monitoring": True
        }
    }
    
    decision_engine, workflow_engine = await initialize_intelligent_automation(config)
    
    # Check decision engine
    assert isinstance(decision_engine, IntelligentProcessingDecisionEngine)
    assert decision_engine.learning_rate == 0.15
    assert decision_engine.confidence_threshold == 0.85
    
    # Check workflow engine
    assert isinstance(workflow_engine, AdaptiveWorkflowEngine)
    assert workflow_engine.decision_engine == decision_engine
    assert "default_hipaa_processing" in workflow_engine.workflows
    
    # Test that default workflow is properly configured
    default_workflow = workflow_engine.workflows["default_hipaa_processing"]
    assert isinstance(default_workflow, AdaptiveWorkflow)
    assert len(default_workflow.steps) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])